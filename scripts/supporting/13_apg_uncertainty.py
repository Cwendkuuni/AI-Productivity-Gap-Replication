#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EARLY_TASKS = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_productivity_gap_tasks.csv"
)

LATE_TASKS = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_productivity_gap_tasks.csv"
)

OUT_TABLE = (
    ROOT
    / "results"
    / "tables"
    / "apg_uncertainty.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "apg_uncertainty.txt"
)


N_BOOT = 20000
SEED = 42


# ============================================================
# Observed METR time effects and 95% CIs
#
# Values already reproduced in our previous analyses.
# These are in TIME space.
# ============================================================

OBSERVED_EFFECTS = {
    "Early-2025": {
        "time_effect": 0.1884,
        "ci_low": 0.0130,
        "ci_high": 0.3941,
    },

    "Late-2025 — All": {
        "time_effect": -0.0582,
        "ci_low": -0.1585,
        "ci_high": 0.0540,
    },

    "Late-2025 — Panel": {
        "time_effect": -0.1810,
        "ci_low": -0.3836,
        "ci_high": 0.0880,
    },

    "Late-2025 — New": {
        "time_effect": -0.0356,
        "ci_low": -0.1466,
        "ci_high": 0.0899,
    },
}


def time_effect_to_log_beta(time_effect):
    """
    METR reports:
        exp(beta) - 1 = time_effect

    Therefore:
        beta = log(1 + time_effect)
    """
    return np.log1p(time_effect)


def log_beta_to_productivity_gain(beta):
    """
    If time multiplier = exp(beta),
    productivity multiplier = exp(-beta).

    Productivity gain:
        exp(-beta) - 1
    """
    return np.exp(-beta) - 1.0


def infer_beta_se(effect, ci_low, ci_high):
    """
    Recover approximate SE of beta from reported 95% CI.

    CI is given after exp(beta)-1 transformation.

    Convert endpoints back to beta space and use:
        SE ≈ (upper_beta - lower_beta) / (2 * 1.96)
    """

    beta = np.log1p(effect)

    beta_low = np.log1p(ci_low)
    beta_high = np.log1p(ci_high)

    se = (
        beta_high - beta_low
    ) / (2.0 * 1.96)

    return beta, se


def bootstrap_geometric_gain(
    log_ratios,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Bootstrap geometric productivity gain:
        exp(mean(log_ratio)) - 1

    Returns bootstrap distribution.
    """

    x = np.asarray(
        log_ratios,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    rng = np.random.default_rng(
        seed
    )

    out = np.empty(n_boot)

    for i in range(n_boot):

        sample = rng.choice(
            x,
            size=len(x),
            replace=True
        )

        out[i] = (
            np.exp(
                sample.mean()
            )
            - 1.0
        )

    return out


def simulate_observed_productivity(
    effect,
    ci_low,
    ci_high,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Monte Carlo uncertainty for observed treatment effect.

    Draw beta from an approximate normal distribution
    on the original log-model scale, then convert to
    productivity gain.
    """

    beta, se = infer_beta_se(
        effect,
        ci_low,
        ci_high,
    )

    rng = np.random.default_rng(
        seed
    )

    beta_draws = rng.normal(
        loc=beta,
        scale=se,
        size=n_boot,
    )

    productivity_draws = (
        log_beta_to_productivity_gain(
            beta_draws
        )
    )

    return productivity_draws


def summarize_distribution(x):
    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "ci_low": np.percentile(
            x,
            2.5
        ),
        "ci_high": np.percentile(
            x,
            97.5
        ),
    }


def compute_apg(
    label,
    subjective_log_ratios,
    observed_effect,
    observed_ci_low,
    observed_ci_high,
    subjective_type,
    seed_offset=0,
):
    """
    Combine uncertainty from:
      1. subjective expected/perceived gain
      2. observed treatment effect

    APG = subjective gain - observed gain
    """

    subjective_draws = (
        bootstrap_geometric_gain(
            subjective_log_ratios,
            seed=SEED + seed_offset,
        )
    )

    observed_draws = (
        simulate_observed_productivity(
            observed_effect,
            observed_ci_low,
            observed_ci_high,
            seed=SEED + 1000 + seed_offset,
        )
    )

    apg_draws = (
        subjective_draws
        - observed_draws
    )

    subjective_summary = (
        summarize_distribution(
            subjective_draws
        )
    )

    observed_summary = (
        summarize_distribution(
            observed_draws
        )
    )

    apg_summary = (
        summarize_distribution(
            apg_draws
        )
    )

    probability_positive = (
        np.mean(
            apg_draws > 0
        )
    )

    return {
        "study_cohort": label,
        "subjective_type": subjective_type,

        "subjective_gain_mean_pct": (
            100.0
            * subjective_summary[
                "mean"
            ]
        ),

        "subjective_gain_median_pct": (
            100.0
            * subjective_summary[
                "median"
            ]
        ),

        "subjective_ci_low_pct": (
            100.0
            * subjective_summary[
                "ci_low"
            ]
        ),

        "subjective_ci_high_pct": (
            100.0
            * subjective_summary[
                "ci_high"
            ]
        ),

        "observed_gain_mean_pct": (
            100.0
            * observed_summary[
                "mean"
            ]
        ),

        "observed_gain_median_pct": (
            100.0
            * observed_summary[
                "median"
            ]
        ),

        "observed_ci_low_pct": (
            100.0
            * observed_summary[
                "ci_low"
            ]
        ),

        "observed_ci_high_pct": (
            100.0
            * observed_summary[
                "ci_high"
            ]
        ),

        "apg_mean_pct": (
            100.0
            * apg_summary[
                "mean"
            ]
        ),

        "apg_median_pct": (
            100.0
            * apg_summary[
                "median"
            ]
        ),

        "apg_ci_low_pct": (
            100.0
            * apg_summary[
                "ci_low"
            ]
        ),

        "apg_ci_high_pct": (
            100.0
            * apg_summary[
                "ci_high"
            ]
        ),

        "prob_apg_positive": (
            probability_positive
        ),
    }


def main():

    early = pd.read_csv(
        EARLY_TASKS
    )

    late = pd.read_csv(
        LATE_TASKS
    )

    rows = []

    # =========================================================
    # EARLY
    # =========================================================

    early_log_ratio = (
        early[
            "expected_log_ratio"
        ]
        .dropna()
        .to_numpy()
    )

    e = OBSERVED_EFFECTS[
        "Early-2025"
    ]

    rows.append(
        compute_apg(
            label="Early-2025",
            subjective_log_ratios=(
                early_log_ratio
            ),
            observed_effect=(
                e["time_effect"]
            ),
            observed_ci_low=(
                e["ci_low"]
            ),
            observed_ci_high=(
                e["ci_high"]
            ),
            subjective_type=(
                "Pre-task expected"
            ),
            seed_offset=0,
        )
    )

    # =========================================================
    # LATE helper
    # =========================================================

    late[
        "pre_log_ratio"
    ] = np.log(
        late[
            "estimatedTimeWithoutAI"
        ]
        / late[
            "estimatedTimeWithAI"
        ]
    )

    # Reconstruct post-task ratio.
    late[
        "post_ratio"
    ] = np.nan

    valid_post = (
        late[
            "estimatedTimeOtherTreatment"
        ].notna()
        & (
            late[
                "estimatedTimeOtherTreatment"
            ] > 0
        )
        & late[
            "totalImplementationTime"
        ].notna()
        & (
            late[
                "totalImplementationTime"
            ] > 0
        )
    )

    ai_mask = (
        valid_post
        & (
            late[
                "treatmentAssignment"
            ] == "AI-allowed"
        )
    )

    control_mask = (
        valid_post
        & (
            late[
                "treatmentAssignment"
            ] == "AI-disallowed"
        )
    )

    late.loc[
        ai_mask,
        "post_ratio"
    ] = (
        late.loc[
            ai_mask,
            "estimatedTimeOtherTreatment"
        ]
        / late.loc[
            ai_mask,
            "totalImplementationTime"
        ]
    )

    late.loc[
        control_mask,
        "post_ratio"
    ] = (
        late.loc[
            control_mask,
            "totalImplementationTime"
        ]
        / late.loc[
            control_mask,
            "estimatedTimeOtherTreatment"
        ]
    )

    late[
        "post_log_ratio"
    ] = np.log(
        late[
            "post_ratio"
        ]
    )

    groups = [
        (
            "Late-2025 — All",
            late,
            10,
        ),

        (
            "Late-2025 — Panel",
            late.loc[
                late[
                    "isPanelDeveloper"
                ]
            ].copy(),
            20,
        ),

        (
            "Late-2025 — New",
            late.loc[
                ~late[
                    "isPanelDeveloper"
                ]
            ].copy(),
            30,
        ),
    ]

    # =========================================================
    # LATE pre + post
    # =========================================================

    for label, df, seed_offset in groups:

        obs = OBSERVED_EFFECTS[
            label
        ]

        pre_log = (
            df[
                "pre_log_ratio"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
            .to_numpy()
        )

        rows.append(
            compute_apg(
                label=label,
                subjective_log_ratios=(
                    pre_log
                ),
                observed_effect=(
                    obs[
                        "time_effect"
                    ]
                ),
                observed_ci_low=(
                    obs[
                        "ci_low"
                    ]
                ),
                observed_ci_high=(
                    obs[
                        "ci_high"
                    ]
                ),
                subjective_type=(
                    "Pre-task expected"
                ),
                seed_offset=(
                    seed_offset
                ),
            )
        )

        post_log = (
            df[
                "post_log_ratio"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
            .to_numpy()
        )

        rows.append(
            compute_apg(
                label=label,
                subjective_log_ratios=(
                    post_log
                ),
                observed_effect=(
                    obs[
                        "time_effect"
                    ]
                ),
                observed_ci_low=(
                    obs[
                        "ci_low"
                    ]
                ),
                observed_ci_high=(
                    obs[
                        "ci_high"
                    ]
                ),
                subjective_type=(
                    "Post-task perceived"
                ),
                seed_offset=(
                    seed_offset + 1
                ),
            )
        )

    results = pd.DataFrame(
        rows
    )

    # =========================================================
    # Save
    # =========================================================

    OUT_TABLE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUT_TABLE,
        index=False
    )

    # =========================================================
    # Human-readable report
    # =========================================================

    lines = []

    lines.append(
        "=" * 112
    )

    lines.append(
        "AI PRODUCTIVITY GAP — "
        "UNCERTAINTY PROPAGATION"
    )

    lines.append(
        "=" * 112
    )

    for _, row in results.iterrows():

        lines.append("")
        lines.append(
            f"{row['study_cohort']} — "
            f"{row['subjective_type']}"
        )

        lines.append(
            "-" * 112
        )

        lines.append(
            "Subjective productivity gain: "
            f"{row['subjective_gain_median_pct']:+.2f}% "
            f"[{row['subjective_ci_low_pct']:+.2f}%, "
            f"{row['subjective_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Observed productivity gain: "
            f"{row['observed_gain_median_pct']:+.2f}% "
            f"[{row['observed_ci_low_pct']:+.2f}%, "
            f"{row['observed_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "AI Productivity Gap: "
            f"{row['apg_median_pct']:+.2f} pp "
            f"[{row['apg_ci_low_pct']:+.2f}, "
            f"{row['apg_ci_high_pct']:+.2f}]"
        )

        lines.append(
            "Monte-Carlo probability APG > 0: "
            f"{100.0 * row['prob_apg_positive']:.1f}%"
        )

    lines.append("")
    lines.append(
        "INTERPRETATION NOTES"
    )

    lines.append(
        "-" * 112
    )

    lines.append(
        "Expected and perceived gains are bootstrapped "
        "from task-level log productivity ratios."
    )

    lines.append(
        "Observed treatment-effect uncertainty is approximated "
        "on the original METR log-regression coefficient scale "
        "using the reproduced 95% confidence intervals."
    )

    lines.append(
        "Each Monte-Carlo draw combines one subjective-gain "
        "draw and one observed-effect draw to produce an "
        "AI Productivity Gap draw."
    )

    lines.append(
        "The reported APG interval therefore reflects uncertainty "
        "from both components rather than treating the observed "
        "treatment effect as fixed."
    )

    lines.append(
        "Late-2025 cohort comparisons remain descriptive because "
        "panel membership and study-wave composition are not randomized."
    )

    LOG_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_OUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        "\n".join(lines)
    )

    print()

    print(
        f"Results written to: "
        f"{OUT_TABLE}"
    )

    print(
        f"Log written to: "
        f"{LOG_OUT}"
    )


if __name__ == "__main__":
    main()
