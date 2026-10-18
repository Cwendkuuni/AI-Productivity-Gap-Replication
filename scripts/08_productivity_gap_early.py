#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import trim_mean


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "metr_early"
    / "data_complete.csv"
)

TABLE_TASKS = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_productivity_gap_tasks.csv"
)

TABLE_DEVS = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_productivity_gap_by_developer.csv"
)

SUMMARY_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_productivity_gap_summary.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_early_productivity_gap.txt"
)


# Reproduced METR Early adjusted time effect:
# AI-assisted tasks take approximately 18.84% more time.
METR_TIME_EFFECT = 0.1884


def bootstrap_mean_ci(
    values,
    n_boot=10000,
    seed=42,
):
    """
    Bootstrap 95% CI for the arithmetic mean.
    """
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        boot_means[i] = sample.mean()

    low, high = np.percentile(
        boot_means,
        [2.5, 97.5]
    )

    return (
        values.mean(),
        low,
        high
    )


def bootstrap_geometric_gain(
    log_values,
    n_boot=10000,
    seed=42,
):
    """
    Bootstrap 95% CI for the geometric/log-ratio
    productivity gain:

        exp(mean(log(T_noAI / T_AI))) - 1
    """

    x = np.asarray(
        log_values,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    rng = np.random.default_rng(seed)

    estimates = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True
        )

        estimates[i] = (
            np.exp(sample.mean()) - 1.0
        )

    point = (
        np.exp(x.mean()) - 1.0
    )

    low, high = np.percentile(
        estimates,
        [2.5, 97.5]
    )

    return (
        point,
        low,
        high
    )


def safe_spearman(x, y):
    """
    Spearman correlation with NaN handling.
    """

    tmp = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if len(tmp) < 3:
        return (
            np.nan,
            np.nan
        )

    rho, p = stats.spearmanr(
        tmp["x"],
        tmp["y"]
    )

    return rho, p


def time_effect_to_productivity_gain(
    time_effect
):
    """
    Convert a relative time effect into an
    equivalent productivity effect.

    Example:

        +18.84% time
        => 1 / 1.1884 - 1
        => approximately -15.85% productivity
    """

    return (
        1.0
        / (
            1.0 + time_effect
        )
    ) - 1.0


def main():

    df = pd.read_csv(
        INPUT
    ).copy()

    # =========================================================
    # 1. Legacy expected time-saving metric
    # =========================================================
    #
    # This was used in the earlier version:
    #
    #   1 - T_AI / T_noAI
    #
    # It represents expected proportional time saving,
    # not exactly productivity gain.
    #

    df[
        "legacy_expected_time_saving"
    ] = (
        1.0
        - (
            df[
                "predicted_time_ai_allowed"
            ]
            / df[
                "predicted_time_no_ai"
            ]
        )
    )

    df[
        "legacy_expected_time_saving_pct"
    ] = (
        100.0
        * df[
            "legacy_expected_time_saving"
        ]
    )

    (
        legacy_mean,
        legacy_low,
        legacy_high
    ) = bootstrap_mean_ci(
        df[
            "legacy_expected_time_saving_pct"
        ]
    )

    legacy_median = (
        df[
            "legacy_expected_time_saving_pct"
        ]
        .median()
    )

    # =========================================================
    # 2. Harmonized productivity-space expectation
    # =========================================================
    #
    # Productivity is proportional to inverse time.
    #
    # Therefore:
    #
    #   productivity ratio
    #       = T_noAI / T_AI
    #
    #   productivity gain
    #       = T_noAI / T_AI - 1
    #
    # This is directly comparable to the observed
    # productivity effect derived from METR's time model.
    #

    df[
        "expected_productivity_ratio"
    ] = (
        df[
            "predicted_time_no_ai"
        ]
        / df[
            "predicted_time_ai_allowed"
        ]
    )

    df[
        "expected_productivity_gain"
    ] = (
        df[
            "expected_productivity_ratio"
        ]
        - 1.0
    )

    df[
        "expected_productivity_gain_pct"
    ] = (
        100.0
        * df[
            "expected_productivity_gain"
        ]
    )

    df[
        "expected_log_ratio"
    ] = np.log(
        df[
            "expected_productivity_ratio"
        ]
    )

    # =========================================================
    # 3. Robust summaries of expected productivity gain
    # =========================================================

    expected_values = (
        df[
            "expected_productivity_gain"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    arithmetic_expected_gain_pct = (
        100.0
        * expected_values.mean()
    )

    median_expected_gain_pct = (
        100.0
        * expected_values.median()
    )

    trimmed_expected_gain_pct = (
        100.0
        * trim_mean(
            expected_values.to_numpy(),
            proportiontocut=0.10
        )
    )

    (
        geometric_expected_gain,
        geometric_low,
        geometric_high
    ) = bootstrap_geometric_gain(
        df[
            "expected_log_ratio"
        ]
    )

    geometric_expected_gain_pct = (
        100.0
        * geometric_expected_gain
    )

    geometric_low_pct = (
        100.0
        * geometric_low
    )

    geometric_high_pct = (
        100.0
        * geometric_high
    )

    expected_quantiles = (
        expected_values.quantile(
            [
                0.05,
                0.25,
                0.75,
                0.95,
            ]
        )
    )

    # =========================================================
    # 4. Observed METR effect
    # =========================================================

    observed_productivity_gain = (
        time_effect_to_productivity_gain(
            METR_TIME_EFFECT
        )
    )

    observed_productivity_gain_pct = (
        100.0
        * observed_productivity_gain
    )

    # =========================================================
    # 5. Harmonized AI Productivity Gap
    # =========================================================
    #
    # Primary APG:
    #
    #   geometric expected productivity gain
    #   minus
    #   observed productivity gain
    #

    harmonized_apg_pct = (
        geometric_expected_gain_pct
        - observed_productivity_gain_pct
    )

    median_apg_pct = (
        median_expected_gain_pct
        - observed_productivity_gain_pct
    )

    trimmed_apg_pct = (
        trimmed_expected_gain_pct
        - observed_productivity_gain_pct
    )

    arithmetic_apg_pct = (
        arithmetic_expected_gain_pct
        - observed_productivity_gain_pct
    )

    # =========================================================
    # 6. Prediction error for assigned treatment
    # =========================================================

    df[
        "predicted_time_for_assigned_condition"
    ] = np.where(
        df[
            "ai_treatment"
        ] == 1,
        df[
            "predicted_time_ai_allowed"
        ],
        df[
            "predicted_time_no_ai"
        ],
    )

    df[
        "prediction_error_ratio"
    ] = (
        df[
            "initial_implementation_time"
        ]
        / df[
            "predicted_time_for_assigned_condition"
        ]
    )

    df[
        "prediction_error_pct"
    ] = (
        100.0
        * (
            df[
                "prediction_error_ratio"
            ]
            - 1.0
        )
    )

    # =========================================================
    # 7. Correlations
    # =========================================================

    (
        rho_expected_vs_observed,
        p_expected_vs_observed
    ) = safe_spearman(
        df[
            "expected_productivity_gain_pct"
        ],
        df[
            "initial_implementation_time"
        ],
    )

    (
        rho_expected_vs_prediction_error,
        p_expected_vs_prediction_error
    ) = safe_spearman(
        df[
            "expected_productivity_gain_pct"
        ],
        df[
            "prediction_error_pct"
        ],
    )

    (
        rho_predicted_noai_vs_actual,
        p_predicted_noai_vs_actual
    ) = safe_spearman(
        df[
            "predicted_time_no_ai"
        ],
        df[
            "initial_implementation_time"
        ],
    )

    # =========================================================
    # 8. Treatment-level descriptives
    # =========================================================

    treatment_summary = (
        df.groupby(
            "ai_treatment"
        )
        .agg(
            n=(
                "issue_id",
                "count"
            ),

            expected_productivity_gain_mean_pct=(
                "expected_productivity_gain_pct",
                "mean"
            ),

            expected_productivity_gain_median_pct=(
                "expected_productivity_gain_pct",
                "median"
            ),

            observed_time_mean=(
                "initial_implementation_time",
                "mean"
            ),

            observed_time_median=(
                "initial_implementation_time",
                "median"
            ),

            prediction_error_mean_pct=(
                "prediction_error_pct",
                "mean"
            ),

            prediction_error_median_pct=(
                "prediction_error_pct",
                "median"
            ),
        )
        .reset_index()
    )

    # =========================================================
    # 9. Developer-level analysis
    # =========================================================

    by_dev = (
        df.groupby(
            "dev_id"
        )
        .agg(
            n_tasks=(
                "issue_id",
                "count"
            ),

            n_ai_tasks=(
                "ai_treatment",
                lambda x: int(
                    (x == 1).sum()
                )
            ),

            n_control_tasks=(
                "ai_treatment",
                lambda x: int(
                    (x == 0).sum()
                )
            ),

            expected_gain_mean_pct=(
                "expected_productivity_gain_pct",
                "mean"
            ),

            expected_gain_median_pct=(
                "expected_productivity_gain_pct",
                "median"
            ),

            mean_expected_log_ratio=(
                "expected_log_ratio",
                "mean"
            ),

            predicted_no_ai_mean=(
                "predicted_time_no_ai",
                "mean"
            ),

            predicted_ai_mean=(
                "predicted_time_ai_allowed",
                "mean"
            ),

            observed_initial_mean=(
                "initial_implementation_time",
                "mean"
            ),

            prediction_error_mean_pct=(
                "prediction_error_pct",
                "mean"
            ),
        )
        .reset_index()
    )

    by_dev[
        "expected_gain_geometric_pct"
    ] = (
        100.0
        * (
            np.exp(
                by_dev[
                    "mean_expected_log_ratio"
                ]
            )
            - 1.0
        )
    )

    by_dev[
        "productivity_gap_geometric_pct"
    ] = (
        by_dev[
            "expected_gain_geometric_pct"
        ]
        - observed_productivity_gain_pct
    )

    # =========================================================
    # 10. Save task-level data
    # =========================================================

    keep_cols = [
        "dev_id",
        "issue_id",
        "ai_treatment",

        "predicted_time_no_ai",
        "predicted_time_ai_allowed",

        "legacy_expected_time_saving",
        "legacy_expected_time_saving_pct",

        "expected_productivity_ratio",
        "expected_productivity_gain",
        "expected_productivity_gain_pct",
        "expected_log_ratio",

        "initial_implementation_time",

        "predicted_time_for_assigned_condition",
        "prediction_error_ratio",
        "prediction_error_pct",

        "prior_task_exposure_1_to_5",
        "external_resource_needs_1_to_3",
    ]

    TABLE_TASKS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df[
        keep_cols
    ].to_csv(
        TABLE_TASKS,
        index=False
    )

    by_dev.to_csv(
        TABLE_DEVS,
        index=False
    )

    # =========================================================
    # 11. Save summary table
    # =========================================================

    summary = pd.DataFrame(
        [
            {
                "n_tasks": len(df),

                "legacy_time_saving_mean_pct": (
                    legacy_mean
                ),

                "legacy_time_saving_median_pct": (
                    legacy_median
                ),

                "legacy_time_saving_ci_low_pct": (
                    legacy_low
                ),

                "legacy_time_saving_ci_high_pct": (
                    legacy_high
                ),

                "expected_productivity_arithmetic_mean_pct": (
                    arithmetic_expected_gain_pct
                ),

                "expected_productivity_median_pct": (
                    median_expected_gain_pct
                ),

                "expected_productivity_trimmed_mean_10_pct": (
                    trimmed_expected_gain_pct
                ),

                "expected_productivity_geometric_pct": (
                    geometric_expected_gain_pct
                ),

                "expected_productivity_geometric_ci_low_pct": (
                    geometric_low_pct
                ),

                "expected_productivity_geometric_ci_high_pct": (
                    geometric_high_pct
                ),

                "observed_time_effect_pct": (
                    100.0
                    * METR_TIME_EFFECT
                ),

                "observed_productivity_effect_pct": (
                    observed_productivity_gain_pct
                ),

                "apg_arithmetic_pct": (
                    arithmetic_apg_pct
                ),

                "apg_median_pct": (
                    median_apg_pct
                ),

                "apg_trimmed_mean_pct": (
                    trimmed_apg_pct
                ),

                "apg_geometric_pct": (
                    harmonized_apg_pct
                ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False
    )

    # =========================================================
    # 12. Human-readable output
    # =========================================================

    lines = []

    lines.append(
        "=" * 100
    )

    lines.append(
        "METR EARLY-2025 — "
        "HARMONIZED AI PRODUCTIVITY GAP ANALYSIS"
    )

    lines.append(
        "=" * 100
    )

    # ---------------------------------------------------------
    # Legacy metric
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "LEGACY EXPECTED TIME-SAVING MEASURE"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        f"N tasks: {len(df)}"
    )

    lines.append(
        f"Mean expected time saving: "
        f"{legacy_mean:+.2f}%"
    )

    lines.append(
        f"Bootstrap 95% CI: "
        f"[{legacy_low:+.2f}%, "
        f"{legacy_high:+.2f}%]"
    )

    lines.append(
        f"Median expected time saving: "
        f"{legacy_median:+.2f}%"
    )

    lines.append(
        "Note: this metric is retained for traceability "
        "but is not the primary productivity-space measure."
    )

    # ---------------------------------------------------------
    # Harmonized expectation
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "HARMONIZED PRODUCTIVITY-SPACE EXPECTATION"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        f"Arithmetic mean productivity gain: "
        f"{arithmetic_expected_gain_pct:+.2f}%"
    )

    lines.append(
        f"Median productivity gain: "
        f"{median_expected_gain_pct:+.2f}%"
    )

    lines.append(
        f"10% trimmed mean productivity gain: "
        f"{trimmed_expected_gain_pct:+.2f}%"
    )

    lines.append(
        f"Geometric/log-ratio expected gain: "
        f"{geometric_expected_gain_pct:+.2f}%"
    )

    lines.append(
        "Geometric bootstrap 95% CI: "
        f"[{geometric_low_pct:+.2f}%, "
        f"{geometric_high_pct:+.2f}%]"
    )

    lines.append("")
    lines.append(
        "EXPECTED-GAIN DISTRIBUTION"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        f"P05: "
        f"{100.0 * expected_quantiles.loc[0.05]:+.2f}%"
    )

    lines.append(
        f"P25: "
        f"{100.0 * expected_quantiles.loc[0.25]:+.2f}%"
    )

    lines.append(
        f"P75: "
        f"{100.0 * expected_quantiles.loc[0.75]:+.2f}%"
    )

    lines.append(
        f"P95: "
        f"{100.0 * expected_quantiles.loc[0.95]:+.2f}%"
    )

    lines.append(
        f"Min: "
        f"{100.0 * expected_values.min():+.2f}%"
    )

    lines.append(
        f"Max: "
        f"{100.0 * expected_values.max():+.2f}%"
    )

    lines.append(
        f"Expected slowdowns: "
        f"{int((expected_values < 0).sum())}/"
        f"{len(expected_values)} "
        f"({100.0 * (expected_values < 0).mean():.1f}%)"
    )

    lines.append(
        f"Expected gains >100%: "
        f"{int((expected_values > 1.0).sum())}/"
        f"{len(expected_values)} "
        f"({100.0 * (expected_values > 1.0).mean():.1f}%)"
    )

    # ---------------------------------------------------------
    # Observed effect
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "OBSERVED METR EFFECT"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        f"Reproduced adjusted time effect: "
        f"{100.0 * METR_TIME_EFFECT:+.2f}%"
    )

    lines.append(
        f"Equivalent productivity effect: "
        f"{observed_productivity_gain_pct:+.2f}%"
    )

    # ---------------------------------------------------------
    # APG
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "AI PRODUCTIVITY GAP"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        f"Arithmetic-mean APG: "
        f"{arithmetic_apg_pct:+.2f} pp"
    )

    lines.append(
        f"Median-based APG: "
        f"{median_apg_pct:+.2f} pp"
    )

    lines.append(
        f"10% trimmed-mean APG: "
        f"{trimmed_apg_pct:+.2f} pp"
    )

    lines.append(
        f"PRIMARY — geometric/log-ratio APG: "
        f"{harmonized_apg_pct:+.2f} pp"
    )

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "CORRELATIONS"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        "Expected productivity gain vs observed initial time: "
        f"rho={rho_expected_vs_observed:+.3f}, "
        f"p={p_expected_vs_observed:.5f}"
    )

    lines.append(
        "Expected productivity gain vs prediction error: "
        f"rho={rho_expected_vs_prediction_error:+.3f}, "
        f"p={p_expected_vs_prediction_error:.5f}"
    )

    lines.append(
        "Predicted no-AI time vs observed initial time: "
        f"rho={rho_predicted_noai_vs_actual:+.3f}, "
        f"p={p_predicted_noai_vs_actual:.5f}"
    )

    # ---------------------------------------------------------
    # Treatment descriptives
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "TREATMENT-LEVEL DESCRIPTIVES"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        treatment_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Developer-level range
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "DEVELOPER-LEVEL EXPECTATION RANGE"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        "Geometric expected productivity gain by developer:"
    )

    lines.append(
        f"  Min: "
        f"{by_dev['expected_gain_geometric_pct'].min():+.2f}%"
    )

    lines.append(
        f"  Median: "
        f"{by_dev['expected_gain_geometric_pct'].median():+.2f}%"
    )

    lines.append(
        f"  Max: "
        f"{by_dev['expected_gain_geometric_pct'].max():+.2f}%"
    )

    lines.append("")
    lines.append(
        "Geometric AI Productivity Gap by developer:"
    )

    lines.append(
        f"  Min: "
        f"{by_dev['productivity_gap_geometric_pct'].min():+.2f} pp"
    )

    lines.append(
        f"  Median: "
        f"{by_dev['productivity_gap_geometric_pct'].median():+.2f} pp"
    )

    lines.append(
        f"  Max: "
        f"{by_dev['productivity_gap_geometric_pct'].max():+.2f} pp"
    )

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    lines.append("")
    lines.append(
        "INTERPRETATION NOTES"
    )
    lines.append(
        "-" * 100
    )

    lines.append(
        "The legacy metric 1 - T_AI/T_noAI measures expected "
        "time saving and is retained only for compatibility "
        "with the earlier exploratory analysis."
    )

    lines.append(
        "The primary expected productivity measure is "
        "T_noAI/T_AI - 1."
    )

    lines.append(
        "The geometric/log-ratio estimate is preferred because "
        "it summarizes multiplicative time ratios and is "
        "naturally aligned with METR's log-linear treatment model."
    )

    lines.append(
        "The observed METR treatment effect is reported in "
        "time space and is transformed into productivity space "
        "using 1/(1 + time_effect) - 1."
    )

    lines.append(
        "The primary AI Productivity Gap is the difference "
        "between the geometric expected productivity gain "
        "and the observed productivity effect."
    )

    lines.append(
        "Task-level causal AI gains are not observable because "
        "each task is assigned to only one treatment."
    )

    # =========================================================
    # 13. Save log
    # =========================================================

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
        f"Summary written to: "
        f"{SUMMARY_OUT}"
    )

    print(
        f"Task-level table written to: "
        f"{TABLE_TASKS}"
    )

    print(
        f"Developer-level table written to: "
        f"{TABLE_DEVS}"
    )

    print(
        f"Log written to: "
        f"{LOG_OUT}"
    )


if __name__ == "__main__":
    main()
