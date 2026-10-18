#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "metr_late"
    / "data.csv"
)

TABLE_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_measurement_boundary.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_late_measurement_boundary.txt"
)


def time_effect_to_productivity_gain(time_effect):
    """
    Convert relative time effect to equivalent productivity effect.

    Example:
        -10% time
        => 1 / 0.90 - 1
        => +11.11% productivity
    """
    return (
        1.0
        / (1.0 + time_effect)
        - 1.0
    )


def prepare_base(df):
    """
    Replicate the common preprocessing used by METR Late.

    1. Missing post-review time -> 0
    2. totalImplementationTime = initial + post-review
    3. drop ditched issues
    """

    work = df.copy()

    work["postReviewImplementationTime"] = (
        work["postReviewImplementationTime"]
        .fillna(0)
    )

    work["totalImplementationTime"] = (
        work["initialImplementationTime"]
        + work["postReviewImplementationTime"]
    )

    work = work.loc[
        work["status"] != "ditched"
    ].copy()

    return work


def filter_initial_sample(df):
    """
    Replicate METR's Initial Impl Started sample.
    """

    return df.loc[
        df["initialImplementationTime"].notna()
        & (df["initialImplementationTime"] != 0)
        & df["estimatedTimeWithoutAI"].notna()
        & (df["estimatedTimeWithoutAI"] > 0)
    ].copy()


def filter_total_sample(df):
    """
    Replicate METR's Total Time Merged sample.
    """

    return df.loc[
        (df["status"] == "complete")
        & df["totalImplementationTime"].notna()
        & (df["totalImplementationTime"] > 0)
        & df["estimatedTimeWithoutAI"].notna()
        & (df["estimatedTimeWithoutAI"] > 0)
    ].copy()


def run_regression(df, outcome):
    """
    METR-style HC3 regression:

        log(outcome)
        ~ ai_treatment
        + log(estimatedTimeWithoutAI)
    """

    work = df.copy()

    work = work.loc[
        work[outcome].notna()
        & (work[outcome] > 0)
        & work["estimatedTimeWithoutAI"].notna()
        & (work["estimatedTimeWithoutAI"] > 0)
    ].copy()

    work["ai_treatment"] = (
        work["treatmentAssignment"]
        == "AI-allowed"
    ).astype(int)

    work["log_outcome"] = np.log(
        work[outcome]
    )

    work["log_predicted_no_ai"] = np.log(
        work["estimatedTimeWithoutAI"]
    )

    model = smf.ols(
        "log_outcome ~ "
        "ai_treatment + log_predicted_no_ai",
        data=work
    )

    fit = model.fit(
        cov_type="HC3"
    )

    beta = fit.params["ai_treatment"]
    se = fit.bse["ai_treatment"]

    z = norm.ppf(0.975)

    low_beta = beta - z * se
    high_beta = beta + z * se

    time_effect = (
        np.exp(beta) - 1.0
    )

    time_low = (
        np.exp(low_beta) - 1.0
    )

    time_high = (
        np.exp(high_beta) - 1.0
    )

    productivity_effect = (
        time_effect_to_productivity_gain(
            time_effect
        )
    )

    # Important:
    # productivity transform reverses CI direction.
    productivity_low = (
        time_effect_to_productivity_gain(
            time_high
        )
    )

    productivity_high = (
        time_effect_to_productivity_gain(
            time_low
        )
    )

    return {
        "n": len(work),

        "n_developers": (
            work["developerId"]
            .nunique()
        ),

        "beta_log": beta,

        "time_effect": time_effect,

        "time_ci_low": time_low,

        "time_ci_high": time_high,

        "productivity_effect": (
            productivity_effect
        ),

        "productivity_ci_low": (
            productivity_low
        ),

        "productivity_ci_high": (
            productivity_high
        ),

        "p_value": (
            fit.pvalues["ai_treatment"]
        ),

        "r_squared": (
            fit.rsquared
        ),
    }


def descriptive_post_review(df):
    """
    Post-review is heavily zero-inflated, so we do NOT use it
    as a primary log-regression outcome.

    Report descriptive values only.
    """

    rows = []

    for treatment, label in [
        ("AI-disallowed", "Control"),
        ("AI-allowed", "AI"),
    ]:

        x = (
            df.loc[
                df["treatmentAssignment"]
                == treatment,
                "postReviewImplementationTime"
            ]
            .dropna()
        )

        rows.append(
            {
                "group": label,
                "n": len(x),
                "mean": x.mean(),
                "median": x.median(),
                "p75": x.quantile(0.75),
                "p90": x.quantile(0.90),
                "max": x.max(),
                "zero_pct": (
                    100.0
                    * (x == 0).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def analyze_cohort(
    base,
    cohort_name,
    cohort_filter=None
):
    """
    Run initial and total boundary models for one cohort.
    """

    if cohort_filter is None:
        cohort = base.copy()
    else:
        cohort = base.loc[
            cohort_filter(base)
        ].copy()

    initial_sample = (
        filter_initial_sample(
            cohort
        )
    )

    total_sample = (
        filter_total_sample(
            cohort
        )
    )

    initial = run_regression(
        initial_sample,
        "initialImplementationTime"
    )

    total = run_regression(
        total_sample,
        "totalImplementationTime"
    )

    # ---------------------------------------------------------
    # Boundary shift in TIME space
    #
    # Positive shift:
    # effect becomes less negative / more positive
    # when moving to total delivery time.
    #
    # Example:
    # initial = -24%
    # total   = -18%
    # shift   = +6 pp
    #
    # => some apparent time saving disappears.
    # ---------------------------------------------------------

    time_boundary_shift = (
        100.0
        * (
            total["time_effect"]
            - initial["time_effect"]
        )
    )

    # ---------------------------------------------------------
    # Boundary shift in PRODUCTIVITY space
    #
    # Negative shift:
    # productivity gain decreases after widening boundary.
    # ---------------------------------------------------------

    productivity_boundary_shift = (
        100.0
        * (
            total["productivity_effect"]
            - initial["productivity_effect"]
        )
    )

    return {
        "cohort": cohort_name,

        # Initial
        "initial_n": (
            initial["n"]
        ),

        "initial_n_developers": (
            initial["n_developers"]
        ),

        "initial_time_effect_pct": (
            100.0
            * initial["time_effect"]
        ),

        "initial_time_ci_low_pct": (
            100.0
            * initial["time_ci_low"]
        ),

        "initial_time_ci_high_pct": (
            100.0
            * initial["time_ci_high"]
        ),

        "initial_productivity_effect_pct": (
            100.0
            * initial["productivity_effect"]
        ),

        "initial_productivity_ci_low_pct": (
            100.0
            * initial["productivity_ci_low"]
        ),

        "initial_productivity_ci_high_pct": (
            100.0
            * initial["productivity_ci_high"]
        ),

        "initial_p_value": (
            initial["p_value"]
        ),

        "initial_r_squared": (
            initial["r_squared"]
        ),

        # Total
        "total_n": (
            total["n"]
        ),

        "total_n_developers": (
            total["n_developers"]
        ),

        "total_time_effect_pct": (
            100.0
            * total["time_effect"]
        ),

        "total_time_ci_low_pct": (
            100.0
            * total["time_ci_low"]
        ),

        "total_time_ci_high_pct": (
            100.0
            * total["time_ci_high"]
        ),

        "total_productivity_effect_pct": (
            100.0
            * total["productivity_effect"]
        ),

        "total_productivity_ci_low_pct": (
            100.0
            * total["productivity_ci_low"]
        ),

        "total_productivity_ci_high_pct": (
            100.0
            * total["productivity_ci_high"]
        ),

        "total_p_value": (
            total["p_value"]
        ),

        "total_r_squared": (
            total["r_squared"]
        ),

        # Boundary shifts
        "time_boundary_shift_pp": (
            time_boundary_shift
        ),

        "productivity_boundary_shift_pp": (
            productivity_boundary_shift
        ),
    }


def main():

    raw = pd.read_csv(
        INPUT
    )

    base = prepare_base(
        raw
    )

    rows = []

    rows.append(
        analyze_cohort(
            base,
            "All developers",
        )
    )

    rows.append(
        analyze_cohort(
            base,
            "Panel developers",
            lambda x: (
                x["isPanelDeveloper"]
            ),
        )
    )

    rows.append(
        analyze_cohort(
            base,
            "New developers",
            lambda x: (
                ~x["isPanelDeveloper"]
            ),
        )
    )

    results = pd.DataFrame(
        rows
    )

    # =========================================================
    # Descriptive post-review analysis
    # =========================================================

    complete_sample = (
        filter_total_sample(
            base
        )
    )

    post_all = (
        descriptive_post_review(
            complete_sample
        )
    )

    post_all[
        "cohort"
    ] = "All developers"

    post_panel = (
        descriptive_post_review(
            complete_sample.loc[
                complete_sample[
                    "isPanelDeveloper"
                ]
            ]
        )
    )

    post_panel[
        "cohort"
    ] = "Panel developers"

    post_new = (
        descriptive_post_review(
            complete_sample.loc[
                ~complete_sample[
                    "isPanelDeveloper"
                ]
            ]
        )
    )

    post_new[
        "cohort"
    ] = "New developers"

    post_review = pd.concat(
        [
            post_all,
            post_panel,
            post_new,
        ],
        ignore_index=True
    )

    # =========================================================
    # Save
    # =========================================================

    TABLE_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        TABLE_OUT,
        index=False
    )

    post_review.to_csv(
        TABLE_OUT.with_name(
            "metr_late_post_review_descriptives.csv"
        ),
        index=False
    )

    # =========================================================
    # Human-readable report
    # =========================================================

    lines = []

    lines.append(
        "=" * 108
    )

    lines.append(
        "METR LATE-2025 — "
        "MEASUREMENT BOUNDARY ANALYSIS"
    )

    lines.append(
        "=" * 108
    )

    for _, row in results.iterrows():

        lines.append("")
        lines.append(
            row["cohort"].upper()
        )

        lines.append(
            "-" * 108
        )

        lines.append(
            "INITIAL IMPLEMENTATION"
        )

        lines.append(
            f"N={int(row['initial_n'])}, "
            f"developers="
            f"{int(row['initial_n_developers'])}"
        )

        lines.append(
            "Time effect: "
            f"{row['initial_time_effect_pct']:+.2f}% "
            f"[{row['initial_time_ci_low_pct']:+.2f}%, "
            f"{row['initial_time_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Equivalent productivity effect: "
            f"{row['initial_productivity_effect_pct']:+.2f}% "
            f"[{row['initial_productivity_ci_low_pct']:+.2f}%, "
            f"{row['initial_productivity_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            f"p={row['initial_p_value']:.5f}, "
            f"R2={row['initial_r_squared']:.3f}"
        )

        lines.append("")
        lines.append(
            "INITIAL + POST-REVIEW"
        )

        lines.append(
            f"N={int(row['total_n'])}, "
            f"developers="
            f"{int(row['total_n_developers'])}"
        )

        lines.append(
            "Time effect: "
            f"{row['total_time_effect_pct']:+.2f}% "
            f"[{row['total_time_ci_low_pct']:+.2f}%, "
            f"{row['total_time_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Equivalent productivity effect: "
            f"{row['total_productivity_effect_pct']:+.2f}% "
            f"[{row['total_productivity_ci_low_pct']:+.2f}%, "
            f"{row['total_productivity_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            f"p={row['total_p_value']:.5f}, "
            f"R2={row['total_r_squared']:.3f}"
        )

        lines.append("")
        lines.append(
            "BOUNDARY SHIFT"
        )

        lines.append(
            "Time-effect shift "
            "(total - initial): "
            f"{row['time_boundary_shift_pp']:+.2f} pp"
        )

        lines.append(
            "Productivity-effect shift "
            "(total - initial): "
            f"{row['productivity_boundary_shift_pp']:+.2f} pp"
        )

    # =========================================================
    # Post-review descriptives
    # =========================================================

    lines.append("")
    lines.append(
        "=" * 108
    )

    lines.append(
        "POST-REVIEW TIME — DESCRIPTIVE ONLY"
    )

    lines.append(
        "=" * 108
    )

    for cohort in (
        post_review[
            "cohort"
        ].unique()
    ):

        lines.append("")
        lines.append(
            cohort.upper()
        )

        lines.append(
            "-" * 108
        )

        subset = (
            post_review.loc[
                post_review[
                    "cohort"
                ] == cohort
            ]
        )

        for _, row in subset.iterrows():

            lines.append(
                f"{row['group']}: "
                f"N={int(row['n'])}, "
                f"mean={row['mean']:.2f}, "
                f"median={row['median']:.2f}, "
                f"P75={row['p75']:.2f}, "
                f"P90={row['p90']:.2f}, "
                f"zero={row['zero_pct']:.1f}%, "
                f"max={row['max']:.2f}"
            )

    # =========================================================
    # Compact table
    # =========================================================

    lines.append("")
    lines.append(
        "COMPACT BOUNDARY COMPARISON"
    )

    lines.append(
        "-" * 108
    )

    compact_cols = [
        "cohort",
        "initial_time_effect_pct",
        "total_time_effect_pct",
        "time_boundary_shift_pp",
        "initial_productivity_effect_pct",
        "total_productivity_effect_pct",
        "productivity_boundary_shift_pp",
    ]

    lines.append(
        results[
            compact_cols
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            )
        )
    )

    # =========================================================
    # Interpretation notes
    # =========================================================

    lines.append("")
    lines.append(
        "INTERPRETATION NOTES"
    )

    lines.append(
        "-" * 108
    )

    lines.append(
        "The initial and total outcomes reproduce METR Late's "
        "official preprocessing and HC3 log-linear regression."
    )

    lines.append(
        "The initial and total regressions use different eligible "
        "samples, matching the original analysis."
    )

    lines.append(
        "Therefore, the boundary shift is a descriptive difference "
        "between estimands and should not be interpreted as a "
        "within-identical-sample causal decomposition."
    )

    lines.append(
        "A positive time-effect boundary shift means the estimated "
        "AI time saving becomes smaller when the boundary expands "
        "from initial implementation to total implementation."
    )

    lines.append(
        "A negative productivity-effect boundary shift means the "
        "estimated productivity benefit becomes smaller under the "
        "broader boundary."
    )

    lines.append(
        "Post-review time is reported descriptively because the "
        "distribution contains many zero values and is strongly "
        "right-skewed."
    )

    lines.append(
        "Panel and new developers are reported separately because "
        "cohort composition differs in METR Late."
    )

    # =========================================================
    # Write log
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
        f"Boundary table written to: "
        f"{TABLE_OUT}"
    )

    print(
        "Post-review descriptives written to: "
        f"{TABLE_OUT.with_name('metr_late_post_review_descriptives.csv')}"
    )

    print(
        f"Log written to: "
        f"{LOG_OUT}"
    )


if __name__ == "__main__":
    main()
