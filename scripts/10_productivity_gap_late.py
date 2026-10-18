#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "metr_late" / "data.csv"

SUMMARY_OUT = (
    ROOT / "results" / "tables"
    / "metr_late_productivity_gap_summary.csv"
)

TASK_OUT = (
    ROOT / "results" / "tables"
    / "metr_late_productivity_gap_tasks.csv"
)

LOG_OUT = (
    ROOT / "results" / "logs"
    / "metr_late_productivity_gap.txt"
)


def time_effect_to_productivity_gain(time_effect):
    """
    Convert a relative time effect to productivity space.

    Example:
        time effect = -0.10  (10% less time)
        productivity gain = 1/0.90 - 1 = +11.11%
    """
    return (1.0 / (1.0 + time_effect)) - 1.0


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    rng = np.random.default_rng(seed)

    means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )
        means[i] = sample.mean()

    low, high = np.percentile(
        means,
        [2.5, 97.5]
    )

    return values.mean(), low, high


def prepare_official_total_sample(df):
    """
    Reproduce METR Late preprocessing for the
    'Total Time Merged' regression.
    """
    work = df.copy()

    # Official script fills missing post-review time with zero.
    work["postReviewImplementationTime"] = (
        work["postReviewImplementationTime"]
        .fillna(0)
    )

    work["totalImplementationTime"] = (
        work["initialImplementationTime"]
        + work["postReviewImplementationTime"]
    )

    # Official script drops ditched issues first.
    work = work[
        work["status"] != "ditched"
    ].copy()

    # Official total-time regression eligibility.
    work = work[
        (work["status"] == "complete")
        & work["totalImplementationTime"].notna()
        & (work["totalImplementationTime"] > 0)
        & work["estimatedTimeWithoutAI"].notna()
        & (work["estimatedTimeWithoutAI"] > 0)
    ].copy()

    return work


def run_official_regression(df):
    """
    Replicate the METR Late HC3 regression.
    """
    work = df.copy()

    work["ai_treatment"] = (
        work["treatmentAssignment"]
        == "AI-allowed"
    ).astype(int)

    work["log_total_time"] = np.log(
        work["totalImplementationTime"]
    )

    work["log_predicted_no_ai"] = np.log(
        work["estimatedTimeWithoutAI"]
    )

    model = smf.ols(
        "log_total_time ~ "
        "ai_treatment + log_predicted_no_ai",
        data=work
    )

    result = model.fit(
        cov_type="HC3"
    )

    beta = result.params["ai_treatment"]
    se = result.bse["ai_treatment"]

    z = norm.ppf(0.975)

    low_beta = beta - z * se
    high_beta = beta + z * se

    time_effect = np.exp(beta) - 1.0
    low = np.exp(low_beta) - 1.0
    high = np.exp(high_beta) - 1.0

    productivity_gain = (
        time_effect_to_productivity_gain(
            time_effect
        )
    )

    return {
        "n": len(work),
        "n_developers": work[
            "developerId"
        ].nunique(),
        "time_effect": time_effect,
        "time_ci_low": low,
        "time_ci_high": high,
        "productivity_gain": productivity_gain,
        "p_value": result.pvalues[
            "ai_treatment"
        ],
    }


def add_pre_task_expected_gain(df):
    """
    Productivity-space expected gain:

        P_AI / P_noAI - 1
        = T_noAI / T_AI - 1

    This is preferable to 1 - T_AI/T_noAI when
    comparing directly with productivity effects.
    """
    df = df.copy()

    valid = (
        (df["estimatedTimeWithAI"] > 0)
        & (df["estimatedTimeWithoutAI"] > 0)
    )

    df["pre_expected_productivity_gain"] = np.nan

    df.loc[
        valid,
        "pre_expected_productivity_gain"
    ] = (
        df.loc[
            valid,
            "estimatedTimeWithoutAI"
        ]
        / df.loc[
            valid,
            "estimatedTimeWithAI"
        ]
        - 1.0
    )

    return df


def add_post_task_perceived_gain(df):
    """
    Derive perceived AI productivity gain after completing
    the task using estimatedTimeOtherTreatment.

    AI-allowed:
        observed AI time = totalImplementationTime
        estimated no-AI time = estimatedTimeOtherTreatment

        perceived productivity gain =
            T_noAI / T_AI - 1

    AI-disallowed:
        observed no-AI time = totalImplementationTime
        estimated AI time = estimatedTimeOtherTreatment

        perceived productivity gain =
            T_noAI / T_AI - 1
    """
    df = df.copy()

    df["post_perceived_productivity_gain"] = np.nan

    valid = (
        df["estimatedTimeOtherTreatment"].notna()
        & (df["estimatedTimeOtherTreatment"] > 0)
        & df["totalImplementationTime"].notna()
        & (df["totalImplementationTime"] > 0)
    )

    ai_mask = (
        valid
        & (
            df["treatmentAssignment"]
            == "AI-allowed"
        )
    )

    control_mask = (
        valid
        & (
            df["treatmentAssignment"]
            == "AI-disallowed"
        )
    )

    # AI condition:
    # estimated no-AI / observed AI - 1
    df.loc[
        ai_mask,
        "post_perceived_productivity_gain"
    ] = (
        df.loc[
            ai_mask,
            "estimatedTimeOtherTreatment"
        ]
        / df.loc[
            ai_mask,
            "totalImplementationTime"
        ]
        - 1.0
    )

    # No-AI condition:
    # observed no-AI / estimated AI - 1
    df.loc[
        control_mask,
        "post_perceived_productivity_gain"
    ] = (
        df.loc[
            control_mask,
            "totalImplementationTime"
        ]
        / df.loc[
            control_mask,
            "estimatedTimeOtherTreatment"
        ]
        - 1.0
    )

    return df


def analyze_group(df, label):
    result = run_official_regression(df)

    actual_gain_pct = (
        100.0
        * result["productivity_gain"]
    )

    pre = (
        100.0
        * df[
            "pre_expected_productivity_gain"
        ].dropna()
    )

    post = (
        100.0
        * df[
            "post_perceived_productivity_gain"
        ].dropna()
    )

    pre_mean, pre_low, pre_high = (
        bootstrap_mean_ci(pre)
    )

    if len(post) > 0:
        post_mean, post_low, post_high = (
            bootstrap_mean_ci(post)
        )
        post_median = post.median()
    else:
        post_mean = np.nan
        post_low = np.nan
        post_high = np.nan
        post_median = np.nan

    return {
        "group": label,

        "n": result["n"],
        "n_developers": (
            result["n_developers"]
        ),

        "observed_time_effect_pct": (
            100.0
            * result["time_effect"]
        ),

        "observed_time_ci_low_pct": (
            100.0
            * result["time_ci_low"]
        ),

        "observed_time_ci_high_pct": (
            100.0
            * result["time_ci_high"]
        ),

        "observed_productivity_gain_pct": (
            actual_gain_pct
        ),

        "pre_expected_gain_mean_pct": (
            pre_mean
        ),

        "pre_expected_gain_median_pct": (
            pre.median()
        ),

        "pre_expected_ci_low_pct": (
            pre_low
        ),

        "pre_expected_ci_high_pct": (
            pre_high
        ),

        "pre_productivity_gap_pct": (
            pre_mean - actual_gain_pct
        ),

        "n_post_counterfactual": (
            len(post)
        ),

        "post_perceived_gain_mean_pct": (
            post_mean
        ),

        "post_perceived_gain_median_pct": (
            post_median
        ),

        "post_perceived_ci_low_pct": (
            post_low
        ),

        "post_perceived_ci_high_pct": (
            post_high
        ),

        "post_productivity_gap_pct": (
            post_mean - actual_gain_pct
            if np.isfinite(post_mean)
            else np.nan
        ),

        "regression_p_value": (
            result["p_value"]
        ),
    }


def main():
    raw = pd.read_csv(INPUT)

    df = prepare_official_total_sample(
        raw
    )

    df = add_pre_task_expected_gain(
        df
    )

    df = add_post_task_perceived_gain(
        df
    )

    # ---------------------------------------------------------
    # Analyze all / panel / new separately
    # ---------------------------------------------------------

    rows = []

    rows.append(
        analyze_group(
            df,
            "All developers"
        )
    )

    rows.append(
        analyze_group(
            df[
                df["isPanelDeveloper"]
            ].copy(),
            "Panel developers"
        )
    )

    rows.append(
        analyze_group(
            df[
                ~df["isPanelDeveloper"]
            ].copy(),
            "New developers"
        )
    )

    summary = pd.DataFrame(rows)

    SUMMARY_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Save task-level derived data
    # ---------------------------------------------------------

    keep_cols = [
        "id",
        "developerId",
        "treatmentAssignment",
        "isPanelDeveloper",
        "priorExperience",
        "estimatedTimeWithAI",
        "estimatedTimeWithoutAI",
        "totalImplementationTime",
        "estimatedTimeOtherTreatment",
        "perceivedEffort",
        "pre_expected_productivity_gain",
        "post_perceived_productivity_gain",
    ]

    df[keep_cols].to_csv(
        TASK_OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------

    lines = []

    lines.append("=" * 100)
    lines.append(
        "METR LATE-2025 — AI PRODUCTIVITY GAP ANALYSIS"
    )
    lines.append("=" * 100)

    for _, row in summary.iterrows():

        lines.append("")
        lines.append(
            row["group"].upper()
        )
        lines.append("-" * 100)

        lines.append(
            f"N issues: {int(row['n'])}, "
            f"N developers: "
            f"{int(row['n_developers'])}"
        )

        lines.append(
            "Observed adjusted time effect: "
            f"{row['observed_time_effect_pct']:+.2f}% "
            f"[{row['observed_time_ci_low_pct']:+.2f}%, "
            f"{row['observed_time_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Equivalent observed productivity gain: "
            f"{row['observed_productivity_gain_pct']:+.2f}%"
        )

        lines.append("")
        lines.append(
            "PRE-TASK EXPECTATION"
        )

        lines.append(
            "Mean expected productivity gain: "
            f"{row['pre_expected_gain_mean_pct']:+.2f}%"
        )

        lines.append(
            "Median expected productivity gain: "
            f"{row['pre_expected_gain_median_pct']:+.2f}%"
        )

        lines.append(
            "Bootstrap 95% CI: "
            f"[{row['pre_expected_ci_low_pct']:+.2f}%, "
            f"{row['pre_expected_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Pre-task AI Productivity Gap: "
            f"{row['pre_productivity_gap_pct']:+.2f} pp"
        )

        lines.append("")
        lines.append(
            "POST-TASK PERCEIVED COUNTERFACTUAL"
        )

        lines.append(
            "N with post-task counterfactual: "
            f"{int(row['n_post_counterfactual'])}"
        )

        lines.append(
            "Mean perceived productivity gain: "
            f"{row['post_perceived_gain_mean_pct']:+.2f}%"
        )

        lines.append(
            "Median perceived productivity gain: "
            f"{row['post_perceived_gain_median_pct']:+.2f}%"
        )

        lines.append(
            "Bootstrap 95% CI: "
            f"[{row['post_perceived_ci_low_pct']:+.2f}%, "
            f"{row['post_perceived_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Post-task AI Productivity Gap: "
            f"{row['post_productivity_gap_pct']:+.2f} pp"
        )

    lines.append("")
    lines.append("INTERPRETATION NOTES")
    lines.append("-" * 100)

    lines.append(
        "Pre-task expectations are measured before randomization."
    )

    lines.append(
        "Post-task perceived gains use the developer's estimate "
        "of the unobserved alternative treatment after completing "
        "the assigned treatment."
    )

    lines.append(
        "Observed productivity gain is derived from the adjusted "
        "METR treatment effect and is therefore group-level, not "
        "task-level causal evidence."
    )

    lines.append(
        "Late-2025 results are subject to the selection concerns "
        "identified by METR and should not be interpreted as a "
        "clean temporal replication of Early-2025."
    )

    LOG_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_OUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))

    print()
    print(
        f"Summary written to: {SUMMARY_OUT}"
    )

    print(
        f"Task-level data written to: {TASK_OUT}"
    )

    print(
        f"Log written to: {LOG_OUT}"
    )


if __name__ == "__main__":
    main()
