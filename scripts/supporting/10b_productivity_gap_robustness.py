#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "metr_late"
    / "data.csv"
)

SUMMARY_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_productivity_gap_robustness.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_late_productivity_gap_robustness.txt"
)


def prepare_total_sample(df):
    """
    Reproduce the analysis-ready complete-task sample
    used for the METR Late total-time regression.
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

    # Remove ditched issues.
    work = work.loc[
        work["status"] != "ditched"
    ].copy()

    # Keep usable completed tasks.
    work = work.loc[
        (work["status"] == "complete")
        & work["totalImplementationTime"].notna()
        & (work["totalImplementationTime"] > 0)
        & work["estimatedTimeWithoutAI"].notna()
        & (work["estimatedTimeWithoutAI"] > 0)
        & work["estimatedTimeWithAI"].notna()
        & (work["estimatedTimeWithAI"] > 0)
    ].copy()

    return work


def add_pre_task_metrics(df):
    """
    Pre-task estimates are available for both hypothetical
    conditions before treatment assignment.

    ratio:
        T_noAI / T_AI

    productivity gain:
        ratio - 1

    log ratio:
        log(T_noAI / T_AI)
    """

    df = df.copy()

    df["pre_ratio"] = (
        df["estimatedTimeWithoutAI"]
        / df["estimatedTimeWithAI"]
    )

    df["pre_gain"] = (
        df["pre_ratio"] - 1.0
    )

    df["pre_log_ratio"] = np.log(
        df["pre_ratio"]
    )

    return df


def add_post_task_metrics(df):
    """
    Construct a perceived counterfactual productivity ratio
    after completing the task.

    AI-allowed:
        observed AI time = totalImplementationTime
        estimated no-AI time = estimatedTimeOtherTreatment

        ratio = estimated_noAI / observed_AI

    AI-disallowed:
        observed no-AI time = totalImplementationTime
        estimated AI time = estimatedTimeOtherTreatment

        ratio = observed_noAI / estimated_AI
    """

    df = df.copy()

    df["post_ratio"] = np.nan

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

    df.loc[
        ai_mask,
        "post_ratio"
    ] = (
        df.loc[
            ai_mask,
            "estimatedTimeOtherTreatment"
        ]
        / df.loc[
            ai_mask,
            "totalImplementationTime"
        ]
    )

    df.loc[
        control_mask,
        "post_ratio"
    ] = (
        df.loc[
            control_mask,
            "totalImplementationTime"
        ]
        / df.loc[
            control_mask,
            "estimatedTimeOtherTreatment"
        ]
    )

    df["post_gain"] = (
        df["post_ratio"] - 1.0
    )

    df["post_log_ratio"] = np.log(
        df["post_ratio"]
    )

    return df


def bootstrap_geometric_gain(
    log_values,
    n_boot=10000,
    seed=42,
):
    """
    Bootstrap the geometric/log-ratio productivity gain.

        exp(mean(log_ratio)) - 1
    """

    x = np.asarray(
        log_values,
        dtype=float
    )

    x = x[np.isfinite(x)]

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

    return point, low, high


def summarize_metric(
    df,
    gain_col,
    log_col,
    label,
    group,
):
    """
    Produce robust summaries of the productivity-gain
    distribution.
    """

    values = (
        df[gain_col]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    logs = (
        df[log_col]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(values) == 0:
        return None

    geometric_gain, geo_low, geo_high = (
        bootstrap_geometric_gain(logs)
    )

    arithmetic_mean = values.mean()
    median = values.median()

    trimmed = trim_mean(
        values.to_numpy(),
        proportiontocut=0.10
    )

    quantiles = values.quantile(
        [
            0.05,
            0.25,
            0.75,
            0.95,
        ]
    )

    return {
        "group": group,
        "measure": label,

        "n": len(values),

        "arithmetic_mean_pct": (
            100.0 * arithmetic_mean
        ),

        "median_pct": (
            100.0 * median
        ),

        "trimmed_mean_10_pct": (
            100.0 * trimmed
        ),

        "geometric_gain_pct": (
            100.0 * geometric_gain
        ),

        "geometric_ci_low_pct": (
            100.0 * geo_low
        ),

        "geometric_ci_high_pct": (
            100.0 * geo_high
        ),

        "p05_pct": (
            100.0 * quantiles.loc[0.05]
        ),

        "p25_pct": (
            100.0 * quantiles.loc[0.25]
        ),

        "p75_pct": (
            100.0 * quantiles.loc[0.75]
        ),

        "p95_pct": (
            100.0 * quantiles.loc[0.95]
        ),

        "min_pct": (
            100.0 * values.min()
        ),

        "max_pct": (
            100.0 * values.max()
        ),

        "n_gt_100pct": int(
            (values > 1.0).sum()
        ),

        "pct_gt_100pct": (
            100.0
            * (values > 1.0).mean()
        ),

        "n_gt_200pct": int(
            (values > 2.0).sum()
        ),

        "pct_gt_200pct": (
            100.0
            * (values > 2.0).mean()
        ),

        "n_gt_500pct": int(
            (values > 5.0).sum()
        ),

        "pct_gt_500pct": (
            100.0
            * (values > 5.0).mean()
        ),

        "n_negative_gain": int(
            (values < 0).sum()
        ),

        "pct_negative_gain": (
            100.0
            * (values < 0).mean()
        ),
    }


def analyze_group(df, group_name):
    rows = []

    pre = summarize_metric(
        df=df,
        gain_col="pre_gain",
        log_col="pre_log_ratio",
        label="Pre-task expected gain",
        group=group_name,
    )

    if pre is not None:
        rows.append(pre)

    post = summarize_metric(
        df=df,
        gain_col="post_gain",
        log_col="post_log_ratio",
        label="Post-task perceived gain",
        group=group_name,
    )

    if post is not None:
        rows.append(post)

    return rows


def main():

    raw = pd.read_csv(INPUT)

    df = prepare_total_sample(raw)

    df = add_pre_task_metrics(df)

    df = add_post_task_metrics(df)

    rows = []

    # ---------------------------------------------------------
    # All developers
    # ---------------------------------------------------------

    rows += analyze_group(
        df,
        "All developers"
    )

    # ---------------------------------------------------------
    # Panel developers
    # ---------------------------------------------------------

    rows += analyze_group(
        df.loc[
            df["isPanelDeveloper"]
        ].copy(),
        "Panel developers"
    )

    # ---------------------------------------------------------
    # New developers
    # ---------------------------------------------------------

    rows += analyze_group(
        df.loc[
            ~df["isPanelDeveloper"]
        ].copy(),
        "New developers"
    )

    results = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Save table
    # ---------------------------------------------------------

    SUMMARY_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        SUMMARY_OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------

    lines = []

    lines.append("=" * 104)

    lines.append(
        "METR LATE-2025 — "
        "PRODUCTIVITY GAP ROBUSTNESS ANALYSIS"
    )

    lines.append("=" * 104)

    for group in results["group"].unique():

        lines.append("")
        lines.append(
            group.upper()
        )
        lines.append("=" * 104)

        subset = results.loc[
            results["group"] == group
        ]

        for _, row in subset.iterrows():

            lines.append("")
            lines.append(
                row["measure"].upper()
            )

            lines.append("-" * 104)

            lines.append(
                f"N: {int(row['n'])}"
            )

            lines.append(
                f"Arithmetic mean: "
                f"{row['arithmetic_mean_pct']:+.2f}%"
            )

            lines.append(
                f"Median: "
                f"{row['median_pct']:+.2f}%"
            )

            lines.append(
                f"10% trimmed mean: "
                f"{row['trimmed_mean_10_pct']:+.2f}%"
            )

            lines.append(
                f"Geometric/log-ratio gain: "
                f"{row['geometric_gain_pct']:+.2f}%"
            )

            lines.append(
                "Geometric bootstrap 95% CI: "
                f"[{row['geometric_ci_low_pct']:+.2f}%, "
                f"{row['geometric_ci_high_pct']:+.2f}%]"
            )

            lines.append(
                "Distribution:"
            )

            lines.append(
                f"  P05: "
                f"{row['p05_pct']:+.2f}%"
            )

            lines.append(
                f"  P25: "
                f"{row['p25_pct']:+.2f}%"
            )

            lines.append(
                f"  P75: "
                f"{row['p75_pct']:+.2f}%"
            )

            lines.append(
                f"  P95: "
                f"{row['p95_pct']:+.2f}%"
            )

            lines.append(
                f"  Min: "
                f"{row['min_pct']:+.2f}%"
            )

            lines.append(
                f"  Max: "
                f"{row['max_pct']:+.2f}%"
            )

            lines.append(
                "Extreme positive gains:"
            )

            lines.append(
                f"  >100%: "
                f"{int(row['n_gt_100pct'])} "
                f"({row['pct_gt_100pct']:.1f}%)"
            )

            lines.append(
                f"  >200%: "
                f"{int(row['n_gt_200pct'])} "
                f"({row['pct_gt_200pct']:.1f}%)"
            )

            lines.append(
                f"  >500%: "
                f"{int(row['n_gt_500pct'])} "
                f"({row['pct_gt_500pct']:.1f}%)"
            )

            lines.append(
                "Expected/perceived slowdowns:"
            )

            lines.append(
                f"  Negative gains: "
                f"{int(row['n_negative_gain'])} "
                f"({row['pct_negative_gain']:.1f}%)"
            )

    # ---------------------------------------------------------
    # Interpretation notes
    # ---------------------------------------------------------

    lines.append("")
    lines.append("INTERPRETATION NOTES")
    lines.append("-" * 104)

    lines.append(
        "Arithmetic productivity gains are highly sensitive "
        "to extreme time ratios."
    )

    lines.append(
        "The geometric/log-ratio estimate computes "
        "exp(mean(log(T_noAI / T_AI))) - 1 and therefore "
        "summarizes multiplicative differences on the same "
        "general scale used by log-linear time models."
    )

    lines.append(
        "The median and 10% trimmed mean are included as "
        "additional robustness summaries."
    )

    lines.append(
        "Post-task counterfactual gains are subjective "
        "developer estimates and must not be interpreted "
        "as task-level causal treatment effects."
    )

    lines.append(
        "Panel and new developers are reported separately "
        "because METR Late is subject to cohort-selection "
        "differences."
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
        f"Table written to: "
        f"{SUMMARY_OUT}"
    )

    print(
        f"Log written to: "
        f"{LOG_OUT}"
    )


if __name__ == "__main__":
    main()
