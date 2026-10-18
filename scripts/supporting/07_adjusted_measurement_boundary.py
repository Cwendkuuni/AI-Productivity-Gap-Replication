#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "metr_early" / "data_complete.csv"

TABLE_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_adjusted_measurement_boundary.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_early_adjusted_measurement_boundary.txt"
)


def impute_post_review_by_treatment(df):
    """
    Replicate METR's treatment-group mean imputation
    for missing post-review implementation times.
    """
    df = df.copy()

    means = (
        df.loc[df["post_review_implementation_time"].notna()]
        .groupby("ai_treatment")["post_review_implementation_time"]
        .mean()
        .to_dict()
    )

    missing = df["post_review_implementation_time"].isna()

    df.loc[
        missing,
        "post_review_implementation_time"
    ] = (
        df.loc[missing, "ai_treatment"]
        .map(means)
    )

    return df


def run_model(df, outcome, label):
    """
    Replicates METR's regression specification:

        log(outcome) ~ ai_treatment
                       + log(predicted_time_no_ai)

    Reports three standard-error specifications.
    """

    work = df.copy()

    work = work.loc[
        (work[outcome].notna())
        & (work[outcome] > 0)
        & (work["predicted_time_no_ai"] > 0)
    ].copy()

    work["log_outcome"] = np.log(work[outcome])

    work["log_predicted_time_no_ai"] = np.log(
        work["predicted_time_no_ai"]
    )

    formula = (
        "log_outcome ~ "
        "ai_treatment + log_predicted_time_no_ai"
    )

    specs = [
        (
            "Homoskedastic",
            "nonrobust",
            None,
        ),
        (
            "Robust_HC3",
            "HC3",
            None,
        ),
        (
            "Clustered_By_Dev",
            "cluster",
            {"groups": work["dev_id"]},
        ),
    ]

    rows = []

    for name, cov_type, cov_kwds in specs:

        model = smf.ols(
            formula,
            data=work,
            missing="drop"
        )

        result = model.fit(
            cov_type=cov_type,
            cov_kwds=cov_kwds
        )

        beta = result.params["ai_treatment"]
        se = result.bse["ai_treatment"]

        effect = np.exp(beta) - 1

        z = norm.ppf(0.975)

        low_beta = beta - z * se
        high_beta = beta + z * se

        low = np.exp(low_beta) - 1
        high = np.exp(high_beta) - 1

        rows.append(
            {
                "boundary": label,
                "outcome": outcome,
                "stderr": name,
                "n": len(work),
                "beta": beta,
                "effect": effect,
                "effect_pct": effect * 100,
                "ci_low": low,
                "ci_high": high,
                "ci_low_pct": low * 100,
                "ci_high_pct": high * 100,
                "p_value": result.pvalues["ai_treatment"],
                "r_squared": result.rsquared,
            }
        )

    return rows


def main():

    df = pd.read_csv(INPUT)

    # ---------------------------------------------------------
    # Boundary 1:
    # initial implementation only
    # ---------------------------------------------------------

    results = []

    results += run_model(
        df=df,
        outcome="initial_implementation_time",
        label="Initial implementation",
    )

    # ---------------------------------------------------------
    # Boundary 2:
    # complete-case total implementation
    # ---------------------------------------------------------

    complete = df.loc[
        df["post_review_implementation_time"].notna()
    ].copy()

    complete["total_implementation_time_complete_case"] = (
        complete["initial_implementation_time"]
        + complete["post_review_implementation_time"]
    )

    results += run_model(
        df=complete,
        outcome="total_implementation_time_complete_case",
        label="Initial + post-review (complete cases)",
    )

    # ---------------------------------------------------------
    # Boundary 3:
    # METR-style imputed total implementation
    # ---------------------------------------------------------

    imputed = impute_post_review_by_treatment(df)

    imputed["total_implementation_time"] = (
        imputed["initial_implementation_time"]
        + imputed["post_review_implementation_time"]
    )

    results += run_model(
        df=imputed,
        outcome="total_implementation_time",
        label="Initial + post-review (METR imputation)",
    )

    result_df = pd.DataFrame(results)

    TABLE_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        TABLE_OUT,
        index=False
    )

    # ---------------------------------------------------------
    # Human-readable report
    # ---------------------------------------------------------

    lines = []

    lines.append("=" * 92)
    lines.append(
        "METR EARLY-2025 — "
        "ADJUSTED MEASUREMENT BOUNDARY ANALYSIS"
    )
    lines.append("=" * 92)

    for boundary in result_df["boundary"].unique():

        lines.append("")
        lines.append(boundary.upper())
        lines.append("-" * 92)

        subset = result_df.loc[
            result_df["boundary"] == boundary
        ]

        for _, row in subset.iterrows():

            lines.append(
                f"{row['stderr']}: "
                f"N={int(row['n'])}, "
                f"effect={row['effect_pct']:+.2f}%, "
                f"95% CI="
                f"[{row['ci_low_pct']:+.2f}%, "
                f"{row['ci_high_pct']:+.2f}%], "
                f"p={row['p_value']:.5f}, "
                f"R2={row['r_squared']:.3f}"
            )

    # ---------------------------------------------------------
    # Boundary shift using the same HC3 specification
    # ---------------------------------------------------------

    hc3 = result_df.loc[
        result_df["stderr"] == "Robust_HC3"
    ].copy()

    initial_effect = hc3.loc[
        hc3["boundary"] == "Initial implementation",
        "effect_pct",
    ].iloc[0]

    total_effect = hc3.loc[
        hc3["boundary"]
        == "Initial + post-review (METR imputation)",
        "effect_pct",
    ].iloc[0]

    shift = total_effect - initial_effect

    lines.append("")
    lines.append("BOUNDARY SHIFT")
    lines.append("-" * 92)

    lines.append(
        f"Initial implementation effect: "
        f"{initial_effect:+.2f}%"
    )

    lines.append(
        f"Total implementation effect: "
        f"{total_effect:+.2f}%"
    )

    lines.append(
        f"Change after widening measurement boundary: "
        f"{shift:+.2f} percentage points"
    )

    LOG_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_OUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))

    print()
    print(f"Table written to: {TABLE_OUT}")
    print(f"Log written to:   {LOG_OUT}")


if __name__ == "__main__":
    main()
