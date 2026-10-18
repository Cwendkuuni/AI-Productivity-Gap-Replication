#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "metr_early" / "data_complete.csv"

TABLE_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_estimation_bias.csv"
)

DEV_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_estimation_bias_by_developer.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_early_estimation_bias.txt"
)


def bootstrap_diff_mean(ai, control, n_boot=10000, seed=42):
    """
    Bootstrap difference in mean prediction error:
        mean(AI) - mean(Control)

    Positive values mean stronger underestimation
    in the AI condition.
    """
    rng = np.random.default_rng(seed)

    ai = np.asarray(ai, dtype=float)
    control = np.asarray(control, dtype=float)

    ai = ai[~np.isnan(ai)]
    control = control[~np.isnan(control)]

    diffs = np.empty(n_boot)

    for i in range(n_boot):
        ai_s = rng.choice(ai, size=len(ai), replace=True)
        ctl_s = rng.choice(control, size=len(control), replace=True)

        diffs[i] = ai_s.mean() - ctl_s.mean()

    low, high = np.percentile(diffs, [2.5, 97.5])

    return diffs.mean(), low, high


def main():
    df = pd.read_csv(INPUT).copy()

    # ---------------------------------------------------------
    # 1. Predicted time for actually assigned treatment
    # ---------------------------------------------------------

    df["predicted_time_assigned"] = np.where(
        df["ai_treatment"] == 1,
        df["predicted_time_ai_allowed"],
        df["predicted_time_no_ai"],
    )

    # ---------------------------------------------------------
    # 2. Prediction error
    #
    # ratio > 1  => actual took longer than predicted
    # ratio < 1  => actual took less time than predicted
    # ---------------------------------------------------------

    df["prediction_error_ratio"] = (
        df["initial_implementation_time"]
        / df["predicted_time_assigned"]
    )

    df["prediction_error_pct"] = (
        100.0
        * (
            df["prediction_error_ratio"]
            - 1.0
        )
    )

    # log-scale calibration error
    # positive = actual > predicted
    # negative = actual < predicted
    df["log_prediction_error"] = (
        np.log(df["initial_implementation_time"])
        - np.log(df["predicted_time_assigned"])
    )

    # ---------------------------------------------------------
    # 3. Descriptive AI vs control
    # ---------------------------------------------------------

    control = df.loc[
        df["ai_treatment"] == 0,
        "prediction_error_pct"
    ]

    ai = df.loc[
        df["ai_treatment"] == 1,
        "prediction_error_pct"
    ]

    control_log = df.loc[
        df["ai_treatment"] == 0,
        "log_prediction_error"
    ]

    ai_log = df.loc[
        df["ai_treatment"] == 1,
        "log_prediction_error"
    ]

    raw_bias = ai.mean() - control.mean()

    boot_mean, boot_low, boot_high = bootstrap_diff_mean(
        ai,
        control
    )

    mw = stats.mannwhitneyu(
        ai,
        control,
        alternative="two-sided"
    )

    # ---------------------------------------------------------
    # 4. Regression on log calibration error
    #
    # Core model:
    # log(actual/predicted_assigned) ~ ai_treatment
    #
    # If beta_AI > 0:
    # AI condition is systematically more underestimated.
    # ---------------------------------------------------------

    model = smf.ols(
        "log_prediction_error ~ ai_treatment",
        data=df
    )

    fit_homo = model.fit()

    fit_hc3 = model.fit(
        cov_type="HC3"
    )

    fit_cluster = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df["dev_id"]
        }
    )

    regression_rows = []

    for name, fit in [
        ("Homoskedastic", fit_homo),
        ("Robust_HC3", fit_hc3),
        ("Clustered_By_Dev", fit_cluster),
    ]:

        beta = fit.params["ai_treatment"]
        se = fit.bse["ai_treatment"]

        ci = fit.conf_int().loc["ai_treatment"]

        # Convert log difference to multiplicative percentage difference
        effect = np.exp(beta) - 1.0
        ci_low = np.exp(ci.iloc[0]) - 1.0
        ci_high = np.exp(ci.iloc[1]) - 1.0

        regression_rows.append(
            {
                "model": name,
                "beta_log": beta,
                "effect_ratio": effect,
                "effect_pct": 100.0 * effect,
                "ci_low_pct": 100.0 * ci_low,
                "ci_high_pct": 100.0 * ci_high,
                "p_value": fit.pvalues["ai_treatment"],
                "r_squared": fit.rsquared,
            }
        )

    regression_df = pd.DataFrame(regression_rows)

    # ---------------------------------------------------------
    # 5. Adjusted model controlling for task difficulty
    #
    # We use predicted no-AI time as a difficulty proxy.
    # ---------------------------------------------------------

    df["log_predicted_no_ai"] = np.log(
        df["predicted_time_no_ai"]
    )

    adjusted_model = smf.ols(
        "log_prediction_error ~ ai_treatment + log_predicted_no_ai",
        data=df
    )

    adjusted_hc3 = adjusted_model.fit(
        cov_type="HC3"
    )

    adjusted_cluster = adjusted_model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": df["dev_id"]
        }
    )

    adjusted_rows = []

    for name, fit in [
        ("Adjusted_HC3", adjusted_hc3),
        ("Adjusted_Clustered_By_Dev", adjusted_cluster),
    ]:

        beta = fit.params["ai_treatment"]
        ci = fit.conf_int().loc["ai_treatment"]

        effect = np.exp(beta) - 1.0
        ci_low = np.exp(ci.iloc[0]) - 1.0
        ci_high = np.exp(ci.iloc[1]) - 1.0

        adjusted_rows.append(
            {
                "model": name,
                "beta_log": beta,
                "effect_ratio": effect,
                "effect_pct": 100.0 * effect,
                "ci_low_pct": 100.0 * ci_low,
                "ci_high_pct": 100.0 * ci_high,
                "p_value": fit.pvalues["ai_treatment"],
                "r_squared": fit.rsquared,
            }
        )

    adjusted_df = pd.DataFrame(adjusted_rows)

    # ---------------------------------------------------------
    # 6. Developer-level analysis
    # ---------------------------------------------------------

    by_dev = (
        df.groupby("dev_id")
        .agg(
            n_tasks=("issue_id", "count"),
            n_ai_tasks=(
                "ai_treatment",
                lambda x: int((x == 1).sum())
            ),
            n_control_tasks=(
                "ai_treatment",
                lambda x: int((x == 0).sum())
            ),
            mean_prediction_error_pct=(
                "prediction_error_pct",
                "mean"
            ),
            median_prediction_error_pct=(
                "prediction_error_pct",
                "median"
            ),
        )
        .reset_index()
    )

    ai_by_dev = (
        df[df["ai_treatment"] == 1]
        .groupby("dev_id")["prediction_error_pct"]
        .mean()
        .rename("ai_error_mean_pct")
    )

    control_by_dev = (
        df[df["ai_treatment"] == 0]
        .groupby("dev_id")["prediction_error_pct"]
        .mean()
        .rename("control_error_mean_pct")
    )

    by_dev = (
        by_dev
        .merge(ai_by_dev, on="dev_id", how="left")
        .merge(control_by_dev, on="dev_id", how="left")
    )

    by_dev["ai_estimation_bias_pct"] = (
        by_dev["ai_error_mean_pct"]
        - by_dev["control_error_mean_pct"]
    )

    # ---------------------------------------------------------
    # 7. Save tables
    # ---------------------------------------------------------

    summary_rows = [
        {
            "analysis": "Raw mean prediction error",
            "control_value": control.mean(),
            "ai_value": ai.mean(),
            "difference_ai_minus_control": raw_bias,
            "ci_low": boot_low,
            "ci_high": boot_high,
            "p_value": mw.pvalue,
        },
        {
            "analysis": "Median prediction error",
            "control_value": control.median(),
            "ai_value": ai.median(),
            "difference_ai_minus_control": (
                ai.median() - control.median()
            ),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    TABLE_OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        TABLE_OUT.with_suffix(".xlsx")
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="descriptives",
            index=False
        )

        regression_df.to_excel(
            writer,
            sheet_name="regression",
            index=False
        )

        adjusted_df.to_excel(
            writer,
            sheet_name="adjusted_regression",
            index=False
        )

    by_dev.to_csv(
        DEV_OUT,
        index=False
    )

    # ---------------------------------------------------------
    # 8. Human-readable report
    # ---------------------------------------------------------

    lines = []

    lines.append("=" * 92)
    lines.append(
        "METR EARLY-2025 — AI ESTIMATION BIAS ANALYSIS"
    )
    lines.append("=" * 92)

    lines.append("")
    lines.append("DESCRIPTIVE PREDICTION ERROR")
    lines.append("-" * 92)

    lines.append(
        f"Control mean prediction error: "
        f"{control.mean():+.2f}%"
    )

    lines.append(
        f"AI mean prediction error: "
        f"{ai.mean():+.2f}%"
    )

    lines.append(
        f"AI Estimation Bias "
        f"(AI - Control): "
        f"{raw_bias:+.2f} percentage points"
    )

    lines.append(
        f"Bootstrap 95% CI: "
        f"[{boot_low:+.2f}, {boot_high:+.2f}]"
    )

    lines.append(
        f"Mann-Whitney p-value: "
        f"{mw.pvalue:.6f}"
    )

    lines.append("")
    lines.append("MEDIANS")
    lines.append("-" * 92)

    lines.append(
        f"Control median prediction error: "
        f"{control.median():+.2f}%"
    )

    lines.append(
        f"AI median prediction error: "
        f"{ai.median():+.2f}%"
    )

    lines.append(
        f"Median difference: "
        f"{ai.median() - control.median():+.2f} "
        f"percentage points"
    )

    lines.append("")
    lines.append("LOG-CALIBRATION REGRESSION")
    lines.append("-" * 92)

    for _, row in regression_df.iterrows():
        lines.append(
            f"{row['model']}: "
            f"AI estimation error multiplier="
            f"{row['effect_pct']:+.2f}%, "
            f"95% CI="
            f"[{row['ci_low_pct']:+.2f}%, "
            f"{row['ci_high_pct']:+.2f}%], "
            f"p={row['p_value']:.5f}"
        )

    lines.append("")
    lines.append("ADJUSTED FOR PREDICTED TASK DIFFICULTY")
    lines.append("-" * 92)

    for _, row in adjusted_df.iterrows():
        lines.append(
            f"{row['model']}: "
            f"AI estimation error multiplier="
            f"{row['effect_pct']:+.2f}%, "
            f"95% CI="
            f"[{row['ci_low_pct']:+.2f}%, "
            f"{row['ci_high_pct']:+.2f}%], "
            f"p={row['p_value']:.5f}"
        )

    lines.append("")
    lines.append("DEVELOPER-LEVEL AI ESTIMATION BIAS")
    lines.append("-" * 92)

    valid_dev_bias = by_dev[
        "ai_estimation_bias_pct"
    ].dropna()

    lines.append(
        f"Developers with both AI and control tasks: "
        f"{len(valid_dev_bias)}"
    )

    if len(valid_dev_bias) > 0:
        lines.append(
            f"Median developer-level bias: "
            f"{valid_dev_bias.median():+.2f} pp"
        )

        lines.append(
            f"Min developer-level bias: "
            f"{valid_dev_bias.min():+.2f} pp"
        )

        lines.append(
            f"Max developer-level bias: "
            f"{valid_dev_bias.max():+.2f} pp"
        )

        lines.append(
            f"Developers with positive AI estimation bias: "
            f"{(valid_dev_bias > 0).sum()}/"
            f"{len(valid_dev_bias)}"
        )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 92)

    lines.append(
        "Positive prediction error means the task took longer "
        "than the developer predicted."
    )

    lines.append(
        "A positive AI Estimation Bias means prediction errors "
        "are systematically larger when AI assistance is allowed."
    )

    lines.append(
        "This analysis does not claim that developers are poor "
        "estimators overall; it tests whether estimation error "
        "changes specifically under AI assistance."
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
        f"Summary workbook written to: "
        f"{TABLE_OUT.with_suffix('.xlsx')}"
    )

    print(
        f"Developer table written to: {DEV_OUT}"
    )

    print(
        f"Log written to: {LOG_OUT}"
    )


if __name__ == "__main__":
    main()


