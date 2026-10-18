import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


def drop_ditched_issues(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[df["status"] != "ditched"].copy()
    print(f"Dropped {n_before - len(df)} ditched issues")
    return df


def filter_to_complete_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to complete issues with valid total implementation time and predicted time.

    Note that some issues have some invalid total implementation time or predicted time, due to fat-finger
    errors that we have not corrected for. These are excluded from the regression by default.
    """
    return df[
        (df["status"] == "complete")
        & (df["totalImplementationTime"].notna())
        & (df["totalImplementationTime"] > 0)
        & (df["estimatedTimeWithoutAI"].notna())
        & (df["estimatedTimeWithoutAI"] > 0)
    ].copy()


def filter_to_started_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to issues with initial implementation time set and valid predicted time.

    Note that some issues have some invalid total implementation time or predicted time, due to fat-finger
    errors that we have not corrected for. These are excluded from the regression by default.
    """
    return df[
        (df["initialImplementationTime"].notna())
        & (df["initialImplementationTime"] != 0)
        & (df["estimatedTimeWithoutAI"].notna())
        & (df["estimatedTimeWithoutAI"] > 0)
    ].copy()


def zero_fill_post_review_time(df: pd.DataFrame) -> pd.DataFrame:
    df["postReviewImplementationTime"] = df["postReviewImplementationTime"].fillna(0)
    return df


def add_total_implementation_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df["totalImplementationTime"] = (
        df["initialImplementationTime"] + df["postReviewImplementationTime"]
    )
    return df


def run_regression(df: pd.DataFrame) -> tuple[float, tuple[float, float]]:
    """
    Estimate the speedup of developers when using AI via OLS regression
    with HC3 (heteroskedasticity-robust) standard errors.
    """
    df["ai_treatment"] = (df["treatmentAssignment"] == "AI-allowed").astype(int)

    df["log_total_implementation_time"] = np.log(df["totalImplementationTime"])
    df["log_predicted_time_no_ai"] = np.log(df["estimatedTimeWithoutAI"])

    regression_formula = (
        "log_total_implementation_time ~ ai_treatment + log_predicted_time_no_ai"
    )

    model = smf.ols(regression_formula, data=df)
    res = model.fit(cov_type="HC3")

    beta = res.params["ai_treatment"]
    se_beta = res.bse["ai_treatment"]

    estimand = np.exp(beta) - 1

    z = norm.ppf(0.975)

    beta_ci = (
        float(beta - z * se_beta),
        float(beta + z * se_beta),
    )

    ci = (
        float(np.exp(beta_ci[0]) - 1),
        float(np.exp(beta_ci[1]) - 1),
    )

    return estimand, ci


def run_regression_on_subset(
    df: pd.DataFrame, label: str
) -> tuple[float, tuple[float, float]]:
    estimand, ci = run_regression(df)
    n_issues = len(df)
    n_developers = df["developerId"].nunique()
    print(f"=== {label} ===")
    print(f"N issues: {n_issues}, N developers: {n_developers}")
    print(f"Regression calculated speedup of: {round(estimand, 3)}")
    print(f"95% Confidence Interval: ({round(ci[0], 3)}, {round(ci[1], 3)})")
    print()
    return estimand, ci


def run_all_regressions(df: pd.DataFrame, suffix: str) -> None:
    run_regression_on_subset(df.copy(), f"All Developers ({suffix})")

    df_panel = df[df["isPanelDeveloper"] == True].copy()
    run_regression_on_subset(df_panel, f"Panel Developers ({suffix})")

    df_new = df[df["isPanelDeveloper"] == False].copy()
    run_regression_on_subset(df_new, f"New Developers ({suffix})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_data)

    df = zero_fill_post_review_time(df)
    df = add_total_implementation_time_column(df)
    df = drop_ditched_issues(df)

    # Total Time (Merged): complete issues with valid time data
    df_complete = filter_to_complete_issues(df)
    run_all_regressions(df_complete, "Total Time Merged")

    # Initial Implementation Started: started issues with valid predicted time
    df_started = filter_to_started_issues(df)
    run_all_regressions(df_started, "Initial Impl Started")


if __name__ == "__main__":
    main()
