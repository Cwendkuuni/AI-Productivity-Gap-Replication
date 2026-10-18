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
    / "metr_late_common_sample_boundary.csv"
)

BOOT_OUT = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_common_sample_boundary_bootstrap.csv"
)

LOG_OUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_late_common_sample_boundary.txt"
)


N_BOOT = 5000
SEED = 42


def time_to_productivity(time_effect):
    return (
        1.0 / (1.0 + time_effect)
    ) - 1.0


def prepare_data(df):
    """
    Construct one common sample on which BOTH outcomes
    are valid:

        initialImplementationTime
        totalImplementationTime
    """

    work = df.copy()

    # Match METR treatment of missing review time.
    work["postReviewImplementationTime"] = (
        work["postReviewImplementationTime"]
        .fillna(0)
    )

    work["totalImplementationTime"] = (
        work["initialImplementationTime"]
        + work["postReviewImplementationTime"]
    )

    work = work.loc[
        (work["status"] != "ditched")
        & (work["status"] == "complete")
        & work["initialImplementationTime"].notna()
        & (work["initialImplementationTime"] > 0)
        & work["totalImplementationTime"].notna()
        & (work["totalImplementationTime"] > 0)
        & work["estimatedTimeWithoutAI"].notna()
        & (work["estimatedTimeWithoutAI"] > 0)
    ].copy()

    work["ai_treatment"] = (
        work["treatmentAssignment"]
        == "AI-allowed"
    ).astype(int)

    work["log_predicted_no_ai"] = np.log(
        work["estimatedTimeWithoutAI"]
    )

    return work


def fit_effect(df, outcome, cov_type="HC3"):
    """
    Fit:
        log(outcome)
        ~ ai_treatment + log_predicted_no_ai
    """

    work = df.copy()

    work["log_outcome"] = np.log(
        work[outcome]
    )

    model = smf.ols(
        "log_outcome ~ "
        "ai_treatment + log_predicted_no_ai",
        data=work
    )

    if cov_type == "cluster":
        fit = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": work["developerId"]
            }
        )
    else:
        fit = model.fit(
            cov_type="HC3"
        )

    beta = fit.params["ai_treatment"]

    ci = fit.conf_int().loc[
        "ai_treatment"
    ]

    time_effect = (
        np.exp(beta) - 1.0
    )

    low_time = (
        np.exp(ci.iloc[0]) - 1.0
    )

    high_time = (
        np.exp(ci.iloc[1]) - 1.0
    )

    productivity = (
        time_to_productivity(
            time_effect
        )
    )

    # Monotonic decreasing transform:
    prod_low = (
        time_to_productivity(
            high_time
        )
    )

    prod_high = (
        time_to_productivity(
            low_time
        )
    )

    return {
        "beta": beta,

        "time_effect": (
            time_effect
        ),

        "time_ci_low": (
            low_time
        ),

        "time_ci_high": (
            high_time
        ),

        "productivity_effect": (
            productivity
        ),

        "productivity_ci_low": (
            prod_low
        ),

        "productivity_ci_high": (
            prod_high
        ),

        "p_value": (
            fit.pvalues[
                "ai_treatment"
            ]
        ),

        "r_squared": (
            fit.rsquared
        ),
    }


def point_estimates(df, cohort):
    """
    Fit both boundaries using the EXACT SAME rows.
    """

    initial = fit_effect(
        df,
        "initialImplementationTime"
    )

    total = fit_effect(
        df,
        "totalImplementationTime"
    )

    return {
        "cohort": cohort,

        "n": len(df),

        "n_developers": (
            df["developerId"]
            .nunique()
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

        "time_boundary_shift_pp": (
            100.0
            * (
                total["time_effect"]
                - initial["time_effect"]
            )
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

        "productivity_boundary_shift_pp": (
            100.0
            * (
                total["productivity_effect"]
                - initial["productivity_effect"]
            )
        ),

        "initial_p_value": (
            initial["p_value"]
        ),

        "total_p_value": (
            total["p_value"]
        ),
    }


def cluster_bootstrap(
    df,
    n_boot=N_BOOT,
    seed=SEED
):
    """
    Cluster bootstrap by developer.

    Resample developers with replacement, preserving all
    of each selected developer's tasks.

    Returns distribution of:
        total effect - initial effect

    in both time and productivity space.
    """

    rng = np.random.default_rng(
        seed
    )

    dev_ids = (
        df["developerId"]
        .unique()
    )

    time_shifts = []
    productivity_shifts = []

    successful = 0
    attempts = 0

    max_attempts = (
        n_boot * 3
    )

    while (
        successful < n_boot
        and attempts < max_attempts
    ):

        attempts += 1

        sampled_devs = rng.choice(
            dev_ids,
            size=len(dev_ids),
            replace=True
        )

        pieces = []

        # Reassign bootstrap cluster IDs so duplicated
        # developers are treated as distinct bootstrap draws.
        for bootstrap_id, dev in enumerate(
            sampled_devs
        ):

            tmp = (
                df.loc[
                    df["developerId"]
                    == dev
                ]
                .copy()
            )

            tmp[
                "bootstrapDeveloperId"
            ] = bootstrap_id

            pieces.append(
                tmp
            )

        sample = pd.concat(
            pieces,
            ignore_index=True
        )

        # Need variation in treatment.
        if (
            sample["ai_treatment"]
            .nunique()
            < 2
        ):
            continue

        try:

            initial = fit_effect(
                sample,
                "initialImplementationTime"
            )

            total = fit_effect(
                sample,
                "totalImplementationTime"
            )

            time_shift = (
                100.0
                * (
                    total["time_effect"]
                    - initial["time_effect"]
                )
            )

            productivity_shift = (
                100.0
                * (
                    total[
                        "productivity_effect"
                    ]
                    - initial[
                        "productivity_effect"
                    ]
                )
            )

            if (
                np.isfinite(time_shift)
                and np.isfinite(
                    productivity_shift
                )
            ):

                time_shifts.append(
                    time_shift
                )

                productivity_shifts.append(
                    productivity_shift
                )

                successful += 1

        except Exception:
            continue

    time_shifts = np.asarray(
        time_shifts
    )

    productivity_shifts = np.asarray(
        productivity_shifts
    )

    return {
        "n_boot": successful,

        "time_shift_boot_mean_pp": (
            time_shifts.mean()
        ),

        "time_shift_ci_low_pp": (
            np.percentile(
                time_shifts,
                2.5
            )
        ),

        "time_shift_ci_high_pp": (
            np.percentile(
                time_shifts,
                97.5
            )
        ),

        "productivity_shift_boot_mean_pp": (
            productivity_shifts.mean()
        ),

        "productivity_shift_ci_low_pp": (
            np.percentile(
                productivity_shifts,
                2.5
            )
        ),

        "productivity_shift_ci_high_pp": (
            np.percentile(
                productivity_shifts,
                97.5
            )
        ),
    }


def analyze(
    df,
    cohort_name
):

    point = point_estimates(
        df,
        cohort_name
    )

    boot = cluster_bootstrap(
        df
    )

    return (
        point,
        {
            "cohort": cohort_name,
            **boot
        }
    )


def main():

    raw = pd.read_csv(
        INPUT
    )

    common = prepare_data(
        raw
    )

    groups = [
        (
            "All developers",
            common
        ),

        (
            "Panel developers",
            common.loc[
                common[
                    "isPanelDeveloper"
                ]
            ].copy()
        ),

        (
            "New developers",
            common.loc[
                ~common[
                    "isPanelDeveloper"
                ]
            ].copy()
        ),
    ]

    point_rows = []
    boot_rows = []

    for name, data in groups:

        point, boot = analyze(
            data,
            name
        )

        point_rows.append(
            point
        )

        boot_rows.append(
            boot
        )

    points = pd.DataFrame(
        point_rows
    )

    boots = pd.DataFrame(
        boot_rows
    )

    results = points.merge(
        boots,
        on="cohort",
        how="left"
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

    boots.to_csv(
        BOOT_OUT,
        index=False
    )

    # =========================================================
    # Report
    # =========================================================

    lines = []

    lines.append(
        "=" * 108
    )

    lines.append(
        "METR LATE-2025 — "
        "COMMON-SAMPLE MEASUREMENT BOUNDARY ANALYSIS"
    )

    lines.append(
        "=" * 108
    )

    for _, row in results.iterrows():

        lines.append("")
        lines.append(
            row[
                "cohort"
            ].upper()
        )

        lines.append(
            "-" * 108
        )

        lines.append(
            f"Common sample: "
            f"N={int(row['n'])}, "
            f"developers="
            f"{int(row['n_developers'])}"
        )

        lines.append("")
        lines.append(
            "INITIAL IMPLEMENTATION"
        )

        lines.append(
            "Time effect: "
            f"{row['initial_time_effect_pct']:+.2f}% "
            f"[{row['initial_time_ci_low_pct']:+.2f}%, "
            f"{row['initial_time_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Productivity effect: "
            f"{row['initial_productivity_effect_pct']:+.2f}% "
            f"[{row['initial_productivity_ci_low_pct']:+.2f}%, "
            f"{row['initial_productivity_ci_high_pct']:+.2f}%]"
        )

        lines.append("")
        lines.append(
            "INITIAL + POST-REVIEW"
        )

        lines.append(
            "Time effect: "
            f"{row['total_time_effect_pct']:+.2f}% "
            f"[{row['total_time_ci_low_pct']:+.2f}%, "
            f"{row['total_time_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            "Productivity effect: "
            f"{row['total_productivity_effect_pct']:+.2f}% "
            f"[{row['total_productivity_ci_low_pct']:+.2f}%, "
            f"{row['total_productivity_ci_high_pct']:+.2f}%]"
        )

        lines.append("")
        lines.append(
            "BOUNDARY SHIFT"
        )

        lines.append(
            "Point estimate — time: "
            f"{row['time_boundary_shift_pp']:+.2f} pp"
        )

        lines.append(
            "Cluster-bootstrap 95% CI — time: "
            f"[{row['time_shift_ci_low_pp']:+.2f}, "
            f"{row['time_shift_ci_high_pp']:+.2f}] pp"
        )

        lines.append(
            "Point estimate — productivity: "
            f"{row['productivity_boundary_shift_pp']:+.2f} pp"
        )

        lines.append(
            "Cluster-bootstrap 95% CI — productivity: "
            f"[{row['productivity_shift_ci_low_pp']:+.2f}, "
            f"{row['productivity_shift_ci_high_pp']:+.2f}] pp"
        )

        lines.append(
            f"Successful bootstrap samples: "
            f"{int(row['n_boot'])}"
        )

    lines.append("")
    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "-" * 108
    )

    lines.append(
        "Both outcome models use exactly the same tasks "
        "within each cohort."
    )

    lines.append(
        "The initial model measures time to initial implementation; "
        "the broader model adds post-review implementation time."
    )

    lines.append(
        "A positive shift in time space indicates that the AI "
        "time-saving effect becomes smaller under the broader boundary."
    )

    lines.append(
        "A negative shift in productivity space indicates that the "
        "estimated AI productivity benefit becomes smaller under "
        "the broader boundary."
    )

    lines.append(
        "Cluster bootstrap confidence intervals resample developers "
        "rather than individual issues to preserve within-developer "
        "task dependence."
    )

    lines.append(
        "These estimates characterize boundary sensitivity within "
        "the observed common sample; they should not be interpreted "
        "as a causal effect of review itself."
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
        f"{TABLE_OUT}"
    )

    print(
        f"Bootstrap summary written to: "
        f"{BOOT_OUT}"
    )

    print(
        f"Log written to: "
        f"{LOG_OUT}"
    )


if __name__ == "__main__":
    main()
