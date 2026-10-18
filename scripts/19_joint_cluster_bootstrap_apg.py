#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EARLY_FILE = (
    ROOT
    / "data"
    / "raw"
    / "metr_early"
    / "data_complete.csv"
)

LATE_FILE = (
    ROOT
    / "data"
    / "raw"
    / "metr_late"
    / "data.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "tables"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_SUMMARY = (
    OUT_DIR
    / "apg_joint_cluster_bootstrap.csv"
)

OUT_DRAWS = (
    OUT_DIR
    / "apg_joint_cluster_bootstrap_draws.csv"
)


# Start with 500 for validation.
# Once point estimates are reproduced, set to 20000.
N_BOOT = 20000

SEED = 20260922


# ============================================================
# General helpers
# ============================================================

def geometric_gain_from_log_ratios(log_ratios):
    """
    Productivity gain from log(T_noAI / T_AI):

        exp(mean(log_ratio)) - 1
    """

    x = np.asarray(
        log_ratios,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return (
        np.exp(
            x.mean()
        )
        - 1.0
    )


def productivity_from_log_time_beta(beta):
    """
    If:

        log(T) = ... + beta * AI

    then exp(beta) is the AI/control time ratio.

    Productivity gain:

        control productivity / AI productivity - 1
        = exp(-beta) - 1
    """

    return (
        np.exp(-beta)
        - 1.0
    )


def percentile_ci(values):
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan
        )

    return tuple(
        np.percentile(
            values,
            [2.5, 97.5]
        )
    )


# ============================================================
# Lightweight adjusted treatment regression
# ============================================================

def treatment_beta_fwl(
    outcome,
    treatment,
    predictor
):
    """
    Estimate treatment coefficient from:

        outcome ~ 1 + treatment + predictor

    using the Frisch-Waugh-Lovell theorem.

    This avoids statsmodels / repeated BLAS-heavy regression
    calls inside the bootstrap loop.

    All inputs must already be on the desired scale.
    """

    y = np.asarray(
        outcome,
        dtype=float
    )

    d = np.asarray(
        treatment,
        dtype=float
    )

    x = np.asarray(
        predictor,
        dtype=float
    )

    valid = (
        np.isfinite(y)
        & np.isfinite(d)
        & np.isfinite(x)
    )

    y = y[valid]
    d = d[valid]
    x = x[valid]

    if len(y) < 3:
        return np.nan

    if np.unique(d).size < 2:
        return np.nan

    # --------------------------------------------------------
    # Residualize y on intercept + x
    # --------------------------------------------------------

    x_mean = x.mean()
    y_mean = y.mean()
    d_mean = d.mean()

    x_centered = (
        x - x_mean
    )

    ss_x = np.sum(
        x_centered ** 2
    )

    if ss_x <= 0:
        return np.nan

    slope_y_x = (
        np.sum(
            x_centered
            * (
                y - y_mean
            )
        )
        / ss_x
    )

    slope_d_x = (
        np.sum(
            x_centered
            * (
                d - d_mean
            )
        )
        / ss_x
    )

    y_resid = (
        y
        - (
            y_mean
            + slope_y_x
            * x_centered
        )
    )

    d_resid = (
        d
        - (
            d_mean
            + slope_d_x
            * x_centered
        )
    )

    denominator = np.sum(
        d_resid ** 2
    )

    if denominator <= 0:
        return np.nan

    beta = (
        np.sum(
            d_resid
            * y_resid
        )
        / denominator
    )

    return float(beta)


# ============================================================
# Cluster bootstrap helpers
# ============================================================

def build_cluster_index(
    df,
    cluster_col
):
    """
    Map each developer to row indexes.
    """

    mapping = {}

    for dev, idx in (
        df.groupby(
            cluster_col,
            sort=False
        )
        .indices
        .items()
    ):
        mapping[dev] = np.asarray(
            idx,
            dtype=int
        )

    developers = np.asarray(
        list(mapping.keys()),
        dtype=object
    )

    return developers, mapping


def cluster_bootstrap_indices(
    developers,
    mapping,
    rng
):
    """
    Sample developers with replacement and return all
    rows belonging to each sampled developer.

    If the same developer is sampled twice, their rows appear
    twice, as required by a cluster bootstrap.
    """

    sampled_devs = rng.choice(
        developers,
        size=len(developers),
        replace=True
    )

    sampled_rows = [
        mapping[dev]
        for dev in sampled_devs
    ]

    return np.concatenate(
        sampled_rows
    )


# ============================================================
# Early-2025
# ============================================================

def prepare_early(df):
    """
    Reproduce METR Early's primary total-time outcome.

    IMPORTANT:
    Missing post-review implementation time is imputed as the
    treatment-group mean, exactly as in the official regression.
    """

    work = df.copy()

    # --------------------------------------------------------
    # Treatment-group mean imputation
    # --------------------------------------------------------

    means = {}

    for treatment in [0, 1]:

        vals = work.loc[
            (
                work["ai_treatment"]
                == treatment
            )
            & work[
                "post_review_implementation_time"
            ].notna(),
            "post_review_implementation_time"
        ]

        means[treatment] = (
            vals.mean()
        )

    missing = (
        work[
            "post_review_implementation_time"
        ]
        .isna()
    )

    work.loc[
        missing,
        "post_review_implementation_time"
    ] = (
        work.loc[
            missing,
            "ai_treatment"
        ]
        .map(means)
    )

    # --------------------------------------------------------
    # Total implementation time
    # --------------------------------------------------------

    work[
        "total_implementation_time"
    ] = (
        work[
            "initial_implementation_time"
        ]
        +
        work[
            "post_review_implementation_time"
        ]
    )

    # --------------------------------------------------------
    # Pre-task subjective log ratio
    # --------------------------------------------------------

    work[
        "pre_log_ratio"
    ] = np.log(
        work[
            "predicted_time_no_ai"
        ]
        /
        work[
            "predicted_time_ai_allowed"
        ]
    )

    return work


def early_point_estimates(df):

    work = prepare_early(
        df
    )

    subjective = geometric_gain_from_log_ratios(
        work[
            "pre_log_ratio"
        ]
    )

    reg = work.loc[
        work[
            "total_implementation_time"
        ].notna()
        & (
            work[
                "total_implementation_time"
            ]
            > 0
        )
        & work[
            "predicted_time_no_ai"
        ].notna()
        & (
            work[
                "predicted_time_no_ai"
            ]
            > 0
        )
        & work[
            "ai_treatment"
        ].notna()
    ].copy()

    beta = treatment_beta_fwl(
        outcome=np.log(
            reg[
                "total_implementation_time"
            ]
        ),
        treatment=reg[
            "ai_treatment"
        ],
        predictor=np.log(
            reg[
                "predicted_time_no_ai"
            ]
        ),
    )

    observed = (
        productivity_from_log_time_beta(
            beta
        )
    )

    return subjective, observed


def bootstrap_early(
    raw_df,
    n_boot,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    developers, mapping = (
        build_cluster_index(
            raw_df,
            "dev_id"
        )
    )

    point_subjective, point_observed = (
        early_point_estimates(
            raw_df
        )
    )

    point_apg = (
        point_subjective
        - point_observed
    )

    draws = []

    failed = 0

    for b in range(
        n_boot
    ):

        idx = cluster_bootstrap_indices(
            developers,
            mapping,
            rng
        )

        boot_raw = (
            raw_df.iloc[idx]
            .copy()
            .reset_index(drop=True)
        )

        try:

            subjective, observed = (
                early_point_estimates(
                    boot_raw
                )
            )

            apg = (
                subjective
                - observed
            )

            if not all(
                np.isfinite(
                    [
                        subjective,
                        observed,
                        apg,
                    ]
                )
            ):
                failed += 1
                continue

            draws.append(
                {
                    "analysis":
                        "Early-2025 | Pre-task expected",

                    "bootstrap_id": b,

                    "subjective_gain":
                        subjective,

                    "observed_gain":
                        observed,

                    "apg":
                        apg,
                }
            )

        except Exception:
            failed += 1

    draws = pd.DataFrame(
        draws
    )

    return (
        point_subjective,
        point_observed,
        point_apg,
        draws,
        failed,
    )


# ============================================================
# Late-2025
# ============================================================

def prepare_late_total_sample(df):
    """
    Reproduce the analysis-ready complete-task sample used
    in scripts 10/10b and the METR Late total-time analysis.
    """

    work = df.copy()

    # --------------------------------------------------------
    # Missing post-review time is treated as zero.
    # This matches our validated scripts 10/10b.
    # --------------------------------------------------------

    work[
        "postReviewImplementationTime"
    ] = (
        work[
            "postReviewImplementationTime"
        ]
        .fillna(0)
    )

    work[
        "totalImplementationTime"
    ] = (
        work[
            "initialImplementationTime"
        ]
        +
        work[
            "postReviewImplementationTime"
        ]
    )

    # --------------------------------------------------------
    # Remove ditched issues
    # --------------------------------------------------------

    work = work.loc[
        work["status"]
        != "ditched"
    ].copy()

    # --------------------------------------------------------
    # Complete-task analysis sample
    # --------------------------------------------------------

    work = work.loc[
        (
            work["status"]
            == "complete"
        )
        & work[
            "totalImplementationTime"
        ].notna()
        & (
            work[
                "totalImplementationTime"
            ]
            > 0
        )
        & work[
            "estimatedTimeWithoutAI"
        ].notna()
        & (
            work[
                "estimatedTimeWithoutAI"
            ]
            > 0
        )
        & work[
            "estimatedTimeWithAI"
        ].notna()
        & (
            work[
                "estimatedTimeWithAI"
            ]
            > 0
        )
    ].copy()

    # --------------------------------------------------------
    # Treatment coding
    # --------------------------------------------------------

    work[
        "ai_treatment"
    ] = (
        work[
            "treatmentAssignment"
        ]
        .map(
            {
                "AI-allowed": 1,
                "AI-disallowed": 0,
            }
        )
    )

    # --------------------------------------------------------
    # Pre-task log ratio
    # --------------------------------------------------------

    work[
        "pre_log_ratio"
    ] = np.log(
        work[
            "estimatedTimeWithoutAI"
        ]
        /
        work[
            "estimatedTimeWithAI"
        ]
    )

    # --------------------------------------------------------
    # Post-task counterfactual log ratio
    # --------------------------------------------------------

    work[
        "post_ratio"
    ] = np.nan

    valid_post = (
        work[
            "estimatedTimeOtherTreatment"
        ].notna()
        & (
            work[
                "estimatedTimeOtherTreatment"
            ]
            > 0
        )
        & work[
            "totalImplementationTime"
        ].notna()
        & (
            work[
                "totalImplementationTime"
            ]
            > 0
        )
    )

    ai_mask = (
        valid_post
        & (
            work[
                "treatmentAssignment"
            ]
            == "AI-allowed"
        )
    )

    control_mask = (
        valid_post
        & (
            work[
                "treatmentAssignment"
            ]
            == "AI-disallowed"
        )
    )

    # AI allowed:
    # estimated no-AI / observed AI
    work.loc[
        ai_mask,
        "post_ratio"
    ] = (
        work.loc[
            ai_mask,
            "estimatedTimeOtherTreatment"
        ]
        /
        work.loc[
            ai_mask,
            "totalImplementationTime"
        ]
    )

    # AI disallowed:
    # observed no-AI / estimated AI
    work.loc[
        control_mask,
        "post_ratio"
    ] = (
        work.loc[
            control_mask,
            "totalImplementationTime"
        ]
        /
        work.loc[
            control_mask,
            "estimatedTimeOtherTreatment"
        ]
    )

    work[
        "post_log_ratio"
    ] = np.log(
        work[
            "post_ratio"
        ]
    )

    return work


def late_estimates_from_prepared(df):
    """
    Compute pre-task subjective gain, post-task perceived
    gain, and experimental observed gain from the SAME
    analysis-ready sample.
    """

    pre = geometric_gain_from_log_ratios(
        df[
            "pre_log_ratio"
        ]
    )

    post = geometric_gain_from_log_ratios(
        df[
            "post_log_ratio"
        ]
    )

    reg = df.loc[
        df[
            "ai_treatment"
        ].notna()
    ].copy()

    beta = treatment_beta_fwl(
        outcome=np.log(
            reg[
                "totalImplementationTime"
            ]
        ),
        treatment=reg[
            "ai_treatment"
        ],
        predictor=np.log(
            reg[
                "estimatedTimeWithoutAI"
            ]
        ),
    )

    observed = (
        productivity_from_log_time_beta(
            beta
        )
    )

    return (
        pre,
        post,
        observed,
    )


def bootstrap_late_cohort(
    prepared_df,
    cohort_name,
    n_boot,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    developers, mapping = (
        build_cluster_index(
            prepared_df,
            "developerId"
        )
    )

    (
        point_pre,
        point_post,
        point_observed,
    ) = late_estimates_from_prepared(
        prepared_df
    )

    point_pre_apg = (
        point_pre
        - point_observed
    )

    point_post_apg = (
        point_post
        - point_observed
    )

    pre_draws = []
    post_draws = []

    failed = 0

    for b in range(
        n_boot
    ):

        idx = cluster_bootstrap_indices(
            developers,
            mapping,
            rng
        )

        boot = (
            prepared_df.iloc[idx]
            .copy()
            .reset_index(drop=True)
        )

        try:

            (
                pre,
                post,
                observed,
            ) = late_estimates_from_prepared(
                boot
            )

            pre_apg = (
                pre
                - observed
            )

            post_apg = (
                post
                - observed
            )

            if not all(
                np.isfinite(
                    [
                        pre,
                        post,
                        observed,
                        pre_apg,
                        post_apg,
                    ]
                )
            ):
                failed += 1
                continue

            pre_draws.append(
                {
                    "analysis":
                        (
                            f"{cohort_name} | "
                            f"Pre-task expected"
                        ),

                    "bootstrap_id": b,

                    "subjective_gain":
                        pre,

                    "observed_gain":
                        observed,

                    "apg":
                        pre_apg,
                }
            )

            post_draws.append(
                {
                    "analysis":
                        (
                            f"{cohort_name} | "
                            f"Post-task perceived"
                        ),

                    "bootstrap_id": b,

                    "subjective_gain":
                        post,

                    "observed_gain":
                        observed,

                    "apg":
                        post_apg,
                }
            )

        except Exception:
            failed += 1

    pre_draws = pd.DataFrame(
        pre_draws
    )

    post_draws = pd.DataFrame(
        post_draws
    )

    return {
        "point_pre":
            point_pre,

        "point_post":
            point_post,

        "point_observed":
            point_observed,

        "point_pre_apg":
            point_pre_apg,

        "point_post_apg":
            point_post_apg,

        "pre_draws":
            pre_draws,

        "post_draws":
            post_draws,

        "failed":
            failed,

        "developers":
            prepared_df[
                "developerId"
            ].nunique(),

        "rows":
            len(
                prepared_df
            ),
    }


# ============================================================
# Summary helpers
# ============================================================

def make_summary_row(
    analysis,
    rows,
    developers,
    subjective_point,
    observed_point,
    apg_point,
    draws,
    n_boot,
    failed,
):

    subj_low, subj_high = percentile_ci(
        draws[
            "subjective_gain"
        ]
    )

    obs_low, obs_high = percentile_ci(
        draws[
            "observed_gain"
        ]
    )

    apg_low, apg_high = percentile_ci(
        draws[
            "apg"
        ]
    )

    return {
        "analysis":
            analysis,

        "rows":
            rows,

        "developers":
            developers,

        "bootstrap_requested":
            n_boot,

        "bootstrap_valid":
            len(draws),

        "bootstrap_failed":
            failed,

        "subjective_point_pct":
            100.0
            * subjective_point,

        "subjective_ci_low_pct":
            100.0
            * subj_low,

        "subjective_ci_high_pct":
            100.0
            * subj_high,

        "observed_point_pct":
            100.0
            * observed_point,

        "observed_ci_low_pct":
            100.0
            * obs_low,

        "observed_ci_high_pct":
            100.0
            * obs_high,

        "apg_point_pp":
            100.0
            * apg_point,

        "apg_ci_low_pp":
            100.0
            * apg_low,

        "apg_ci_high_pp":
            100.0
            * apg_high,
    }


def print_result(
    row
):

    print()
    print("=" * 78)
    print(
        row["analysis"]
    )
    print("=" * 78)

    print(
        f"Rows:                "
        f"{row['rows']}"
    )

    print(
        f"Developers:          "
        f"{row['developers']}"
    )

    print(
        f"Subjective gain:     "
        f"{row['subjective_point_pct']:+.2f}%"
    )

    print(
        f"Observed gain:       "
        f"{row['observed_point_pct']:+.2f}%"
    )

    print(
        f"APG:                 "
        f"{row['apg_point_pp']:+.2f} pp"
    )

    print(
        f"Valid bootstrap:     "
        f"{row['bootstrap_valid']}/"
        f"{row['bootstrap_requested']}"
    )

    print(
        "Subjective 95% CI: "
        f"[{row['subjective_ci_low_pct']:+.2f}, "
        f"{row['subjective_ci_high_pct']:+.2f}]%"
    )

    print(
        "Observed 95% CI:   "
        f"[{row['observed_ci_low_pct']:+.2f}, "
        f"{row['observed_ci_high_pct']:+.2f}]%"
    )

    print(
        "Joint APG 95% CI:  "
        f"[{row['apg_ci_low_pp']:+.2f}, "
        f"{row['apg_ci_high_pp']:+.2f}] pp"
    )


# ============================================================
# Main
# ============================================================

def main():

    all_summary_rows = []
    all_draws = []

    # ========================================================
    # Early
    # ========================================================

    early_raw = pd.read_csv(
        EARLY_FILE
    )

    (
        early_subjective,
        early_observed,
        early_apg,
        early_draws,
        early_failed,
    ) = bootstrap_early(
        raw_df=early_raw,
        n_boot=N_BOOT,
        seed=SEED,
    )

    early_row = make_summary_row(
        analysis=(
            "Early-2025 | "
            "Pre-task expected"
        ),
        rows=len(early_raw),
        developers=early_raw[
            "dev_id"
        ].nunique(),
        subjective_point=(
            early_subjective
        ),
        observed_point=(
            early_observed
        ),
        apg_point=(
            early_apg
        ),
        draws=early_draws,
        n_boot=N_BOOT,
        failed=early_failed,
    )

    all_summary_rows.append(
        early_row
    )

    all_draws.append(
        early_draws
    )

    print_result(
        early_row
    )

    # ========================================================
    # Late
    # ========================================================

    late_raw = pd.read_csv(
        LATE_FILE
    )

    late = prepare_late_total_sample(
        late_raw
    )

    # --------------------------------------------------------
    # IMPORTANT validation output
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("LATE ANALYSIS SAMPLE CHECK")
    print("=" * 78)

    print(
        f"Raw rows:            "
        f"{len(late_raw)}"
    )

    print(
        f"Prepared rows:       "
        f"{len(late)}"
    )

    print(
        f"Prepared developers: "
        f"{late['developerId'].nunique()}"
    )

    # --------------------------------------------------------
    # Cohorts
    # --------------------------------------------------------

    cohorts = {
        "Late-2025 All":
            late,

        "Late-2025 Panel":
            late.loc[
                late[
                    "isPanelDeveloper"
                ]
                == True
            ].copy(),

        "Late-2025 New":
            late.loc[
                late[
                    "isPanelDeveloper"
                ]
                == False
            ].copy(),
    }

    for cohort_index, (
        cohort_name,
        cohort_df
    ) in enumerate(
        cohorts.items()
    ):

        result = bootstrap_late_cohort(
            prepared_df=cohort_df,
            cohort_name=cohort_name,
            n_boot=N_BOOT,
            seed=(
                SEED
                + 100
                + cohort_index
            ),
        )

        # ----------------------------------------------------
        # Pre-task row
        # ----------------------------------------------------

        pre_analysis = (
            f"{cohort_name} | "
            f"Pre-task expected"
        )

        pre_row = make_summary_row(
            analysis=pre_analysis,
            rows=result[
                "rows"
            ],
            developers=result[
                "developers"
            ],
            subjective_point=result[
                "point_pre"
            ],
            observed_point=result[
                "point_observed"
            ],
            apg_point=result[
                "point_pre_apg"
            ],
            draws=result[
                "pre_draws"
            ],
            n_boot=N_BOOT,
            failed=result[
                "failed"
            ],
        )

        all_summary_rows.append(
            pre_row
        )

        all_draws.append(
            result[
                "pre_draws"
            ]
        )

        print_result(
            pre_row
        )

        # ----------------------------------------------------
        # Post-task row
        # ----------------------------------------------------

        post_analysis = (
            f"{cohort_name} | "
            f"Post-task perceived"
        )

        post_row = make_summary_row(
            analysis=post_analysis,
            rows=result[
                "rows"
            ],
            developers=result[
                "developers"
            ],
            subjective_point=result[
                "point_post"
            ],
            observed_point=result[
                "point_observed"
            ],
            apg_point=result[
                "point_post_apg"
            ],
            draws=result[
                "post_draws"
            ],
            n_boot=N_BOOT,
            failed=result[
                "failed"
            ],
        )

        all_summary_rows.append(
            post_row
        )

        all_draws.append(
            result[
                "post_draws"
            ]
        )

        print_result(
            post_row
        )

    # ========================================================
    # Save
    # ========================================================

    summary_df = pd.DataFrame(
        all_summary_rows
    )

    draws_df = pd.concat(
        all_draws,
        ignore_index=True
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False
    )

    draws_df.to_csv(
        OUT_DRAWS,
        index=False
    )

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    display_cols = [
        "analysis",
        "rows",
        "developers",
        "subjective_point_pct",
        "observed_point_pct",
        "apg_point_pp",
        "apg_ci_low_pp",
        "apg_ci_high_pp",
        "bootstrap_valid",
    ]

    print(
        summary_df[
            display_cols
        ]
        .to_string(
            index=False,
            float_format=(
                lambda x: f"{x:.2f}"
            )
        )
    )

    print()
    print(
        f"Summary written to: {OUT_SUMMARY}"
    )

    print(
        f"Draws written to:   {OUT_DRAWS}"
    )


if __name__ == "__main__":
    main()
