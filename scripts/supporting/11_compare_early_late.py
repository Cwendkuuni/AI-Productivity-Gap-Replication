#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EARLY_SUMMARY = (
    ROOT
    / "results"
    / "tables"
    / "metr_early_productivity_gap_summary.csv"
)

LATE_ROBUSTNESS = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_productivity_gap_robustness.csv"
)

LATE_GAP = (
    ROOT
    / "results"
    / "tables"
    / "metr_late_productivity_gap_summary.csv"
)

OUT_TABLE = (
    ROOT
    / "results"
    / "tables"
    / "early_late_productivity_gap_comparison.csv"
)

OUT_LOG = (
    ROOT
    / "results"
    / "logs"
    / "early_late_productivity_gap_comparison.txt"
)


def get_late_metric(
    robustness,
    group,
    measure,
    column,
):
    row = robustness.loc[
        (robustness["group"] == group)
        & (robustness["measure"] == measure)
    ]

    if len(row) != 1:
        return np.nan

    return row.iloc[0][column]


def get_late_observed(
    gap,
    group,
    column,
):
    row = gap.loc[
        gap["group"] == group
    ]

    if len(row) != 1:
        return np.nan

    return row.iloc[0][column]


def main():

    early = pd.read_csv(
        EARLY_SUMMARY
    )

    late_robust = pd.read_csv(
        LATE_ROBUSTNESS
    )

    late_gap = pd.read_csv(
        LATE_GAP
    )

    rows = []

    # =========================================================
    # EARLY
    # =========================================================

    e = early.iloc[0]

    early_expected = (
        e[
            "expected_productivity_geometric_pct"
        ]
    )

    early_observed = (
        e[
            "observed_productivity_effect_pct"
        ]
    )

    early_apg = (
        early_expected
        - early_observed
    )

    rows.append(
        {
            "study_cohort": "Early-2025",
            "study": "Early-2025",
            "cohort": "Original developers",

            "pre_expected_gain_pct": (
                early_expected
            ),

            "pre_expected_ci_low_pct": (
                e[
                    "expected_productivity_geometric_ci_low_pct"
                ]
            ),

            "pre_expected_ci_high_pct": (
                e[
                    "expected_productivity_geometric_ci_high_pct"
                ]
            ),

            "post_perceived_gain_pct": np.nan,
            "post_perceived_ci_low_pct": np.nan,
            "post_perceived_ci_high_pct": np.nan,

            "observed_productivity_gain_pct": (
                early_observed
            ),

            "pre_apg_pct": (
                early_apg
            ),

            "post_apg_pct": np.nan,
        }
    )

    # =========================================================
    # LATE
    # =========================================================

    group_map = {
        "All developers": "All",
        "Panel developers": "Panel",
        "New developers": "New",
    }

    for full_group, short_group in group_map.items():

        pre = get_late_metric(
            late_robust,
            full_group,
            "Pre-task expected gain",
            "geometric_gain_pct",
        )

        pre_low = get_late_metric(
            late_robust,
            full_group,
            "Pre-task expected gain",
            "geometric_ci_low_pct",
        )

        pre_high = get_late_metric(
            late_robust,
            full_group,
            "Pre-task expected gain",
            "geometric_ci_high_pct",
        )

        post = get_late_metric(
            late_robust,
            full_group,
            "Post-task perceived gain",
            "geometric_gain_pct",
        )

        post_low = get_late_metric(
            late_robust,
            full_group,
            "Post-task perceived gain",
            "geometric_ci_low_pct",
        )

        post_high = get_late_metric(
            late_robust,
            full_group,
            "Post-task perceived gain",
            "geometric_ci_high_pct",
        )

        observed = get_late_observed(
            late_gap,
            full_group,
            "observed_productivity_gain_pct",
        )

        rows.append(
            {
                "study_cohort": (
                    f"Late-2025 — {short_group}"
                ),

                "study": "Late-2025",

                "cohort": short_group,

                "pre_expected_gain_pct": (
                    pre
                ),

                "pre_expected_ci_low_pct": (
                    pre_low
                ),

                "pre_expected_ci_high_pct": (
                    pre_high
                ),

                "post_perceived_gain_pct": (
                    post
                ),

                "post_perceived_ci_low_pct": (
                    post_low
                ),

                "post_perceived_ci_high_pct": (
                    post_high
                ),

                "observed_productivity_gain_pct": (
                    observed
                ),

                "pre_apg_pct": (
                    pre - observed
                ),

                "post_apg_pct": (
                    post - observed
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    OUT_TABLE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison.to_csv(
        OUT_TABLE,
        index=False
    )

    # =========================================================
    # Extra derived changes
    # =========================================================

    comparison[
        "belief_revision_pct"
    ] = (
        comparison[
            "post_perceived_gain_pct"
        ]
        - comparison[
            "pre_expected_gain_pct"
        ]
    )

    # =========================================================
    # Human-readable output
    # =========================================================

    lines = []

    lines.append("=" * 108)

    lines.append(
        "METR EARLY vs LATE — "
        "HARMONIZED AI PRODUCTIVITY GAP COMPARISON"
    )

    lines.append("=" * 108)

    for _, row in comparison.iterrows():

        lines.append("")
        lines.append(
            row[
                "study_cohort"
            ].upper()
        )

        lines.append(
            "-" * 108
        )

        lines.append(
            "Pre-task expected productivity gain: "
            f"{row['pre_expected_gain_pct']:+.2f}%"
        )

        lines.append(
            "Pre-task expectation 95% CI: "
            f"[{row['pre_expected_ci_low_pct']:+.2f}%, "
            f"{row['pre_expected_ci_high_pct']:+.2f}%]"
        )

        if np.isfinite(
            row[
                "post_perceived_gain_pct"
            ]
        ):

            lines.append(
                "Post-task perceived productivity gain: "
                f"{row['post_perceived_gain_pct']:+.2f}%"
            )

            lines.append(
                "Post-task perception 95% CI: "
                f"[{row['post_perceived_ci_low_pct']:+.2f}%, "
                f"{row['post_perceived_ci_high_pct']:+.2f}%]"
            )

            lines.append(
                "Belief revision "
                "(post - pre): "
                f"{row['belief_revision_pct']:+.2f} pp"
            )

        else:

            lines.append(
                "Post-task perceived productivity gain: "
                "Not available"
            )

        lines.append(
            "Observed productivity effect: "
            f"{row['observed_productivity_gain_pct']:+.2f}%"
        )

        lines.append(
            "Pre-task AI Productivity Gap: "
            f"{row['pre_apg_pct']:+.2f} pp"
        )

        if np.isfinite(
            row[
                "post_apg_pct"
            ]
        ):

            lines.append(
                "Post-task AI Productivity Gap: "
                f"{row['post_apg_pct']:+.2f} pp"
            )

    # =========================================================
    # Compact comparison table
    # =========================================================

    lines.append("")
    lines.append(
        "COMPACT COMPARISON"
    )

    lines.append(
        "-" * 108
    )

    display_cols = [
        "study_cohort",
        "pre_expected_gain_pct",
        "post_perceived_gain_pct",
        "observed_productivity_gain_pct",
        "pre_apg_pct",
        "post_apg_pct",
        "belief_revision_pct",
    ]

    lines.append(
        comparison[
            display_cols
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
        "Expected and perceived productivity gains use the "
        "geometric/log-ratio estimator in both datasets."
    )

    lines.append(
        "Observed productivity effects are derived from "
        "the corresponding METR log-linear treatment estimates."
    )

    lines.append(
        "The AI Productivity Gap is defined as expected or "
        "perceived productivity gain minus observed productivity gain."
    )

    lines.append(
        "Early-2025 and Late-2025 should not be interpreted "
        "as a clean longitudinal experiment because participant "
        "selection and cohort composition differ."
    )

    lines.append(
        "Late Panel and Late New are therefore reported separately."
    )

    lines.append(
        "Differences across cohorts are descriptive and must not "
        "be interpreted as causal effects of prior AI experience, "
        "panel membership, or calendar time."
    )

    # =========================================================
    # Save
    # =========================================================

    OUT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_LOG.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(
        "\n".join(lines)
    )

    print()

    print(
        f"Comparison table written to: "
        f"{OUT_TABLE}"
    )

    print(
        f"Log written to: "
        f"{OUT_LOG}"
    )


if __name__ == "__main__":
    main()
