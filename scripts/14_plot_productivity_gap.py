#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "tables"
    / "apg_joint_cluster_bootstrap.csv"
)

FIG_DIR = (
    ROOT
    / "results"
    / "figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_PDF = (
    FIG_DIR
    / "fig_productivity_gap_v4.pdf"
)

OUT_PNG = (
    FIG_DIR
    / "fig_productivity_gap_v4.png"
)


# ============================================================
# Visual encoding
# ============================================================
#
# Color encodes estimate type consistently across cohorts.
#
# Expected            -> blue
# Post-task perceived -> red
# Observed            -> orange
#
# Marker shape provides redundant encoding:
#
# Expected            -> circle
# Post-task perceived -> square
# Observed            -> triangle
# ============================================================

EXPECTED_COLOR = "C0"
POST_COLOR = "C3"
OBSERVED_COLOR = "C1"

EXPECTED_MARKER = "o"
POST_MARKER = "s"
OBSERVED_MARKER = "^"


# ============================================================
# Load final bootstrap results
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT}"
    )

df = pd.read_csv(
    INPUT
)


# ============================================================
# Required columns
# ============================================================

required_columns = [
    "analysis",
    "subjective_point_pct",
    "subjective_ci_low_pct",
    "subjective_ci_high_pct",
    "observed_point_pct",
    "observed_ci_low_pct",
    "observed_ci_high_pct",
    "apg_point_pp",
    "apg_ci_low_pp",
    "apg_ci_high_pp",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns in "
        f"{INPUT}:\n{missing_columns}"
    )


# ============================================================
# Select populations shown in Figure 1
# ============================================================
#
# Late-2025 All is reported in tables and text but is omitted
# here because it aggregates the Panel and New cohorts and would
# add visual redundancy.
# ============================================================

required_analyses = [
    "Early-2025 | Pre-task expected",
    "Late-2025 Panel | Pre-task expected",
    "Late-2025 Panel | Post-task perceived",
    "Late-2025 New | Pre-task expected",
    "Late-2025 New | Post-task perceived",
]


missing_analyses = [
    name
    for name in required_analyses
    if name not in set(
        df["analysis"]
    )
]

if missing_analyses:
    raise ValueError(
        "Missing required analyses in bootstrap results:\n"
        + "\n".join(
            missing_analyses
        )
    )


plot_df = (
    df.loc[
        df["analysis"].isin(
            required_analyses
        )
    ]
    .copy()
)


# ============================================================
# Row extraction helper
# ============================================================

def get_row(name):
    """
    Return exactly one result row for a named analysis.
    """

    row = plot_df.loc[
        plot_df["analysis"]
        == name
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected exactly one row for '{name}', "
            f"found {len(row)}."
        )

    return row.iloc[0]


early = get_row(
    "Early-2025 | Pre-task expected"
)

panel_pre = get_row(
    "Late-2025 Panel | Pre-task expected"
)

panel_post = get_row(
    "Late-2025 Panel | Post-task perceived"
)

new_pre = get_row(
    "Late-2025 New | Pre-task expected"
)

new_post = get_row(
    "Late-2025 New | Post-task perceived"
)


# ============================================================
# Validation
# ============================================================

numeric_columns = [
    "subjective_point_pct",
    "subjective_ci_low_pct",
    "subjective_ci_high_pct",
    "observed_point_pct",
    "observed_ci_low_pct",
    "observed_ci_high_pct",
]

if not np.isfinite(
    plot_df[
        numeric_columns
    ].to_numpy(
        dtype=float
    )
).all():
    raise ValueError(
        "Non-finite values detected in plotting data."
    )


# ============================================================
# Helper for asymmetric confidence intervals
# ============================================================

def asymmetric_error(
    point,
    low,
    high
):
    """
    Convert lower/upper confidence bounds into the asymmetric
    error representation expected by matplotlib.errorbar().
    """

    lower_error = (
        point - low
    )

    upper_error = (
        high - point
    )

    if (
        lower_error < 0
        or upper_error < 0
    ):
        raise ValueError(
            "Invalid confidence interval: "
            f"point={point}, low={low}, high={high}"
        )

    return np.array(
        [
            [lower_error],
            [upper_error],
        ]
    )


# ============================================================
# Plotting helpers
# ============================================================

def plot_subjective(
    ax,
    x,
    row,
    marker,
    color,
    label=None,
):
    """
    Plot an expected or post-task perceived productivity
    estimate with its joint cluster-bootstrap interval.
    """

    point = float(
        row[
            "subjective_point_pct"
        ]
    )

    low = float(
        row[
            "subjective_ci_low_pct"
        ]
    )

    high = float(
        row[
            "subjective_ci_high_pct"
        ]
    )

    ax.errorbar(
        x,
        point,
        yerr=asymmetric_error(
            point,
            low,
            high
        ),
        fmt=marker,
        color=color,
        ecolor=color,
        markersize=7,
        markeredgewidth=1.0,
        capsize=4,
        capthick=1.3,
        elinewidth=1.5,
        linewidth=1.5,
        label=label,
        zorder=3,
    )


def plot_observed(
    ax,
    x,
    row,
    label=None,
):
    """
    Plot the experimentally observed productivity estimate
    with its joint cluster-bootstrap interval.
    """

    point = float(
        row[
            "observed_point_pct"
        ]
    )

    low = float(
        row[
            "observed_ci_low_pct"
        ]
    )

    high = float(
        row[
            "observed_ci_high_pct"
        ]
    )

    ax.errorbar(
        x,
        point,
        yerr=asymmetric_error(
            point,
            low,
            high
        ),
        fmt=OBSERVED_MARKER,
        color=OBSERVED_COLOR,
        ecolor=OBSERVED_COLOR,
        markersize=7,
        markeredgewidth=1.0,
        capsize=4,
        capthick=1.3,
        elinewidth=1.5,
        linewidth=1.5,
        label=label,
        zorder=3,
    )


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8.6,
        5.3
    )
)


# ============================================================
# X-axis positions
# ============================================================

x_early = 0.0
x_panel = 1.3
x_new = 2.6

offset = 0.18


# ============================================================
# Early-2025
# ============================================================

plot_subjective(
    ax=ax,
    x=x_early - offset,
    row=early,
    marker=EXPECTED_MARKER,
    color=EXPECTED_COLOR,
    label="Expected",
)

plot_observed(
    ax=ax,
    x=x_early + offset,
    row=early,
    label="Observed",
)


# ============================================================
# Late-2025 Panel
# ============================================================

plot_subjective(
    ax=ax,
    x=x_panel - offset,
    row=panel_pre,
    marker=EXPECTED_MARKER,
    color=EXPECTED_COLOR,
)

plot_subjective(
    ax=ax,
    x=x_panel,
    row=panel_post,
    marker=POST_MARKER,
    color=POST_COLOR,
    label="Post-task perceived",
)

plot_observed(
    ax=ax,
    x=x_panel + offset,
    row=panel_pre,
)


# ============================================================
# Late-2025 New
# ============================================================

plot_subjective(
    ax=ax,
    x=x_new - offset,
    row=new_pre,
    marker=EXPECTED_MARKER,
    color=EXPECTED_COLOR,
)

plot_subjective(
    ax=ax,
    x=x_new,
    row=new_post,
    marker=POST_MARKER,
    color=POST_COLOR,
)

plot_observed(
    ax=ax,
    x=x_new + offset,
    row=new_pre,
)


# ============================================================
# Zero-effect reference line
# ============================================================

ax.axhline(
    y=0,
    linestyle="--",
    linewidth=1.2,
    color="0.35",
    alpha=0.8,
    zorder=1,
)


# ============================================================
# Axes
# ============================================================

ax.set_xticks(
    [
        x_early,
        x_panel,
        x_new,
    ]
)

ax.set_xticklabels(
    [
        "Early-2025",
        "Late-2025\nPanel",
        "Late-2025\nNew",
    ]
)

ax.set_ylabel(
    "Productivity effect (%)"
)

ax.set_xlabel(
    ""
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="y",
    alpha=0.22,
    linewidth=0.8,
    zorder=0,
)


# ============================================================
# Legend
# ============================================================
#
# Matplotlib records labels according to plotting order.
# We explicitly impose the conceptual sequence:
#
# Expected -> Post-task perceived -> Observed
# ============================================================

handles, labels = (
    ax.get_legend_handles_labels()
)

legend_lookup = dict(
    zip(
        labels,
        handles
    )
)

legend_order = [
    "Expected",
    "Post-task perceived",
    "Observed",
]

missing_legend_items = [
    label
    for label in legend_order
    if label not in legend_lookup
]

if missing_legend_items:
    raise ValueError(
        "Missing legend items: "
        f"{missing_legend_items}"
    )

ax.legend(
    [
        legend_lookup[
            label
        ]
        for label in legend_order
    ],
    legend_order,
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        1.15
    ),
    columnspacing=1.8,
    handletextpad=0.6,
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout()


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# Console report
# ============================================================

print(
    f"Input: {INPUT}"
)

print(
    f"Saved: {OUT_PDF}"
)

print(
    f"Saved: {OUT_PNG}"
)

print()

print(
    "Figure data:"
)

print(
    f"Early expected:       "
    f"{early['subjective_point_pct']:+.2f}%"
)

print(
    f"Early observed:       "
    f"{early['observed_point_pct']:+.2f}%"
)

print(
    f"Panel expected:       "
    f"{panel_pre['subjective_point_pct']:+.2f}%"
)

print(
    f"Panel perceived:      "
    f"{panel_post['subjective_point_pct']:+.2f}%"
)

print(
    f"Panel observed:       "
    f"{panel_pre['observed_point_pct']:+.2f}%"
)

print(
    f"New expected:         "
    f"{new_pre['subjective_point_pct']:+.2f}%"
)

print(
    f"New perceived:        "
    f"{new_post['subjective_point_pct']:+.2f}%"
)

print(
    f"New observed:         "
    f"{new_pre['observed_point_pct']:+.2f}%"
)
