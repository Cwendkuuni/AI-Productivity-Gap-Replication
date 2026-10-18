#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "metr_late" / "data.csv"

OUTPUT = (
    ROOT
    / "results"
    / "logs"
    / "metr_late_audit.txt"
)


def pct(x, n):
    if n == 0:
        return 0.0
    return 100.0 * x / n


def describe_variable(df, col, lines):
    if col not in df.columns:
        return

    s = df[col].dropna()

    lines.append("")
    lines.append(col)
    lines.append(f"  N: {len(s):,}")
    lines.append(
        f"  Missing: {df[col].isna().sum():,}"
    )

    if len(s) == 0:
        return

    if pd.api.types.is_numeric_dtype(s):
        lines.append(f"  Mean: {s.mean():.2f}")
        lines.append(f"  Median: {s.median():.2f}")
        lines.append(f"  SD: {s.std():.2f}")
        lines.append(f"  Min: {s.min():.2f}")
        lines.append(f"  Max: {s.max():.2f}")


def main():
    df = pd.read_csv(INPUT)

    lines = []

    lines.append("=" * 90)
    lines.append("METR LATE-2025 DATA AUDIT")
    lines.append("=" * 90)

    lines.append(f"\nInput: {INPUT}")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")

    # ---------------------------------------------------------
    # Columns
    # ---------------------------------------------------------

    lines.append("")
    lines.append("COLUMNS")
    lines.append("-" * 90)

    for col in df.columns:
        lines.append(
            f"- {col} [{df[col].dtype}]"
        )

    # ---------------------------------------------------------
    # Missingness
    # ---------------------------------------------------------

    lines.append("")
    lines.append("MISSING VALUES")
    lines.append("-" * 90)

    for col in df.columns:
        n_missing = df[col].isna().sum()

        lines.append(
            f"{col}: "
            f"{n_missing:,}/{len(df):,} "
            f"({pct(n_missing, len(df)):.1f}%)"
        )

    # ---------------------------------------------------------
    # Developers / issues
    # ---------------------------------------------------------

    lines.append("")
    lines.append("DATASET SIZE")
    lines.append("-" * 90)

    if "developerId" in df.columns:
        lines.append(
            "Unique developers: "
            f"{df['developerId'].nunique(dropna=True)}"
        )

    if "id" in df.columns:
        lines.append(
            "Unique issues: "
            f"{df['id'].nunique(dropna=True)}"
        )

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    if "status" in df.columns:
        lines.append("")
        lines.append("STATUS COUNTS")
        lines.append("-" * 90)

        lines.append(
            df["status"]
            .value_counts(dropna=False)
            .to_string()
        )

    # ---------------------------------------------------------
    # Treatment
    # ---------------------------------------------------------

    if "treatmentAssignment" in df.columns:
        lines.append("")
        lines.append("TREATMENT COUNTS")
        lines.append("-" * 90)

        lines.append(
            df["treatmentAssignment"]
            .value_counts(dropna=False)
            .to_string()
        )

    # ---------------------------------------------------------
    # Panel vs new developers
    # ---------------------------------------------------------

    if "isPanelDeveloper" in df.columns:
        lines.append("")
        lines.append("PANEL STATUS")
        lines.append("-" * 90)

        lines.append(
            df["isPanelDeveloper"]
            .value_counts(dropna=False)
            .to_string()
        )

    # ---------------------------------------------------------
    # Numeric summary
    # ---------------------------------------------------------

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    lines.append("")
    lines.append("NUMERIC SUMMARY")
    lines.append("-" * 90)

    if numeric_cols:
        lines.append(
            df[numeric_cols]
            .describe()
            .transpose()
            .to_string()
        )

    # ---------------------------------------------------------
    # Core variables
    # ---------------------------------------------------------

    lines.append("")
    lines.append("CORE VARIABLES")
    lines.append("-" * 90)

    core_cols = [
        "estimatedTimeWithoutAI",
        "estimatedTimeWithAI",
        "initialImplementationTime",
        "postReviewImplementationTime",
        "estimatedTimeOtherTreatment",
        "priorExperience",
        "perceivedEffort",
    ]

    for col in core_cols:
        describe_variable(
            df,
            col,
            lines
        )

    # ---------------------------------------------------------
    # Measurement boundary availability
    # ---------------------------------------------------------

    if "postReviewImplementationTime" in df.columns:

        complete_review = (
            df["postReviewImplementationTime"]
            .notna()
        )

        lines.append("")
        lines.append("MEASUREMENT-BOUNDARY AVAILABILITY")
        lines.append("-" * 90)

        lines.append(
            "Rows with post-review implementation time: "
            f"{complete_review.sum():,}/{len(df):,} "
            f"({pct(complete_review.sum(), len(df)):.1f}%)"
        )

        if "treatmentAssignment" in df.columns:

            by_treatment = (
                df.assign(
                    has_post_review=complete_review
                )
                .groupby(
                    "treatmentAssignment"
                )["has_post_review"]
                .agg(
                    ["sum", "count", "mean"]
                )
            )

            by_treatment["percent"] = (
                100.0 * by_treatment["mean"]
            )

            lines.append("")
            lines.append(
                "Post-review availability by treatment:"
            )

            lines.append(
                by_treatment[
                    ["sum", "count", "percent"]
                ].to_string()
            )

    # ---------------------------------------------------------
    # Panel vs new treatment balance
    # ---------------------------------------------------------

    if {
        "isPanelDeveloper",
        "treatmentAssignment",
    }.issubset(df.columns):

        lines.append("")
        lines.append(
            "TREATMENT COUNTS BY PANEL STATUS"
        )
        lines.append("-" * 90)

        cross = pd.crosstab(
            df["isPanelDeveloper"],
            df["treatmentAssignment"],
            margins=True
        )

        lines.append(
            cross.to_string()
        )

    # ---------------------------------------------------------
    # Completion status x treatment
    # ---------------------------------------------------------

    if {
        "status",
        "treatmentAssignment",
    }.issubset(df.columns):

        lines.append("")
        lines.append(
            "STATUS BY TREATMENT"
        )
        lines.append("-" * 90)

        cross = pd.crosstab(
            df["status"],
            df["treatmentAssignment"],
            margins=True
        )

        lines.append(
            cross.to_string()
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))

    print()
    print(
        f"Audit written to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()