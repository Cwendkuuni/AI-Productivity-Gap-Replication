#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "metr_early" / "data_complete.csv"
OUTPUT = ROOT / "results" / "logs" / "metr_early_audit.txt"


def pct(x, n):
    if n == 0:
        return 0.0
    return 100.0 * x / n


def main():
    df = pd.read_csv(INPUT)

    lines = []

    lines.append("=" * 80)
    lines.append("METR EARLY-2025 DATA AUDIT")
    lines.append("=" * 80)

    lines.append(f"\nInput: {INPUT}")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")

    lines.append("\nCOLUMNS")
    lines.append("-" * 80)

    for col in df.columns:
        lines.append(f"- {col} [{df[col].dtype}]")

    lines.append("\nMISSING VALUES")
    lines.append("-" * 80)

    for col in df.columns:
        n_missing = df[col].isna().sum()

        lines.append(
            f"{col}: "
            f"{n_missing:,}/{len(df):,} "
            f"({pct(n_missing, len(df)):.1f}%)"
        )

    if "dev_id" in df.columns:
        lines.append("\nDEVELOPERS")
        lines.append("-" * 80)
        lines.append(
            f"Unique developers: "
            f"{df['dev_id'].nunique(dropna=True)}"
        )

    if "issue_id" in df.columns:
        lines.append(
            f"Unique issues: "
            f"{df['issue_id'].nunique(dropna=True)}"
        )

    if "ai_treatment" in df.columns:
        lines.append("\nTREATMENT COUNTS")
        lines.append("-" * 80)

        counts = df["ai_treatment"].value_counts(
            dropna=False
        ).sort_index()

        lines.append(counts.to_string())

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    lines.append("\nNUMERIC SUMMARY")
    lines.append("-" * 80)

    if numeric_cols:
        lines.append(
            df[numeric_cols]
            .describe()
            .transpose()
            .to_string()
        )

    core_cols = [
        "predicted_time_no_ai",
        "predicted_time_ai_allowed",
        "initial_implementation_time",
        "post_review_implementation_time",
    ]

    lines.append("\nCORE VARIABLES")
    lines.append("-" * 80)

    for col in core_cols:
        if col not in df.columns:
            continue

        s = df[col].dropna()

        lines.append(f"\n{col}")
        lines.append(f"  N: {len(s):,}")
        lines.append(f"  Missing: {df[col].isna().sum():,}")

        if len(s) > 0:
            lines.append(f"  Mean: {s.mean():.2f}")
            lines.append(f"  Median: {s.median():.2f}")
            lines.append(f"  SD: {s.std():.2f}")
            lines.append(f"  Min: {s.min():.2f}")
            lines.append(f"  Max: {s.max():.2f}")

    if {
        "initial_implementation_time",
        "post_review_implementation_time",
    }.issubset(df.columns):

        complete_review = df[
            "post_review_implementation_time"
        ].notna()

        lines.append("\nMEASUREMENT-BOUNDARY AVAILABILITY")
        lines.append("-" * 80)

        lines.append(
            "Rows with post-review implementation time: "
            f"{complete_review.sum():,}/{len(df):,} "
            f"({pct(complete_review.sum(), len(df)):.1f}%)"
        )

        by_treatment = (
            df.assign(has_post_review=complete_review)
            .groupby("ai_treatment")["has_post_review"]
            .agg(["sum", "count", "mean"])
        )

        by_treatment["percent"] = (
            by_treatment["mean"] * 100
        )

        lines.append("\nPost-review availability by treatment:")
        lines.append(
            by_treatment[
                ["sum", "count", "percent"]
            ].to_string()
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Audit written to: {OUTPUT}")


if __name__ == "__main__":
    main()