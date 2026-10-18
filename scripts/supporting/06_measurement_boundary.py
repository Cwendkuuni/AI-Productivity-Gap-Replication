#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "metr_early" / "data_complete.csv"
OUTPUT = ROOT / "results" / "tables" / "metr_early_measurement_boundary.csv"
LOG = ROOT / "results" / "logs" / "metr_early_measurement_boundary.txt"


RNG = np.random.default_rng(42)


def pct_effect(ai, control):
    """
    Relative change in mean time:
    positive = AI takes longer
    negative = AI is faster
    """
    return (ai.mean() / control.mean()) - 1.0


def bootstrap_effect(ai, control, n_boot=10000):
    ai = np.asarray(ai)
    control = np.asarray(control)

    effects = np.empty(n_boot)

    for i in range(n_boot):
        ai_sample = RNG.choice(ai, size=len(ai), replace=True)
        control_sample = RNG.choice(control, size=len(control), replace=True)

        effects[i] = (
            ai_sample.mean() / control_sample.mean()
        ) - 1.0

    low, high = np.percentile(effects, [2.5, 97.5])

    return effects.mean(), low, high


def summarize(df, column, label):
    tmp = df[["ai_treatment", column]].dropna()

    control = tmp.loc[
        tmp["ai_treatment"] == 0, column
    ].to_numpy()

    ai = tmp.loc[
        tmp["ai_treatment"] == 1, column
    ].to_numpy()

    raw_effect = pct_effect(ai, control)

    boot_mean, ci_low, ci_high = bootstrap_effect(
        ai,
        control
    )

    # Mann-Whitney is useful here because times are skewed.
    mw = stats.mannwhitneyu(
        ai,
        control,
        alternative="two-sided"
    )

    return {
        "boundary": label,
        "n_total": len(tmp),
        "n_control": len(control),
        "n_ai": len(ai),
        "control_mean": control.mean(),
        "control_median": np.median(control),
        "ai_mean": ai.mean(),
        "ai_median": np.median(ai),
        "relative_effect": raw_effect,
        "relative_effect_pct": 100 * raw_effect,
        "bootstrap_effect": boot_mean,
        "bootstrap_ci_low_pct": 100 * ci_low,
        "bootstrap_ci_high_pct": 100 * ci_high,
        "mannwhitney_u": mw.statistic,
        "mannwhitney_p": mw.pvalue,
    }


def main():
    df = pd.read_csv(INPUT)

    # Boundary 1: initial implementation
    initial = summarize(
        df,
        "initial_implementation_time",
        "Initial implementation"
    )

    # Boundary 2: post-review only
    post_review = summarize(
        df,
        "post_review_implementation_time",
        "Post-review implementation"
    )

    # Boundary 3: total observed delivery effort
    df["observed_delivery_time"] = (
        df["initial_implementation_time"]
        + df["post_review_implementation_time"]
    )

    delivery = summarize(
        df,
        "observed_delivery_time",
        "Initial + post-review"
    )

    results = pd.DataFrame(
        [initial, post_review, delivery]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    lines = []

    lines.append("=" * 88)
    lines.append("METR EARLY-2025 — MEASUREMENT BOUNDARY ANALYSIS")
    lines.append("=" * 88)

    for _, row in results.iterrows():
        lines.append("")
        lines.append(row["boundary"].upper())
        lines.append("-" * 88)

        lines.append(
            f"N = {int(row['n_total'])} "
            f"(control={int(row['n_control'])}, "
            f"AI={int(row['n_ai'])})"
        )

        lines.append(
            f"Control mean/median: "
            f"{row['control_mean']:.2f} / "
            f"{row['control_median']:.2f}"
        )

        lines.append(
            f"AI mean/median: "
            f"{row['ai_mean']:.2f} / "
            f"{row['ai_median']:.2f}"
        )

        lines.append(
            f"Relative AI effect: "
            f"{row['relative_effect_pct']:+.2f}%"
        )

        lines.append(
            "Bootstrap 95% CI: "
            f"[{row['bootstrap_ci_low_pct']:+.2f}%, "
            f"{row['bootstrap_ci_high_pct']:+.2f}%]"
        )

        lines.append(
            f"Mann-Whitney p-value: "
            f"{row['mannwhitney_p']:.6f}"
        )

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Table written to: {OUTPUT}")
    print(f"Log written to:   {LOG}")


if __name__ == "__main__":
    main()