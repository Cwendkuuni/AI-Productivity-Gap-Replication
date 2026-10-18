#!/usr/bin/env python3
"""Validate headline manuscript results against reproduced CSVs."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

checks = 0
failures = []


def check(actual, expected, label, tolerance=0.0051):
    global checks

    checks += 1
    actual = float(actual)

    passed = abs(actual - expected) <= tolerance

    print(
        f"{'PASS' if passed else 'FAIL'} "
        f"{label}: {actual:.6f} "
        f"(expected {expected})"
    )

    if not passed:
        failures.append(label)


print("\n=== RQ1: JOINT APG ===")

apg = pd.read_csv(
    TABLES / "apg_joint_cluster_bootstrap.csv"
).set_index("analysis")

expected_apg = {
    "Early-2025 | Pre-task expected":
        (246, 16, 48.90, 33.45, 65.29),

    "Late-2025 All | Pre-task expected":
        (827, 53, 28.32, 5.98, 49.32),

    "Late-2025 All | Post-task perceived":
        (827, 53, 23.18, 2.85, 41.73),

    "Late-2025 Panel | Pre-task expected":
        (137, 9, 3.03, -42.41, 44.07),

    "Late-2025 Panel | Post-task perceived":
        (137, 9, -2.81, -48.01, 31.66),

    "Late-2025 New | Pre-task expected":
        (690, 44, 32.76, 7.97, 55.40),

    "Late-2025 New | Post-task perceived":
        (690, 44, 27.74, 5.24, 47.27),
}

assert apg.index.is_unique
assert set(apg.index) == set(expected_apg)

for name, values in expected_apg.items():
    row = apg.loc[name]
    n, devs, point, low, high = values

    check(row["rows"], n, name + " N", 0)
    check(row["developers"], devs, name + " developers", 0)

    check(row["apg_point_pp"], point, name + " APG")
    check(row["apg_ci_low_pp"], low, name + " CI low")
    check(row["apg_ci_high_pp"], high, name + " CI high")

    check(row["bootstrap_requested"], 20000,
          name + " requested", 0)
    check(row["bootstrap_valid"], 20000,
          name + " valid", 0)
    check(row["bootstrap_failed"], 0,
          name + " failed", 0)

    check(
        row["subjective_point_pct"]
        - row["observed_point_pct"],
        row["apg_point_pp"],
        name + " APG identity",
        1e-8,
    )


print("\n=== RQ2: COMMON-SAMPLE BOUNDARY ===")

boundary = pd.read_csv(
    TABLES / "metr_late_common_sample_boundary.csv"
).set_index("cohort")

expected_boundary = {
    "All developers":
        (824, 52, -3.23, -11.56, 1.65),

    "Panel developers":
        (137, 9, -3.28, -16.42, 7.34),

    "New developers":
        (687, 43, -3.36, -13.76, 1.94),
}

assert boundary.index.is_unique
assert set(boundary.index) == set(expected_boundary)

for name, values in expected_boundary.items():
    row = boundary.loc[name]

    n, devs, shift, low, high = values

    check(row["n"], n, name + " N", 0)
    check(row["n_developers"], devs,
          name + " developers", 0)

    check(
        row["productivity_boundary_shift_pp"],
        shift,
        name + " shift",
    )

    check(
        row["productivity_shift_ci_low_pp"],
        low,
        name + " CI low",
    )

    check(
        row["productivity_shift_ci_high_pp"],
        high,
        name + " CI high",
    )

    check(row["n_boot"], 5000,
          name + " bootstrap", 0)

    check(
        row["total_productivity_effect_pct"]
        - row["initial_productivity_effect_pct"],
        row["productivity_boundary_shift_pp"],
        name + " boundary identity",
        1e-8,
    )

print("\n=== BOOTSTRAP SUMMARY CONSISTENCY ===")

boot = pd.read_csv(
    TABLES / "metr_late_common_sample_boundary_bootstrap.csv"
).set_index("cohort")

assert boot.index.is_unique
assert set(boot.index) == set(boundary.index)

boot_columns = [
    "n_boot",
    "time_shift_boot_mean_pp",
    "time_shift_ci_low_pp",
    "time_shift_ci_high_pp",
    "productivity_shift_boot_mean_pp",
    "productivity_shift_ci_low_pp",
    "productivity_shift_ci_high_pp",
]

for name in boundary.index:
    for col in boot_columns:
        check(
            boot.loc[name, col],
            boundary.loc[name, col],
            f"{name} bootstrap {col}",
            1e-8,
        )


print("\n=== RQ3: WORKFLOW BOUNDARIES ===")

slr = pd.read_csv(
    ROOT / "data/derived/slr_boundary_current39_final.csv"
)

assert slr["current_study_id"].is_unique

eligibility = slr["verified_eligibility"].value_counts()
boundaries = slr["verified_boundary"].value_counts()

check(len(slr), 39, "RQ3 total studies", 0)
check(slr["current_study_id"].nunique(), 39, "RQ3 unique IDs", 0)

for category, expected in {
    "ELIGIBLE": 19,
    "NOT_ELIGIBLE": 18,
    "UNRESOLVED": 2,
}.items():
    check(
        eligibility.get(category, 0),
        expected,
        f"RQ3 {category}",
        0,
    )

assert set(eligibility.index) == {
    "ELIGIBLE", "NOT_ELIGIBLE", "UNRESOLVED"
}

for boundary, expected in {
    "B1": 3,
    "B2": 11,
    "B3": 3,
    "B4": 2,
}.items():
    check(
        boundaries.get(boundary, 0),
        expected,
        f"RQ3 {boundary}",
        0,
    )

assert set(boundaries.index) == {"B1", "B2", "B3", "B4"}

eligible = slr["verified_eligibility"].eq("ELIGIBLE")
coded = slr["verified_boundary"].notna()

check(
    (eligible == coded).sum(),
    39,
    "RQ3 eligible/boundary consistency",
    0,
)

check(
    (boundaries.get("B3", 0) + boundaries.get("B4", 0))
    / eligible.sum() * 100,
    26.31578947368421,
    "RQ3 broad boundary percentage",
    1e-8,
)

print("\n=== VALIDATION SUMMARY ===")
print("Checks:", checks)
print("Failures:", len(failures))

if failures:
    for item in failures:
        print("FAILED:", item)
    raise SystemExit(1)

print("MANUSCRIPT_NUMERICAL_VALIDATION_PASSED")
