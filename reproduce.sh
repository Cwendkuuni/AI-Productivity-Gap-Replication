#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "=========================================="
echo " AI Productivity Gap - Reproduction"
echo "=========================================="
echo

echo "[1] Checking environment"

"$PYTHON" --version

"$PYTHON" - <<'PY'
import importlib

packages = [
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "matplotlib",
    "openpyxl",
]

for package in packages:
    module = importlib.import_module(package)
    print(
        package,
        getattr(module, "__version__", "installed")
    )
PY

command -v pdftotext >/dev/null 2>&1 || {
    echo "ERROR: pdftotext is required."
    echo "On Ubuntu/Debian: sudo apt install poppler-utils"
    exit 1
}

echo
echo "[2] Checking input files"

INPUTS=(
    "data/raw/metr_early/data_complete.csv"
    "data/raw/metr_late/data.csv"
    "data/raw/slr_artifact/SLR_all_included_studies.xlsx"
    "data/raw/slr_artifact/Tables.pdf"
    "data/raw/slr_artifact/Readme.md"
)

for file in "${INPUTS[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Missing input: $file"
        exit 1
    fi

    echo "OK: $file"
done

mkdir -p \
    results/tables \
    results/figures \
    results/logs

echo
echo "[3] Running canonical pipeline"

run_script() {
    local script="$1"

    echo
    echo "=========================================="
    echo "RUNNING: $script"
    echo "=========================================="

    if [[ -n "${CPU_AFFINITY:-}" ]]; then
        taskset -c "$CPU_AFFINITY" \
            "$PYTHON" -X faulthandler "scripts/$script"
    else
        "$PYTHON" -X faulthandler "scripts/$script"
    fi

    echo "SUCCESS: $script"
}

run_script "01_audit_metr_early.py"
run_script "02_audit_metr_late.py"

run_script "08_productivity_gap_early.py"
run_script "09_estimation_bias_early.py"
run_script "10_productivity_gap_late.py"

run_script "12_measurement_boundary_late.py"
run_script "12b_common_sample_boundary_late.py"

run_script "19_joint_cluster_bootstrap_apg.py"
run_script "14_plot_productivity_gap.py"

run_script "16_audit_slr_artifact.py"
run_script "17_audit_slr_tables.py"
run_script "18_reconcile_slr_space.py"

echo
echo "=========================================="
echo " Canonical pipeline completed successfully"
echo "=========================================="
