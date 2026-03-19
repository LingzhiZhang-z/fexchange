#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

OUTPUT_ROOT="$REPO_ROOT/outputs/nature_fig3_particle"
FIG3_OUTPUT_ROOT="$OUTPUT_ROOT/fig3"
TABLE2_OUTPUT_ROOT="$OUTPUT_ROOT/table2"
WORK_ROOT="$OUTPUT_ROOT/workdirs"
FIG3_WORK_ROOT="$WORK_ROOT/fig3"
TABLE2_WORK_ROOT="$WORK_ROOT/table2"
PLOT_SCRIPT="$REPO_ROOT/tests/validation/plot_nature_fig3_abef.gp"

FIG3_CASES=(f1_ce_g7 f3_nd_g6 f11_er_g7 f13_yb_g6)
TABLE2_CASES=(f1_ce_g7 f3_nd_g6 f11_er_g7 f13_yb_g6)
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

SMOKE=0
SKIP_FIG3=0
SKIP_TABLE2=0
PREPARE_ONLY=0
CUSTOM_FIG3_CASES=0
CUSTOM_TABLE2_CASES=0

while (($#)); do
    case "$1" in
        --smoke)
            SMOKE=1
            ;;
        --skip-fig3)
            SKIP_FIG3=1
            ;;
        --skip-table2)
            SKIP_TABLE2=1
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            ;;
        --fig3-case)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value for --fig3-case" >&2
                exit 2
            fi
            if [ "$CUSTOM_FIG3_CASES" -eq 0 ]; then
                FIG3_CASES=()
                CUSTOM_FIG3_CASES=1
            fi
            FIG3_CASES+=("$1")
            ;;
        --table2-case)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value for --table2-case" >&2
                exit 2
            fi
            if [ "$CUSTOM_TABLE2_CASES" -eq 0 ]; then
                TABLE2_CASES=()
                CUSTOM_TABLE2_CASES=1
            fi
            TABLE2_CASES+=("$1")
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if [ "$SMOKE" -eq 1 ]; then
    if [ "$CUSTOM_FIG3_CASES" -eq 0 ]; then
        FIG3_CASES=(f1_ce_g7)
    fi
    if [ "$CUSTOM_TABLE2_CASES" -eq 0 ]; then
        TABLE2_CASES=(f1_ce_g7)
    fi
    RATIOS=(0.1)
fi

PLOT_TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fexchange-nature.XXXXXX")"
trap 'rm -rf "$PLOT_TMP_ROOT"' EXIT

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

join_csv() {
    local IFS=,
    echo "$*"
}

prepare_assets() {
    local fig3_cases_csv=""
    local table2_cases_csv=""
    local ratios_csv=""

    if [ "$SKIP_FIG3" -eq 0 ]; then
        fig3_cases_csv="$(join_csv "${FIG3_CASES[@]}")"
        ratios_csv="$(join_csv "${RATIOS[@]}")"
    fi
    if [ "$SKIP_TABLE2" -eq 0 ]; then
        table2_cases_csv="$(join_csv "${TABLE2_CASES[@]}")"
    fi

    FIG3_CASES_ENV="$fig3_cases_csv" \
    TABLE2_CASES_ENV="$table2_cases_csv" \
    RATIOS_ENV="$ratios_csv" \
    "$PYTHON_BIN" - <<'PY'
import os

from tests.validation import run_nature_fig3_particle as runner


def parse_cases(env_name: str):
    raw = os.environ.get(env_name, "")
    if not raw:
        return ()
    return tuple(runner.case_by_key(key) for key in raw.split(",") if key)


def parse_ratios():
    raw = os.environ.get("RATIOS_ENV", "")
    if not raw:
        return runner.RATIO_VALUES
    return tuple(float(value) for value in raw.split(",") if value)


runner.prepare_static_assets()

fig3_cases = parse_cases("FIG3_CASES_ENV")
if fig3_cases:
    runner.prepare_fig3_assets(cases=fig3_cases, ratios=parse_ratios())

table2_cases = parse_cases("TABLE2_CASES_ENV")
if table2_cases:
    runner.prepare_table2_assets(cases=table2_cases)
PY
}

run_cli() {
    local run_input="$1"
    local workdir="$2"
    local log_file="$3"
    local local_run_input="$workdir/run_input.toml"

    mkdir -p "$workdir"
    (
        cd "$workdir"
        cp "$run_input" "$local_run_input"
        "$PYTHON_BIN" -m fexchange.cli run run_input.toml
    ) >"$log_file" 2>&1
}

copy_point_result() {
    local workdir="$1"
    local log_file="$2"
    local dest="$3"
    local point_rel

    point_rel="$(awk -F= '/point_result=/{print $2}' "$log_file" | tail -n1 | tr -d '\r')"
    if [ -z "$point_rel" ]; then
        echo "Missing point_result in $log_file" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dest")"
    cp "$workdir/$point_rel" "$dest"
}

emit_fig3_row() {
    local ratio="$1"
    local point_result="$2"

    RATIO="$ratio" POINT_RESULT="$point_result" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

from tests.validation import run_nature_fig3_particle as runner

row = runner.fig3_row_from_point_result(
    ratio=float(os.environ["RATIO"]),
    path=Path(os.environ["POINT_RESULT"]),
)
print(" ".join(f"{value:.12f}" for value in row))
PY
}

write_fig3_case_summary() {
    local case_key="$1"
    local curve_path="$2"
    local case_root="$3"

    CASE_KEY="$case_key" CURVE_PATH="$curve_path" CASE_ROOT="$case_root" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

from tests.validation import run_nature_fig3_particle as runner

case = runner.case_by_key(os.environ["CASE_KEY"])
curve_path = Path(os.environ["CURVE_PATH"])
case_root = Path(os.environ["CASE_ROOT"])
curves = np.loadtxt(curve_path, comments="#", dtype=float)
curves = np.atleast_2d(curves)
payload = {
    "case": case.key,
    "panel": case.panel,
    "curve_path": str(curve_path),
    "reference_path": str(runner.REFERENCE_ROOT / case.reference_file) if case.reference_file else "",
    "compare": None,
}
if case.reference_file:
    payload["compare"] = runner.compare_curves(curves[:, :5], runner.load_reference_curve(case))
(case_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

write_fig3_root_summary() {
    FIG3_OUTPUT_ROOT_ENV="$FIG3_OUTPUT_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FIG3_OUTPUT_ROOT_ENV"])
cases = []
for summary in sorted(root.glob("*/summary.json")):
    cases.append(json.loads(summary.read_text(encoding="utf-8")))
payload = {
    "figure_png": str(root / "nature_fig3_abef.png"),
    "figure_pdf": str(root / "nature_fig3_abef.pdf"),
    "cases": cases,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

write_table2_entry() {
    local case_key="$1"
    local point_result="$2"
    local result_json="$3"

    CASE_KEY="$case_key" POINT_RESULT="$point_result" RESULT_JSON="$result_json" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

from tests.validation import run_nature_fig3_particle as runner

case = runner.case_by_key(os.environ["CASE_KEY"])
entry = runner.table2_from_point_result(case, path=Path(os.environ["POINT_RESULT"]))
Path(os.environ["RESULT_JSON"]).write_text(json.dumps(entry, indent=2), encoding="utf-8")
print(
    f"{entry['case']}\t{entry['panel']}\t{entry['ratio_abs_pfpi_pf_sigma']:.1f}\t"
    f"{entry['J_meV']:.12f}\t{entry['K_meV']:.12f}\t{entry['Gamma_meV']:.12f}\t"
    f"{entry['GammaPrime_meV']:.12f}\t{entry['abs_K_over_J']:.12f}\t"
    f"{entry.get('literature_J_meV', float('nan')):.12f}\t{entry.get('literature_K_meV', float('nan')):.12f}\t"
    f"{entry.get('literature_abs_K_over_J', float('nan')):.12f}\t"
    f"{entry.get('delta_J_meV', float('nan')):.12f}\t{entry.get('delta_K_meV', float('nan')):.12f}\t"
    f"{entry.get('delta_abs_K_over_J', float('nan')):.12f}"
)
PY
}

write_table2_root_summary() {
    local output_root="$1"

    TABLE2_OUTPUT_ROOT_ENV="$output_root" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["TABLE2_OUTPUT_ROOT_ENV"])
cases = []
for result in sorted(root.glob("*/result.json")):
    cases.append(json.loads(result.read_text(encoding="utf-8")))

lines = [
    "case\tpanel\tratio_abs_pfpi_pf_sigma\tJ_meV\tK_meV\tGamma_meV\tGammaPrime_meV\tabs_K_over_J\tliterature_J_meV\tliterature_K_meV\tliterature_abs_K_over_J\tdelta_J_meV\tdelta_K_meV\tdelta_abs_K_over_J"
]
for entry in cases:
    lines.append(
        f"{entry['case']}\t{entry['panel']}\t{entry['ratio_abs_pfpi_pf_sigma']:.1f}\t"
        f"{entry['J_meV']:.12f}\t{entry['K_meV']:.12f}\t{entry['Gamma_meV']:.12f}\t"
        f"{entry['GammaPrime_meV']:.12f}\t{entry['abs_K_over_J']:.12f}\t"
        f"{entry.get('literature_J_meV', float('nan')):.12f}\t{entry.get('literature_K_meV', float('nan')):.12f}\t"
        f"{entry.get('literature_abs_K_over_J', float('nan')):.12f}\t"
        f"{entry.get('delta_J_meV', float('nan')):.12f}\t{entry.get('delta_K_meV', float('nan')):.12f}\t"
        f"{entry.get('delta_abs_K_over_J', float('nan')):.12f}"
    )
(root / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

payload = {
    "summary_tsv": str(root / "summary.tsv"),
    "cases": cases,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_gnuplot() {
    local output_file="$1"
    local term_spec="$2"
    local wrapper="$PLOT_TMP_ROOT/$(basename "$output_file").gp"

    cat >"$wrapper" <<EOF
curve_root = "$FIG3_OUTPUT_ROOT"
reference_root = "$REPO_ROOT/data/test/nature_fig3_particle/reference"
output_file = "$output_file"
term_spec = "$term_spec"
load "$PLOT_SCRIPT"
EOF
    gnuplot "$wrapper" >>"$FIG3_OUTPUT_ROOT/plot.log" 2>&1
}

plot_fig3() {
    : >"$FIG3_OUTPUT_ROOT/plot.log"
    run_gnuplot "$FIG3_OUTPUT_ROOT/nature_fig3_abef.png" "pngcairo size 1400,980 enhanced font 'Helvetica,18'"
    run_gnuplot "$FIG3_OUTPUT_ROOT/nature_fig3_abef.pdf" "pdfcairo enhanced font 'Helvetica,12' size 12in,8in"
}

run_fig3() {
    local case_key ratio run_input workdir case_root log_file point_file curve_file row

    mkdir -p "$FIG3_OUTPUT_ROOT"
    for case_key in "${FIG3_CASES[@]}"; do
        case_root="$FIG3_OUTPUT_ROOT/$case_key"
        rm -rf "$case_root" "$FIG3_WORK_ROOT/$case_key"
        mkdir -p "$case_root/logs" "$case_root/point_results"
        curve_file="$case_root/curve.tsv"
        echo "#ratio J K Gamma GammaPrime symmetry_leakage mapping_residual" >"$curve_file"
        for ratio in "${RATIOS[@]}"; do
            run_input="$REPO_ROOT/data/test/nature_fig3_particle/fig3/$case_key/run_input_ratio_${ratio}.toml"
            workdir="$FIG3_WORK_ROOT/$case_key/r${ratio}"
            log_file="$case_root/logs/run_ratio_${ratio}.log"
            point_file="$case_root/point_results/point_ratio_${ratio}.txt"
            run_cli "$run_input" "$workdir" "$log_file"
            copy_point_result "$workdir" "$log_file" "$point_file"
            row="$(emit_fig3_row "$ratio" "$point_file")"
            echo "$row" >>"$curve_file"
        done
        write_fig3_case_summary "$case_key" "$curve_file" "$case_root"
    done
    plot_fig3
    write_fig3_root_summary
}

run_table2() {
    local case_key run_input workdir case_root log_file point_file result_json row

    mkdir -p "$TABLE2_OUTPUT_ROOT"
    for case_key in "${TABLE2_CASES[@]}"; do
        case_root="$TABLE2_OUTPUT_ROOT/$case_key"
        workdir="$TABLE2_WORK_ROOT/$case_key"
        rm -rf "$case_root" "$workdir"
        mkdir -p "$case_root/logs"
        run_input="$REPO_ROOT/data/test/nature_fig3_particle/table2/$case_key/run_input.toml"
        log_file="$case_root/logs/run.log"
        point_file="$case_root/point_result.txt"
        result_json="$case_root/result.json"
        run_cli "$run_input" "$workdir" "$log_file"
        copy_point_result "$workdir" "$log_file" "$point_file"
        row="$(write_table2_entry "$case_key" "$point_file" "$result_json")"
    done
    write_table2_root_summary "$TABLE2_OUTPUT_ROOT"
}

prepare_assets
mkdir -p "$OUTPUT_ROOT" "$WORK_ROOT" "$FIG3_OUTPUT_ROOT" "$TABLE2_OUTPUT_ROOT"

if [ "$PREPARE_ONLY" -eq 1 ]; then
    echo "Assets prepared under $REPO_ROOT/data/test/nature_fig3_particle"
    exit 0
fi

if [ "$SKIP_FIG3" -eq 0 ]; then
    run_fig3
fi

if [ "$SKIP_TABLE2" -eq 0 ]; then
    run_table2
fi

echo "Outputs written to $OUTPUT_ROOT"
