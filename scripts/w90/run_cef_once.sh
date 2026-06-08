#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home2/k0282/k028230/optz/miniforge3/envs/fexchange/bin/python}

if [[ $# -ne 2 ]]; then
  cat <<'EOF'
Usage:
  scripts/w90/run_cef_once.sh path/to/material bond_label

Example:
  scripts/w90/run_cef_once.sh data/data-DFT-input/REOCl/ErOCl J1-z
  scripts/w90/run_cef_once.sh data/data-DFT-input/REX3-C2m/DyBr3 x

Env:
  PYTHON=/path/to/python
  CEF_SYMMETRY=C3v
  CEF_MODE_Q3=sin
  CEF_RE=auto
  CEF_N_ELE=
  CEF_NO_CEF=0
EOF
  exit 2
fi

material_dir=$1
bond_label=$2
onsite="$material_dir/$bond_label/w90/onsite.txt"
out_dir="$material_dir/cef"

if [[ ! -f "$onsite" ]]; then
  echo "missing $onsite; run extract for the representative bond first" >&2
  exit 1
fi

args=("$onsite" "--symmetry" "${CEF_SYMMETRY:-C3v}" "--mode-q3" "${CEF_MODE_Q3:-sin}" "--output-dir" "$out_dir")
if [[ "${CEF_NO_CEF:-0}" = "1" ]]; then
  args+=("--no-cef")
fi
if [[ -n "${CEF_RE:-}" ]]; then
  args+=("--RE" "$CEF_RE")
fi
if [[ -n "${CEF_N_ELE:-}" ]]; then
  args+=("--n-ele" "$CEF_N_ELE")
fi

"$PYTHON" -m fexchange.tools.w90_onsite "${args[@]}"
if [[ "${CEF_NO_CEF:-0}" = "1" ]]; then
  rm -f "$out_dir/kramer_projector.txt"
fi
{
  printf 'material_dir %s\n' "$material_dir"
  printf 'bond_label %s\n' "$bond_label"
  printf 'onsite %s\n' "$onsite"
} > "$out_dir/source.txt"
