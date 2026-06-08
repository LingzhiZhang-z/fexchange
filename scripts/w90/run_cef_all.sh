#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-/home2/k0282/k028230/optz/miniforge3/envs/fexchange/bin/python}
INPUT_ROOT=${1:-"$ROOT/data/data-DFT-input"}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "missing INPUT_ROOT: $INPUT_ROOT" >&2
  exit 1
fi

mapfile -t material_dirs < <(
  find "$INPUT_ROOT" -path '*/bond_info.txt' -not -path '*/kramer/*' \
    | sed 's#/[^/]*/bond_info.txt$##' \
    | sort -u
)

if [[ "${#material_dirs[@]}" -eq 0 ]]; then
  echo "no material directories with bond_info.txt under $INPUT_ROOT" >&2
  exit 1
fi

echo "INPUT_ROOT: $INPUT_ROOT"
echo "materials: ${#material_dirs[@]}"

done_materials=0
start_time=$SECONDS

for material_idx in "${!material_dirs[@]}"; do
  material_dir=${material_dirs[$material_idx]}
  rel_material=${material_dir#"$INPUT_ROOT"/}
  if [[ -f "$material_dir/x/bond_info.txt" ]]; then
    cef_bond=x
  elif [[ -f "$material_dir/J1-z/bond_info.txt" ]]; then
    cef_bond=J1-z
  else
    echo "missing representative CEF bond x or J1-z under $material_dir" >&2
    exit 1
  fi

  echo
  if [[ "$rel_material" == REX3-*/* ]]; then
    echo "=== onsite [$((material_idx + 1))/${#material_dirs[@]}]: $rel_material / $cef_bond ==="
    CEF_NO_CEF=1 "$SCRIPT_DIR/run_cef_once.sh" "$material_dir" "$cef_bond"
    rm -f "$material_dir/cef/kramer_projector.txt"
  else
    echo "=== cef [$((material_idx + 1))/${#material_dirs[@]}]: $rel_material / $cef_bond ==="
    "$SCRIPT_DIR/run_cef_once.sh" "$material_dir" "$cef_bond"
    "$PYTHON" -m fexchange.tools.cef_states \
      "$material_dir/cef/cef_REChX_C3v_sin.toml" \
      --format projector \
      --output "$material_dir/cef/kramer_projector.txt"
  fi
  done_materials=$((done_materials + 1))
  echo "=== done [$done_materials/${#material_dirs[@]}]: $rel_material elapsed=$((SECONDS - start_time))s ==="
done

echo
echo "done: $done_materials/${#material_dirs[@]} materials elapsed=$((SECONDS - start_time))s"
