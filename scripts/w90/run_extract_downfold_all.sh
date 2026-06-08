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

total_bonds=$(find "$INPUT_ROOT" -path '*/bond_info.txt' -not -path '*/kramer/*' | wc -l)
done_bonds=0
start_time=$SECONDS

for material_idx in "${!material_dirs[@]}"; do
  material_dir=${material_dirs[$material_idx]}
  rel_material=${material_dir#"$INPUT_ROOT"/}
  echo
  echo "=== material [$((material_idx + 1))/${#material_dirs[@]}]: $rel_material ==="

  mapfile -t bond_infos < <(find "$material_dir" -mindepth 2 -maxdepth 2 -name bond_info.txt | sort)
  if [[ "${#bond_infos[@]}" -eq 0 ]]; then
    echo "no bond_info.txt under $material_dir" >&2
    exit 1
  fi

  for bond_idx in "${!bond_infos[@]}"; do
    bond_info=${bond_infos[$bond_idx]}
    bond_dir=$(dirname "$bond_info")
    w90_dir="$bond_dir/w90"
    bond_label=$(basename "$bond_dir")
    echo "-- bond [$((done_bonds + 1))/$total_bonds] material-bond [$((bond_idx + 1))/${#bond_infos[@]}]: $bond_label"
    "$PYTHON" "$SCRIPT_DIR/prepare_extract.py" "$bond_info"
    "$PYTHON" -m fexchange.tools.w90_extract "$w90_dir/extract.toml"
    "$PYTHON" "$SCRIPT_DIR/prepare_downfold.py" "$w90_dir"
    "$PYTHON" -m fexchange.tools.w90_downfold "$w90_dir/downfold.toml"
    done_bonds=$((done_bonds + 1))
    echo "-- done [$done_bonds/$total_bonds]: $rel_material/$bond_label elapsed=$((SECONDS - start_time))s"
  done

done

echo
echo "done: $done_bonds/$total_bonds bonds elapsed=$((SECONDS - start_time))s"
