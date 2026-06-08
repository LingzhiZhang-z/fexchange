#!/usr/bin/env bash
set -euo pipefail

input_root="${1:-data/data-DFT-input}"
dft_root="${2:-data/data-DFT}"
out_name="enumerate_nn_bonds.txt"

count=0
missing=0

while IFS= read -r material_dir; do
  category="$(basename "$(dirname "$material_dir")")"
  material="$(basename "$material_dir")"
  [ "$category" = "kramer" ] && continue

  src_dir="$dft_root/$category/$material/wannier"
  wout="$(find "$src_dir" -maxdepth 1 -type f -name '*.wout' ! -name '*.pp.wout' | sort | head -n 1 || true)"
  if [ -z "$wout" ]; then
    printf 'missing_wout\t%s/%s\n' "$category" "$material" >&2
    missing=$((missing + 1))
    continue
  fi

  python scripts/bond/enumerate.py "$wout" > "$material_dir/$out_name"
  count=$((count + 1))
done < <(find "$input_root" -mindepth 2 -maxdepth 2 -type d | sort)

printf 'wrote\t%s\nmissing\t%s\n' "$count" "$missing"
