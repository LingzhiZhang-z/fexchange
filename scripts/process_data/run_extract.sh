#!/usr/bin/env bash
# Choose which runs to extract, then extract each into the organized tree.
#
# By default it discovers every run (the parent of the U<NN> dirs) under SRC and
# extracts them all. To run a subset, replace the CASES discovery with a literal
# list, e.g.:
#   CASES=( REX3-C2m/DyBr3/y/sopt/f9_dy_g6  REX3-R3/YbI3/z/fopt/f9_yb_g6 )
#
# SRC must be the outputs ROOT (the dir holding the families) so the case
# relpath keeps family/material/bond/branch/run_id.
#
# Usage:  [SRC=... DST=...] scripts/process_data/run_extract.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${SRC:-/work/k0282/k028230/code/fexchange/outputs}"
DST="${DST:-/work/k0282/k028230/code/fexchange/outputs_organized}"

# U<NN> dirs sit at a fixed depth (family/material/bond/branch/run_id/U = 6 below
# SRC); bound the find to that depth so it never crawls the huge jh*/L3 subtrees.
mapfile -t CASES < <(
  find "$SRC" -mindepth 6 -maxdepth 6 -type d -name 'U[0-9]*' \
    | sed -E 's#/U[0-9]+$##' | sort -u \
    | while read -r d; do echo "${d#"$SRC"/}"; done
)

echo "# ${#CASES[@]} cases  ->  $DST  (parallel x${JOBS:-8})"
# cases are independent (each writes its own subtree) -> parallelise the I/O-bound
# reads across JOBS workers; override with JOBS=N.
printf '%s\n' "${CASES[@]}" \
  | xargs -P "${JOBS:-8}" -I{} "$HERE/extract_exchange.sh" "$SRC" "$DST" {}
echo "# done: $(find "$DST" -name 'U_*.txt' 2>/dev/null | wc -l) files in $DST"
