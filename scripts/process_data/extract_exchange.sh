#!/usr/bin/env bash
# Extract ONE run's Jh/U sweeps into compact per-U files, preserving the tree:
#
#   <SRC>/<case>/U<NN>/jh<NN>/L3/exchange.txt
#       ->  <DST>/<case>/U_<NN>.txt          case = family/material/bond/branch/run_id
#
# Each output file is one Jh/U sweep -- a header, then one row per jh:
#
#   <ratio> total <residual> Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz
#
# ratio = Jh/U = jh index / 100 (jh00..jh40 -> 0.00..0.40). Only the 'total' row
# is kept (grep drops fopt's P1..P5 process rows).
#
# Usage:  extract_exchange.sh <src_root> <dst_root> <case_relpath>
set -euo pipefail

SRC="${1:?usage: extract_exchange.sh <src_root> <dst_root> <case_relpath>}"
DST="${2:?dst_root}"
CASE="${3:?case relpath: family/material/bond/branch/run_id}"
HEADER="# ratio label residual Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz"

run="$SRC/$CASE"
[ -d "$run" ] || { echo "missing run dir: $run" >&2; exit 1; }

for udir in "$run"/U[0-9]*; do
  [ -d "$udir" ] || continue
  U="$(basename "$udir")"                       # U06
  out="$DST/$CASE/U_${U#U}.txt"                 # .../U_06.txt
  mkdir -p "$(dirname "$out")"
  # one grep over the whole U sweep; sed turns each .../jhNN/... path into ratio 0.NN
  printf '%s\n' "$HEADER" > "$out"
  grep -H '^total' "$udir"/jh[0-9]*/L3/exchange.txt 2>/dev/null \
    | sed -E 's#.*/jh([0-9]+)/L3/exchange.txt:#0.\1 #' >> "$out" || true
done
