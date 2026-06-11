#!/usr/bin/env bash
# Build the fopt_plus_sopt_direct branch = elementwise sum of the fopt and
# sopt_direct exchange tensors, written in the SAME organized U_NN.txt format:
#
#   <DST>/<fam>/<mat>/<bond>/fopt/<run>/U_NN.txt
# + <DST>/<fam>/<mat>/<bond>/sopt_direct/<run>/U_NN.txt
# = <DST>/<fam>/<mat>/<bond>/fopt_plus_sopt_direct/<run>/U_NN.txt
#
# Effective exchange tensors are additive, so we sum the 9 J components (and the
# residual) row by row (the two branches share ratio order and run_id/U layout).
#
# Usage:  combine_branches.sh <organized_root>
set -euo pipefail

DST="${1:?usage: combine_branches.sh <organized_root>}"

find "$DST" -type d -name fopt | while read -r fdir; do          # .../<bond>/fopt
  sdir="${fdir%/fopt}/sopt_direct"
  cdir="${fdir%/fopt}/fopt_plus_sopt_direct"
  [ -d "$sdir" ] || continue
  find "$fdir" -name 'U_*.txt' | while read -r ff; do            # fopt/<run>/U_NN.txt
    rel="${ff#"$fdir"/}"                                          # <run>/U_NN.txt
    sf="$sdir/$rel"; cf="$cdir/$rel"
    [ -f "$sf" ] || continue
    mkdir -p "$(dirname "$cf")"
    # sum residual (col 3) and Jxx..Jzz (cols 4..12), keep ratio/label from file 2
    awk '
      NR==FNR { a[FNR]=$0; next }
      /^#/    { print; next }
      { split(a[FNR], A); printf "%s total %.12g", $1, A[3]+$3
        for (i = 4; i <= 12; i++) printf " %.12g", A[i]+$i
        printf "\n" }
    ' "$ff" "$sf" > "$cf"
  done
done

echo "# combined: $(find "$DST" -path '*fopt_plus_sopt_direct*' -name 'U_*.txt' | wc -l) files"
