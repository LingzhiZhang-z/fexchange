#!/usr/bin/env bash
set -euo pipefail

repo=/home/k0282/k028230/code/fexchange
python_bin=${PYTHON:-/home2/k0282/k028230/optz/miniforge3/envs/fexchange/bin/python}
input_dir="$repo/scripts/sweep-dft/kramer"
output_dir="$repo/data/data-DFT-input/kramer/REChX"

for name in \
  YbOCl_exp_baseline_Yb_J7_2 \
  ErOCl_exp_baseline_Er_J15_2 \
  NdOF_exp_baseline_Nd_J9_2 \
  YbOCl_exp_baseline_Dy_J15_2 \
  ErOCl_exp_baseline_Dy_J15_2 \
  NdOF_exp_baseline_Dy_J15_2
do
  "$python_bin" -B -m fexchange.tools.cef_states \
    "$input_dir/$name.toml" \
    --format projector \
    --output "$output_dir/${name}_projector.txt" \
    --silent
  echo "wrote $output_dir/${name}_projector.txt"
done
