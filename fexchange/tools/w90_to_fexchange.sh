#!/usr/bin/env bash
set -euo pipefail
cd /Users/lingzhi/Documents/Code/fexchange


python ./fexchange/tools/w90_extract.py ./data/YbOCl/SOC/config.toml

python ./fexchange/tools/w90_decompose_h_local.py \
    --h_local_dat ./data/YbOCl/test/w90/w90_soc_h_local.dat \
    --n_ele 13 \
    --symmetry C3v --mode_q3 sin
