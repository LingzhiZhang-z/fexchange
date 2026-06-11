#!/bin/bash
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 00:29:00
#SBATCH -J fxe_tb_gpt_2d
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

set -uo pipefail

REPO=/home/k0282/k028230/code/fexchange
PYTHON=${PYTHON:-/home2/k0282/k028230/optz/miniforge3/envs/fexchange/bin/python}

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$REPO"
LOG_DIR="logs/tb-gpt-strict2d-${SLURM_JOB_ID:-manual}"
mkdir -p "$LOG_DIR"

targets=${TB_GPT_TARGETS:-$(cd data/data-DFT-input && ls */*/enumerate_nn_bonds.txt | xargs -n1 dirname)}
echo "[$(date)] launching $(echo "$targets" | wc -w) materials (extra flags: ${TB_GPT_FLAGS:-none})"
for target in $targets; do
    "$PYTHON" -B scripts/tb/tb_gpt_strict2d.py "$target" ${TB_GPT_FLAGS:-} \
        > "$LOG_DIR/$(echo "$target" | tr / _).log" 2>&1 &
done

rc=0
for pid in $(jobs -p); do
    wait "$pid" || rc=$?
done
echo "[$(date)] all done rc=$rc"
exit "$rc"
