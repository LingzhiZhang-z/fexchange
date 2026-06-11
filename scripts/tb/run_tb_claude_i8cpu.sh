#!/bin/bash
#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 00:29:00
#SBATCH -J fxe_tb_claude
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

# Run the tb-claude shell-extension analysis for all materials in parallel
# (one single-process python per material, 4 BLAS threads each; plots are
# rendered afterwards on the login node where gnuplot exists).

set -uo pipefail
REPO=/home/k0282/k028230/code/fexchange
PYTHON=${PYTHON:-/home2/k0282/k028230/optz/miniforge3/envs/fexchange/bin/python}

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$REPO"
LOG_DIR="logs/tb-claude-${SLURM_JOB_ID:-manual}"
mkdir -p "$LOG_DIR"

targets=${TB_CLAUDE_TARGETS:-$(cd data/data-DFT-input && ls */*/enumerate_nn_bonds.txt | xargs -n1 dirname)}
echo "[$(date)] launching $(echo "$targets" | wc -l) materials (extra flags: ${TB_CLAUDE_FLAGS:-none})"
for t in $targets; do
    "$PYTHON" -B scripts/tb/tb_claude_shells.py "$t" \
        --dmax 9 --max-pp 10 --max-fp 6 --max-ff 6 --no-plot ${TB_CLAUDE_FLAGS:-} \
        > "$LOG_DIR/$(echo "$t" | tr / _).log" 2>&1 &
done

rc=0
for p in $(jobs -p); do
    wait "$p" || rc=$?
done
echo "[$(date)] all done rc=$rc"
exit "$rc"
