#!/bin/bash
#SBATCH -p i8cpu
#SBATCH -N 5
#SBATCH --ntasks-per-node=128
#SBATCH -J fxe_rechx5
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err

set -euo pipefail

REPO=/home/k0282/k028230/code/fexchange
PYTHON=${PYTHON:-/home2/k0282/k028230/optz/miniforge3/envs/fexchange/bin/python}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$REPO"
mkdir -p logs
LOG_DIR="logs/sweep-dft-${SLURM_JOB_ID:-manual}"
mkdir -p "$LOG_DIR"

run_material() {
    local label=$1
    local root=$2
    local limit=$3
    local mpi=$4
    local omp=$5

    (
        set -euo pipefail
        cd "$REPO"

        mapfile -t tomls < <(find "$root" -path '*/sweep-dft/*/*.toml' -type f | sort)
        if [ "$limit" -gt 0 ]; then
            tomls=("${tomls[@]:0:$limit}")
        fi

        echo "[$(date)] $label: ${#tomls[@]} sweep inputs, mpi=$mpi, omp=$omp"
        for toml in "${tomls[@]}"; do
            echo "[$(date)] $label START $toml"
            OMP_NUM_THREADS=$omp \
            OPENBLAS_NUM_THREADS=$omp \
            MKL_NUM_THREADS=$omp \
            BLIS_NUM_THREADS=$omp \
            NUMEXPR_NUM_THREADS=$omp \
                srun --exclusive -N 1 -n "$mpi" -c "$omp" \
                "$PYTHON" -B -m fexchange.cli sweep "$toml" --log-level INFO
            echo "[$(date)] $label DONE  $toml"
        done
        echo "[$(date)] $label complete"
    ) > "$LOG_DIR/${label}.log" 2>&1 &
}

run_material YbOCl "$REPO/data/data-DFT-input/REOCl/YbOCl" 0 128 1
PID1=$!

run_material ErOCl "$REPO/data/data-DFT-input/REOCl/ErOCl" 0 128 1
PID2=$!

run_material NdSI "$REPO/data/data-DFT-input/RESI/NdSI-re" 0 128 1
PID3=$!

run_material DyOF "$REPO/data/data-DFT-input/REOF/DyOF" 2 64 2
PID4=$!

run_material SmSI "$REPO/data/data-DFT-input/RESI/SmSI-re" 2 64 2
PID5=$!

rc=0
for p in $PID1 $PID2 $PID3 $PID4 $PID5; do
    wait "$p" || rc=$?
done

echo "[$(date)] All done (rc=$rc)"
exit "$rc"
