#!/usr/bin/env bash
#SBATCH --job-name=l200_timing
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=logs/l200_timing_benchmark/%j.out
#SBATCH --error=logs/l200_timing_benchmark/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

# One single-core job (trivially within the <= 1000-job cap) that times every
# (v, solver, dt) combo on the same node, so the wall-times are directly comparable.
# The 24h wall bounds the combo count; keep the sweep small and the operator reads the
# resulting table by hand to size Stage-1 and Stage-2.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

if ! command -v julia >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
        module load julia
    fi
fi

command -v julia >/dev/null 2>&1 || {
    echo "julia is not available; load the Julia module or set JULIA_BIN before submitting" >&2
    exit 1
}

L="${L:-200}"
GAMMA="${GAMMA:-1}"
J_VALUE="${J_VALUE:-2}"
VELOCITIES="${VELOCITIES:-0.1,1.0,10.0}"
SOLVERS="${SOLVERS:-SRA1,SRA2}"
DTS="${DTS:-0.001,0.0005}"
DR="${DR:-0.25}"
R_MAX="${R_MAX:-45}"
T_MAX="${T_MAX:-16}"
NTIMES="${NTIMES:-8}"
REPS="${REPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-results/l200_timing_benchmark}"

mkdir -p logs/l200_timing_benchmark "${OUTPUT_DIR}"

julia --project=. scripts/run_l200_timing_benchmark_cluster.jl \
    --L "${L}" \
    --gamma "${GAMMA}" \
    --J "${J_VALUE}" \
    --velocities "${VELOCITIES}" \
    --solvers "${SOLVERS}" \
    --dts "${DTS}" \
    --dr "${DR}" \
    --r-max "${R_MAX}" \
    --T-max "${T_MAX}" \
    --ntimes "${NTIMES}" \
    --reps "${REPS}" \
    --output-dir "${OUTPUT_DIR}"
