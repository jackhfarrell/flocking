#!/usr/bin/env bash
#SBATCH --job-name=v10_equil
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=logs/v10_equilibration/%j.out
#SBATCH --error=logs/v10_equilibration/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

# One single-core job that anneals a from-scratch v=10 state on L=200 until the energy
# density and magnetization plateau, recording blocks / steps / model-time / wall-time to
# equilibration — the seed cost for the Stage-1 down-sweep and the L=200 adequacy check at
# the hard end. Trivially within the <= 1000-job cap; the block loop stops as soon as the
# state settles, well inside the 24h wall.

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
V_VALUE="${V_VALUE:-10.0}"
DT="${DT:-0.001}"
SOLVER="${SOLVER:-SRIW1}"
INIT="${INIT:-ordered}"
EQUIL_BLOCK_STEPS="${EQUIL_BLOCK_STEPS:-1000}"
EQUIL_MAX_BLOCKS="${EQUIL_MAX_BLOCKS:-10000}"
BASE_SEED="${BASE_SEED:-810000}"
OUTPUT="${OUTPUT:-results/v10_equilibration/v10_equilibration.jld2}"

mkdir -p logs/v10_equilibration "$(dirname "${OUTPUT}")"

julia --project=. scripts/run_v10_equilibration_cluster.jl \
    --L "${L}" \
    --gamma "${GAMMA}" \
    --J "${J_VALUE}" \
    --v "${V_VALUE}" \
    --dt "${DT}" \
    --solver "${SOLVER}" \
    --init "${INIT}" \
    --equil-block-steps "${EQUIL_BLOCK_STEPS}" \
    --equil-max-blocks "${EQUIL_MAX_BLOCKS}" \
    --base-seed "${BASE_SEED}" \
    --output "${OUTPUT}"
