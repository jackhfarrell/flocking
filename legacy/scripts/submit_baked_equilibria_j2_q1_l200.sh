#!/usr/bin/env bash
#SBATCH --job-name=bake_eq_J2
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --array=1-500
#SBATCH --output=logs/baked_equilibria_J2_Q1_L200/%A_%a.out
#SBATCH --error=logs/baked_equilibria_J2_Q1_L200/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

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
DT="${DT:-0.001}"
BURNIN_TIME="${BURNIN_TIME:-1000}"
ARRAY_COUNT="${ARRAY_COUNT:-500}"
BASE_SEED="${BASE_SEED:-1}"
BURNIN_LOG_TIME="${BURNIN_LOG_TIME:-10.0}"
INITIAL_CONDITION="${INITIAL_CONDITION:-ordered}"
OUTPUT_DIR="${OUTPUT_DIR:-equilibria/J2_Q1_L200}"

mkdir -p "${OUTPUT_DIR}"

julia --project=. scripts/run_baked_equilibrium_cluster.jl \
    --L "${L}" \
    --gamma "${GAMMA}" \
    --J "${J_VALUE}" \
    --dt "${DT}" \
    --burnin-time "${BURNIN_TIME}" \
    --array-count "${ARRAY_COUNT}" \
    --base-seed "${BASE_SEED}" \
    --burnin-log-time "${BURNIN_LOG_TIME}" \
    --initial-condition "${INITIAL_CONDITION}" \
    --output-dir "${OUTPUT_DIR}"
