#!/usr/bin/env bash
#SBATCH --job-name=spinF_L200
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --array=1-20
#SBATCH --output=logs/spin_aligned_f_L200/%A_%a.out
#SBATCH --error=logs/spin_aligned_f_L200/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

L="${L:-200}"
GAMMA="${GAMMA:-1}"
J_VALUE="${J_VALUE:?set J_VALUE}"
V_VALUE="${V_VALUE:?set V_VALUE}"
DT="${DT:-0.001}"
BURNIN_TIME="${BURNIN_TIME:-100}"
T_MAX="${T_MAX:-16}"
NTIMES="${NTIMES:-8}"
NCHUNKS="${NCHUNKS:-10}"
ARRAY_COUNT="${ARRAY_COUNT:-20}"
BURNIN_LOG_TIME="${BURNIN_LOG_TIME:-1.0}"
WINDOW_LOG_EVERY="${WINDOW_LOG_EVERY:-1}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

julia --project=. scripts/run_spin_aligned_f_correlator_cluster.jl \
    --L "${L}" \
    --gamma "${GAMMA}" \
    --J "${J_VALUE}" \
    --v "${V_VALUE}" \
    --dt "${DT}" \
    --burnin-time "${BURNIN_TIME}" \
    --T-max "${T_MAX}" \
    --ntimes "${NTIMES}" \
    --nchunks "${NCHUNKS}" \
    --array-count "${ARRAY_COUNT}" \
    --burnin-log-time "${BURNIN_LOG_TIME}" \
    --window-log-every "${WINDOW_LOG_EVERY}" \
    --output-dir "${OUTPUT_DIR}"
