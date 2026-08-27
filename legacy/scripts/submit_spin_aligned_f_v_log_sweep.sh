#!/usr/bin/env bash
#SBATCH --job-name=spinF_vlog
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --array=1-500
#SBATCH --output=logs/spin_aligned_f_L200_J2_Q1_v_log_sweep/%A_%a.out
#SBATCH --error=logs/spin_aligned_f_L200_J2_Q1_v_log_sweep/%A_%a.err
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

EQUILIBRIUM_DIR="${EQUILIBRIUM_DIR:-equilibria/J2_Q1_L200}"
OUTPUT_DIR="${OUTPUT_DIR:-results/spin_aligned_f_correlator_L200_J2_Q1_v_log_sweep}"
ARRAY_COUNT="${ARRAY_COUNT:-500}"
BASE_SEED="${BASE_SEED:-100000}"
DT="${DT:-0.001}"
DR="${DR:-0.25}"
T_MAX="${T_MAX:-16}"
NTIMES="${NTIMES:-8}"
V_MIN="${V_MIN:-0.01}"
V_MAX="${V_MAX:-1.0}"
NV="${NV:-30}"
V_VALUES="${V_VALUES:-}"
WINDOW_LOG_EVERY="${WINDOW_LOG_EVERY:-4}"

mkdir -p "${OUTPUT_DIR}"

cmd=(
    julia --project=. scripts/run_spin_aligned_f_v_sweep_cluster.jl
    --equilibrium-dir "${EQUILIBRIUM_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --array-count "${ARRAY_COUNT}"
    --base-seed "${BASE_SEED}"
    --dt "${DT}"
    --dr "${DR}"
    --T-max "${T_MAX}"
    --ntimes "${NTIMES}"
    --v-min "${V_MIN}"
    --v-max "${V_MAX}"
    --nv "${NV}"
    --window-log-every "${WINDOW_LOG_EVERY}"
)

if [[ -n "${V_VALUES}" ]]; then
    cmd+=(--v-values "${V_VALUES}")
fi

"${cmd[@]}"
