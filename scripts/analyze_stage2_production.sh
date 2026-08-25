#!/usr/bin/env bash
#SBATCH --job-name=analyze_zeta_v
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=logs/stage2_zeta_v/%j_analysis.out
#SBATCH --error=logs/stage2_zeta_v/%j_analysis.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

if ! command -v julia >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    module load julia
fi
command -v julia >/dev/null 2>&1 || {
    echo "julia is not available; load the Julia module before submitting" >&2
    exit 1
}

OUTPUT_DIR="${OUTPUT_DIR:-results/spin_aligned_f_stage2_L200_J2_Q1}"
FIGURES_DIR="${FIGURES_DIR:-figures/spin_aligned_f_stage2_L200_J2_Q1}"

julia --project=. scripts/plot_log_sweep_zeta_vs_logv_two_direction.jl \
    --up-dir "${OUTPUT_DIR}/up" \
    --down-dir "${OUTPUT_DIR}/down" \
    --results-dir "${OUTPUT_DIR}/zeta_vs_logv_two_direction" \
    --figures-dir "${FIGURES_DIR}"

for direction in up down; do
    julia --project=. scripts/analyze_log_sweep_zeta_fit_window_robustness.jl \
        --input-dir "${OUTPUT_DIR}/${direction}" \
        --results-dir "${OUTPUT_DIR}/zeta_fit_window_robustness/${direction}" \
        --figures-dir "${FIGURES_DIR}/zeta_fit_window_robustness/${direction}"
done
