#!/usr/bin/env bash
#SBATCH --job-name=flock_pmeasure
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=16G
#SBATCH --time=23:30:00
#SBATCH --output=logs/exponent_precision/%A_%a_measure.out
#SBATCH --error=logs/exponent_precision/%A_%a_measure.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-22}"

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

temperature="${TEMPERATURE:?TEMPERATURE is required}"
direction="${DIRECTION:?DIRECTION is required}"
ntrajectories="${NTRAJECTORIES:-80}"
trajectories_per_task="${TRAJECTORIES_PER_TASK:-1}"
trajectory_start="${TRAJECTORY_START:-1}"
first_trajectory=$((
    trajectory_start + (SLURM_ARRAY_TASK_ID - 1) * trajectories_per_task
))
last_trajectory=$((first_trajectory + trajectories_per_task - 1))
((last_trajectory > ntrajectories)) && last_trajectory="${ntrajectories}"
tag="T_${temperature//./p}"

for ((trajectory = first_trajectory; trajectory <= last_trajectory; trajectory++)); do
    julia --startup-file=no --project=. --threads="${JULIA_NUM_THREADS}" \
        scripts/measure_exponent_sweep.jl \
        --temperature "${temperature}" \
        --direction "${direction}" \
        --trajectory "${trajectory}" \
        --ntrajectories "${ntrajectories}" \
        --L "${L:-200}" --Q "${Q:-1}" --dt "${DT:-0.0009765625}" \
        --v-min "${V_MIN:-0.1}" --v-max "${V_MAX:-10}" --nv "${NV:-30}" \
        --velocity-index-min "${VI_MIN:-1}" --velocity-index-max "${VI_MAX:-22}" \
        --dr "${DR:-0.5}" --r-max "${R_MAX:-60}" \
        --T-max "${T_MAX:-64}" --ntimes "${NTIMES:-16}" \
        --windows-low-v "${WINDOWS_LOW_V:-6}" \
        --windows-high-v "${WINDOWS_HIGH_V:-2}" \
        --budget-power "${BUDGET_POWER:-1}" \
        --base-seed "${BASE_SEED:-2000000}" \
        --library-dir "library/exponent_sweep/${tag}" \
        --output-dir "results/exponent_precision/${tag}/${direction}"
done
