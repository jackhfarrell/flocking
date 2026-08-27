#!/usr/bin/env bash
#SBATCH --job-name=flock_measure
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exponent_sweep/%A_%a_measure.out
#SBATCH --error=logs/exponent_sweep/%A_%a_measure.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

temperature="${TEMPERATURE:?TEMPERATURE is required}"
direction="${DIRECTION:?DIRECTION is required}"
tag="T_${temperature//./p}"

julia --startup-file=no --project=. --threads="${JULIA_NUM_THREADS}" \
    scripts/measure_exponent_sweep.jl \
    --temperature "${temperature}" \
    --direction "${direction}" \
    --trajectory "${SLURM_ARRAY_TASK_ID}" \
    --ntrajectories "${NTRAJECTORIES:-20}" \
    --L "${L:-200}" --Q "${Q:-1}" --dt "${DT:-0.0009765625}" \
    --v-min "${V_MIN:-0.1}" --v-max "${V_MAX:-10}" --nv "${NV:-30}" \
    --dr "${DR:-0.5}" --r-max "${R_MAX:-60}" \
    --T-max "${T_MAX:-16}" --ntimes "${NTIMES:-8}" \
    --windows-low-v "${WINDOWS_LOW_V:-10}" \
    --windows-high-v "${WINDOWS_HIGH_V:-2}" \
    --budget-power "${BUDGET_POWER:-1}" \
    --base-seed "${BASE_SEED:-1000000}" \
    --library-dir "library/exponent_sweep/${tag}" \
    --output-dir "results/exponent_sweep/${tag}/${direction}"
