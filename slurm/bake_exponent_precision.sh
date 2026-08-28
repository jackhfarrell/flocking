#!/usr/bin/env bash
#SBATCH --job-name=flock_pbake
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/exponent_precision/%A_%a_bake.out
#SBATCH --error=logs/exponent_precision/%A_%a_bake.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

temperature="${TEMPERATURE:?TEMPERATURE is required}"
direction="${DIRECTION:?DIRECTION is required}"
trajectory_start="${TRAJECTORY_START:-21}"
trajectory=$((trajectory_start + SLURM_ARRAY_TASK_ID - 1))
tag="T_${temperature//./p}"

julia --startup-file=no --project=. --threads="${JULIA_NUM_THREADS}" \
    scripts/bake_exponent_sweep.jl \
    --temperature "${temperature}" \
    --direction "${direction}" \
    --trajectory "${trajectory}" \
    --L "${L:-200}" --Q "${Q:-1}" --dt "${DT:-0.0009765625}" \
    --v-min "${V_MIN:-0.1}" --v-max "${V_MAX:-10}" --nv "${NV:-30}" \
    --block-steps "${BLOCK_STEPS:-2048}" --max-blocks "${MAX_BLOCKS:-10000}" \
    --window-time "${WINDOW_TIME:-50}" \
    --energy-threshold "${ENERGY_THRESHOLD:-0.02}" \
    --magnetization-threshold "${MAGNETIZATION_THRESHOLD:-0.02}" \
    --base-seed "${BASE_SEED:-100000}" \
    --output-dir "library/exponent_sweep/${tag}"
