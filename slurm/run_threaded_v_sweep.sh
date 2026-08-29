#!/usr/bin/env bash
#SBATCH --job-name=flock_vconv
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/threaded_v_sweep/%A_%a.out
#SBATCH --error=logs/threaded_v_sweep/%A_%a.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

IFS=':' read -r -a temperatures <<< "${TEMPERATURES:?TEMPERATURES is required}"
IFS=':' read -r -a velocities <<< "${VELOCITIES:?VELOCITIES is required}"

task_index=$((SLURM_ARRAY_TASK_ID - 1))
velocity_count=${#velocities[@]}
temperature_index=$((task_index / velocity_count))
velocity_index=$((task_index % velocity_count))
temperature=${temperatures[temperature_index]}
v=${velocities[velocity_index]}

temperature_tag=${temperature//./p}
v_tag=${v//./p}
seed=$((${BASE_SEED:-2400000} + 100000 * temperature_index + 1000 * velocity_index))
output="results/threaded_v_sweep/T_${temperature_tag}/v_$(printf '%02d' "$((velocity_index + 1))")_${v_tag}.jld2"

julia --startup-file=no --project=. --threads="${JULIA_NUM_THREADS}" \
    scripts/converge_single_v_threaded.jl \
    --temperature "${temperature}" --v "${v}" \
    --L "${L:-128}" --Q "${Q:-1}" --dt "${DT:-0.001953125}" \
    --chains "${CHAINS:-8}" --tolerance "${TOLERANCE:-0.01}" \
    --minimum-blocks "${MINIMUM_BLOCKS:-20}" \
    --stable-checks "${STABLE_CHECKS:-5}" \
    --chain-error-mode standard-error \
    --maximum-rounds "${MAXIMUM_ROUNDS:-10000}" \
    --report-seconds "${REPORT_SECONDS:-30}" \
    --seed "${seed}" --output "${output}"
