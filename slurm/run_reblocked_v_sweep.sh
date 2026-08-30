#!/usr/bin/env bash
#SBATCH --job-name=flock_zeta_v
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reblocked_v_sweep/%A_%a.out
#SBATCH --error=logs/reblocked_v_sweep/%A_%a.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

IFS=':' read -r -a velocities <<< "${VELOCITIES:?VELOCITIES is required}"
velocity_index=$((SLURM_ARRAY_TASK_ID - 1))
v=${velocities[velocity_index]}
temperature="${TEMPERATURE:-0.8}"
L="${L:-256}"
temperature_tag=${temperature//./p}
seed=$((${BASE_SEED:-3800000} + SLURM_ARRAY_TASK_ID))
output_dir="results/reblocked_v_sweep/T_${temperature_tag}/L_${L}"
output="${output_dir}/v_$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").jld2"

julia --startup-file=no --project=. --threads="${JULIA_NUM_THREADS}" \
    scripts/converge_single_v_reblocked.jl \
    --temperature "${temperature}" --v "${v}" --L "${L}" --Q "${Q:-1}" \
    --dt "${DT:-0}" --tolerance "${TOLERANCE:-0.005}" \
    --fit-step "${FIT_STEP:-0.0025}" \
    --zeta-min "${ZETA_MIN:-0.05}" --zeta-max "${ZETA_MAX:-1.0}" \
    --minimum-blocks "${MINIMUM_BLOCKS:-64}" \
    --maximum-blocks "${MAXIMUM_BLOCKS:-100000}" \
    --windows-per-block "${WINDOWS_PER_BLOCK:-2}" \
    --minimum-reblock-groups "${MINIMUM_REBLOCK_GROUPS:-8}" \
    --stable-checks "${STABLE_CHECKS:-5}" --check-every "${CHECK_EVERY:-4}" \
    --dr "${DR:-1}" --r-max "${R_MAX:-96}" --fit-rmax "${FIT_RMAX:-80}" \
    --T-max "${T_MAX:-8}" --ntimes "${NTIMES:-8}" \
    --reference-points "${REFERENCE_POINTS:-5}" \
    --spatial-samples "${SPATIAL_SAMPLES:-0}" \
    --report-seconds "${REPORT_SECONDS:-30}" \
    --maximum-wall-seconds "${MAXIMUM_WALL_SECONDS:-85500}" \
    --seed "${seed}" --output "${output}"
