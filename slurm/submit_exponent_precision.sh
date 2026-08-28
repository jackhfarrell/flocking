#!/usr/bin/env bash

# Add independent velocity ladders through trajectory 80, then measure the complete
# paper range with longer time lags. Each measurement task handles one trajectory so
# its runtime stays well below Alpine's 24 hour limit.

set -euo pipefail

temperatures_csv="${TEMPERATURES_CSV:-0.25,0.5,0.8}"
IFS=',' read -r -a temperatures <<< "${temperatures_csv}"
ntrajectories="${NTRAJECTORIES:-80}"
existing_trajectories="${EXISTING_TRAJECTORIES:-20}"
new_trajectories=$((ntrajectories - existing_trajectories))
trajectories_per_task="${TRAJECTORIES_PER_TASK:-1}"
measure_trajectory_start="${MEASURE_TRAJECTORY_START:-1}"
measurement_trajectories=$((ntrajectories - measure_trajectory_start + 1))
measure_tasks=$(((measurement_trajectories + trajectories_per_task - 1) /
    trajectories_per_task))
nv="${NV:-30}"
vi_min="${VI_MIN:-1}"
vi_max="${VI_MAX:-22}"
t_max="${T_MAX:-64}"
ntimes="${NTIMES:-16}"
windows_low_v="${WINDOWS_LOW_V:-6}"
windows_high_v="${WINDOWS_HIGH_V:-2}"
measure_cpus="${MEASURE_CPUS:-$((vi_max - vi_min + 1))}"
max_bake_concurrent="${MAX_BAKE_CONCURRENT:-24}"
max_measure_concurrent="${MAX_MEASURE_CONCURRENT:-4}"

((new_trajectories > 0)) || {
    echo "NTRAJECTORIES must exceed EXISTING_TRAJECTORIES" >&2
    exit 1
}

mkdir -p logs/exponent_precision
setup_job="$(sbatch --parsable slurm/prepare_julia.sh)"
measure_jobs=()

for temperature in "${temperatures[@]}"; do
    for direction in up down; do
        bake_job="$(sbatch --parsable \
            --dependency="afterok:${setup_job}" \
            --array="1-${new_trajectories}%${max_bake_concurrent}" \
            --export="ALL,TEMPERATURE=${temperature},DIRECTION=${direction},\
TRAJECTORY_START=$((existing_trajectories + 1)),NV=${nv}" \
            slurm/bake_exponent_precision.sh)"
        measure_job="$(sbatch --parsable \
            --dependency="afterok:${bake_job}" \
            --array="1-${measure_tasks}%${max_measure_concurrent}" \
            --cpus-per-task="${measure_cpus}" \
            --export="ALL,TEMPERATURE=${temperature},DIRECTION=${direction},\
NTRAJECTORIES=${ntrajectories},TRAJECTORIES_PER_TASK=${trajectories_per_task},\
TRAJECTORY_START=${measure_trajectory_start},NV=${nv},VI_MIN=${vi_min},VI_MAX=${vi_max},\
T_MAX=${t_max},NTIMES=${ntimes},WINDOWS_LOW_V=${windows_low_v},\
WINDOWS_HIGH_V=${windows_high_v}" \
            slurm/measure_exponent_precision.sh)"
        measure_jobs+=("${measure_job}")
        echo "T=${temperature} ${direction} bake ${bake_job}, measure ${measure_job}"
    done
done

dependency="$(IFS=:; echo "${measure_jobs[*]}")"
analysis_job="$(sbatch --parsable \
    --dependency="afterok:${dependency}" \
    slurm/analyze_exponent_precision.sh "${temperatures_csv}" 500)"
echo "analysis ${analysis_job}"
