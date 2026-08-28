#!/usr/bin/env bash

# Submit one setup job, then independent up and down ladders at each temperature.
# Measurement tasks split over trajectories and use Julia threads for velocities, so the
# full campaign stays below Alpine's submitted-job limit.

set -euo pipefail

temperatures_csv="${TEMPERATURES_CSV:-0.25,0.5,0.8}"
IFS=',' read -r -a temperatures <<< "${temperatures_csv}"
ntrajectories="${NTRAJECTORIES:-20}"
nv="${NV:-30}"
max_bake_concurrent="${MAX_BAKE_CONCURRENT:-100}"
max_measure_concurrent="${MAX_MEASURE_CONCURRENT:-4}"
measure_cpus="${MEASURE_CPUS:-${nv}}"

mkdir -p logs/exponent_sweep
setup_job="$(sbatch --parsable slurm/prepare_julia.sh)"
measure_jobs=()

for temperature in "${temperatures[@]}"; do
    for direction in up down; do
        bake_job="$(sbatch --parsable \
            --dependency="afterok:${setup_job}" \
            --array="1-${ntrajectories}%${max_bake_concurrent}" \
            --export="ALL,TEMPERATURE=${temperature},DIRECTION=${direction},\
NTRAJECTORIES=${ntrajectories},NV=${nv}" \
            slurm/bake_exponent_sweep.sh)"
        measure_job="$(sbatch --parsable \
            --dependency="afterok:${bake_job}" \
            --array="1-${ntrajectories}%${max_measure_concurrent}" \
            --cpus-per-task="${measure_cpus}" \
            --export="ALL,TEMPERATURE=${temperature},DIRECTION=${direction},\
NTRAJECTORIES=${ntrajectories},NV=${nv}" \
            slurm/measure_exponent_sweep.sh)"
        measure_jobs+=("${measure_job}")
        echo "T=${temperature} ${direction} bake ${bake_job}, measure ${measure_job}"
    done
done

dependency="$(IFS=:; echo "${measure_jobs[*]}")"
analysis_job="$(sbatch --parsable \
    --dependency="afterok:${dependency}" \
    slurm/analyze_exponent_sweep.sh "${temperatures_csv}")"
echo "analysis ${analysis_job}"
