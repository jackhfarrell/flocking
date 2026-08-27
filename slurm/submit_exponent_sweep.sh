#!/usr/bin/env bash

# Submit one setup job, then independent up and down ladders at each temperature. Each
# measurement array waits for its matching ladder, and the final fit waits for every array.

set -euo pipefail

temperatures_csv="${TEMPERATURES_CSV:-0.25,0.5,0.8}"
IFS=',' read -r -a temperatures <<< "${temperatures_csv}"
ntrajectories="${NTRAJECTORIES:-20}"
nv="${NV:-30}"
max_concurrent="${MAX_CONCURRENT:-100}"
measure_count=$((ntrajectories * nv))

(( measure_count <= 1000 )) || {
    echo "measurement array needs ${measure_count} tasks, above Alpine's 1000-task cap" >&2
    exit 1
}

mkdir -p logs/exponent_sweep
setup_job="$(sbatch --parsable slurm/prepare_julia.sh)"
measure_jobs=()

for temperature in "${temperatures[@]}"; do
    for direction in up down; do
        bake_job="$(sbatch --parsable \
            --dependency="afterok:${setup_job}" \
            --array="1-${ntrajectories}%${max_concurrent}" \
            --export="ALL,TEMPERATURE=${temperature},DIRECTION=${direction},\
NTRAJECTORIES=${ntrajectories},NV=${nv}" \
            slurm/bake_exponent_sweep.sh)"
        measure_job="$(sbatch --parsable \
            --dependency="afterok:${bake_job}" \
            --array="1-${measure_count}%${max_concurrent}" \
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
    --export="ALL,TEMPERATURES_CSV=${temperatures_csv}" \
    slurm/analyze_exponent_sweep.sh)"
echo "analysis ${analysis_job}"
