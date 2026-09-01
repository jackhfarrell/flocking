#!/usr/bin/env bash

# Submit restartable reblocked velocity sweeps at several temperatures.

set -euo pipefail

temperatures="${TEMPERATURES:-0.25:0.5}"
IFS=':' read -r -a temperature_values <<< "${temperatures}"

base_seed="${BASE_SEED:-4800000}"
seed_stride="${TEMPERATURE_SEED_STRIDE:-100000}"
max_concurrent="${MAX_CONCURRENT_PER_TEMPERATURE:-12}"

mkdir -p logs/exponent_sweep logs/reblocked_v_sweep
setup_job=$(sbatch --parsable slurm/prepare_julia.sh)

echo "setup ${setup_job}"
for temperature_index in "${!temperature_values[@]}"; do
    temperature=${temperature_values[temperature_index]}
    temperature_seed=$((base_seed + seed_stride * temperature_index))
    TEMPERATURE="${temperature}" \
    BASE_SEED="${temperature_seed}" \
    MAX_CONCURRENT="${max_concurrent}" \
    SETUP_JOB_ID="${setup_job}" \
        bash slurm/submit_reblocked_v_sweep.sh
done
