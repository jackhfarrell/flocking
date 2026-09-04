#!/usr/bin/env bash

# Extend the T=0.8 reblocked sweep beyond v=10 to test whether z=1/zeta
# plateaus at 3 or continues to grow.

set -euo pipefail

velocities="${VELOCITIES:-12.589254117941675:15.848931924611133:19.952623149688797:25.118864315095795:31.622776601683793}"
IFS=':' read -r -a velocity_values <<< "${velocities}"
task_count=${#velocity_values[@]}
array_spec="1-${task_count}%${MAX_CONCURRENT:-5}"

temperature="${TEMPERATURE:-0.8}"
L="${L:-256}"
temperature_tag=${temperature//./p}
output_dir="${OUTPUT_DIR:-results/reblocked_v_sweep/T_${temperature_tag}/L_${L}/z3_extension}"

setup_job="${SETUP_JOB_ID:-}"
if [[ -z "${setup_job}" ]]; then
    setup_job=$(sbatch --parsable slurm/prepare_julia.sh)
fi

sweep_job=$(sbatch --parsable \
    --dependency="afterok:${setup_job}" \
    --array="${array_spec}" \
    --export="ALL,VELOCITIES=${velocities},TEMPERATURE=${temperature},L=${L},OUTPUT_DIR=${output_dir},TOLERANCE=${TOLERANCE:-0.005},FIT_STEP=${FIT_STEP:-0.0025},ZETA_MIN=${ZETA_MIN:-0.05},ZETA_MAX=${ZETA_MAX:-1.0},MINIMUM_BLOCKS=${MINIMUM_BLOCKS:-64},SPATIAL_SAMPLES=${SPATIAL_SAMPLES:-0},MAXIMUM_WALL_SECONDS=${MAXIMUM_WALL_SECONDS:-85500},BASE_SEED=${BASE_SEED:-3900000}" \
    slurm/run_reblocked_v_sweep.sh)

echo "setup ${setup_job}"
echo "z=3 extension ${sweep_job}: ${task_count} velocities, ${velocity_values[0]} to ${velocity_values[task_count - 1]}"
echo "T=${temperature}, L=${L}, output=${output_dir}"
