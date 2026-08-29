#!/usr/bin/env bash

set -euo pipefail

temperatures="${TEMPERATURES:-0.25:0.5:0.8}"
velocities="${VELOCITIES:-0.10000000000000002:0.11721022975334806:0.13738237958832633:0.16102620275609397:0.18873918221350977:0.22122162910704496:0.25929437974046676:0.30391953823131984:0.3562247890262443:0.4175318936560402:0.48939009184774945:0.573615251044868:0.6723357536499338:0.7880462815669914:0.9236708571873864:1.0826367338740548:1.2689610031679224:1.4873521072935116:1.7433288221999885:2.0433597178569425:2.395026619987486:2.8072162039411777}"

IFS=':' read -r -a temperature_values <<< "${temperatures}"
IFS=':' read -r -a velocity_values <<< "${velocities}"
task_count=$((${#temperature_values[@]} * ${#velocity_values[@]}))
max_concurrent="${MAX_CONCURRENT:-12}"

mkdir -p logs/exponent_sweep logs/threaded_v_sweep
setup_job=$(sbatch --parsable slurm/prepare_julia.sh)
sweep_job=$(sbatch --parsable \
    --dependency="afterok:${setup_job}" \
    --array="1-${task_count}%${max_concurrent}" \
    --export="ALL,TEMPERATURES=${temperatures},VELOCITIES=${velocities},L=${L:-128},\
CHAINS=${CHAINS:-8},TOLERANCE=${TOLERANCE:-0.01},MINIMUM_BLOCKS=${MINIMUM_BLOCKS:-20},\
STABLE_CHECKS=${STABLE_CHECKS:-5},MAXIMUM_ROUNDS=${MAXIMUM_ROUNDS:-10000},\
BASE_SEED=${BASE_SEED:-2400000}" \
    slurm/run_threaded_v_sweep.sh)
analysis_job=$(sbatch --parsable \
    --dependency="afterany:${sweep_job}" \
    --export="ALL,EXPECTED_POINTS=${task_count}" \
    slurm/analyze_threaded_v_sweep.sh)

echo "setup ${setup_job}"
echo "sweep ${sweep_job}: ${#temperature_values[@]} temperatures x ${#velocity_values[@]} velocities = ${task_count} tasks"
echo "analysis ${analysis_job}"
