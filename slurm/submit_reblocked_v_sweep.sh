#!/usr/bin/env bash

# Submit one independent, restartable single-chain job at each logarithmic velocity.

set -euo pipefail

default_velocities="0.1:0.11220184543019636:0.12589254117941673:"
default_velocities+="0.14125375446227545:0.15848931924611134:0.1778279410038923:"
default_velocities+="0.19952623149688797:0.223872113856834:0.251188643150958:"
default_velocities+="0.28183829312644537:0.31622776601683794:0.35481338923357547:"
default_velocities+="0.3981071705534972:0.44668359215096315:0.5011872336272722:"
default_velocities+="0.5623413251903491:0.6309573444801932:0.7079457843841379:"
default_velocities+="0.7943282347242815:0.8912509381337456:1.0:"
default_velocities+="1.1220184543019633:1.2589254117941673:1.4125375446227544:"
default_velocities+="1.5848931924611136:1.7782794100389228:1.9952623149688795:"
default_velocities+="2.2387211385683394:2.51188643150958:2.8183829312644537:"
default_velocities+="3.1622776601683795:3.548133892335755:3.9810717055349722:"
default_velocities+="4.466835921509632:5.011872336272722:5.623413251903491:"
default_velocities+="6.309573444801933:7.079457843841379:7.943282347242816:"
default_velocities+="8.912509381337454:10.0"
velocities="${VELOCITIES:-${default_velocities}}"
IFS=':' read -r -a velocity_values <<< "${velocities}"
task_count=${#velocity_values[@]}
array_spec="1-${task_count}"
if [[ -n "${MAX_CONCURRENT:-}" ]]; then
    array_spec+="%${MAX_CONCURRENT}"
fi

temperature="${TEMPERATURE:-0.8}"
L="${L:-256}"
tolerance="${TOLERANCE:-0.005}"
sweep_export="ALL,VELOCITIES=${velocities},TEMPERATURE=${temperature},L=${L},\
TOLERANCE=${tolerance},FIT_STEP=${FIT_STEP:-0.0025},ZETA_MIN=${ZETA_MIN:-0.05},\
ZETA_MAX=${ZETA_MAX:-1.0},MINIMUM_BLOCKS=${MINIMUM_BLOCKS:-64},\
SPATIAL_SAMPLES=${SPATIAL_SAMPLES:-0},\
MAXIMUM_WALL_SECONDS=${MAXIMUM_WALL_SECONDS:-85500},\
BASE_SEED=${BASE_SEED:-3800000}"
analysis_export="ALL,TEMPERATURE=${temperature},L=${L},TOLERANCE=${tolerance},\
EXPECTED_POINTS=${task_count}"

mkdir -p logs/exponent_sweep logs/reblocked_v_sweep
setup_job="${SETUP_JOB_ID:-}"
if [[ -z "${setup_job}" ]]; then
    setup_job=$(sbatch --parsable slurm/prepare_julia.sh)
fi
sweep_job=$(sbatch --parsable \
    --dependency="afterok:${setup_job}" \
    --array="${array_spec}" \
    --export="${sweep_export}" \
    slurm/run_reblocked_v_sweep.sh)
monitor_export="${analysis_export},SWEEP_JOB_ID=${sweep_job},\
SNAPSHOT_SECONDS=${SNAPSHOT_SECONDS:-300}"
monitor_job=$(sbatch --parsable \
    --dependency="afterok:${setup_job}" \
    --export="${monitor_export}" \
    slurm/monitor_reblocked_v_sweep.sh)
analysis_job=$(sbatch --parsable \
    --dependency="afterany:${sweep_job}:${monitor_job}" \
    --export="${analysis_export}" \
    slurm/analyze_reblocked_v_sweep.sh)

echo "setup ${setup_job}"
echo "sweep ${sweep_job}: ${task_count} velocities, ${velocity_values[0]} to" \
    "${velocity_values[task_count - 1]}"
echo "live monitor ${monitor_job}, final collector ${analysis_job}"
echo "T=${temperature}, L=${L}, target uncertainty=${tolerance}"
echo "one core and one chain per velocity"
