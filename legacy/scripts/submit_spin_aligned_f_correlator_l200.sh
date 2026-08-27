#!/usr/bin/env bash
#SBATCH --job-name=spinF_launch
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --output=logs/spin_aligned_f_L200_launcher/%j.out
#SBATCH --error=logs/spin_aligned_f_L200_launcher/%j.err

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

mkdir -p logs/spin_aligned_f_L200_launcher

timestamp="$(date +"%Y%m%d_%H%M%S")"
dataset_tag="spin_aligned_f_correlator_L200_J2_v1_gamma1_${timestamp}"
log_tag="spin_aligned_f_L200_v1_${timestamp}"
output_dir="results/${dataset_tag}"
log_dir="logs/${log_tag}"

mkdir -p "${output_dir}" "${log_dir}"

sbatch \
    "--job-name=spinF_v1_${timestamp}" \
    "--account=ucb792_asc1" \
    "--partition=amilan" \
    "--qos=normal" \
    "--array=1-500" \
    "--output=${log_dir}/%A_%a.out" \
    "--error=${log_dir}/%A_%a.err" \
    "--nodes=1" \
    "--ntasks=1" \
    "--cpus-per-task=1" \
    "--mem-per-cpu=4G" \
    "--time=24:00:00" \
    --export=ALL,L=200,GAMMA=1,J_VALUE=2,V_VALUE=1,DT=0.001,DR=0.25,BURNIN_TIME=1000,T_MAX=16,NTIMES=8,NCHUNKS=10,ARRAY_COUNT=500,BURNIN_LOG_TIME=1.0,WINDOW_LOG_EVERY=1,OUTPUT_DIR="${output_dir}" \
    scripts/submit_spin_aligned_f_correlator_point.sh
