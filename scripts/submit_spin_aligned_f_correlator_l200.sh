#!/usr/bin/env bash
#SBATCH --job-name=spinF_L200
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --array=1-500
#SBATCH --output=logs/spin_aligned_f_L200/%A_%a.out
#SBATCH --error=logs/spin_aligned_f_L200/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/spin_aligned_f_L200 results/spin_aligned_f_correlator_L200_J2_v1_gamma1

julia --project=. scripts/run_spin_aligned_f_correlator_cluster.jl \
    --L 200 \
    --gamma 1 \
    --J 2 \
    --v 1 \
    --dt 0.001 \
    --burnin-time 100 \
    --T-max 16 \
    --ntimes 8 \
    --nchunks 10 \
    --array-count 500 \
    --burnin-log-time 1.0 \
    --window-log-every 1 \
    --output-dir results/spin_aligned_f_correlator_L200_J2_v1_gamma1
