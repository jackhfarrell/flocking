#!/usr/bin/env bash
#SBATCH --job-name=flock_vsummary
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/threaded_v_sweep/%j_analysis.out
#SBATCH --error=logs/threaded_v_sweep/%j_analysis.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

julia --startup-file=no --project=. --threads="${JULIA_NUM_THREADS}" \
    scripts/analyze_threaded_v_sweep.jl \
    --expected-points "${EXPECTED_POINTS:-66}"
