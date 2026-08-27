#!/usr/bin/env bash
#SBATCH --job-name=flock_setup
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exponent_sweep/%j_setup.out
#SBATCH --error=logs/exponent_sweep/%j_setup.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JULIA_NUM_PRECOMPILE_TASKS=2

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
