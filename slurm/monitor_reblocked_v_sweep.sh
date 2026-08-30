#!/usr/bin/env bash
#SBATCH --job-name=flock_zeta_live
#SBATCH --account=ucb819_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reblocked_v_sweep/%j_live.out
#SBATCH --error=logs/reblocked_v_sweep/%j_live.err

set -euo pipefail

cluster_root=/scratch/alpine/jafa3629
export PATH="${cluster_root}/software/julia-1.12.7/bin:${PATH}"
export JULIA_DEPOT_PATH="${cluster_root}/depots/julia-1.12"
export JULIA_NUM_THREADS=1

repo_root="${SLURM_SUBMIT_DIR:-${cluster_root}/src/flocking}"
cd "${repo_root}"

temperature="${TEMPERATURE:-0.8}"
L="${L:-256}"
temperature_tag=${temperature//./p}
sweep_job_id="${SWEEP_JOB_ID:?SWEEP_JOB_ID is required}"
interval_seconds="${SNAPSHOT_SECONDS:-300}"

while true; do
    julia --startup-file=no --project=. --threads=1 \
        scripts/analyze_reblocked_v_sweep.jl \
        --input-dir "results/reblocked_v_sweep/T_${temperature_tag}/L_${L}" \
        --output-dir "analysis/reblocked_v_sweep/T_${temperature_tag}_L${L}" \
        --expected-points "${EXPECTED_POINTS:-41}" \
        --tolerance "${TOLERANCE:-0.005}"

    [[ -n "$(squeue --noheader --jobs="${sweep_job_id}")" ]] || break

    elapsed=0
    while ((elapsed < interval_seconds)); do
        sleep 60
        elapsed=$((elapsed + 60))
    done
done
