#!/usr/bin/env bash
#SBATCH --job-name=stage2_zeta_v
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=logs/stage2_zeta_v/%A_%a.out
#SBATCH --error=logs/stage2_zeta_v/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

LIBRARY_DIR="${LIBRARY_DIR:-library/J2_Q1_L200}"
OUTPUT_DIR="${OUTPUT_DIR:-results/spin_aligned_f_stage2_L200_J2_Q1}"
NTRAJECTORIES="${NTRAJECTORIES:-}"
TRAJ_FIRST="${TRAJ_FIRST:-1}"
BATCH_NTRAJ="${BATCH_NTRAJ:-1}"
DIRECTION="${DIRECTION:-up}"
BASE_SEED_UP="${BASE_SEED_UP:-200000}"
BASE_SEED_DOWN="${BASE_SEED_DOWN:-1200000}"
DT="${DT:-0.001}"
DR="${DR:-0.25}"
R_MAX="${R_MAX:-60}"
T_MAX="${T_MAX:-16}"
NTIMES="${NTIMES:-8}"
V_MIN="${V_MIN:-0.1}"
V_MAX="${V_MAX:-10}"
NV="${NV:-30}"
WINDOW_BUDGET="${WINDOW_BUDGET:-}"
BUDGET_TOTAL="${BUDGET_TOTAL:-600}"
BUDGET_POWER="${BUDGET_POWER:-1}"
CHUNKS_PER_JOB="${CHUNKS_PER_JOB:-20}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-100}"

if ! command -v julia >/dev/null 2>&1 && type module >/dev/null 2>&1; then
    module load julia
fi
command -v julia >/dev/null 2>&1 || {
    echo "julia is not available; load the Julia module before submitting" >&2
    exit 1
}

common_args=(
    --library-dir "${LIBRARY_DIR}"
    --L 200 --gamma 1 --J 2
    --vmin "${V_MIN}" --vmax "${V_MAX}" --nv "${NV}"
    --budget-total "${BUDGET_TOTAL}" --budget-power "${BUDGET_POWER}"
    --chunks-per-job "${CHUNKS_PER_JOB}" --max-jobs "${MAX_ARRAY_TASKS}"
    --dt "${DT}" --dr "${DR}" --r-max "${R_MAX}"
    --lag-spacing geometric --solver SRA1
    --T-max "${T_MAX}" --ntimes "${NTIMES}"
)
if [[ -n "${WINDOW_BUDGET}" ]]; then
    common_args+=(--window-budget "${WINDOW_BUDGET}")
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ -d "${LIBRARY_DIR}" ]] || {
        echo "missing Stage-1 library: ${LIBRARY_DIR}" >&2
        exit 1
    }
    first_v_dir="$(find "${LIBRARY_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'v_01_*' -print -quit)"
    [[ -n "${first_v_dir}" ]] || {
        echo "missing Stage-1 library rung v_01_* below ${LIBRARY_DIR}" >&2
        exit 1
    }
    if [[ -z "${NTRAJECTORIES}" ]]; then
        NTRAJECTORIES="$(find "${first_v_dir}" -maxdepth 1 -type f -name 'up_traj_*.jld2' | wc -l | tr -d ' ')"
    fi
    (( NTRAJECTORIES > 0 )) || {
        echo "no Stage-1 trajectories found in ${first_v_dir}" >&2
        exit 1
    }

    rung_count="$(find "${LIBRARY_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'v_*' | wc -l | tr -d ' ')"
    [[ "${rung_count}" == "${NV}" ]] || {
        echo "library has ${rung_count} velocity rungs; expected ${NV}" >&2
        exit 1
    }
    for v_dir in "${LIBRARY_DIR}"/v_*; do
        for direction in up down; do
            for traj in $(seq 1 "${NTRAJECTORIES}"); do
                checkpoint="${v_dir}/${direction}_traj_$(printf '%03d' "${traj}").jld2"
                [[ -f "${checkpoint}" ]] || {
                    echo "incomplete Stage-1 library; missing ${checkpoint}" >&2
                    exit 1
                }
            done
        done
    done

    jobs_per_trajectory="$(julia --project=. scripts/run_stage2_measurement_cluster.jl \
        "${common_args[@]}" --direction up --ntrajectories 1 --plan-only)"
    trajectories_per_batch=$((MAX_ARRAY_TASKS / jobs_per_trajectory))
    (( trajectories_per_batch > 0 )) || {
        echo "one trajectory needs ${jobs_per_trajectory} jobs, above the ${MAX_ARRAY_TASKS}-job cap" >&2
        exit 1
    }

    mkdir -p logs/stage2_zeta_v "${OUTPUT_DIR}"
    dependency=""
    for direction in up down; do
        first=1
        while (( first <= NTRAJECTORIES )); do
            remaining=$((NTRAJECTORIES - first + 1))
            batch_n=$((remaining < trajectories_per_batch ? remaining : trajectories_per_batch))
            task_count=$((batch_n * jobs_per_trajectory))
            submit_args=(--parsable --array="1-${task_count}%${MAX_CONCURRENT}")
            [[ -n "${dependency}" ]] && submit_args+=(--dependency="afterok:${dependency}")
            dependency="$(sbatch "${submit_args[@]}" \
                --export="ALL,DIRECTION=${direction},TRAJ_FIRST=${first},BATCH_NTRAJ=${batch_n},NTRAJECTORIES=${NTRAJECTORIES}" \
                "$0")"
            echo "submitted ${direction} trajectories ${first}-$((first + batch_n - 1)): ${dependency}"
            first=$((first + batch_n))
        done
    done

    analysis_job="$(sbatch --parsable --dependency="afterok:${dependency}" \
        --export="ALL,OUTPUT_DIR=${OUTPUT_DIR}" scripts/analyze_stage2_production.sh)"
    echo "submitted production analysis: ${analysis_job}"
    exit 0
fi

base_seed="${BASE_SEED_UP}"
[[ "${DIRECTION}" == "down" ]] && base_seed="${BASE_SEED_DOWN}"

julia --project=. scripts/run_stage2_measurement_cluster.jl \
    "${common_args[@]}" \
    --direction "${DIRECTION}" \
    --traj-first "${TRAJ_FIRST}" \
    --ntrajectories "${BATCH_NTRAJ}" \
    --trajectory-subdirs \
    --array-count "${SLURM_ARRAY_TASK_COUNT}" \
    --base-seed "${base_seed}" \
    --output-dir "${OUTPUT_DIR}/${DIRECTION}"
