#!/usr/bin/env bash
#SBATCH --job-name=spinF_jv_sweep
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --output=logs/spin_aligned_f_L200_gamma1_jv_sweep_launcher/%j.out
#SBATCH --error=logs/spin_aligned_f_L200_gamma1_jv_sweep_launcher/%j.err

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    dry_run=0
else
    dry_run=1
fi
max_points=25

while (($# > 0)); do
    case "$1" in
        --submit)
            dry_run=0
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --max-points)
            max_points="$2"
            shift 2
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 1
            ;;
    esac
done

[[ "$max_points" =~ ^[0-9]+$ ]] || {
    echo "--max-points must be a nonnegative integer" >&2
    exit 1
}

gamma=1
array_count=20
L=200
dt=0.001
burnin_time=100
T_max=16
ntimes=8
nchunks=10
burnin_log_time=1.0
window_log_every=1

script="scripts/submit_spin_aligned_f_correlator_point.sh"
root_results="results/spin_aligned_f_correlator_L200_gamma1_jv_sweep"
root_logs="logs/spin_aligned_f_L200_gamma1_jv_sweep"
launcher_logs="logs/spin_aligned_f_L200_gamma1_jv_sweep_launcher"

j_values=(0.1 0.31622776601683794 1.0 3.1622776601683795 10.0)
j_tags=(0p1 0p316227766 1 3p16227766 10)
v_values=(0.1 0.31622776601683794 1.0 3.1622776601683795 10.0)
v_tags=(0p1 0p316227766 1 3p16227766 10)

mkdir -p "$root_results" "$root_logs" "$launcher_logs"

count=0
for j_idx in "${!j_values[@]}"; do
    for v_idx in "${!v_values[@]}"; do
        ((count += 1))
        if ((count > max_points)); then
            break 2
        fi

        J_VALUE="${j_values[j_idx]}"
        J_TAG="${j_tags[j_idx]}"
        V_VALUE="${v_values[v_idx]}"
        V_TAG="${v_tags[v_idx]}"

        point_name="J${J_TAG}_v${V_TAG}"
        output_dir="${root_results}/${point_name}"
        log_dir="${root_logs}/${point_name}"

        mkdir -p "$output_dir" "$log_dir"

        cmd=(
            sbatch
            "--job-name=spinF_${point_name}"
            "--array=1-${array_count}"
            "--output=${log_dir}/%A_%a.out"
            "--error=${log_dir}/%A_%a.err"
            "--export=ALL,L=${L},GAMMA=${gamma},J_VALUE=${J_VALUE},V_VALUE=${V_VALUE},DT=${dt},BURNIN_TIME=${burnin_time},T_MAX=${T_max},NTIMES=${ntimes},NCHUNKS=${nchunks},ARRAY_COUNT=${array_count},BURNIN_LOG_TIME=${burnin_log_time},WINDOW_LOG_EVERY=${window_log_every},OUTPUT_DIR=${output_dir}"
            "$script"
        )

        printf "[%02d/25] J=%s v=%s array_count=%s output=%s\n" \
            "$count" "$J_VALUE" "$V_VALUE" "$array_count" "$output_dir"
        if ((dry_run)); then
            printf "DRY-RUN:"
            printf " %q" "${cmd[@]}"
            printf "\n"
        else
            "${cmd[@]}"
        fi
    done
done

printf "planned_points=%d total_array_tasks=%d mode=%s\n" \
    "$count" "$((count * array_count))" "$([[ $dry_run -eq 1 ]] && echo dry-run || echo submit)"
