#!/usr/bin/env bash
#SBATCH --job-name=dt_convergence
#SBATCH --account=ucb792_asc1
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --output=logs/dt_convergence/%A_%a.out
#SBATCH --error=logs/dt_convergence/%A_%a.err
#SBATCH --array=1-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Memory note: the CRN fine solve runs with save_noise=true, storing the full Wiener path
# (an L^2-vector per internal step) so the coarse solve can replay it. At L=200 with the
# dt=0.001 candidate that path is ~10 GB, and SRA1's RSWM keeps two noise arrays, so peak
# RSS lands in the low tens of GB. 4G/core OOM-killed it, then 32G OOM-killed it too because
# nchunks=4 chunks' worth of noise arrays weren't collected before the next chunk allocated
# (see the explicit GC.gc() per chunk in dt_convergence_crn.jl). 64G gives headroom on the
# worst (finest-dt, v=10) task even with prompt collection. Pass `sbatch --mem=... ` to
# override further if a candidate dt goes finer.

# One array task per candidate dt: each runs the CRN dt-convergence bake-off across the
# anchor velocities (worst case v=10, then 1 and 0.1) at L=200, coupling dt and dt/2 on a
# shared Wiener path and reporting |zeta(dt) - zeta(dt/2)|. Each task writes its own CSV;
# the operator assembles dt(v) by taking the coarsest dt whose gap is within tol per anchor
# (scripts/calibration_schedule.jl:select_production_dt). Two candidates keep this trivially
# within the <= 1000-job cap and each task within the 24h wall.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "${script_dir}/.." && pwd)}"
cd "${repo_root}"

if ! command -v julia >/dev/null 2>&1; then
    if type module >/dev/null 2>&1; then
        module load julia
    fi
fi

command -v julia >/dev/null 2>&1 || {
    echo "julia is not available; load the Julia module or set JULIA_BIN before submitting" >&2
    exit 1
}

L="${L:-200}"
GAMMA="${GAMMA:-1}"
J_VALUE="${J_VALUE:-2}"
VELOCITIES="${VELOCITIES:-10.0,1.0,0.1}"
SOLVER="${SOLVER:-SRA1}"
DT_CANDIDATES="${DT_CANDIDATES:-0.001,0.002}"
DR="${DR:-0.25}"
R_MAX="${R_MAX:-45}"
BURNIN_TIME="${BURNIN_TIME:-40}"
T_MAX="${T_MAX:-16}"
NTIMES="${NTIMES:-8}"
NCHUNKS="${NCHUNKS:-4}"
TOL="${TOL:-0.005}"
BASE_SEED="${BASE_SEED:-800000}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dt_convergence}"

task_id="${SLURM_ARRAY_TASK_ID:-1}"
IFS=',' read -r -a dt_list <<< "${DT_CANDIDATES}"
# Keep --array in sync with DT_CANDIDATES: overriding one without the other would
# silently skip a candidate or index past the list. Fail loudly instead.
if (( task_id < 1 || task_id > ${#dt_list[@]} )); then
    echo "array task ${task_id} has no matching dt in DT_CANDIDATES (${#dt_list[@]} entries); set --array=1-${#dt_list[@]}" >&2
    exit 1
fi
dt="${dt_list[$((task_id - 1))]}"

mkdir -p logs/dt_convergence "${OUTPUT_DIR}"

julia --project=. scripts/dt_convergence_crn.jl \
    --L "${L}" \
    --gamma "${GAMMA}" \
    --J "${J_VALUE}" \
    --velocities "${VELOCITIES}" \
    --solver "${SOLVER}" \
    --dt "${dt}" \
    --dr "${DR}" \
    --r-max "${R_MAX}" \
    --burnin-time "${BURNIN_TIME}" \
    --T-max "${T_MAX}" \
    --ntimes "${NTIMES}" \
    --nchunks "${NCHUNKS}" \
    --tol "${TOL}" \
    --base-seed "${BASE_SEED}" \
    --csv "${OUTPUT_DIR}/dt_${dt}.csv" \
    --output "${OUTPUT_DIR}/dt_${dt}.jld2"
