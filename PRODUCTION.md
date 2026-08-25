# Production ζ(v) run

The production pipeline measures the spin-aligned correlator at 30 log-spaced velocities
from `v = 0.1` to `v = 10`, separately for the up- and down-sweep equilibrium libraries,
then writes the two-direction ζ(v) CSV, plot, and fit-window robustness results.

## Prerequisite

`library/J2_Q1_L200` must contain the completed Stage-1 library. Every `v_*` directory must
have the same numbered `up_traj_*.jld2` and `down_traj_*.jld2` checkpoints. The launcher
checks the rung and checkpoint counts before submitting anything.

The calibrated production values of `DT` and `T_MAX` must also be chosen before launch.
Their current script defaults are `DT=0.001` and `T_MAX=16`. The launcher measures through
`r = 60`: the primary exponent fit uses `r <= 40`, while the robustness scan compares
cutoffs at `r <= 20`, `40`, and `60`.

## Submit the complete run

From the repository root on the CU cluster:

```bash
ssh jafa3629@login.rc.colorado.edu
cd /path/to/flocking
julia --project=. -e 'using Pkg; Pkg.instantiate()'
DT=0.001 T_MAX=16 bash scripts/submit_stage2_production.sh
```

The launcher discovers the number of Stage-1 trajectories, computes the χ²-shaped array
size, and submits batches of at most 1000 tasks. Batches run in sequence, up sweep before
down sweep. A final dependent Slurm job runs both the ζ(v) comparison and the robustness
scan. Cluster defaults are allocation `ucb792_asc1`, partition `amilan`, and QoS `normal`.

With the defaults, the per-trajectory velocity budgets are

```text
89,76,65,55,47,40,34,29,25,21,18,15,13,11,10,8,7,6,5,4,4,3,3,2,2,2,1,1,1,1
```

This is 598 windows and 49 array tasks per trajectory at 20 chunks per job. The launcher
therefore places at most 20 trajectories in each 980-task batch.

To use an explicit trajectory count or a larger budget:

```bash
NTRAJECTORIES=40 \
BUDGET_TOTAL=1200 \
DT=0.0005 T_MAX=12 \
bash scripts/submit_stage2_production.sh
```

## Monitor

```bash
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.6D %R'
```

Logs are written below `logs/stage2_zeta_v/`.

## Run only the final analysis

If the measurements already exist:

```bash
sbatch --account=ucb792_asc1 --partition=amilan --qos=normal \
  scripts/analyze_stage2_production.sh
```

The main deliverables are:

- `results/spin_aligned_f_stage2_L200_J2_Q1/zeta_vs_logv_two_direction/zeta_vs_logv_two_direction.csv`
- `figures/spin_aligned_f_stage2_L200_J2_Q1/zeta_vs_logv_two_direction_trace_collapse.png`
- `results/spin_aligned_f_stage2_L200_J2_Q1/zeta_fit_window_robustness/{up,down}/zeta_fit_window_robustness.csv`

The analysis first combines all chunks belonging to one baked trajectory, then computes
the standard error across independent Stage-1 trajectories. It does not treat correlated
within-trajectory chunks as independent samples.
