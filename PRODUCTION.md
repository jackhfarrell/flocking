# Alpine exponent sweep

The production run measures `ζ(v, T)` from the spin-aligned correlator. It uses independent
equilibrated states as statistical samples and runs each velocity ladder in both directions.

## Submit

Keep the checkout under `/scratch/alpine/jafa3629/src` and submit from an Alpine login node.

```bash
ssh jafa3629@login.rc.colorado.edu
cd /scratch/alpine/jafa3629/src/flocking
bash slurm/submit_exponent_sweep.sh
```

The launcher first instantiates and precompiles the project inside an `acpu` allocation. It
then submits six equilibrium arrays, one for each temperature and direction, six dependent
measurement arrays, and one final analysis job.

The defaults use allocation `ucb819_asc1`, partition `acpu`, QoS `cpu-normal`, and the
installed Julia 1.12.7 binary. The job scripts set the versioned depot and thread count
inside each allocation.

## Production design

The default campaign has 3 temperatures, 30 velocities, 20 trajectories, and 2 directions.
Each measurement array has 20 tasks, one per trajectory. A task uses 30 Julia threads for
the velocity points, which keeps the full campaign below Alpine's submitted-job limit. Low
velocity gets 10 windows per trajectory and high velocity gets 2, with a linear budget in
`log(v)`.

The temperatures stay in the ordered passive phase while spanning a useful range

```text
T = 0.25, 0.5, 0.8
J = 4.0, 2.0, 1.25
```

Override a campaign at submission time with environment variables.

```bash
TEMPERATURES_CSV=0.2,0.35,0.5,0.65,0.8 \
NTRAJECTORIES=24 NV=30 MAX_MEASURE_CONCURRENT=4 MEASURE_CPUS=30 \
bash slurm/submit_exponent_sweep.sh
```

The measurement scripts use a conservative fixed step `dt = 2^-10`, geometric lags through
`T_max = 16`, and radii through `r = 60`. The reference fit uses `r <= 40`. The robustness
scan also uses `r <= 20`, `r <= 60`, and fits with the first or last positive lag removed.

## Outputs

```text
library/exponent_sweep/T_*/
results/exponent_sweep/T_*/{up,down}/
analysis/exponent_sweep/zeta_by_direction.csv
analysis/exponent_sweep/zeta_vs_v_temperature.csv
analysis/exponent_sweep/zeta_vs_v_temperature.{png,pdf}
analysis/exponent_sweep/summary.md
logs/exponent_sweep/
```

The main curve is the midpoint of the up and down fits. Its plotted uncertainty is the
largest of the local collapse sensitivity, the fit-window half-spread, and the up-down
half-difference. Filled and open points retain the two directions, so a hysteretic region is
visible rather than folded into an error bar.

## Late-time convergence analysis

The stricter analysis averages the two preparation directions at the correlator level and
fits only the last five positive lags. It compares complete traces on a common scaled-radius
grid and resamples whole trajectories within each direction.

```bash
sbatch slurm/analyze_exponent_convergence.sh 0.25,0.5,0.8 500
```

The job writes

```text
analysis/exponent_convergence/zeta_late_time.csv
analysis/exponent_convergence/zeta_time_convergence.csv
analysis/exponent_convergence/summary.md
```

After copying this directory to the local checkout, render the RevTeX figure with

```bash
python3 paper/zeta-convergence.py
```

The main panel shows the late-time bootstrap estimate. The two smaller panels show the fit
drift as early lags are removed at one passive and one active velocity.

Monitor the run with

```bash
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.6D %R'
```

Scratch is not backed up. Copy the final analysis directory and any unique equilibrium
library to durable storage when the dependent analysis job finishes.

## Adaptive threaded velocity sweep

The streaming convergence workflow runs one velocity and temperature per array task. Each
task requests eight CPUs, advances eight independent chains, checkpoints after every round,
and stops when its conservative uncertainty remains at or below `0.01`. The default sweep
uses `L = 128`, at least 20 temporal blocks, and the three paper temperatures
`T = 0.25, 0.5, 0.8` across the 22 logarithmic velocities from `v = 0.1` through `v = 2.807`.

Submit all 66 resumable points with

```bash
bash slurm/submit_threaded_v_sweep.sh
```

The array allows 12 simultaneous points by default. The internal 10,000-round cap is high
enough that the 24-hour Slurm allocation normally controls unconverged points. Override the
concurrency limit or other run controls at submission time:

```bash
MAX_CONCURRENT=8 L=160 MINIMUM_BLOCKS=30 \
bash slurm/submit_threaded_v_sweep.sh
```

The worker uses the between-chain standard error for stopping; the full chain range remains
in the output as a mixing diagnostic. The delete-group jackknife, half-run drift, time-window
sensitivity, radius-window sensitivity, and recent checkpoint drift must also all fall below
the tolerance. A point that reaches the 24-hour or round limit remains checkpointed and will
resume when the same sweep is submitted again.

Outputs are collected after the array finishes:

```text
results/threaded_v_sweep/T_*/v_*.{jld2,csv}
analysis/threaded_v_sweep/zeta_vs_v_temperature.csv
analysis/threaded_v_sweep/summary.md
logs/threaded_v_sweep/
```

## Precision campaign

The precision campaign keeps the original velocity ladders and extends them from 20 to 80
independent trajectories in each direction. It remeasures all 80 trajectories at the 22
velocities shown in the paper figure, through `log10(v) = 0.448`. The 16 positive lags run
from `t = 4` through `t = 64`.

```bash
bash slurm/submit_exponent_precision.sh
```

The launcher submits 360 new equilibrium tasks and 480 measurement tasks. Each measurement
task processes one trajectory, so its longer measurement stays below the 24 hour job limit.
The complete campaign remains below Alpine's submitted-job limit. Completed equilibrium
rungs and measurements are reused if the launcher is run again.

The longer measurements are separate from the original sweep.

```text
results/exponent_precision/T_*/{up,down}/
analysis/exponent_precision/zeta_late_time.csv
analysis/exponent_precision/zeta_time_convergence.csv
analysis/exponent_precision/summary.md
logs/exponent_precision/
```

The final analysis pools 80 up and 80 down trajectories at the correlator level, uses the
last six time lags for the reference fit, and draws 500 stratified trajectory bootstraps.

After this campaign and its analysis finish, a short second campaign extends every plotted
velocity to 100 trajectories per direction without repeating the first 80 measurements.

```bash
NTRAJECTORIES=100 EXISTING_TRAJECTORIES=80 \
MEASURE_TRAJECTORY_START=81 \
bash slurm/submit_exponent_precision.sh
```

This second launch adds trajectories 81 through 100. Do not submit it while the first
campaign is still queued.
