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

Monitor the run with

```bash
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.6D %R'
```

Scratch is not backed up. Copy the final analysis directory and any unique equilibrium
library to durable storage when the dependent analysis job finishes.
