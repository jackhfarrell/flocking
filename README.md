# Lattice Flocking SDE

Julia scripts for simulating a stochastic angle field on a 2D periodic square lattice using `DifferentialEquations.jl` / `StochasticDiffEq.jl`.

The implemented convention is

```text
d theta_r = drift_r(theta) dt + sqrt(Q) dW_r
drift_r = -(Q/2) mu_r - (v/2) active_terms(theta, mu)
```

with white noise correlation `<xi(r,t) xi(r',t')> = Q delta_rr' delta(t-t')`.
Positive `v` therefore follows the current sign convention implemented in
`src/model.jl`.

## Setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Run a simulation

```bash
julia --project=. scripts/run_sim.jl --L 32 --Q 1.0 --J 2.0 --v 0.0 \
  --dt 0.01 --burnin-steps 1000 --sample-stride 10 --nsamples 100 \
  --ntrajectories 4 --init random --output results/passive_L32.jld2
```

The output is a compact JLD2 summary with parameters, seeds, diagnostics, averaged radial correlation, and the fitted power-law exponent. Full trajectories are not saved.

For passive XY baseline checks, use `v=0` and compare equal-time correlations
against known XY expectations. The code reports `eta_fit` from
`C(r) ~ r^(-eta)` and the low-temperature spin-wave reference
`eta_lowT_spin_wave = 1/(2*pi*J)`.

For a live look at equilibration noise, run
`scripts/equilibration_diagnostic_live.jl` to watch the trailing energy-density
trajectory update block by block.

Suggested organization for future baseline runs:

```text
results/baseline_xy/
figures/baseline_xy/
```

Useful subfolders are `lowT_J2/`, `kt_J1p11982/`, `dt_scan/`, and
`equilibration/`.

## Plot results

```bash
julia --project=. scripts/plot_results.jl results/passive_L32.jld2 --output-dir figures/passive_L32
```

## Snapshot movie

```bash
julia --project=. scripts/run_snapshot_movie.jl --J 8 --Q 1e-4 --dt 0.0009765625 \
  --output figures/snapshot_dataset/snapshot_movie.mp4
```

By default the movie now targets a clean low-temperature advection demo: a
localized upward bump is seeded on a uniform right-pointing background, the
runtime is auto-calibrated from the measured early-time drift speed, and a JLD2
sidecar with the bump `x_centroids` diagnostic is saved next to the movie. If
you set `--equilibration-time` to a positive value, the script first evolves
the background and then inserts the same localized perturbation into that
already-fluctuating state.

## Snapshot dataset

```bash
julia --project=. scripts/run_snapshot_dataset.jl --J 8 --Q 1e-4 --dt 0.0009765625 \
  --output results/snapshot_dataset_L200_v2.jld2
```

This is the canonical advection dataset path. If `--times` is omitted, the
script measures the initial bump drift speed and chooses four early-time
snapshots from that calibration. The saved dataset includes `x_centroids` so
rightward transport is tracked numerically as well as visually.

One important detail: in this model `Q` is not a standalone "noise knob". It
appears in both the passive drift and the stochastic term, so changing `Q`
mainly rescales the passive relaxation timescale. If you want a visually clean
uniform background, keep `--equilibration-time 0` rather than expecting smaller
`Q` alone to suppress fluctuations. Low `Q` also does not fix active snapshot
artifacts when the fixed timestep is too large; for `v ≈ 2`, use a small step
such as `dt = 2^-10`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Production exponent sweep

See [PRODUCTION.md](PRODUCTION.md) for the exact CU-cluster command that measures and plots
the spin-aligned scaling exponent ζ as a function of `v`.

## Current Output Index

See `analysis/run_index.md` for a human-readable summary of existing legacy
results and figures.
