# Lattice Flocking SDE

Julia scripts for simulating a stochastic angle field on a 2D periodic square lattice using `DifferentialEquations.jl` / `StochasticDiffEq.jl`.

The implemented convention is

```text
d theta_r = drift_r(theta) dt + sqrt(Q) dW_r
drift_r = -(Q/2) mu_r + active_v_terms
```

with white noise correlation `<xi(r,t) xi(r',t')> = Q delta_rr' delta(t-t')`.

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

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Current Output Index

See `analysis/run_index.md` for a human-readable summary of existing legacy
results and figures.
