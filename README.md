# Lattice Flocking SDE

Julia code for the stochastic angle field

```text
dθᵣ = [-(Q/2) μᵣ - (v/2) activeᵣ] dt + √Q dWᵣ.
```

The passive steady state is `exp(-H)`. The coupling `J` is therefore the inverse
dimensionless temperature, `J = 1/T`, while `Q` sets the time scale. Temperature sweeps vary
`J`, not `Q`.

## Local setup

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

A small equal-time simulation is still available through
[`scripts/run_sim.jl`](scripts/run_sim.jl).

## Exponent sweep

The active production path measures the spin-aligned spreading exponent `ζ(v, T)` at
three temperatures. Each temperature is prepared along independent up and down velocity
ladders. Their difference measures hysteresis and catches biased equilibration.

The defaults are

- `T = 0.25, 0.5, 0.8`, or `J = 4, 2, 1.25`
- 30 logarithmic velocities from `v = 0.1` to `v = 10`
- 20 independent trajectories in each direction
- `L = 200`, `Q = 1`, `dt = 2^-10`, and the additive-noise `SRA1` solver

See [PRODUCTION.md](PRODUCTION.md) for the Alpine submission command and output layout.

## Repository layout

```text
src/       model, observables, correlator, and collapse fit
scripts/   current local and production entry points
slurm/     Alpine setup, arrays, dependencies, and analysis
test/      package and production smoke tests
```

Generated libraries, results, figures, and logs are ignored by Git. Scratch is not backed
up, so copy the final CSV, figure, and any unique equilibrated libraries to durable storage.
