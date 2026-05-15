# Run Index

This file summarizes the exploratory outputs currently in `results/` and
`figures/`. New passive XY baseline runs should use:

```text
results/baseline_xy/
figures/baseline_xy/
```

Suggested subfolders:

```text
lowT_J2/
kt_J1p11982/
dt_scan/
equilibration/
```

## Baseline Conventions

- Stationary passive XY check: set `v = 0`.
- Temperature convention: `T = 1/J`.
- Low-temperature reference: `eta_lowT_spin_wave = 1/(2*pi*J)`.
- KT reference: `J_c ≈ 1.11982`, `T_c ≈ 0.893`, `eta_KT = 1/4`.
- Prefer seeded outputs after the SciML noise seed fix.

## Existing Outputs

| Label | Files | Purpose | Notes |
|---|---|---|---|
| Smoke test | `results/smoke.jld2`, `figures/smoke/` | End-to-end pipeline check | Tiny `L=8`; not a physics run. |
| Equilibration smoke | `results/equilibration_smoke.jld2`, `figures/equilibration_smoke/` | Equilibration plotting check | Tiny run; not a physics run. |
| Low-T XY production | `results/passive_L24_J2.jld2`, `figures/passive_L24_J2/` | Passive `v=0`, `L=24`, `J=2`, smooth `C(r)` | Older exploratory output; use seeded reruns for reproducible comparisons. |
| Low-T seeded timestep scan | `results/passive_L24_J2_dt001_seeded.jld2`, `results/passive_L24_J2_dt002_seeded.jld2`, `results/passive_L24_J2_dt005_seeded.jld2` | Compare `dt=0.01,0.02,0.05` at fixed physical burn-in and sampling spacing | Preferred timestep-scan outputs. |
| Older timestep scan | `results/passive_L24_J2_dt002.jld2`, `results/passive_L24_J2_dt005.jld2` | Early timestep checks | Generated before the SciML noise seed fix; keep as legacy only. |
| Removed-helicity exploratory run | `results/passive_L24_J2_dt002_helicity.jld2` | Temporary helicity diagnostic | Legacy; helicity is no longer part of the workflow. |
| Low-T equilibration | `results/equilibration_L24_J2_seeded.jld2`, `results/equilibration_L32_J2_seeded.jld2`, `results/equilibration_L48_J2_seeded.jld2` | Random-start relaxation scaling at `J=2` | Preferred reproducible equilibration diagnostics. |
| Older low-T equilibration | `results/equilibration_L24_J2.jld2`, `results/equilibration_L32_J2.jld2`, `results/equilibration_L48_J2.jld2`, `figures/equilibration_L24_J2/`, `figures/equilibration_L32_J2/`, `figures/equilibration_L48_J2/` | Initial equilibration diagnostics | Generated before the seed fix; useful qualitatively. |
| Ordered-start check | `results/equilibration_L48_J2_ordered.jld2` | Compare ordered vs random initialization | Ordered starts relax faster at low T but should not be the only equilibrium validation. |
| KT small-system check | `results/passive_L24_J1p12_KT.jld2`, `figures/passive_L24_J1p12_KT/` | Passive `L=24`, `J≈1.12` near KT | Small system; showed `eta_fit` moving toward `1/4` but not precision KT scaling. |

## Recommended Next Baseline Runs

Low-temperature check:

```bash
julia --project=. scripts/run_sim.jl \
  --L 32 --Q 1.0 --J 2.0 --v 0.0 \
  --dt 0.02 --burnin-steps 7500 --sample-stride 25 \
  --nsamples 1000 --ntrajectories 8 \
  --fit-rmin 3 --fit-rmax 10 --init random \
  --output results/baseline_xy/lowT_J2/L32_production.jld2
```

KT finite-size check:

```bash
julia --project=. scripts/run_sim.jl \
  --L 64 --Q 1.0 --J 1.11982 --v 0.0 \
  --dt 0.02 --burnin-steps 60000 --sample-stride 50 \
  --nsamples 2500 --ntrajectories 8 \
  --fit-rmin 4 --fit-rmax 20 --init random \
  --output results/baseline_xy/kt_J1p11982/L64_production.jld2
```
