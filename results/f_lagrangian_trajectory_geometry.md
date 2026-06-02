# F-weighted Lagrangian trajectory geometry

- Observable: tracer paths through the time-dependent spin field from the midpoint time.
- Weight: accumulated absolute time-antisymmetric spin overlap between forward and backward traced paths.
- Parameters: `L=64`, `Q=1.0`, `J=2.0`, `v=1.0`, `Tmax=8.0`
- Windows: `4`, seed grid `10x10`, paths `800`
- Fit: `R2(s) ~ s^alpha`, using lag time `>= 1.0`

## Estimates

- All paths: `alpha = 1.95759`, `d_f = 1.02166`
- Time-antisymmetric weighted paths: `alpha = 1.95168`, `d_f = 1.02476`
- Time-even weighted paths: `alpha = 1.95913`, `d_f = 1.02086`
- Reference SAW: `alpha = 1.5`, `d_f = 1.33333`
