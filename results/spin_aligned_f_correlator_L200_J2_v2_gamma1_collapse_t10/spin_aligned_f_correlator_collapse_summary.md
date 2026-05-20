# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v2_gamma1`
- Number of runs: 500
- Radius window: `0.0 <= r <= 80.0`
- Time cutoff: `t >= 10.0`
- Included times: `10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.2400`, `zeta = 0.3400`, reduced `chi^2 = 112.8748`
- Refined best fit: `eta_F = 0.2250`, `zeta = 0.3300`, reduced `chi^2 = 112.5464`
- Collapse uncertainty from sensitivity band: `eta_F = 0.2400 ± 0.0600`, `zeta = 0.3400 ± 0.0600`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.4677, 32.0428]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.1800, 0.3000]`, `zeta in [0.2800, 0.4000]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.3884 ± 0.0426`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.4291 ± 0.0786`

## Comparison

- Collapse minus feature estimate: `Δeta_F = -0.2041`, `Δzeta = -0.0584`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 10.000 | 21.000 | 0.000539 |
| 12.000 | 22.000 | 0.000522 |
| 14.000 | 24.000 | 0.000465 |
| 16.000 | 25.000 | 0.000447 |
