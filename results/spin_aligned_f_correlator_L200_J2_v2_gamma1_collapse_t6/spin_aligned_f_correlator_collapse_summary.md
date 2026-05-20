# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v2_gamma1`
- Number of runs: 500
- Radius window: `0.0 <= r <= 80.0`
- Time cutoff: `t >= 6.0`
- Included times: `6, 8, 10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.2000`, `zeta = 0.3600`, reduced `chi^2 = 94.4471`
- Refined best fit: `eta_F = 0.2000`, `zeta = 0.3600`, reduced `chi^2 = 94.4471`
- Collapse uncertainty from sensitivity band: `eta_F = 0.2000 ± 0.0450`, `zeta = 0.3350 ± 0.0350`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.5246, 29.4854]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.1550, 0.2450]`, `zeta in [0.3000, 0.3700]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.3954 ± 0.0131`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.3554 ± 0.0314`

## Comparison

- Collapse minus feature estimate: `Δeta_F = -0.1554`, `Δzeta = -0.0354`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 6.000 | 17.000 | 0.000631 |
| 8.000 | 19.000 | 0.000586 |
| 10.000 | 21.000 | 0.000539 |
| 12.000 | 22.000 | 0.000522 |
| 14.000 | 24.000 | 0.000465 |
| 16.000 | 25.000 | 0.000447 |
