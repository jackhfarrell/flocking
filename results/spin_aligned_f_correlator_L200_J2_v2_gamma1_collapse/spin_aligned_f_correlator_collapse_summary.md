# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v2_gamma1`
- Number of runs: 500
- Radius window: `0.0 <= r <= 80.0`
- Time cutoff: `t >= 0.0`
- Included times: `2, 4, 6, 8, 10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.2200`, `zeta = 0.3600`, reduced `chi^2 = 108.5165`
- Refined best fit: `eta_F = 0.2250`, `zeta = 0.3600`, reduced `chi^2 = 107.9888`
- Collapse uncertainty from sensitivity band: `eta_F = 0.2275 ± 0.0175`, `zeta = 0.3650 ± 0.0050`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.7792, 29.4854]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.2100, 0.2450]`, `zeta in [0.3600, 0.3700]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.3892 ± 0.0083`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.3046 ± 0.0171`

## Comparison

- Collapse minus feature estimate: `Δeta_F = -0.0796`, `Δzeta = -0.0292`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 2.000 | 11.000 | 0.000842 |
| 4.000 | 15.000 | 0.000723 |
| 6.000 | 17.000 | 0.000631 |
| 8.000 | 19.000 | 0.000586 |
| 10.000 | 21.000 | 0.000539 |
| 12.000 | 22.000 | 0.000522 |
| 14.000 | 24.000 | 0.000465 |
| 16.000 | 25.000 | 0.000447 |
