# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v1_gamma1`
- Number of runs: 500
- Radius window: `0.0 <= r <= 80.0`
- Time cutoff: `t >= 0.0`
- Included times: `2, 4, 6, 8, 10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.3000`, `zeta = 0.3800`, reduced `chi^2 = 104.9916`
- Refined best fit: `eta_F = 0.2950`, `zeta = 0.3900`, reduced `chi^2 = 103.6271`
- Collapse uncertainty from sensitivity band: `eta_F = 0.2900 ± 0.0250`, `zeta = 0.3750 ± 0.0150`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.7631, 27.1321]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.2650, 0.3150]`, `zeta in [0.3600, 0.3900]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.3816 ± 0.0281`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.3228 ± 0.0069`

## Comparison

- Collapse minus feature estimate: `Δeta_F = -0.0278`, `Δzeta = +0.0084`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 2.000 | 3.000 | 0.009873 |
| 4.000 | 4.000 | 0.008090 |
| 6.000 | 5.000 | 0.007082 |
| 8.000 | 5.000 | 0.006494 |
| 10.000 | 6.000 | 0.006012 |
| 12.000 | 6.000 | 0.005652 |
| 14.000 | 6.000 | 0.005292 |
| 16.000 | 7.000 | 0.005046 |
