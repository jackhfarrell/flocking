# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353`
- Number of runs: 500
- Radius window: `0.0 <= r <= 40.0`
- Time cutoff: `t >= 0.0`
- Included times: `2, 4, 6, 8, 10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.3000`, `zeta = 0.3800`, reduced `chi^2 = 227.8920`
- Refined best fit: `eta_F = 0.3100`, `zeta = 0.3900`, reduced `chi^2 = 223.1516`
- Collapse uncertainty from sensitivity band: `eta_F = 0.3100 ± 0.0200`, `zeta = 0.3850 ± 0.0150`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.1908, 13.5660]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.2900, 0.3300]`, `zeta in [0.3700, 0.4000]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.4051 ± 0.0065`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.9296 ± 0.0731`

## Comparison

- Collapse minus feature estimate: `Δeta_F = -0.6196`, `Δzeta = -0.0151`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 2.000 | 9.250 | 0.000544 |
| 4.000 | 12.250 | 0.000352 |
| 6.000 | 14.250 | 0.000268 |
| 8.000 | 16.000 | 0.000198 |
| 10.000 | 17.500 | 0.000133 |
| 12.000 | 19.250 | 0.000111 |
| 14.000 | 20.000 | 0.000087 |
| 16.000 | 21.750 | 0.000095 |
