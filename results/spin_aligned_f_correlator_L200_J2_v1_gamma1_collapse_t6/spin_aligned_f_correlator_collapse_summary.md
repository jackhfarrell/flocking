# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v1_gamma1`
- Number of runs: 500
- Radius window: `0.0 <= r <= 80.0`
- Time cutoff: `t >= 6.0`
- Included times: `6, 8, 10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.2800`, `zeta = 0.3000`, reduced `chi^2 = 79.6140`
- Refined best fit: `eta_F = 0.2800`, `zeta = 0.2950`, reduced `chi^2 = 79.2910`
- Collapse uncertainty from sensitivity band: `eta_F = 0.2800 ± 0.0550`, `zeta = 0.3200 ± 0.0400`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.5894, 35.3081]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.2250, 0.3350]`, `zeta in [0.2800, 0.3600]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.3225 ± 0.0710`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.3480 ± 0.0095`

## Comparison

- Collapse minus feature estimate: `Δeta_F = -0.0680`, `Δzeta = -0.0275`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 6.000 | 5.000 | 0.007082 |
| 8.000 | 5.000 | 0.006494 |
| 10.000 | 6.000 | 0.006012 |
| 12.000 | 6.000 | 0.005652 |
| 14.000 | 6.000 | 0.005292 |
| 16.000 | 7.000 | 0.005046 |
