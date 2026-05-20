# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v1_gamma1`
- Number of runs: 500
- Radius window: `0.0 <= r <= 80.0`
- Time cutoff: `t >= 10.0`
- Included times: `10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.3800`, `zeta = 0.4000`, reduced `chi^2 = 92.7548`
- Refined best fit: `eta_F = 0.3900`, `zeta = 0.4500`, reduced `chi^2 = 92.2872`
- Collapse uncertainty from sensitivity band: `eta_F = 0.3800 ± 0.0600`, `zeta = 0.4000 ± 0.0600`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [0.3548, 22.9740]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [0.3200, 0.4400]`, `zeta in [0.3400, 0.4600]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.2794 ± 0.1829`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = 0.3779 ± 0.0122`

## Comparison

- Collapse minus feature estimate: `Δeta_F = +0.0121`, `Δzeta = +0.1706`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 10.000 | 6.000 | 0.006012 |
| 12.000 | 6.000 | 0.005652 |
| 14.000 | 6.000 | 0.005292 |
| 16.000 | 7.000 | 0.005046 |
