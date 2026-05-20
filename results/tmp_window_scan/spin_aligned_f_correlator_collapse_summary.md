# Spin-aligned F(r,t) collapse analysis

- Input directory: `results/spin_aligned_f_correlator_L200_J2_v2_gamma1`
- Number of runs: 500
- Radius window: `25.0 <= r <= 100.0`
- Time cutoff: `t >= 10.0`
- Included times: `10, 12, 14, 16`
- Polynomial order: `3`
- Collapse bins: `60`

## Primary collapse fit

- Coarse best fit: `eta_F = 0.0000`, `zeta = 0.3600`, reduced `chi^2 = 0.5070`
- Refined best fit: `eta_F = 0.0000`, `zeta = 0.3550`, reduced `chi^2 = 0.4993`
- Collapse uncertainty from sensitivity band: `eta_F = 0.0000 ± 0.0600`, `zeta = 0.3575 ± 0.0075`
- Smooth master curve: weighted polynomial of order `3` fit after exponent selection
- Shared collapsed window: `x in [11.0393, 37.3712]`
- Sensitivity band (`objective <= 1.050 * min`): `eta_F in [-0.0600, 0.0600]`, `zeta in [0.3500, 0.3650]`

## Feature-based sanity check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.0000 ± 0.0000`
- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = -2.3373 ± 0.6310`

## Comparison

- Collapse minus feature estimate: `Δeta_F = +2.3373`, `Δzeta = +0.3550`

## Trough data

| t | r_trough | -F_min |
|---:|---:|---:|
| 10.000 | 25.000 | 0.000148 |
| 12.000 | 25.000 | 0.000326 |
| 14.000 | 25.000 | 0.000416 |
| 16.000 | 25.000 | 0.000447 |
