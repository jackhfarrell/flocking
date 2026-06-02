# F-correlator backbone df preference

- Observable: spin-aligned time-antisymmetric `F(r,t)`
- Input: `/Users/jfarrell/Desktop/flocking/results/spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353`
- Runs: `500`
- Window: `0.0 <= r <= 30.0`, `t >= 2.0`
- Times: `2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0`

## Best collapse

- `eta_F = 0.3000`
- `zeta = 1/z = 0.3850`
- `z = 2.5974`
- Interpreting `z = 2 d_f`: `d_f = 1.2987`
- Reduced `chi^2 = 140.8374`

## Fixed-exponent checks

| target | zeta | z | implied d_f | best eta_F | reduced chi^2 | ratio to best |
|---|---:|---:|---:|---:|---:|---:|
| SAW contour | 0.3750 | 2.6667 | 1.3333 | 0.3000 | 143.6804 | 1.0202 |
| lab diffusion | 0.5000 | 2.0000 | 1.0000 | 0.3300 | 818.9981 | 5.8152 |
| older 1/3 check | 0.3333 | 3.0000 | 1.5000 | 0.2900 | 226.8049 | 1.6104 |

## Trough feature check

- `r_min(t) ~ t^zeta`: `zeta = 0.4051 ± 0.0065`
- Equivalent `z = 2.4684`
- Equivalent `d_f = 1/(2 zeta) = 1.2342`
- Reference SAW: `zeta = 0.3750`, `z = 2.6667`, `d_f = 1.3333`
