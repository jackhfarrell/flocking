# F-correlator backbone df preference

- Observable: spin-aligned time-antisymmetric `F(r,t)`
- Input: `/Users/jfarrell/Desktop/flocking/results/spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353`
- Runs: `500`
- Window: `0.0 <= r <= 40.0`, `t >= 6.0`
- Times: `6.0, 8.0, 10.0, 12.0, 14.0, 16.0`

## Best collapse

- `eta_F = 0.3300`
- `zeta = 1/z = 0.3900`
- `z = 2.5641`
- Interpreting `z = 2 d_f`: `d_f = 1.2821`
- Reduced `chi^2 = 144.0518`

## Fixed-exponent checks

| target | zeta | z | implied d_f | best eta_F | reduced chi^2 | ratio to best |
|---|---:|---:|---:|---:|---:|---:|
| SAW contour | 0.3750 | 2.6667 | 1.3333 | 0.3200 | 144.7306 | 1.0047 |
| lab diffusion | 0.5000 | 2.0000 | 1.0000 | 0.3500 | 200.5785 | 1.3924 |
| straight contour | 0.5000 | 2.0000 | 1.0000 | 0.3500 | 200.5785 | 1.3924 |
| older 1/3 check | 0.3333 | 3.0000 | 1.5000 | 0.3100 | 154.2160 | 1.0706 |

## Trough feature check

- `r_min(t) ~ t^zeta`: `zeta = 0.4241 ± 0.0147`
- Equivalent `z = 2.3577`
- Equivalent `d_f = 1/(2 zeta) = 1.1789`
- Reference SAW: `zeta = 0.3750`, `z = 2.6667`, `d_f = 1.3333`
