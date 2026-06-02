# F-contributing trajectory geometry

- Observable: midpoint spin streamline arclength needed to approach the Euclidean endpoint sampled by spin-aligned `F`.
- Parameters: `L=96`, `Q=1.0`, `J=2.0`, `v=1.0`, lag time `0.5`
- Windows: `4`, seed grid `12x12`, arms per radius `1152`
- Fit: `s_hit(r) ~ r^d_f`, using `r >= 8.0`
- Hit rule: closest streamline distance <= `2.0`

## Estimates

- All pairs: `d_f = 1.01466`
- `abs(F)`-weighted pairs: `d_f = 1.02023`
- Sign-coherent `F`-weighted pairs: `d_f = 1.02103`
- Top `abs(F)` fraction 0.1: `d_f = 0.99332`
- Hit-only `abs(F)`-weighted pairs: `d_f = 1.03666`
- Reference SAW: `d_f = 1.33333`

| r | mean s_hit | abs(F)-weighted s_hit | coherent F-weighted s_hit | top abs(F) s_hit | hit abs(F)-weighted s_hit | hit fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 3.6567 | 3.4496 | 3.4429 | 3.213 | 3.8071 | 0.7804 |
| 8.0 | 7.3108 | 6.8121 | 6.7678 | 6.4261 | 8.2534 | 0.4071 |
| 12.0 | 11.1354 | 10.5778 | 10.4585 | 9.9957 | 12.7576 | 0.2674 |
| 16.0 | 14.8611 | 13.9479 | 13.9495 | 12.987 | 17.1155 | 0.2023 |
| 20.0 | 18.533 | 17.5579 | 17.4453 | 16.487 | 21.6406 | 0.1597 |
| 24.0 | 22.3103 | 21.2358 | 21.1489 | 19.7696 | 26.0563 | 0.1398 |
| 28.0 | 26.2036 | 24.9212 | 24.7166 | 23.7087 | 30.1888 | 0.1181 |
| 32.0 | 29.951 | 27.947 | 27.7686 | 24.5739 | 34.9938 | 0.1024 |
