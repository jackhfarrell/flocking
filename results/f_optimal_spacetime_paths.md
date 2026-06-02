# F optimal spacetime paths

- Observable: dynamic-programming paths maximizing a local time-antisymmetric reward `n(x,t0+k) - n(x,t0-k)` projected onto the seed spin.
- Parameters: `L=48`, `Q=1.0`, `J=2.0`, `v=1.0`, `Tmax=8.0`, `frame_dt=0.5`
- Windows: `3`, seed grid `8x8`, paths `192`
- Move penalty: `0.05`

## Estimates

- All optimal paths: `alpha = 1.46475`, `d_f = 1.36542`
- Score-weighted optimal paths: `alpha = 1.44042`, `d_f = 1.38849`
- Top 10% score paths: `alpha = 1.44117`, `d_f = 1.38777`
- Reference SAW: `alpha = 1.5`, `d_f = 1.33333`

## Diagnostics

- Mean endpoint displacement: `8.3135`
- Mean top-score endpoint displacement: `7.33802`
