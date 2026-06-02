# Streamline arclength scaling

- Observable: `R2(s) = <|x(s0+s)-x(s0)|^2>` on unwrapped active streamlines.
- Parameters: `L=200`, `Q=0.0001`, `J=8.0`, `v=2.0`
- Snapshots: `1`, times `0.0`
- Streamlines: `4`, seed grid `2x2`, jitter `0.0`
- Trace arclength each direction: `20.0`, step `0.25`
- Fit window: `4.0 <= s <= 10.0`

## Primary estimate

- Fit `R2(s) ~ s^alpha`: `alpha = 2.0`
- Fractal dimension `d_f = 2 / alpha`: `1.0`
- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`

## Secondary diagnostics

- Mean per-streamline `d_f`: `1.0`
- Std per-streamline `d_f`: `0.0`
- Mean box-count dimension proxy: `0.84516`
- Mean recurrence fraction in fit window: `0.05`
