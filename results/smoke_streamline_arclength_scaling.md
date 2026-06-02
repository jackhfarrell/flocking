# Streamline arclength scaling

- Observable: `R2(s) = <|x(s0+s)-x(s0)|^2>` on unwrapped active streamlines.
- Parameters: `L=24`, `Q=1.0`, `J=2.0`, `v=1.0`
- Snapshots: `1`, times `0.2`
- Streamlines: `9`, seed grid `3x3`, jitter `0.0`
- Trace arclength each direction: `12.0`, step `0.25`
- Fit window: `4.0 <= s <= 8.0`

## Primary estimate

- Fit `R2(s) ~ s^alpha`: `alpha = 1.98768`
- Fractal dimension `d_f = 2 / alpha`: `1.0062`
- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`

## Secondary diagnostics

- Mean per-streamline `d_f`: `1.00627`
- Std per-streamline `d_f`: `0.0057`
- Mean box-count dimension proxy: `0.86885`
- Mean recurrence fraction in fit window: `0.07114`
