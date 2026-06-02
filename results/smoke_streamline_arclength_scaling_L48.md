# Streamline arclength scaling

- Observable: `R2(s) = <|x(s0+s)-x(s0)|^2>` on unwrapped active streamlines.
- Parameters: `L=48`, `Q=1.0`, `J=2.0`, `v=1.0`
- Snapshots: `1`, times `0.2`
- Streamlines: `9`, seed grid `3x3`, jitter `0.0`
- Trace arclength each direction: `24.0`, step `0.25`
- Fit window: `4.0 <= s <= 16.0`

## Primary estimate

- Fit `R2(s) ~ s^alpha`: `alpha = 1.99176`
- Fractal dimension `d_f = 2 / alpha`: `1.00414`
- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`

## Secondary diagnostics

- Mean per-streamline `d_f`: `1.00416`
- Std per-streamline `d_f`: `0.00298`
- Mean box-count dimension proxy: `0.91313`
- Mean recurrence fraction in fit window: `0.12456`
