# Streamline arclength scaling

- Observable: `R2(s) = <|x(s0+s)-x(s0)|^2>` on unwrapped active streamlines.
- Parameters: `L=200`, `Q=1.0`, `J=2.0`, `v=1.0`
- Snapshots: `4`, times `11.0, 12.0, 13.0, 14.0`
- Streamlines: `1024`, seed grid `16x16`, jitter `0.0`
- Trace arclength each direction: `200.0`, step `0.25`
- Fit window: `4.0 <= s <= 66.66666666666667`

## Primary estimate

- Fit `R2(s) ~ s^alpha`: `alpha = 1.94798`
- Fractal dimension `d_f = 2 / alpha`: `1.02671`
- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`

## Secondary diagnostics

- Mean per-streamline `d_f`: `1.02681`
- Std per-streamline `d_f`: `0.00528`
- Mean box-count dimension proxy: `1.01041`
- Mean recurrence fraction in fit window: `0.08717`
