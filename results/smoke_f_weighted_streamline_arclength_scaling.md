# F-weighted streamline arclength scaling

- Observable: arclength scaling of streamline arms weighted by spin-aligned T-odd `F` contribution.
- Parameters: `L=32`, `Q=1.0`, `J=2.0`, `v=1.0`
- Windows: `1`, lag time `0.1`, target radius `8.0`
- Arms: `18`, top abs(F) fraction `0.1` => `2` arms
- Fit window: `4.0 <= s <= 10.0`

## Estimates

- All sampled arms: `alpha = 1.98544`, `d_f = 1.00733`
- `abs(F)`-weighted arms: `alpha = 1.98448`, `d_f = 1.00782`
- Top `abs(F)` arms: `alpha = 1.9845`, `d_f = 1.00781`
- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`

## Diagnostics

- Mean per-arm `d_f`: `1.00747`
- Mean top-arm `d_f`: `1.00781`
- Mean recurrence fraction in fit window: `0.06181`
- Total signed sampled F: `-9.02467e-5`
- Total absolute sampled F: `0.000336676`
