# F-weighted streamline arclength scaling

- Observable: arclength scaling of streamline arms weighted by spin-aligned T-odd `F` contribution.
- Parameters: `L=200`, `Q=1.0`, `J=2.0`, `v=1.0`
- Windows: `2`, lag time `0.5`, target radius `24.0`
- Arms: `1024`, top abs(F) fraction `0.1` => `102` arms
- Fit window: `4.0 <= s <= 66.66666666666667`

## Estimates

- All sampled arms: `alpha = 1.95045`, `d_f = 1.0254`
- `abs(F)`-weighted arms: `alpha = 1.95099`, `d_f = 1.02512`
- Top `abs(F)` arms: `alpha = 1.95115`, `d_f = 1.02504`
- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`

## Diagnostics

- Mean per-arm `d_f`: `1.02555`
- Mean top-arm `d_f`: `1.02519`
- Mean recurrence fraction in fit window: `0.08744`
- Total signed sampled F: `0.000125059`
- Total absolute sampled F: `0.00118134`
