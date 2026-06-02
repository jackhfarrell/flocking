# SAW F-correlator sampling surrogate

- SAW sampler: 2D square-lattice pivot sampler
- Walk length: `768`, samples: `800`
- Geometry fit window: `16.0 <= s <= 256.0`

## Geometry

- Fit `R2(s) ~ s^alpha`: `alpha = 1.49554`
- Inferred `d_f = 2 / alpha`: `1.33731`
- Reference SAW: `alpha = 1.5`, `d_f = 1.33333`

## Clock comparison

- Contour diffusive clock plus SAW geometry: `r(t) ~ t^0.37388`, `z = 2.67463`
- Formula `z = 2 d_f`: `2.67463`
- Reference `8/3`: `2.66667`
- Euclidean lab-frame diffusion control: `r(t) ~ t^0.5`, `z = 2.0`
