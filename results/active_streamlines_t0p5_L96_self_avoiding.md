# Active streamline dimension check

- Field: `v (cos(theta), sin(theta))`; streamline geometry is independent of the speed scale `v`.
- Snapshot time: `0.5`
- Parameters: `L=96`, `Q=1.0`, `J=2.0`, `v=1.0`
- Streamlines: `256`, arclength each direction `384.0`, step `0.25`
- Fit arclength window: `8.0` to `256.0`
- Box sizes: `2.0, 4.0, 8.0, 16.0, 32.0`
- Divider yardsticks: `2.0, 4.0, 8.0, 16.0, 32.0`
- Self-avoiding selection: coarse box `2.0`, recurrence <= `0.1`, `Rg >= 12.0`, endpoint distance >= `12.0`

## Estimates

- Mean `length ~ Rg^D`: `6.7916`
- Std `length ~ Rg^D`: `21.6823`
- Mean per-streamline divider dimension: `0.4003`
- Std per-streamline divider dimension: `0.3279`
- Dimension from ensemble-mean divider counts: `0.6686`
- Mean per-streamline box dimension: `0.6111`
- Std per-streamline box dimension: `0.2015`
- Dimension from ensemble-mean box counts: `0.6355`
- Reference SAW value `4/3`: `1.3333`

## Self-avoiding/open subset

- Selected streamlines: `0` of `256`
