# Self-avoiding contribution to spin-aligned F

- Observable: spin-aligned `F(r,t) = (C(r,t) - C(r,-t))/2` contribution split by midpoint active-streamline arm class.
- Parameters: `L=16`, `gamma=1.0`, `J=2.0`, `v=1.0`, `dt=0.001`, `lag_time=0.05`
- Windows: `1`, burn-in `0.1`
- Self-avoiding arm rule: recurrence <= `0.0` using box size `2.0`; endpoint distance >= `1.0 * sqrt(r)`

| r | selected arms | abs F fraction | signed selected / total | total F | selected F |
|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.8066 | 0.7977 | 2.4003 | 0.001419 | 0.003406 |
| 8.0 | 0.8066 | 0.8207 | 1.634 | 0.0014927 | 0.002439 |
