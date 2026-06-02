# Self-avoiding contribution to spin-aligned F

- Observable: spin-aligned `F(r,t) = (C(r,t) - C(r,-t))/2` contribution split by midpoint active-streamline arm class.
- Parameters: `L=48`, `gamma=1.0`, `J=2.0`, `v=1.0`, `dt=0.001`, `lag_time=0.5`
- Windows: `8`, burn-in `10.0`
- Self-avoiding arm rule: recurrence <= `0.0` using box size `2.0`; endpoint distance >= `1.0 * sqrt(r)`

| r | selected arms | abs F fraction | signed selected / total | total F | selected F |
|---:|---:|---:|---:|---:|---:|
| 4.0 | 0.8197 | 0.8228 | 0.8966 | 0.017039 | 0.015278 |
| 8.0 | 0.7759 | 0.7802 | 1.4724 | -0.0068559 | -0.010094 |
| 12.0 | 0.7071 | 0.7074 | 1.2252 | -0.0095943 | -0.011755 |
| 16.0 | 0.6305 | 0.6315 | 0.8183 | -0.014666 | -0.012001 |
| 20.0 | 0.5557 | 0.5574 | 0.6975 | 0.0097567 | 0.0068054 |
| 24.0 | 0.4913 | 0.4889 | 0.0555 | 0.0087111 | 0.00048326 |
