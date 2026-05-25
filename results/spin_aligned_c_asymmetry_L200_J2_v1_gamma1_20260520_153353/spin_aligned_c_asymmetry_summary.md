# Spin-aligned C asymmetry analysis

- Quantity: `ΔC_F(r,t) = C(r,t) - C(r,-t) = 2F(r,t)`
- Convention: same space-time symmetrization used by the stored `F_mean`
- Note: stored `C_plus_mean - C_minus_mean` is not used here; those fields pair `forward_plus/backward_plus` and `forward_minus/backward_minus`, which differs from the F-correlator subtraction.
- Input directory: `/Users/jfarrell/Desktop/flocking/results/spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353`
- Number of runs: 500
- Radius window: `0.2 <= r <= 40.0`
- Included times: `2, 4, 6, 8, 10, 12, 14, 16`

## Collapse fit

- Coarse best fit: `eta = 0.3000`, `z = 2.5500`, `zeta = 1/z = 0.3922`, reduced `chi^2 = 226.1132`
- Refined best fit: `eta = 0.3100`, `z = 2.5600`, `zeta = 1/z = 0.3906`, reduced `chi^2 = 223.3993`
- Sensitivity band: `eta = 0.3100 ± 0.0200`, `z = 2.6050 ± 0.1050`
- Shared collapsed window: `x in [0.1907, 13.5426]`

## Feature check

- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = 0.4051 ± 0.0065`, equivalently `z = 2.4684 ± 0.0396`
- Trough-amplitude scaling: `-ΔC_F,min(t) ~ t^{-eta}`, fitted `eta = 0.9296 ± 0.0731`
