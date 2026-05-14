using Flocking
using CairoMakie

params = ModelParams(L=32, J=1.0, Q=0.25, v=0.5)
sol = simulate(params; T=5.0, dt=0.01, saveat=0.1, seed=1)
fig = diagnostic_plot(sol, params)
save("diagnostics.png", fig)
