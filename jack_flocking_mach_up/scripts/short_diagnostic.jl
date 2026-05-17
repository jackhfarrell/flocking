using Flocking
using CairoMakie
using ProgressMeter

params = ModelParams(L=32, J=1.0, Q=0.25, v=0.5)
sol = simulate(params; T=5.0, dt=0.01, saveat=0.1, seed=1, progress=true)

# Compute diagnostics with a progress meter to show progress for long runs
cache = Flocking.LatticeCache(params)
times = collect(sol.t)
n = length(sol.u)
magnetization = Vector{Float64}(undef, n)
energy_density = Vector{Float64}(undef, n)

@showprogress 1 "Computing diagnostics" for i in 1:n
    θ = sol.u[i]
    magnetization[i] = Flocking.magnetization_magnitude(θ)
    energy_density[i] = Flocking.hamiltonian(θ, cache) / length(θ)
end

data = (times=times, magnetization=magnetization, energy_density=energy_density)

# Plot using the precomputed diagnostics (avoids recomputing inside diagnostic_plot)
θ_final = reshape(mod2pi.(sol.u[end]), params.Lx, params.Ly)
stride = max(1, params.Lx ÷ 16)
xs = collect(1:stride:params.Lx)
ys = collect(1:stride:params.Ly)
θc = θ_final[xs, ys]

fig = Figure(size=(1100, 800))

axθ = Axis(fig[1, 1], title="final angle field", aspect=DataAspect())
hm = heatmap!(axθ, θ_final; colormap=:twilight)
Colorbar(fig[1, 2], hm, label="θ")

axn = Axis(fig[1, 3], title="coarse spin field", aspect=DataAspect())
arrows = arrows2d!(
    axn,
    repeat(xs, inner=length(ys)),
    repeat(ys, outer=length(xs)),
    vec(cos.(θc)),
    vec(sin.(θc));
    tipwidth=10,
    tiplength=8,
    lengthscale=0.7 * stride,
)
xlims!(axn, 1, params.Lx)
ylims!(axn, 1, params.Ly)

axm = Axis(fig[2, 1:2], title="magnetization", xlabel="t", ylabel="|m|")
lines!(axm, data.times, data.magnetization)

axe = Axis(fig[2, 3], title="XY energy density", xlabel="t", ylabel="Φ / N")
lines!(axe, data.times, data.energy_density)

save("diagnostics.png", fig)
