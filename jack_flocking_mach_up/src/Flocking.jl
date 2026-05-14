module Flocking

using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Random
using Statistics
using StochasticDiffEq: EM

export ModelParams, simulate, diagnostics, diagnostic_plot

struct ModelParams{T<:Real}
    Lx::Int
    Ly::Int
    J::T
    Q::T
    v::T
end

function ModelParams(; L=32, Lx=L, Ly=L, J=1.0, Q=0.2, v=0.5)
    ModelParams(Int(Lx), Int(Ly), promote(J, Q, v)...)
end

struct LatticeCache{T<:Real}
    params::ModelParams{T}
    xp::Vector{Int}
    xm::Vector{Int}
    yp::Vector{Int}
    ym::Vector{Int}
    μ::Vector{T}
end

function LatticeCache(params::ModelParams{T}) where {T}
    xp, xm, yp, ym = neighbor_indices(params.Lx, params.Ly)
    LatticeCache(params, xp, xm, yp, ym, zeros(T, params.Lx * params.Ly))
end

function neighbor_indices(Lx, Ly)
    N = Lx * Ly
    xp = Vector{Int}(undef, N)
    xm = Vector{Int}(undef, N)
    yp = Vector{Int}(undef, N)
    ym = Vector{Int}(undef, N)

    for y in 1:Ly, x in 1:Lx
        i = x + (y - 1) * Lx
        xp[i] = (x == Lx ? 1 : x + 1) + (y - 1) * Lx
        xm[i] = (x == 1 ? Lx : x - 1) + (y - 1) * Lx
        yp[i] = x + (y == Ly ? 0 : y) * Lx
        ym[i] = x + (y == 1 ? Ly - 1 : y - 2) * Lx
    end

    xp, xm, yp, ym
end

function chemical_potential!(μ, θ, cache::LatticeCache)
    J = cache.params.J
    xp, xm, yp, ym = cache.xp, cache.xm, cache.yp, cache.ym

    @inbounds for i in eachindex(θ)
        θi = θ[i]
        μ[i] = J * (
            -sin(θ[xp[i]] - θi) +
            sin(θi - θ[xm[i]]) -
            sin(θ[yp[i]] - θi) +
            sin(θi - θ[ym[i]])
        )
    end

    μ
end

function hamiltonian(θ, params::ModelParams)
    cache = LatticeCache(params)
    hamiltonian(θ, cache)
end

function hamiltonian(θ, cache::LatticeCache)
    J = cache.params.J
    xp, yp = cache.xp, cache.yp
    Φ = zero(eltype(θ))

    @inbounds for i in eachindex(θ)
        Φ -= J * (cos(θ[i] - θ[xp[i]]) + cos(θ[i] - θ[yp[i]]))
    end

    Φ
end

function drift!(dθ, θ, cache::LatticeCache, t)
    params = cache.params
    xp, xm, yp, ym = cache.xp, cache.xm, cache.yp, cache.ym
    μ = chemical_potential!(cache.μ, θ, cache)

    @inbounds for i in eachindex(θ)
        active_angle = (
            sin(θ[xp[i]]) - sin(θ[xm[i]]) -
            cos(θ[yp[i]]) + cos(θ[ym[i]])
        )

        active_x = (
            cos(θ[i]) * μ[xp[i]] -
            cos(θ[i]) * μ[xm[i]] +
            cos(θ[xp[i]]) * μ[xp[i]] -
            cos(θ[xm[i]]) * μ[xm[i]]
        )

        active_y = (
            sin(θ[i]) * μ[yp[i]] -
            sin(θ[i]) * μ[ym[i]] +
            sin(θ[yp[i]]) * μ[yp[i]] -
            sin(θ[ym[i]]) * μ[ym[i]]
        )

        dθ[i] = -(params.Q / 2) * μ[i] + (params.v / 2) * (active_angle + active_x + active_y)
    end

    nothing
end

function noise!(dθ, θ, cache::LatticeCache, t)
    fill!(dθ, sqrt(cache.params.Q))
    nothing
end

function simulate(params::ModelParams; T=5.0, dt=0.01, saveat=0.1, seed=1)
    rng = MersenneTwister(seed)
    cache = LatticeCache(params)
    θ0 = 2π .* rand(rng, params.Lx * params.Ly)
    problem = SDEProblem(drift!, noise!, θ0, (0.0, T), cache; noise_rate_prototype=similar(θ0))
    solve(problem, EM(); dt, saveat, adaptive=false, rng)
end

function magnetization_magnitude(θ)
    mx = mean(cos, θ)
    my = mean(sin, θ)
    hypot(mx, my)
end

function diagnostics(sol, params::ModelParams)
    cache = LatticeCache(params)
    times = collect(sol.t)
    magnetization = [magnetization_magnitude(θ) for θ in sol.u]
    energy_density = [hamiltonian(θ, cache) / length(θ) for θ in sol.u]
    (; times, magnetization, energy_density)
end

function diagnostic_plot(sol, params::ModelParams)
    data = diagnostics(sol, params)
    θ = reshape(mod2pi.(sol.u[end]), params.Lx, params.Ly)
    stride = max(1, params.Lx ÷ 16)
    xs = collect(1:stride:params.Lx)
    ys = collect(1:stride:params.Ly)
    θc = θ[xs, ys]

    fig = Figure(size=(1100, 800))

    axθ = Axis(fig[1, 1], title="final angle field", aspect=DataAspect())
    hm = heatmap!(axθ, θ; colormap=:twilight)
    Colorbar(fig[1, 2], hm, label="θ")

    axn = Axis(fig[1, 3], title="coarse spin field", aspect=DataAspect())
    arrows2d!(
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

    fig
end

end
