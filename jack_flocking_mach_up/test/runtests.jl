using Flocking
using Test

const F = Flocking

@testset "chemical potential" begin
    params = ModelParams(L=4, J=1.3, Q=0.2, v=0.5)
    cache = F.LatticeCache(params)
    θ = range(0.1, 1.7; length=params.Lx * params.Ly) |> collect
    μ = similar(θ)
    F.chemical_potential!(μ, θ, cache)

    ϵ = 1e-6
    for i in eachindex(θ)
        θp = copy(θ)
        θm = copy(θ)
        θp[i] += ϵ
        θm[i] -= ϵ
        fd = (F.hamiltonian(θp, cache) - F.hamiltonian(θm, cache)) / (2ϵ)
        @test μ[i] ≈ fd atol=1e-7 rtol=1e-7
    end
end

@testset "uniform state" begin
    params = ModelParams(L=5, J=1.0, Q=0.2, v=0.5)
    cache = F.LatticeCache(params)
    θ = fill(0.4, params.Lx * params.Ly)
    μ = similar(θ)
    F.chemical_potential!(μ, θ, cache)
    @test all(abs.(μ) .< 1e-14)
end

@testset "inactive drift" begin
    active = ModelParams(L=4, J=1.0, Q=0.3, v=0.0)
    cache = F.LatticeCache(active)
    θ = range(0.2, 2.1; length=active.Lx * active.Ly) |> collect
    dθ = similar(θ)
    μ = similar(θ)

    F.drift!(dθ, θ, cache, 0.0)
    F.chemical_potential!(μ, θ, cache)

    @test dθ ≈ .-(active.Q / 2) .* μ
end

@testset "short solve" begin
    params = ModelParams(L=4, J=1.0, Q=0.1, v=0.2)
    sol = simulate(params; T=0.03, dt=0.01, saveat=0.01, seed=2)

    @test length(sol.u) == 4
    @test all(length(θ) == params.Lx * params.Ly for θ in sol.u)
    @test all(all(isfinite, θ) for θ in sol.u)

    data = diagnostics(sol, params)
    @test length(data.times) == length(sol.u)
    @test length(data.magnetization) == length(sol.u)
    @test length(data.energy_density) == length(sol.u)
end
