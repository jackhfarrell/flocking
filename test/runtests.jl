using Test
using Random

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

@testset "periodic indexing" begin
    @test site_index(1, 1, 4) == 1
    @test site_index(5, 1, 4) == 1
    @test site_index(0, 1, 4) == 4
    @test site_index(1, 0, 4) == 13
end

@testset "initial conditions" begin
    rng = Random.MersenneTwister(1)
    @test all(initial_angles(rng, 4, :ordered) .== 0)
    @test length(initial_angles(rng, 4, :random)) == 16
    @test_throws ArgumentError initial_angles(rng, 4, :bad)
end

@testset "energy derivative and passive drift" begin
    params = ModelParams(; L=4, Q=1.7, J=2.3, v=0.0)
    theta = collect(range(0.1, 2.4; length=params.L^2))
    mu = zeros(params.L^2)
    compute_mu!(mu, theta, params)

    eps = 1e-6
    for i in eachindex(theta)
        plus = copy(theta)
        minus = copy(theta)
        plus[i] += eps
        minus[i] -= eps
        fd = (xy_energy(plus, params) - xy_energy(minus, params)) / (2eps)
        @test isapprox(mu[i], fd; rtol=1e-6, atol=1e-6)
    end

    du = similar(theta)
    LatticeFlockingSDE.drift!(du, theta, LatticeFlockingSDE.DriftWorkspace(params), 0.0)
    @test du ≈ .-(params.Q / 2) .* mu
end

@testset "correlation normalization" begin
    L = 8
    theta = zeros(L^2)
    _, c, counts = radial_correlation(theta, L)
    @test all(counts .> 0)
    @test all(isapprox.(c, 1.0; atol=1e-12))
end

@testset "small smoke run" begin
    config = SimulationConfig(; L=8, Q=0.1, J=1.5, v=0.0, dt=0.001,
        burnin_steps=2, sample_stride=1, nsamples=2, ntrajectories=1, seed=11,
        fit_rmin=1, fit_rmax=3)
    result = run_ensemble(config)
    @test length(result.times) == 2
    @test length(result.correlation_mean) == config.params.L ÷ 2
    @test haskey(result.fit, :eta)
    @test hasproperty(result, :config)
    @test hasproperty(result, :seeds)
    @test hasproperty(result, :energy_density_mean)
    @test hasproperty(result, :magnetization_mean)
    @test hasproperty(result, :radii)
    @test hasproperty(result, :correlation_mean)
    @test hasproperty(result, :correlation_stderr)
    @test hasproperty(result, :fit)
    @test !hasproperty(result, :helicity_modulus)
    @test !hasproperty(result, :eta_helicity)
end
