using Test
using Random
using DifferentialEquations
using JLD2
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

@testset "model" begin
    @test site_index(5, 1, 4) == 1
    @test site_index(0, 1, 4) == 4

    params = ModelParams(; L=4, Q=1.7, J=2.3, v=0.0)
    theta = collect(range(0.1, 2.4; length=params.L^2))
    mu = zeros(params.L^2)
    compute_mu!(mu, theta, params)

    epsilon = 1e-6
    for index in eachindex(theta)
        plus = copy(theta)
        minus = copy(theta)
        plus[index] += epsilon
        minus[index] -= epsilon
        derivative = (xy_energy(plus, params) - xy_energy(minus, params)) / (2epsilon)
        @test isapprox(mu[index], derivative; rtol=1e-6, atol=1e-6)
    end

    drift = similar(theta)
    LatticeFlockingSDE.drift!(
        drift, theta, LatticeFlockingSDE.DriftWorkspace(params), 0.0)
    @test drift ≈ .-(params.Q / 2) .* mu
    @test expected_eta(params) ≈ inv(params.J) / (2π)
end

@testset "fixed-step window" begin
    schedule = lag_step_schedule(4, 8; spacing=:geometric)
    @test schedule.cum_steps[[1, end]] == [0, 32]
    @test schedule.advance_gaps == [reverse(schedule.gaps); schedule.gaps]

    L = 6
    params = ModelParams(; L, Q=0.0, J=2.0, v=1.0)
    theta = zeros(L^2)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    advanced = LatticeFlockingSDE.advance_fixed(
        theta, 2, 0.001, work, SRA1(), MersenneTwister(2))
    @test advanced == theta

    window = [copy(theta) for _ in 1:5]
    F = LatticeFlockingSDE.spin_aligned_correlators(window, L, 0.5:0.5:3.0)
    @test size(F) == (6, 3)
    @test all(iszero, F)
end

@testset "collapse exponent" begin
    radii = collect(0.5:0.5:60.0)
    times = [0.0; exp.(range(log(0.5), log(8.0); length=8))]
    eta = 0.3
    zeta = 0.4
    mean_F = zeros(length(radii), length(times))
    for tidx in 2:length(times)
        scaled_radius = radii ./ times[tidx]^zeta
        mean_F[:, tidx] .= times[tidx]^(-eta) .* exp.(-((scaled_radius .- 8) ./ 3).^2)
    end
    stderr_F = fill(0.01, size(mean_F))

    fit = best_collapse(radii, times, mean_F, stderr_F)
    @test abs(fit.best.zeta - zeta) <= 0.02
    robustness = fit_window_robustness(radii, times, mean_F, stderr_F)
    @test robustness.halfspread <= 0.05
end

@testset "production scripts" begin
    project = joinpath(@__DIR__, "..")
    library = joinpath(tempname(), "library")
    output = joinpath(tempname(), "measurements")
    bake = joinpath(project, "scripts", "bake_exponent_sweep.jl")
    measure = joinpath(project, "scripts", "measure_exponent_sweep.jl")

    run(`$(Base.julia_cmd()) --project=$project $bake
        --temperature 1 --direction up --trajectory 1 --L 6 --Q 0 --dt 0.001
        --v-min 0.5 --v-max 1 --nv 2 --block-steps 1 --max-blocks 1
        --window-time 0 --window-blocks 1 --energy-threshold 1e9
        --magnetization-threshold 1e9 --output-dir $library`)

    checkpoints = filter(endswith(".jld2"),
        [joinpath(dir, name) for (dir, _, names) in walkdir(library) for name in names])
    @test length(checkpoints) == 2
    @test all(load(path, "result").config.reached for path in checkpoints)

    run(`$(Base.julia_cmd()) --project=$project $measure
        --temperature 1 --direction up --ntrajectories 1 --array-id 1
        --L 6 --Q 0 --dt 0.001 --v-min 0.5 --v-max 1 --nv 2
        --dr 0.5 --r-max 3 --T-max 0.003 --ntimes 3
        --windows-low-v 1 --windows-high-v 1
        --library-dir $library --output-dir $output`)

    result_file = only([
        joinpath(dir, name) for (dir, _, names) in walkdir(output) for name in names
        if name == "measurement.jld2"
    ])
    result = load(result_file, "result")
    @test result.config.temperature == 1.0
    @test result.config.J == 1.0
    @test result.config.v == 0.5
    @test size(result.F_mean) == (6, 4)
    @test all(isfinite, result.F_mean)

    for script in readdir(joinpath(project, "slurm"); join=true)
        endswith(script, ".sh") && run(`bash -n $script`)
    end
end
