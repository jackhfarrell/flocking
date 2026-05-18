using Test
using Random
using StochasticDiffEq
using JLD2

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

@testset "chunked correlator geometry" begin
    L = 6
    shell_data = radial_displacement_shells(L)
    oriented_shell_data = radial_displacement_shells(L; oriented=true)
    @test length(shell_data.radii) == L ÷ 2
    @test length(shell_data.shells) == L ÷ 2
    @test all(!isempty, shell_data.shells)
    @test length(oriented_shell_data.radii) == L ÷ 2
    @test length(oriented_shell_data.shells) == L ÷ 2
    @test all(!isempty, oriented_shell_data.shells)

    for shell in shell_data.shells, (dx, dy) in shell
        @test any(isequal((mod(-dx, L), mod(-dy, L))), shell)
        for y in 1:L, x in 1:L
            forward = site_index(x + dx, y + dy, L)
            backward = site_index(x - dx, y - dy, L)
            @test 1 <= forward <= L^2
            @test 1 <= backward <= L^2
        end
    end

    for shell in oriented_shell_data.shells, (dx, dy) in shell
        @test !any(isequal((mod(-dx, L), mod(-dy, L))), shell)
    end
end

@testset "chunked correlator window" begin
    params = ModelParams(; L=6, Q=1.0, J=2.0, v=1.0)
    shell_data = radial_displacement_shells(params.L; oriented=true)
    window = [zeros(params.L^2) for _ in 1:5]
    F = chunk_correlator(window, params, shell_data)
    @test size(F) == (params.L ÷ 2, 3)
    @test all(isapprox.(F, 0.0; atol=1e-12))

    mean = zeros(size(F))
    m2 = zeros(size(F))
    stderr1 = online_mean_stderr!(mean, m2, F, 1)
    @test all(iszero, stderr1)
    stderr2 = online_mean_stderr!(mean, m2, F .+ 1, 2)
    @test all(stderr2 .>= 0)
end

@testset "eta equilibrium predicate" begin
    @test observable_window_range([0.3, 0.2, 0.25], 3) ≈ 0.1
    @test eta_window_range([0.3, 0.2], 3) == Inf
    @test eta_window_range([0.3, 0.2, 0.25], 3) ≈ 0.1
    @test eta_window_range([0.3, NaN, 0.25], 3) == Inf
    @test eta_equilibrium_reached([0.3, 0.2, 0.25], 3, 0.11)
    @test !eta_equilibrium_reached([0.3, 0.2, 0.25], 3, 0.09)
    @test equilibrium_window_blocks(5.0, 50.0, 5) == 10
    @test equilibrium_window_blocks(5.0, 0.0, 5) == 5
    stationarity = equilibrium_stationarity_reached(
        [0.1, 0.11, 0.105],
        [1.0, 1.01, 1.005],
        [0.8, 0.79, 0.805],
        3,
        0.02,
        0.02,
        0.02,
    )
    @test stationarity.reached
    @test stationarity.window == 3
    @test_throws ArgumentError eta_window_range([0.1], 0)
    @test_throws ArgumentError eta_equilibrium_reached([0.1], 1, -0.1)
    @test_throws ArgumentError equilibrium_window_blocks(0.0, 1.0, 1)
end

@testset "chunked correlator solver smoke" begin
    L = 4
    params = ModelParams(; L, Q=0.1, J=1.0, v=0.2)
    dt = 0.001
    sample_stride = 1
    T_max = 2
    rng = Random.MersenneTwister(7)
    theta = initial_angles(rng, L, :random)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    solver = SRIW1()

    burnin_problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!,
        theta, (0.0, 2dt), work)
    burnin_solution = solve(burnin_problem, solver; dt, adaptive=false,
        save_everystep=false, save_start=false, rng)
    theta = wrap_angles!(collect(burnin_solution.u[end]))

    shell_data = radial_displacement_shells(L; oriented=true)
    F_mean = zeros(Float64, L ÷ 2, T_max + 1)
    F_m2 = zeros(Float64, L ÷ 2, T_max + 1)
    chunk_advance_steps = (2 * T_max + 1) * sample_stride
    saveat = collect(0:sample_stride:chunk_advance_steps) .* dt

    for chunk in 1:2
        chunk_problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!,
            theta, (0.0, chunk_advance_steps * dt), work)
        chunk_solution = solve(chunk_problem, solver; dt, adaptive=false,
            saveat, save_start=true, rng)
        @test length(chunk_solution.u) == 2T_max + 2
        window = [wrap_angles!(collect(state)) for state in chunk_solution.u[1:(end - 1)]]
        @test length(window) == 2T_max + 1
        F_chunk = chunk_correlator(window, params, shell_data)
        F_stderr = online_mean_stderr!(F_mean, F_m2, F_chunk, chunk)
        @test size(F_stderr) == size(F_mean)
        theta = wrap_angles!(collect(chunk_solution.u[end]))
    end

    @test size(F_mean) == (L ÷ 2, T_max + 1)
    @test all(isfinite, F_mean)
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

@testset "production chunked correlator script smoke" begin
    output = tempname() * ".jld2"
    script = joinpath(@__DIR__, "..", "scripts", "run_chunked_correlator_production.jl")
    project = joinpath(@__DIR__, "..")
    run(`$(Base.julia_cmd()) --project=$project $script --L 6 --Q 0.0 --J 1.0 --v 0.0 --dt 0.001 --seed 3 --init ordered --equil-block-steps 1 --equil-max-blocks 1 --equil-window 1 --equil-window-time 0 --equil-eta-threshold 0.0 --equil-energy-threshold 0.0 --equil-magnetization-threshold 0.0 --equil-log-every 1 --sample-stride 1 --T-max 1 --nchunks 1 --chunk-log-samples 1 --fit-rmin 1 --fit-rmax 3 --output $output`)

    loaded = JLD2.load(output)
    @test haskey(loaded, "result")
    result = loaded["result"]
    @test hasproperty(result, :equilibration_history)
    @test hasproperty(result, :equilibrium_reached)
    @test hasproperty(result, :equilibrium_steps)
    @test hasproperty(result, :equilibrium_time)
    @test hasproperty(result, :F_mean)
    @test hasproperty(result, :F_stderr)
    @test hasproperty(result.equilibration_history, :window_blocks)
    @test hasproperty(result.equilibration_history, :window_time)
    @test hasproperty(result.config, :log_every_blocks)
    @test result.config.log_every_blocks == 1
    @test result.equilibrium_reached
    @test result.equilibrium_steps == 1
    @test size(result.F_mean) == (3, 2)
    @test size(result.F_stderr) == (3, 2)
end
