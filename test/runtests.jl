using Test
using Random
using DifferentialEquations
using StochasticDiffEq
using JLD2

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

const DEFAULT_ACTIVE_SNAPSHOT_DT = 2.0^-9

function build_snapshot_theta0(L::Integer; center_x=L / 4, center_y=L / 2,
        radius=L / 10, transition_width=L / 40, base_theta=nothing)
    theta0 = isnothing(base_theta) ? zeros(Float64, L^2) : copy(base_theta)
    seed_upward_bump!(theta0, L; center_x, center_y, radius, transition_width)
    return theta0
end

function run_snapshot_dataset_script(; output, dt, Q, J=8.0, v=2.0, times=nothing, seed=2)
    script = joinpath(@__DIR__, "..", "scripts", "run_snapshot_dataset.jl")
    project = joinpath(@__DIR__, "..")
    args = String[
        "--L", "32",
        "--Q", string(Q),
        "--J", string(J),
        "--v", string(v),
        "--dt", string(dt),
        "--seed", string(seed),
        "--block-size", "4",
        "--output", output,
    ]
    if !isnothing(times)
        append!(args, ["--times", times])
    end
    run(`$(Base.julia_cmd()) --project=$project $script $(args)`)
    loaded = JLD2.load(output)
    return loaded["dataset"]
end

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

@testset "lag step schedule" begin
    ntimes = 8
    lag_steps = 2000

    uniform = lag_step_schedule(ntimes, lag_steps; spacing=:uniform)
    @test uniform.cum_steps == collect(0:ntimes) .* lag_steps
    @test all(==(lag_steps), uniform.advance_gaps)
    @test length(uniform.advance_gaps) == 2ntimes

    geom = lag_step_schedule(ntimes, lag_steps; spacing=:geometric)
    @test length(geom.cum_steps) == ntimes + 1
    @test geom.cum_steps[1] == 0
    @test geom.cum_steps[2] == lag_steps
    @test geom.cum_steps[end] == ntimes * lag_steps
    @test issorted(geom.cum_steps)
    @test all(>(0), geom.gaps)
    # genuinely non-uniform geometric spacing
    @test !all(==(geom.gaps[1]), geom.gaps)
    # advance gaps are symmetric about the window midpoint
    @test geom.advance_gaps == [reverse(geom.gaps); geom.gaps]
    @test geom.advance_gaps[1:ntimes] == reverse(geom.advance_gaps[(ntimes + 1):end])

    # ntimes == 1 degenerates to the uniform single-lag schedule
    @test lag_step_schedule(1, lag_steps; spacing=:geometric).cum_steps == [0, lag_steps]

    @test_throws ArgumentError lag_step_schedule(0, lag_steps)
    @test_throws ArgumentError lag_step_schedule(ntimes, lag_steps; spacing=:linear)
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

@testset "snapshot perturbation drifts right" begin
    L = 80
    params = ModelParams(; L, Q=0.0, J=2.0, v=2.0)
    theta0 = build_snapshot_theta0(L)
    x0 = positive_sin_marker_x(theta0, L)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    problem = ODEProblem(LatticeFlockingSDE.drift!, theta0, (0.0, 8.0), work)
    solution = solve(problem, Tsit5(); saveat=[8.0], save_start=false)
    x1 = positive_sin_marker_x(solution.u[end], L)

    @test x1 > x0 + 8
end

@testset "snapshot timestep smoke" begin
    L = 48
    params = ModelParams(; L, Q=1e-4, J=2.0, v=2.0)
    theta0 = build_snapshot_theta0(L)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    times = [0.0, 1.0, 2.0]
    problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!,
        theta0, (0.0, last(times)), work)
    solution = solve(problem, EM(); dt=DEFAULT_ACTIVE_SNAPSHOT_DT, adaptive=false,
        saveat=times, save_start=true, seed=17)

    @test length(solution.u) == length(times)
    for state in solution.u
        @test all(isfinite, state)
        m = magnetization(state)
        @test isfinite(m)
        @test 0.2 <= m <= 1.0
    end
end

@testset "snapshot dataset script clean advection preset" begin
    output = tempname() * ".jld2"
    dataset = run_snapshot_dataset_script(output=output, dt=2.0^-10, Q=0.0)

    @test length(dataset.times) == 4
    @test length(dataset.x_centroids) == 4
    @test all(isfinite, dataset.x_centroids)
    @test dataset.metadata.calibrated_drift_speed > 0
    @test issorted(dataset.x_centroids)
    @test dataset.x_centroids[end] > dataset.x_centroids[1] + 2
    @test size(dataset.theta_snapshots) == (32, 32, 4)
end

@testset "snapshot dataset script noisy high-J drift" begin
    output = tempname() * ".jld2"
    dataset = run_snapshot_dataset_script(output=output, dt=2.0^-10, Q=1e-4,
        times="0,0.75,1.5,2.25", seed=5)

    @test all(isfinite, dataset.x_centroids)
    @test all(isfinite, dataset.theta_snapshots)
    @test dataset.x_centroids[end] > dataset.x_centroids[1]
end

@testset "snapshot dataset timestep consistency" begin
    output_coarse = tempname() * ".jld2"
    output_fine = tempname() * ".jld2"
    times = "0,0.25,0.5,0.75"
    coarse = run_snapshot_dataset_script(output=output_coarse, dt=2.0^-9, Q=1e-4,
        times=times, seed=7)
    fine = run_snapshot_dataset_script(output=output_fine, dt=2.0^-10, Q=1e-4,
        times=times, seed=7)

    coarse_shift = coarse.x_centroids[end] - coarse.x_centroids[1]
    fine_shift = fine.x_centroids[end] - fine.x_centroids[1]
    @test coarse_shift > 0
    @test fine_shift > 0
    @test abs(coarse_shift - fine_shift) < 0.75
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

@testset "spin-aligned correlator script smoke" begin
    output = tempname() * ".jld2"
    figure = tempname() * ".png"
    script = joinpath(@__DIR__, "..", "scripts", "run_spin_aligned_f_correlator.jl")
    project = joinpath(@__DIR__, "..")
    dr = 0.5

    run(`$(Base.julia_cmd()) --project=$project $script --L 6 --gamma 0.0 --J 1.0 --v 1.0 --dt 0.001 --dr $dr --burnin-time 0.0 --T-max 1 --ntimes 1 --nwindows 1 --seed 2 --burnin-log-time 1.0 --window-log-every 1 --output $output --figure $figure`)

    loaded = JLD2.load(output)
    @test haskey(loaded, "result")
    result = loaded["result"]
    @test result.config.dr == dr
    @test length(result.radii) == 6
    @test size(result.F_mean, 1) == length(result.radii)
    @test size(result.C_plus_mean) == size(result.F_mean)
    @test size(result.C_minus_mean) == size(result.F_mean)
    @test size(result.C_plus_stderr) == size(result.F_mean)
    @test size(result.C_minus_stderr) == size(result.F_mean)
end

@testset "ordinary C correlator cluster script smoke" begin
    output = tempname() * ".jld2"
    script = joinpath(@__DIR__, "..", "scripts", "run_ordinary_c_correlator_cluster.jl")
    project = joinpath(@__DIR__, "..")
    dr = 0.5

    run(`$(Base.julia_cmd()) --project=$project $script --L 6 --gamma 0.0 --J 1.0 --dt 0.001 --dr $dr --nangles 8 --burnin-time 0.0 --T-max 1 --ntimes 1 --nchunks 1 --array-count 1 --array-id 1 --burnin-log-time 1.0 --window-log-every 1 --output $output`)

    loaded = JLD2.load(output)
    @test haskey(loaded, "result")
    result = loaded["result"]
    @test result.passive.config.v == 0.0
    @test result.active.config.v == 1.0
    @test result.passive.config.dr == dr
    @test length(result.passive.times_signed) == 3
    @test first(result.passive.times_signed) ≈ -1.0
    @test last(result.passive.times_signed) ≈ 1.0
    @test size(result.passive.C_mean) == (length(result.passive.radii), 3)
    @test size(result.active.C_mean) == size(result.passive.C_mean)
    @test all(isapprox.(result.passive.C_mean, 1.0; atol=1e-12))
    @test all(isapprox.(result.active.C_mean, 1.0; atol=1e-12))
end

@testset "stage-1 ladder script library and resume" begin
    library = joinpath(tempname(), "ladder")
    script = joinpath(@__DIR__, "..", "scripts", "run_stage1_ladder_cluster.jl")
    project = joinpath(@__DIR__, "..")
    ladder_args = String[
        "--L", "6", "--gamma", "0.0", "--J", "1.0", "--dt", "0.001",
        "--vmin", "0.5", "--vmax", "2.0", "--nv", "3",
        "--directions", "up,down", "--ntrajectories", "2", "--base-seed", "5",
        "--init", "ordered",
        "--equil-block-steps", "1", "--equil-max-blocks", "3", "--equil-window", "1",
        "--equil-window-time", "0", "--equil-energy-threshold", "1e9",
        "--equil-magnetization-threshold", "1e9", "--equil-log-every", "1",
        "--library-dir", library,
    ]
    run(`$(Base.julia_cmd()) --project=$project $script $(ladder_args)`)

    # v-keyed library: one directory per grid point, each holding both directions
    # for every trajectory.
    vkeys = filter(d -> startswith(d, "v_"), readdir(library))
    @test length(vkeys) == 3
    for vkey in vkeys, direction in ("up", "down"), traj in 1:2
        path = joinpath(library, vkey, "$(direction)_traj_$(lpad(traj, 3, '0')).jld2")
        @test isfile(path)
        result = JLD2.load(path, "result")
        @test length(result.theta) == 36
        @test result.config.direction == direction
        @test result.config.traj == traj
        @test result.config.reached
    end

    # Kill-and-resume: drop the last baked rung, keep an earlier one's state, rerun.
    last_rung = joinpath(library, "v_01_0.500000", "down_traj_002.jld2")
    early_rung = joinpath(library, "v_03_2.000000", "up_traj_001.jld2")
    dropped_theta = JLD2.load(last_rung, "result").theta
    early_theta = JLD2.load(early_rung, "result").theta
    early_mtime = mtime(early_rung)
    rm(last_rung)

    run(`$(Base.julia_cmd()) --project=$project $script $(ladder_args)`)

    # The deleted rung is regenerated bit-for-bit (deterministic per-rung seed)...
    @test JLD2.load(last_rung, "result").theta == dropped_theta
    # ...while completed rungs are not recomputed (file untouched).
    @test mtime(early_rung) == early_mtime
    @test JLD2.load(early_rung, "result").theta == early_theta
end

@testset "stage-2 measurement reads per-v library and chi2 budget" begin
    library = joinpath(tempname(), "ladder")
    output_dir = joinpath(tempname(), "stage2")
    ladder = joinpath(@__DIR__, "..", "scripts", "run_stage1_ladder_cluster.jl")
    stage2 = joinpath(@__DIR__, "..", "scripts", "run_stage2_measurement_cluster.jl")
    project = joinpath(@__DIR__, "..")

    run(`$(Base.julia_cmd()) --project=$project $ladder
        --L 6 --gamma 0.0 --J 1.0 --dt 0.001 --vmin 0.5 --vmax 2.0 --nv 3
        --directions up --ntrajectories 1 --base-seed 5 --init ordered
        --equil-block-steps 1 --equil-max-blocks 3 --equil-window 1
        --equil-window-time 0 --equil-energy-threshold 1e9
        --equil-magnetization-threshold 1e9 --equil-log-every 1
        --library-dir $library`)

    # Budget 4,2,1 is chi-squared-shaped (heavier at low v). With chunks-per-job 2 the
    # low-v rung spills into two single-core array tasks; both target v = 0.5.
    stage2_args = String[
        "--library-dir", library, "--direction", "up", "--traj", "1",
        "--L", "6", "--gamma", "0.0", "--J", "1.0",
        "--vmin", "0.5", "--vmax", "2.0", "--nv", "3",
        "--window-budget", "4,2,1", "--chunks-per-job", "2",
        "--dt", "0.001", "--dr", "0.5", "--r-max", "3.0", "--solver", "SRA1",
        "--T-max", "1", "--ntimes", "1", "--output-dir", output_dir,
    ]
    for array_id in (1, 2)
        run(`$(Base.julia_cmd()) --project=$project $stage2 $(stage2_args) --array-id $array_id`)
    end

    # Both tasks land in the v = 0.5 subdirectory as collapse-ready sample files.
    vdir = joinpath(output_dir, "v_01_0.500000")
    files = sort(filter(f -> startswith(f, "sample_"), readdir(vdir)))
    @test files == ["sample_0001.jld2", "sample_0002.jld2"]

    result = JLD2.load(joinpath(vdir, files[1]), "result")
    @test result.v_values == [0.5]
    @test length(result.radii) == 6
    @test result.times == [0.0, 1.0]
    @test size(result.F) == (1, length(result.radii), 2)
    @test size(result.C_plus) == size(result.F)
    @test size(result.C_minus) == size(result.F)
    @test all(isfinite, result.F)
    # Baked state read straight from the per-v library, no burn-in.
    @test result.config.v == 0.5
    @test result.config.direction == "up"
    @test result.config.budget_v == 4
    @test result.config.chunks == 2
    @test !hasproperty(result.config, :burnin_steps)
    @test result.config.equilibrium ==
        joinpath(library, "v_01_0.500000", "up_traj_001.jld2")

    # An array id past the budget's task table is rejected.
    bad = Cmd(`$(Base.julia_cmd()) --project=$project $stage2 $(stage2_args) --array-id 99`;
        ignorestatus=true)
    @test !success(bad)
end

@testset "L=200 timing benchmark writes a budget table" begin
    output = tempname() * ".jld2"
    csv = tempname() * ".csv"
    script = joinpath(@__DIR__, "..", "scripts", "run_l200_timing_benchmark_cluster.jl")
    project = joinpath(@__DIR__, "..")

    run(`$(Base.julia_cmd()) --project=$project $script
        --L 6 --gamma 0.0 --J 1.0 --velocities 0.5,1.0 --solvers SRA1,SRA2
        --dts 0.001 --dr 0.5 --r-max 3.0 --T-max 1 --ntimes 1 --reps 1
        --output $output --csv $csv`)

    result = JLD2.load(output, "result")
    # One row per (velocity, solver, dt) combo: 2 velocities x 2 solvers x 1 dt.
    @test length(result.rows) == 4
    @test result.config.ntimes == 1
    @test length(result.radii) == 6
    for row in result.rows
        @test row.window_steps == 2000  # 2*ntimes advances of lag_steps = T_max/ntimes/dt
        @test row.t_window >= 0
        @test isapprox(row.t_window, row.t_integrate + row.t_correlator)
        @test row.integrate_us_per_step >= 0
    end
    @test Set(getproperty.(result.rows, :solver)) == Set(["SRA1", "SRA2"])

    lines = readlines(csv)
    @test lines[1] ==
        "v,solver,dt,window_steps,t_integrate_s,t_correlator_s,t_window_s,integrate_us_per_step"
    @test length(lines) == 5  # header + one line per combo
end

@testset "CRN dt-coupling reproduces the coarse path" begin
    L = 8
    params = ModelParams(; L, Q=1.0, J=2.0, v=1.0)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    rng = Random.MersenneTwister(3)
    theta0 = initial_angles(rng, L, :ordered)
    solver = SRA1()
    dt = 0.01
    prob = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!,
        theta0, (0.0, 0.5), work)
    fine = solve(prob, solver; dt=dt / 2, adaptive=false, save_everystep=false,
        save_start=false, save_noise=true, seed=7)

    # Re-driving the same Wiener path at dt/2 reproduces the trajectory exactly.
    same = solve(remake(prob; noise=NoiseWrapper(fine.W)), solver; dt=dt / 2,
        adaptive=false, save_everystep=false, save_start=false)
    @test maximum(abs.(fine.u[end] .- same.u[end])) < 1e-12

    # Coarsening that same path to dt gives a small, controlled (nonzero) difference.
    coarse = solve(remake(prob; noise=NoiseWrapper(fine.W)), solver; dt=dt,
        adaptive=false, save_everystep=false, save_start=false)
    gap = maximum(abs.(fine.u[end] .- coarse.u[end]))
    @test 0 < gap < 1.0
end

@testset "dt-convergence CRN harness smoke" begin
    script = joinpath(@__DIR__, "..", "scripts", "dt_convergence_crn.jl")
    project = joinpath(@__DIR__, "..")
    output = read(`$(Base.julia_cmd()) --project=$project $script --L 16 --nchunks 1
        --ntimes 3 --burnin-time 1.0 --T-max 3.0 --dr 1.0 --r-max 7.0
        --velocities 1.0 --base-seed 4`, String)
    @test occursin("zeta(dt/2)", output)
    @test occursin("max |dzeta|", output)
end

@testset "spin-aligned correlator rejects invalid dr" begin
    script = joinpath(@__DIR__, "..", "scripts", "run_spin_aligned_f_correlator.jl")
    project = joinpath(@__DIR__, "..")
    output = tempname() * ".jld2"
    figure = tempname() * ".png"
    cmd = Cmd(`$(Base.julia_cmd()) --project=$project $script --L 6 --gamma 0.0 --J 1.0 --v 1.0 --dt 0.001 --dr 0.0 --burnin-time 0.0 --T-max 1 --ntimes 1 --nwindows 1 --seed 2 --burnin-log-time 1.0 --window-log-every 1 --output $output --figure $figure`; ignorestatus=true)
    @test !success(cmd)
end
