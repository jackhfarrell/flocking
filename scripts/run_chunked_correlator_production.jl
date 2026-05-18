#!/usr/bin/env julia

using ArgParse
using JLD2
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(
        description="Run a production chunked F-correlator calculation after stationarity-based equilibration.",
    )
    @add_arg_table! settings begin
        "--L"
            arg_type = Int
            default = 32
        "--Q"
            arg_type = Float64
            default = 1.0
        "--J"
            arg_type = Float64
            default = 2.0
        "--v"
            arg_type = Float64
            default = 1.0
        "--dt"
            arg_type = Float64
            default = 0.005
        "--seed"
            arg_type = Int
            default = 1
        "--init"
            arg_type = String
            default = "random"
        "--output"
            arg_type = String
            default = "results/chunked_correlator_production.jld2"
        "--equil-block-steps"
            arg_type = Int
            default = 1000
        "--equil-max-blocks"
            arg_type = Int
            default = 10_000
        "--equil-window"
            arg_type = Int
            default = 5
        "--equil-window-time"
            arg_type = Float64
            default = 50.0
        "--equil-eta-threshold"
            arg_type = Float64
            default = 0.01
        "--equil-energy-threshold"
            arg_type = Float64
            default = 0.02
        "--equil-magnetization-threshold"
            arg_type = Float64
            default = 0.02
        "--equil-log-every"
            arg_type = Int
            default = 0
        "--allow-unequilibrated"
            action = :store_true
        "--sample-stride"
            arg_type = Int
            default = 10
        "--T-max"
            arg_type = Int
            default = 10
        "--nchunks"
            arg_type = Int
            default = 100
        "--chunk-log-samples"
            arg_type = Int
            default = 10
        "--log-radius-index"
            arg_type = Int
            default = 1
        "--log-time-index"
            arg_type = Int
            default = 0
        "--fit-rmin"
            arg_type = Float64
            default = 2.0
        "--fit-rmax"
            arg_type = Float64
            default = Inf
    end
    return ArgParse.parse_args(settings)
end

function validate_args!(args)
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["equil-block-steps"] > 0 ||
        throw(ArgumentError("--equil-block-steps must be positive"))
    args["equil-max-blocks"] > 0 ||
        throw(ArgumentError("--equil-max-blocks must be positive"))
    args["equil-window"] > 0 || throw(ArgumentError("--equil-window must be positive"))
    args["equil-window-time"] >= 0 ||
        throw(ArgumentError("--equil-window-time must be nonnegative"))
    args["equil-eta-threshold"] >= 0 ||
        throw(ArgumentError("--equil-eta-threshold must be nonnegative"))
    args["equil-energy-threshold"] >= 0 ||
        throw(ArgumentError("--equil-energy-threshold must be nonnegative"))
    args["equil-magnetization-threshold"] >= 0 ||
        throw(ArgumentError("--equil-magnetization-threshold must be nonnegative"))
    args["equil-log-every"] >= 0 ||
        throw(ArgumentError("--equil-log-every must be nonnegative"))
    args["sample-stride"] > 0 || throw(ArgumentError("--sample-stride must be positive"))
    args["T-max"] >= 0 || throw(ArgumentError("--T-max must be nonnegative"))
    args["nchunks"] > 0 || throw(ArgumentError("--nchunks must be positive"))
    args["chunk-log-samples"] > 0 ||
        throw(ArgumentError("--chunk-log-samples must be positive"))
    Symbol(args["init"]) in (:random, :ordered) ||
        throw(ArgumentError("--init must be random or ordered"))
    return nothing
end

function solve_advance(theta, steps::Integer, dt::Real, work, solver, rng; saveat=nothing,
        save_start::Bool=false)
    prob = SDEProblem(
        LatticeFlockingSDE.drift!,
        LatticeFlockingSDE.noise!,
        theta,
        (0.0, steps * dt),
        work,
    )
    if saveat === nothing
        return solve(prob, solver; dt, adaptive=false, save_everystep=false,
            save_start=false, rng)
    end
    return solve(prob, solver; dt, adaptive=false, saveat, save_start, rng)
end

function equilibrate!(theta, params, args, work, solver, rng)
    L = params.L
    fit_rmax = isfinite(args["fit-rmax"]) ? args["fit-rmax"] : L / 3
    block_time = args["equil-block-steps"] * args["dt"]
    window_blocks = equilibrium_window_blocks(
        block_time, args["equil-window-time"], args["equil-window"])
    window_time = window_blocks * block_time
    log_every_blocks = args["equil-log-every"] == 0 ? window_blocks :
        args["equil-log-every"]
    blocks = Int[]
    steps = Int[]
    times = Float64[]
    energies = Float64[]
    magnetizations = Float64[]
    etas = Float64[]
    eta_ranges = Float64[]
    energy_ranges = Float64[]
    magnetization_ranges = Float64[]
    reached = false

    for block in 1:args["equil-max-blocks"]
        sol = solve_advance(theta, args["equil-block-steps"], args["dt"], work, solver, rng)
        theta = wrap_angles!(collect(sol.u[end]))

        total_steps = block * args["equil-block-steps"]
        r, c, _ = radial_correlation(theta, L)
        fit = fit_power_law(r, c; rmin=args["fit-rmin"], rmax=fit_rmax)
        push!(blocks, block)
        push!(steps, total_steps)
        push!(times, total_steps * args["dt"])
        push!(energies, xy_energy(theta, params) / L^2)
        push!(magnetizations, magnetization(theta))
        push!(etas, fit.eta)
        stationarity = equilibrium_stationarity_reached(
            etas, energies, magnetizations, window_blocks,
            args["equil-eta-threshold"], args["equil-energy-threshold"],
            args["equil-magnetization-threshold"],
        )
        prev_stationarity = length(etas) > window_blocks ? equilibrium_stationarity_reached(
            etas[1:(end - 1)], energies[1:(end - 1)], magnetizations[1:(end - 1)],
            window_blocks, args["equil-eta-threshold"], args["equil-energy-threshold"],
            args["equil-magnetization-threshold"],
        ) : nothing
        push!(eta_ranges, stationarity.eta_range)
        push!(energy_ranges, stationarity.energy_range)
        push!(magnetization_ranges, stationarity.magnetization_range)

        if block == 1 || block % log_every_blocks == 0 || stationarity.reached
            eta_window_delta = prev_stationarity === nothing ? NaN :
                stationarity.eta_range - prev_stationarity.eta_range
            energy_window_delta = prev_stationarity === nothing ? NaN :
                stationarity.energy_range - prev_stationarity.energy_range
            magnetization_window_delta = prev_stationarity === nothing ? NaN :
                stationarity.magnetization_range - prev_stationarity.magnetization_range
            @info "equilibration block" block max_blocks=args["equil-max-blocks"] total_steps time=times[end] window_blocks=window_blocks window_time=window_time energy_density=energies[end] magnetization=magnetizations[end] eta=etas[end] eta_window_range=stationarity.eta_range energy_window_range=stationarity.energy_range magnetization_window_range=stationarity.magnetization_range eta_window_delta energy_window_delta magnetization_window_delta eta_threshold=args["equil-eta-threshold"] energy_threshold=args["equil-energy-threshold"] magnetization_threshold=args["equil-magnetization-threshold"]
        end

        if stationarity.reached
            reached = true
            break
        end
    end

    history = (;
        block=blocks,
        steps,
        time=times,
        energy_density=energies,
        magnetization=magnetizations,
        eta=etas,
        eta_window_range=eta_ranges,
        energy_window_range=energy_ranges,
        magnetization_window_range=magnetization_ranges,
        window_blocks=window_blocks,
        window_time=window_time,
        eta_threshold=args["equil-eta-threshold"],
        energy_threshold=args["equil-energy-threshold"],
        magnetization_threshold=args["equil-magnetization-threshold"],
        log_every_blocks=log_every_blocks,
    )
    return theta, reached, history
end

function integrate_chunk_with_logs!(theta, chunk::Integer, params, args, work, solver, rng)
    chunk_sample_intervals = 2 * args["T-max"] + 1
    remaining_intervals = chunk_sample_intervals
    completed_intervals = 0
    states = Vector{Vector{Float64}}()
    push!(states, copy(theta))

    while remaining_intervals > 0
        segment_intervals = min(args["chunk-log-samples"], remaining_intervals)
        segment_steps = segment_intervals * args["sample-stride"]
        saveat = collect(args["sample-stride"]:args["sample-stride"]:segment_steps) .*
            args["dt"]
        sol = solve_advance(theta, segment_steps, args["dt"], work, solver, rng;
            saveat, save_start=false)

        for state in sol.u
            push!(states, wrap_angles!(collect(state)))
        end

        theta = copy(states[end])
        completed_intervals += segment_intervals
        remaining_intervals -= segment_intervals
        @info "chunk integration progress" chunk nchunks=args["nchunks"] samples=completed_intervals total_samples=chunk_sample_intervals steps=completed_intervals * args["sample-stride"] total_steps=chunk_sample_intervals * args["sample-stride"]
    end

    window = states[1:(end - 1)]
    return theta, window
end

function main()
    args = parse_args()
    validate_args!(args)

    params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
    rng = MersenneTwister(args["seed"])
    theta = initial_angles(rng, params.L, Symbol(args["init"]))
    work = LatticeFlockingSDE.DriftWorkspace(params)
    solver = SRIW1()

    shell_data = radial_displacement_shells(params.L; oriented=true)
    radii = shell_data.radii
    times = collect(0:args["T-max"]) .* args["sample-stride"] .* args["dt"]
    log_radius_index = args["log-radius-index"]
    log_time_index = args["log-time-index"] == 0 ? args["T-max"] + 1 :
        args["log-time-index"]
    1 <= log_radius_index <= length(radii) ||
        throw(ArgumentError("--log-radius-index is out of range"))
    1 <= log_time_index <= length(times) ||
        throw(ArgumentError("--log-time-index is out of range"))

    @info "starting production chunked correlator" L=params.L Q=params.Q J=params.J v=params.v dt=args["dt"] seed=args["seed"] init=args["init"] equil_block_steps=args["equil-block-steps"] equil_max_blocks=args["equil-max-blocks"] equil_window=args["equil-window"] equil_window_time=args["equil-window-time"] equil_eta_threshold=args["equil-eta-threshold"] equil_energy_threshold=args["equil-energy-threshold"] equil_magnetization_threshold=args["equil-magnetization-threshold"] sample_stride=args["sample-stride"] T_max=args["T-max"] nchunks=args["nchunks"]

    theta, equilibrium_reached, equilibration_history =
        equilibrate!(theta, params, args, work, solver, rng)
    equilibrium_steps = isempty(equilibration_history.steps) ? 0 :
        equilibration_history.steps[end]
    equilibrium_time = equilibrium_steps * args["dt"]

    if !equilibrium_reached && !args["allow-unequilibrated"]
        error("equilibrium was not reached after $(args["equil-max-blocks"]) blocks; pass --allow-unequilibrated to continue anyway")
    elseif !equilibrium_reached
        @warn "continuing without stationarity equilibrium" equilibrium_steps equilibrium_time window_blocks=equilibration_history.window_blocks window_time=equilibration_history.window_time
    else
        @info "equilibrium reached" equilibrium_steps equilibrium_time window_blocks=equilibration_history.window_blocks window_time=equilibration_history.window_time eta=equilibration_history.eta[end] eta_window_range=equilibration_history.eta_window_range[end] energy_window_range=equilibration_history.energy_window_range[end] magnetization_window_range=equilibration_history.magnetization_window_range[end]
    end

    F_mean = zeros(Float64, length(radii), args["T-max"] + 1)
    F_m2 = zeros(Float64, length(radii), args["T-max"] + 1)
    F_stderr = zeros(Float64, length(radii), args["T-max"] + 1)

    for chunk in 1:args["nchunks"]
        theta, window = integrate_chunk_with_logs!(
            theta, chunk, params, args, work, solver, rng)
        F_chunk = chunk_correlator(window, params, shell_data)
        F_stderr = online_mean_stderr!(F_mean, F_m2, F_chunk, chunk)

        @info "rolling correlator" chunk nchunks=args["nchunks"] radius=radii[log_radius_index] lag=times[log_time_index] F=F_mean[log_radius_index, log_time_index] stderr=F_stderr[log_radius_index, log_time_index]
    end

    config = (;
        L=params.L,
        Q=params.Q,
        J=params.J,
        v=params.v,
        dt=args["dt"],
        seed=args["seed"],
        initial_condition=Symbol(args["init"]),
        solver=string(typeof(solver)),
        equil_block_steps=args["equil-block-steps"],
        equil_max_blocks=args["equil-max-blocks"],
        equil_window=args["equil-window"],
        equil_window_time=args["equil-window-time"],
        equil_eta_threshold=args["equil-eta-threshold"],
        equil_energy_threshold=args["equil-energy-threshold"],
        equil_magnetization_threshold=args["equil-magnetization-threshold"],
        log_every_blocks=equilibration_history.log_every_blocks,
        allow_unequilibrated=args["allow-unequilibrated"],
        sample_stride=args["sample-stride"],
        T_max=args["T-max"],
        nchunks=args["nchunks"],
        chunk_log_samples=args["chunk-log-samples"],
        fit_rmin=args["fit-rmin"],
        fit_rmax=args["fit-rmax"],
        log_radius_index,
        log_time_index,
    )
    result = (;
        config,
        equilibration_history,
        equilibrium_reached,
        equilibrium_steps,
        equilibrium_time,
        radii,
        times,
        F_mean,
        F_stderr,
    )

    mkpath(dirname(args["output"]))
    jldsave(args["output"]; result)
    @info "saved production chunked correlator" output=args["output"]
end

main()
