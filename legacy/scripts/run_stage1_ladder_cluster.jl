#!/usr/bin/env julia

using ArgParse
using Dates
using JLD2
using Logging
using Printf
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

global_logger(ConsoleLogger(stderr, Logging.Info))

function parse_args()
    settings = ArgParseSettings(
        description="Bake a per-v equilibrium library by annealing theta along a two-direction v-ladder.",
    )
    @add_arg_table! settings begin
        "--L"
            arg_type = Int
            default = 200
        "--gamma"
            arg_type = Float64
            default = 1.0
        "--J"
            arg_type = Float64
            default = 2.0
        "--dt"
            arg_type = Float64
            default = 0.001
        "--vmin"
            arg_type = Float64
            default = 0.1
        "--vmax"
            arg_type = Float64
            default = 10.0
        "--nv"
            arg_type = Int
            default = 30
        "--directions"
            arg_type = String
            default = "up,down"
        "--ntrajectories"
            arg_type = Int
            default = 1
        "--base-seed"
            arg_type = Int
            default = 1
        "--up-seed"
            arg_type = String
            default = ""
        "--down-seed"
            arg_type = String
            default = ""
        "--init"
            arg_type = String
            default = "ordered"
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
        "--equil-energy-threshold"
            arg_type = Float64
            default = 0.02
        "--equil-magnetization-threshold"
            arg_type = Float64
            default = 0.02
        "--equil-log-every"
            arg_type = Int
            default = 0
        "--library-dir"
            arg_type = String
            default = "library/J2_Q1_L200"
    end
    return ArgParse.parse_args(settings)
end

function loginfo(message::String, fields::Pair...)
    pieces = ["$(first(field))=$(repr(last(field)))" for field in fields]
    println(stderr, Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"), " [info] ",
        message, isempty(pieces) ? "" : " " * join(pieces, " "))
    flush(stderr)
    flush(stdout)
    return nothing
end

function validate_args!(args, directions)
    args["L"] > 1 || throw(ArgumentError("--L must be greater than 1"))
    args["gamma"] >= 0 || throw(ArgumentError("--gamma must be nonnegative"))
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["vmin"] > 0 || throw(ArgumentError("--vmin must be positive"))
    args["vmax"] > args["vmin"] || throw(ArgumentError("--vmax must exceed --vmin"))
    args["nv"] > 0 || throw(ArgumentError("--nv must be positive"))
    args["ntrajectories"] > 0 || throw(ArgumentError("--ntrajectories must be positive"))
    args["equil-block-steps"] > 0 ||
        throw(ArgumentError("--equil-block-steps must be positive"))
    args["equil-max-blocks"] > 0 ||
        throw(ArgumentError("--equil-max-blocks must be positive"))
    args["equil-window"] > 0 || throw(ArgumentError("--equil-window must be positive"))
    args["equil-window-time"] >= 0 ||
        throw(ArgumentError("--equil-window-time must be nonnegative"))
    args["equil-energy-threshold"] >= 0 ||
        throw(ArgumentError("--equil-energy-threshold must be nonnegative"))
    args["equil-magnetization-threshold"] >= 0 ||
        throw(ArgumentError("--equil-magnetization-threshold must be nonnegative"))
    args["equil-log-every"] >= 0 ||
        throw(ArgumentError("--equil-log-every must be nonnegative"))
    isempty(directions) && throw(ArgumentError("--directions must be nonempty"))
    all(d -> d in ("up", "down"), directions) ||
        throw(ArgumentError("--directions entries must be up or down"))
    Symbol(args["init"]) in (:random, :ordered) ||
        throw(ArgumentError("--init must be random or ordered"))
    return nothing
end

function advance(theta, steps::Integer, dt::Real, work, solver, rng)
    prob = SDEProblem(
        LatticeFlockingSDE.drift!,
        LatticeFlockingSDE.noise!,
        theta,
        (0.0, steps * dt),
        work,
    )
    sol = solve(prob, solver; dt, adaptive=false, save_everystep=false,
        save_start=false, rng)
    return wrap_angles!(collect(sol.u[end]))
end

# Re-equilibrate at one rung until the energy density and magnetization plateau,
# i.e. their windowed ranges (the equilibration-diagnostic block statistic) both
# fall below threshold. Effort stops as soon as the state has stopped moving.
function reequilibrate!(theta, params, dt, block_steps, max_blocks, window_blocks,
        energy_threshold, magnetization_threshold, log_every, work, solver, rng, context)
    L = params.L
    energies = Float64[]
    mags = Float64[]
    reached = false
    block = 0
    while block < max_blocks
        block += 1
        theta = advance(theta, block_steps, dt, work, solver, rng)
        push!(energies, xy_energy(theta, params) / L^2)
        push!(mags, magnetization(theta))
        energy_range = observable_window_range(energies, window_blocks)
        magnetization_range = observable_window_range(mags, window_blocks)
        reached = energy_range <= energy_threshold &&
            magnetization_range <= magnetization_threshold
        if block == 1 || block % log_every == 0 || reached
            loginfo("re-equilibration block", context...,
                :block => block, :max_blocks => max_blocks,
                :time => block * block_steps * dt,
                :energy_density => energies[end], :magnetization => mags[end],
                :energy_window_range => energy_range,
                :magnetization_window_range => magnetization_range)
        end
        reached && break
    end
    history = (; blocks=block, steps=block * block_steps, energies, magnetizations=mags)
    return theta, reached, history
end

load_theta(path::String) = JLD2.load(path, "result").theta

function checkpoint_path(library_dir, vi::Integer, v::Real, direction::AbstractString,
        traj::Integer)
    key = @sprintf("v_%02d_%.6f", vi, v)
    return joinpath(library_dir, key, @sprintf("%s_traj_%03d.jld2", direction, traj))
end

function seed_theta(direction::AbstractString, args, L, traj::Integer)
    seed_path = direction == "up" ? args["up-seed"] : args["down-seed"]
    if !isempty(seed_path)
        return load_theta(seed_path)
    end
    rng = MersenneTwister(args["base-seed"] + 10_000 * traj)
    return initial_angles(rng, L, Symbol(args["init"]))
end

function main()
    args = parse_args()
    directions = split(args["directions"], ",")
    validate_args!(args, directions)

    L = args["L"]
    gamma = args["gamma"]
    dt = args["dt"]
    nv = args["nv"]
    block_steps = args["equil-block-steps"]
    block_time = block_steps * dt
    window_blocks = equilibrium_window_blocks(
        block_time, args["equil-window-time"], args["equil-window"])
    log_every = args["equil-log-every"] == 0 ? window_blocks : args["equil-log-every"]
    energy_threshold = args["equil-energy-threshold"]
    magnetization_threshold = args["equil-magnetization-threshold"]
    library_dir = args["library-dir"]

    vgrid = exp.(range(log(args["vmin"]), log(args["vmax"]); length=nv))
    solver = SRIW1()

    loginfo("starting stage-1 ladder", :L => L, :gamma => gamma, :J => args["J"],
        :dt => dt, :vmin => args["vmin"], :vmax => args["vmax"], :nv => nv,
        :directions => Tuple(directions), :ntrajectories => args["ntrajectories"],
        :window_blocks => window_blocks, :library_dir => library_dir,
        :solver => string(typeof(solver)))

    for direction in directions
        # Up-sweep walks vmin -> vmax; down-sweep walks vmax -> vmin. Both carry
        # theta forward across the ascending vgrid in their own direction.
        order = direction == "up" ? (1:nv) : (nv:-1:1)
        for traj in 1:args["ntrajectories"]
            theta = nothing
            for (rung, vi) in enumerate(order)
                v = vgrid[vi]
                output = checkpoint_path(library_dir, vi, v, direction, traj)

                if isfile(output)
                    # Resume: this rung is already baked, carry its field forward.
                    theta = load_theta(output)
                    loginfo("resuming from baked rung", :direction => direction,
                        :traj => traj, :rung => rung, :vi => vi, :v => v,
                        :output => output)
                    continue
                end

                if theta === nothing
                    theta = seed_theta(direction, args, L, traj)
                end

                params = ModelParams(; L, Q=gamma, J=args["J"], v)
                work = LatticeFlockingSDE.DriftWorkspace(params)
                seed = args["base-seed"] + (traj - 1) * 2nv +
                    (direction == "down" ? nv : 0) + (vi - 1)
                rng = MersenneTwister(seed)
                context = (:direction => direction, :traj => traj, :rung => rung,
                    :vi => vi, :v => v)

                loginfo("re-equilibrating rung", context...)
                theta, reached, history = reequilibrate!(
                    copy(theta), params, dt, block_steps, args["equil-max-blocks"],
                    window_blocks, energy_threshold, magnetization_threshold,
                    log_every, work, solver, rng, context)

                config = (;
                    L, gamma, Q=gamma, J=params.J, v, dt, direction, traj, rung, vi,
                    seed, base_seed=args["base-seed"], block_steps, window_blocks,
                    energy_threshold, magnetization_threshold,
                    reached, blocks=history.blocks, steps=history.steps,
                    energy_density=history.energies[end],
                    magnetization=history.magnetizations[end],
                    solver=string(typeof(solver)), output,
                )
                result = (; config, theta)

                mkpath(dirname(output))
                jldsave(output; result)
                loginfo("baked rung", :direction => direction, :traj => traj,
                    :rung => rung, :vi => vi, :v => v, :reached => reached,
                    :blocks => history.blocks, :output => output)
            end
        end
    end

    loginfo("finished stage-1 ladder", :library_dir => library_dir)
end

main()
