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
        description="Run one Slurm-array task that saves a baked passive equilibrium state.",
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
        "--burnin-time"
            arg_type = Float64
            default = 1000.0
        "--array-count"
            arg_type = Int
            default = 500
        "--array-id"
            arg_type = Int
            default = 0
        "--base-seed"
            arg_type = Int
            default = 1
        "--burnin-log-time"
            arg_type = Float64
            default = 10.0
        "--initial-condition"
            arg_type = String
            default = "ordered"
        "--output-dir"
            arg_type = String
            default = "equilibria/J2_Q1_L200"
        "--output"
            arg_type = String
            default = ""
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

function steps_for_time(time::Real, dt::Real, name::String)
    steps = round(Int, time / dt)
    isapprox(steps * dt, time; atol=100eps(max(abs(time), abs(dt))), rtol=0) ||
        throw(ArgumentError("$name must be an integer multiple of dt"))
    return steps
end

function slurm_array_id(args)
    if args["array-id"] != 0
        return args["array-id"]
    end
    value = get(ENV, "SLURM_ARRAY_TASK_ID", "")
    return isempty(value) ? 1 : parse(Int, value)
end

function output_path(args, array_id::Integer)
    if !isempty(args["output"])
        return args["output"]
    end
    return joinpath(args["output-dir"], "equilibrium_$(@sprintf("%04d", array_id)).jld2")
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

function validate_args!(args, array_id::Integer)
    args["L"] > 1 || throw(ArgumentError("--L must be greater than 1"))
    args["gamma"] >= 0 || throw(ArgumentError("--gamma must be nonnegative"))
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["burnin-time"] >= 0 || throw(ArgumentError("--burnin-time must be nonnegative"))
    args["array-count"] > 0 || throw(ArgumentError("--array-count must be positive"))
    1 <= array_id <= args["array-count"] ||
        throw(ArgumentError("array id must be in 1:$(args["array-count"])"))
    args["burnin-log-time"] > 0 ||
        throw(ArgumentError("--burnin-log-time must be positive"))
    args["initial-condition"] in ("ordered", "random") ||
        throw(ArgumentError("--initial-condition must be ordered or random"))
    return nothing
end

function main()
    args = parse_args()
    array_id = slurm_array_id(args)
    validate_args!(args, array_id)

    L = args["L"]
    gamma = args["gamma"]
    dt = args["dt"]
    burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
    burnin_log_steps = steps_for_time(args["burnin-log-time"], dt, "burnin-log-time")
    seed = args["base-seed"] + array_id - 1
    output = output_path(args, array_id)
    initial_condition = Symbol(args["initial-condition"])

    params = ModelParams(; L, Q=gamma, J=args["J"], v=0.0)
    rng = MersenneTwister(seed)
    theta = initial_angles(rng, L, initial_condition)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    solver = SRIW1()

    loginfo("starting baked equilibrium",
        :array_id => array_id,
        :array_count => args["array-count"],
        :L => L,
        :gamma => gamma,
        :J => params.J,
        :v => params.v,
        :dt => dt,
        :burnin_time => args["burnin-time"],
        :burnin_steps => burnin_steps,
        :initial_condition => initial_condition,
        :seed => seed,
        :solver => string(typeof(solver)),
        :output => output)

    completed_burnin = 0
    while completed_burnin < burnin_steps
        segment_steps = min(burnin_log_steps, burnin_steps - completed_burnin)
        theta = advance(theta, segment_steps, dt, work, solver, rng)
        completed_burnin += segment_steps
        loginfo("baked equilibrium burn-in progress",
            :array_id => array_id,
            :steps => completed_burnin,
            :total_steps => burnin_steps,
            :time => completed_burnin * dt,
            :total_time => args["burnin-time"])
    end

    config = (;
        L=params.L,
        gamma,
        Q=gamma,
        J=params.J,
        v=params.v,
        dt,
        burnin_time=args["burnin-time"],
        burnin_steps,
        array_id,
        array_count=args["array-count"],
        seed,
        base_seed=args["base-seed"],
        initial_condition,
        solver=string(typeof(solver)),
        output,
    )
    result = (; config, theta)

    mkpath(dirname(output))
    loginfo("saving baked equilibrium", :array_id => array_id, :output => output)
    jldsave(output; result)
    loginfo("saved baked equilibrium", :array_id => array_id, :output => output)
end

main()
