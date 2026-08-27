#!/usr/bin/env julia

# Measure how long a from-scratch v = 10 state takes to equilibrate on L = 200 — the
# single most expensive Stage-1 primitive and the seed source for the down-sweep. It
# also doubles as the sanity check that L = 200 is adequate at the hard end: the energy
# density and magnetization must plateau, not drift, before the plateau time is recorded.
#
# From an ordered (or random) start it advances theta in fixed step-blocks and stops
# when the windowed ranges of both observables fall below threshold, mirroring the
# Stage-1 re-equilibration primitive so the measured time transfers directly. Reports
# blocks / steps / model-time / wall-time to plateau and saves the equilibrated field.

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
        description="Measure from-scratch v=10 equilibration time on L=200.",
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
        "--v"
            arg_type = Float64
            default = 10.0
        "--dt"
            arg_type = Float64
            default = 0.001
        "--solver"
            arg_type = String
            default = "SRIW1"
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
        "--base-seed"
            arg_type = Int
            default = 1
        "--output"
            arg_type = String
            default = "results/v10_equilibration/v10_equilibration.jld2"
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

function select_solver(name::String)
    name == "SRA1" && return SRA1()
    name == "SRA2" && return SRA2()
    name == "SRA3" && return SRA3()
    name == "SRIW1" && return SRIW1()
    name == "EM" && return EM()
    throw(ArgumentError("unknown solver: $name (choose SRA1, SRA2, SRA3, SRIW1, EM)"))
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

# Advance from scratch until the energy density and magnetization plateau (their windowed
# ranges both fall below threshold), recording the wall-clock cost of each block.
function equilibrate(theta, params, dt, block_steps, max_blocks, window_blocks,
        energy_threshold, magnetization_threshold, log_every, work, solver, rng)
    L = params.L
    energies = Float64[]
    mags = Float64[]
    reached = false
    block = 0
    wall_start = time()
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
            loginfo("equilibration block", :block => block, :max_blocks => max_blocks,
                :time => block * block_steps * dt, :wall_s => time() - wall_start,
                :energy_density => energies[end], :magnetization => mags[end],
                :energy_window_range => energy_range,
                :magnetization_window_range => magnetization_range)
        end
        reached && break
    end
    wall_s = time() - wall_start
    history = (; blocks=block, steps=block * block_steps, energies, magnetizations=mags,
        wall_s)
    return theta, reached, history
end

function main()
    args = parse_args()
    L = args["L"]
    dt = args["dt"]
    v = args["v"]
    block_steps = args["equil-block-steps"]
    block_time = block_steps * dt
    window_blocks = equilibrium_window_blocks(
        block_time, args["equil-window-time"], args["equil-window"])
    log_every = args["equil-log-every"] == 0 ? window_blocks : args["equil-log-every"]
    solver = select_solver(args["solver"])

    params = ModelParams(; L, Q=args["gamma"], J=args["J"], v)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    rng = MersenneTwister(args["base-seed"])
    theta = initial_angles(rng, L, Symbol(args["init"]))

    loginfo("starting v10 equilibration", :L => L, :gamma => args["gamma"], :J => args["J"],
        :v => v, :dt => dt, :solver => args["solver"], :init => args["init"],
        :block_steps => block_steps, :window_blocks => window_blocks,
        :max_blocks => args["equil-max-blocks"])

    theta, reached, history = equilibrate(
        theta, params, dt, block_steps, args["equil-max-blocks"], window_blocks,
        args["equil-energy-threshold"], args["equil-magnetization-threshold"],
        log_every, work, solver, rng)

    equil_time = history.steps * dt
    config = (; L, gamma=args["gamma"], Q=args["gamma"], J=args["J"], v, dt,
        solver=args["solver"], init=args["init"], block_steps, window_blocks,
        energy_threshold=args["equil-energy-threshold"],
        magnetization_threshold=args["equil-magnetization-threshold"],
        base_seed=args["base-seed"], reached, blocks=history.blocks, steps=history.steps,
        equil_time, wall_s=history.wall_s, energy_density=history.energies[end],
        magnetization=history.magnetizations[end], output=args["output"])
    result = (; config, theta, energies=history.energies,
        magnetizations=history.magnetizations)

    mkpath(dirname(abspath(args["output"])))
    jldsave(args["output"]; result)
    loginfo("finished v10 equilibration", :reached => reached, :blocks => history.blocks,
        :steps => history.steps, :equil_time => equil_time, :wall_s => history.wall_s,
        :output => args["output"])
    return reached
end

main()
