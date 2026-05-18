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
        description="Run one headless Slurm-array chunk of the spin-aligned F-correlator calculation.",
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
            default = 1.0
        "--dt"
            arg_type = Float64
            default = 0.001
        "--burnin-time"
            arg_type = Float64
            default = 100.0
        "--T-max"
            arg_type = Float64
            default = 16.0
        "--ntimes"
            arg_type = Int
            default = 8
        "--nchunks"
            arg_type = Int
            default = 10
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
            default = 1.0
        "--window-log-every"
            arg_type = Int
            default = 1
        "--log-radius-index"
            arg_type = Int
            default = 1
        "--log-time-index"
            arg_type = Int
            default = 0
        "--output-dir"
            arg_type = String
            default = "results/spin_aligned_f_correlator_L200_J2_v1_gamma1"
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
    return joinpath(args["output-dir"], "job_$(@sprintf("%04d", array_id)).jld2")
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

function interpolated_spin_dot(cos_field, sin_field, L::Integer, px::Real, py::Real,
        ref_cos::Real, ref_sin::Real)
    x0 = floor(Int, px)
    y0 = floor(Int, py)
    fx = px - x0
    fy = py - y0
    wx0 = 1 - fx
    wx1 = fx
    wy0 = 1 - fy
    wy1 = fy

    ix0 = mod1(x0, L)
    ix1 = mod1(x0 + 1, L)
    iy0 = mod1(y0, L)
    iy1 = mod1(y0 + 1, L)

    c = 0.0
    s = 0.0
    @inbounds for (ix, wx) in ((ix0, wx0), (ix1, wx1))
        for (iy, wy) in ((iy0, wy0), (iy1, wy1))
            w = wx * wy
            idx = site_index(ix, iy, L)
            c += w * cos_field[idx]
            s += w * sin_field[idx]
        end
    end
    return ref_cos * c + ref_sin * s
end

function spin_aligned_f_correlator(window::AbstractVector, params::ModelParams,
        radii::AbstractVector)
    L = params.L
    nsites = L * L
    isodd(length(window)) || throw(ArgumentError("window length must be odd"))
    all(length(state) == nsites for state in window) ||
        throw(DimensionMismatch("each window state must have length L^2"))

    ntimes = length(window) ÷ 2
    mid = ntimes + 1
    cos_window = [cos.(state) for state in window]
    sin_window = [sin.(state) for state in window]
    cos_mid = cos_window[mid]
    sin_mid = sin_window[mid]
    F = zeros(Float64, length(radii), ntimes + 1)

    @inbounds for (ridx, r) in enumerate(radii)
        for lag in 0:ntimes
            cos_minus = cos_window[mid - lag]
            sin_minus = sin_window[mid - lag]
            cos_plus = cos_window[mid + lag]
            sin_plus = sin_window[mid + lag]
            accum = 0.0

            for y in 1:L, x in 1:L
                center = site_index(x, y, L)
                cx = cos_mid[center]
                sy = sin_mid[center]
                dx = r * cx
                dy = r * sy

                forward_plus = interpolated_spin_dot(cos_plus, sin_plus, L,
                    x + dx, y + dy, cx, sy)
                backward_minus = interpolated_spin_dot(cos_minus, sin_minus, L,
                    x - dx, y - dy, cx, sy)
                forward_minus = interpolated_spin_dot(cos_minus, sin_minus, L,
                    x + dx, y + dy, cx, sy)
                backward_plus = interpolated_spin_dot(cos_plus, sin_plus, L,
                    x - dx, y - dy, cx, sy)
                accum += 0.25 * (forward_plus + backward_minus -
                    forward_minus - backward_plus)
            end

            F[ridx, lag + 1] = accum / nsites
        end
    end

    return F
end

function validate_args!(args, array_id::Integer)
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["burnin-time"] >= 0 || throw(ArgumentError("--burnin-time must be nonnegative"))
    args["T-max"] > 0 || throw(ArgumentError("--T-max must be positive"))
    args["ntimes"] > 0 || throw(ArgumentError("--ntimes must be positive"))
    args["nchunks"] > 0 || throw(ArgumentError("--nchunks must be positive"))
    args["array-count"] > 0 || throw(ArgumentError("--array-count must be positive"))
    1 <= array_id <= args["array-count"] ||
        throw(ArgumentError("array id must be in 1:$(args["array-count"])"))
    args["burnin-log-time"] > 0 ||
        throw(ArgumentError("--burnin-log-time must be positive"))
    args["window-log-every"] > 0 ||
        throw(ArgumentError("--window-log-every must be positive"))
    return nothing
end

function main()
    args = parse_args()
    array_id = slurm_array_id(args)
    validate_args!(args, array_id)

    L = args["L"]
    gamma = args["gamma"]
    dt = args["dt"]
    ntimes = args["ntimes"]
    nchunks = args["nchunks"]
    lag_time = args["T-max"] / ntimes
    lag_steps = steps_for_time(lag_time, dt, "T-max / ntimes")
    burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
    burnin_log_steps = steps_for_time(args["burnin-log-time"], dt, "burnin-log-time")
    seed = args["base-seed"] + array_id - 1
    output = output_path(args, array_id)

    params = ModelParams(; L, Q=gamma, J=args["J"], v=args["v"])
    rng = MersenneTwister(seed)
    theta = initial_angles(rng, L, :ordered)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    solver = SRIW1()

    radii = collect(1.0:(L ÷ 2))
    times = collect(0:ntimes) .* lag_steps .* dt
    log_radius_index = args["log-radius-index"]
    log_time_index = args["log-time-index"] == 0 ? ntimes + 1 : args["log-time-index"]
    1 <= log_radius_index <= length(radii) ||
        throw(ArgumentError("--log-radius-index is out of range"))
    1 <= log_time_index <= length(times) ||
        throw(ArgumentError("--log-time-index is out of range"))

    loginfo("starting headless spin-aligned F correlator",
        :array_id => array_id,
        :array_count => args["array-count"],
        :L => L,
        :gamma => gamma,
        :J => params.J,
        :v => params.v,
        :dt => dt,
        :burnin_time => args["burnin-time"],
        :burnin_steps => burnin_steps,
        :T_max => args["T-max"],
        :ntimes => ntimes,
        :lag_time => lag_time,
        :lag_steps => lag_steps,
        :nchunks => nchunks,
        :seed => seed,
        :solver => string(typeof(solver)),
        :output => output)

    completed_burnin = 0
    while completed_burnin < burnin_steps
        segment_steps = min(burnin_log_steps, burnin_steps - completed_burnin)
        theta = advance(theta, segment_steps, dt, work, solver, rng)
        completed_burnin += segment_steps
        loginfo("burn-in progress",
            :array_id => array_id,
            :steps => completed_burnin,
            :total_steps => burnin_steps,
            :time => completed_burnin * dt,
            :total_time => args["burnin-time"])
    end

    F_mean = zeros(Float64, length(radii), ntimes + 1)
    F_m2 = zeros(Float64, length(radii), ntimes + 1)
    F_stderr = zeros(Float64, length(radii), ntimes + 1)

    for chunk in 1:nchunks
        window = Vector{Vector{Float64}}(undef, 2ntimes + 1)
        window[1] = copy(theta)

        for sample_index in 1:(2ntimes)
            theta = advance(theta, lag_steps, dt, work, solver, rng)
            window[sample_index + 1] = copy(theta)
            if sample_index == 1 || sample_index % args["window-log-every"] == 0 ||
                    sample_index == 2ntimes
                loginfo("window integration progress",
                    :array_id => array_id,
                    :chunk => chunk,
                    :nchunks => nchunks,
                    :sample => sample_index,
                    :total_samples => 2ntimes,
                    :steps => sample_index * lag_steps,
                    :total_steps => 2ntimes * lag_steps,
                    :time => sample_index * lag_time,
                    :total_time => 2 * args["T-max"])
            end
        end

        loginfo("computing spin-aligned F chunk",
            :array_id => array_id,
            :chunk => chunk,
            :nchunks => nchunks,
            :nradii => length(radii),
            :ntimes => ntimes)
        F_chunk = spin_aligned_f_correlator(window, params, radii)
        F_stderr = online_mean_stderr!(F_mean, F_m2, F_chunk, chunk)
        loginfo("rolling spin-aligned F",
            :array_id => array_id,
            :chunk => chunk,
            :nchunks => nchunks,
            :radius => radii[log_radius_index],
            :lag => times[log_time_index],
            :F => F_mean[log_radius_index, log_time_index],
            :stderr => F_stderr[log_radius_index, log_time_index])
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
        T_max=args["T-max"],
        ntimes,
        lag_time,
        lag_steps,
        array_id,
        array_count=args["array-count"],
        chunks_per_job=nchunks,
        nchunks,
        seed,
        base_seed=args["base-seed"],
        initial_condition=:ordered,
        solver=string(typeof(solver)),
        interpolation=:periodic_bilinear_spin_vector,
        log_radius_index,
        log_time_index,
        output,
    )
    result = (; config, radii, times, F_mean, F_stderr)

    mkpath(dirname(output))
    loginfo("saving headless spin-aligned F correlator",
        :array_id => array_id,
        :output => output)
    jldsave(output; result)
    loginfo("saved headless spin-aligned F correlator",
        :array_id => array_id,
        :output => output)
end

main()
