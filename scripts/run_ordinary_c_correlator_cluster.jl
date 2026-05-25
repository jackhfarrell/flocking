#!/usr/bin/env julia

using ArgParse
using Dates
using FFTW
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
        description="Run one headless Slurm-array chunk of ordinary isotropic C(r,t) for v=0 and v=1.",
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
        "--dr"
            arg_type = Float64
            default = 0.25
        "--nangles"
            arg_type = Int
            default = 256
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
            default = "results/ordinary_c_correlator_L200_J2_v0_v1_gamma1"
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

function bilinear_periodic(field::AbstractMatrix{<:Real}, x::Real, y::Real)
    Lx, Ly = size(field)
    x0 = floor(Int, x)
    y0 = floor(Int, y)
    fx = x - x0
    fy = y - y0
    ix0 = mod(x0, Lx) + 1
    ix1 = mod(x0 + 1, Lx) + 1
    iy0 = mod(y0, Ly) + 1
    iy1 = mod(y0 + 1, Ly) + 1

    return (1 - fx) * (1 - fy) * field[ix0, iy0] +
        fx * (1 - fy) * field[ix1, iy0] +
        (1 - fx) * fy * field[ix0, iy1] +
        fx * fy * field[ix1, iy1]
end

function correlation_field(theta_t::AbstractVector{<:Real}, theta0::AbstractVector{<:Real},
        L::Integer)
    zt = reshape(exp.(im .* theta_t), L, L)
    z0 = reshape(exp.(im .* theta0), L, L)
    return real.(ifft(fft(zt) .* conj.(fft(z0)))) ./ (L * L)
end

function radial_sample(field::AbstractMatrix{<:Real}, radii::AbstractVector{<:Real},
        angles::AbstractVector{<:Real})
    C = zeros(Float64, length(radii))
    @inbounds for (ridx, r) in enumerate(radii)
        accum = 0.0
        for angle in angles
            accum += bilinear_periodic(field, r * cos(angle), r * sin(angle))
        end
        C[ridx] = accum / length(angles)
    end
    return C
end

function ordinary_c_correlator(window::AbstractVector, L::Integer,
        radii::AbstractVector{<:Real}, angles::AbstractVector{<:Real})
    isodd(length(window)) || throw(ArgumentError("window length must be odd"))
    all(length(state) == L * L for state in window) ||
        throw(DimensionMismatch("each window state must have length L^2"))

    ntimes = length(window) ÷ 2
    mid = ntimes + 1
    C = zeros(Float64, length(radii), 2ntimes + 1)
    @inbounds for (tidx, offset) in enumerate(-ntimes:ntimes)
        field = correlation_field(window[mid + offset], window[mid], L)
        C[:, tidx] .= radial_sample(field, radii, angles)
    end
    return C
end

function validate_args!(args, array_id::Integer)
    args["L"] > 1 || throw(ArgumentError("--L must be greater than 1"))
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["dr"] > 0 || throw(ArgumentError("--dr must be positive"))
    args["nangles"] > 0 || throw(ArgumentError("--nangles must be positive"))
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
    args["dr"] <= args["L"] / 2 || throw(ArgumentError("--dr must be at most L / 2"))
    return nothing
end

function velocity_seed(base_seed::Integer, array_id::Integer, velocity_index::Integer)
    return base_seed + 10_000 * (array_id - 1) + 1_000_000 * (velocity_index - 1)
end

function run_velocity(args, v::Real, velocity_index::Integer, array_id::Integer,
        radii, times_signed, angles, output::String)
    L = args["L"]
    gamma = args["gamma"]
    dt = args["dt"]
    ntimes = args["ntimes"]
    nchunks = args["nchunks"]
    lag_time = args["T-max"] / ntimes
    lag_steps = steps_for_time(lag_time, dt, "T-max / ntimes")
    burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
    burnin_log_steps = steps_for_time(args["burnin-log-time"], dt, "burnin-log-time")
    seed = velocity_seed(args["base-seed"], array_id, velocity_index)

    params = ModelParams(; L, Q=gamma, J=args["J"], v)
    rng = MersenneTwister(seed)
    theta = initial_angles(rng, L, :ordered)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    solver = SRIW1()

    log_radius_index = args["log-radius-index"]
    log_time_index = args["log-time-index"] == 0 ? ntimes + 1 : args["log-time-index"]
    1 <= log_radius_index <= length(radii) ||
        throw(ArgumentError("--log-radius-index is out of range"))
    1 <= log_time_index <= length(times_signed) ||
        throw(ArgumentError("--log-time-index is out of range"))

    loginfo("starting ordinary C velocity block",
        :array_id => array_id,
        :array_count => args["array-count"],
        :L => L,
        :gamma => gamma,
        :J => params.J,
        :v => params.v,
        :dt => dt,
        :dr => args["dr"],
        :nangles => args["nangles"],
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
            :v => params.v,
            :steps => completed_burnin,
            :total_steps => burnin_steps,
            :time => completed_burnin * dt,
            :total_time => args["burnin-time"])
    end

    C_mean = zeros(Float64, length(radii), length(times_signed))
    C_m2 = zeros(Float64, length(radii), length(times_signed))
    C_stderr = zeros(Float64, length(radii), length(times_signed))

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
                    :v => params.v,
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

        loginfo("computing ordinary C chunk",
            :array_id => array_id,
            :v => params.v,
            :chunk => chunk,
            :nchunks => nchunks,
            :nradii => length(radii),
            :ntimes_signed => length(times_signed))
        C_chunk = ordinary_c_correlator(window, L, radii, angles)
        C_stderr = online_mean_stderr!(C_mean, C_m2, C_chunk, chunk)
        loginfo("rolling ordinary C",
            :array_id => array_id,
            :v => params.v,
            :chunk => chunk,
            :nchunks => nchunks,
            :radius => radii[log_radius_index],
            :time => times_signed[log_time_index],
            :C => C_mean[log_radius_index, log_time_index],
            :stderr_C => C_stderr[log_radius_index, log_time_index])
    end

    config = (;
        L=params.L,
        gamma,
        Q=gamma,
        J=params.J,
        v=params.v,
        dt,
        dr=args["dr"],
        nangles=args["nangles"],
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
        radial_sampling=:periodic_bilinear_angular_average,
        log_radius_index,
        log_time_index,
        output,
    )
    return (; config, radii, times_signed, C_mean, C_stderr, C_m2)
end

function main()
    args = parse_args()
    array_id = slurm_array_id(args)
    validate_args!(args, array_id)

    ntimes = args["ntimes"]
    lag_time = args["T-max"] / ntimes
    lag_steps = steps_for_time(lag_time, args["dt"], "T-max / ntimes")
    times_signed = collect(-ntimes:ntimes) .* lag_steps .* args["dt"]
    radii = collect(args["dr"]:args["dr"]:(args["L"] / 2))
    angles = (0:(args["nangles"] - 1)) .* (2pi / args["nangles"])
    output = output_path(args, array_id)

    passive = run_velocity(args, 0.0, 1, array_id, radii, times_signed, angles, output)
    active = run_velocity(args, 1.0, 2, array_id, radii, times_signed, angles, output)

    config = (;
        L=args["L"],
        gamma=args["gamma"],
        Q=args["gamma"],
        J=args["J"],
        dt=args["dt"],
        dr=args["dr"],
        nangles=args["nangles"],
        burnin_time=args["burnin-time"],
        T_max=args["T-max"],
        ntimes=args["ntimes"],
        lag_time,
        lag_steps,
        array_id,
        array_count=args["array-count"],
        chunks_per_job=args["nchunks"],
        nchunks=args["nchunks"],
        base_seed=args["base-seed"],
        velocities=(passive.config.v, active.config.v),
        output,
    )
    result = (; config, passive, active)

    mkpath(dirname(output))
    loginfo("saving ordinary C correlator", :array_id => array_id, :output => output)
    jldsave(output; result)
    loginfo("saved ordinary C correlator", :array_id => array_id, :output => output)
end

main()
