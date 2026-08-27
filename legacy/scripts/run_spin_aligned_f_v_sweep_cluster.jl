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
        description="Run one baked-equilibrium spin-aligned F-correlator log-v sweep task.",
    )
    @add_arg_table! settings begin
        "--equilibrium-dir"
            arg_type = String
            default = "equilibria/J2_Q1_L200"
        "--equilibrium"
            arg_type = String
            default = ""
        "--output-dir"
            arg_type = String
            default = "results/spin_aligned_f_correlator_L200_J2_Q1_v_log_sweep"
        "--output"
            arg_type = String
            default = ""
        "--array-count"
            arg_type = Int
            default = 500
        "--array-id"
            arg_type = Int
            default = 0
        "--base-seed"
            arg_type = Int
            default = 100000
        "--dt"
            arg_type = Float64
            default = 0.001
        "--dr"
            arg_type = Float64
            default = 0.25
        "--r-max"
            arg_type = Float64
            default = 45.0
        "--lag-spacing"
            arg_type = String
            default = "geometric"
        "--solver"
            arg_type = String
            default = "SRA1"
        "--T-max"
            arg_type = Float64
            default = 16.0
        "--ntimes"
            arg_type = Int
            default = 8
        "--v-min"
            arg_type = Float64
            default = 0.01
        "--v-max"
            arg_type = Float64
            default = 1.0
        "--nv"
            arg_type = Int
            default = 30
        "--v-values"
            arg_type = String
            default = ""
        "--window-log-every"
            arg_type = Int
            default = 4
        "--log-radius-index"
            arg_type = Int
            default = 1
        "--log-time-index"
            arg_type = Int
            default = 0
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

function equilibrium_path(args, array_id::Integer)
    if !isempty(args["equilibrium"])
        return args["equilibrium"]
    end
    return joinpath(args["equilibrium-dir"], "equilibrium_$(@sprintf("%04d", array_id)).jld2")
end

function output_path(args, array_id::Integer)
    if !isempty(args["output"])
        return args["output"]
    end
    return joinpath(args["output-dir"], "sample_$(@sprintf("%04d", array_id)).jld2")
end

function parse_v_values(args)
    if !isempty(args["v-values"])
        return [parse(Float64, strip(value)) for value in split(args["v-values"], ',')]
    end
    args["nv"] > 0 || throw(ArgumentError("--nv must be positive"))
    args["v-min"] > 0 || throw(ArgumentError("--v-min must be positive for log spacing"))
    args["v-max"] > args["v-min"] || throw(ArgumentError("--v-max must exceed --v-min"))
    return collect(exp.(range(log(args["v-min"]), log(args["v-max"]); length=args["nv"])))
end

# SRA1/SRA2/SRA3 are purpose-built for additive, state-independent noise (the model's
# noise! fills with √Q); SRIW1/EM remain selectable for cross-checks.
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

function spin_aligned_correlators(window::AbstractVector, params::ModelParams,
        radii::AbstractVector)
    L = params.L
    nsites = L * L
    isodd(length(window)) || throw(ArgumentError("window length must be odd"))

    ntimes = length(window) ÷ 2
    mid = ntimes + 1
    cos_window = [cos.(state) for state in window]
    sin_window = [sin.(state) for state in window]
    cos_mid = cos_window[mid]
    sin_mid = sin_window[mid]
    F = zeros(Float64, length(radii), ntimes + 1)
    C_plus = zeros(Float64, length(radii), ntimes + 1)
    C_minus = zeros(Float64, length(radii), ntimes + 1)

    @inbounds for (ridx, r) in enumerate(radii)
        for lag in 0:ntimes
            cos_minus = cos_window[mid - lag]
            sin_minus = sin_window[mid - lag]
            cos_plus = cos_window[mid + lag]
            sin_plus = sin_window[mid + lag]
            accum_f = 0.0
            accum_c_plus = 0.0
            accum_c_minus = 0.0

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

                accum_f += 0.25 * (forward_plus + backward_minus -
                    forward_minus - backward_plus)
                accum_c_plus += 0.5 * (forward_plus + backward_plus)
                accum_c_minus += 0.5 * (forward_minus + backward_minus)
            end

            F[ridx, lag + 1] = accum_f / nsites
            C_plus[ridx, lag + 1] = accum_c_plus / nsites
            C_minus[ridx, lag + 1] = accum_c_minus / nsites
        end
    end

    return (; F, C_plus, C_minus)
end

function validate_args!(args, array_id::Integer)
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["dr"] > 0 || throw(ArgumentError("--dr must be positive"))
    args["T-max"] > 0 || throw(ArgumentError("--T-max must be positive"))
    args["ntimes"] > 0 || throw(ArgumentError("--ntimes must be positive"))
    args["array-count"] > 0 || throw(ArgumentError("--array-count must be positive"))
    1 <= array_id <= args["array-count"] ||
        throw(ArgumentError("array id must be in 1:$(args["array-count"])"))
    args["window-log-every"] > 0 ||
        throw(ArgumentError("--window-log-every must be positive"))
    return nothing
end

function main()
    args = parse_args()
    array_id = slurm_array_id(args)
    validate_args!(args, array_id)

    input = equilibrium_path(args, array_id)
    output = output_path(args, array_id)
    equilibrium = load(input, "result")
    equilibrium_config = equilibrium.config
    theta0 = collect(equilibrium.theta)

    L = equilibrium_config.L
    gamma = equilibrium_config.gamma
    J = equilibrium_config.J
    dt = args["dt"]
    dr = args["dr"]
    ntimes = args["ntimes"]
    lag_spacing = Symbol(args["lag-spacing"])
    lag_time = args["T-max"] / ntimes
    lag_steps = steps_for_time(lag_time, dt, "T-max / ntimes")
    schedule = lag_step_schedule(ntimes, lag_steps; spacing=lag_spacing)
    v_values = parse_v_values(args)
    nv = length(v_values)
    solver = select_solver(args["solver"])

    dr <= L / 2 || throw(ArgumentError("--dr must be at most L / 2"))
    args["r-max"] > 0 || throw(ArgumentError("--r-max must be positive"))
    radii = collect(dr:dr:min(args["r-max"], L / 2))
    times = schedule.cum_steps .* dt
    F = zeros(Float64, nv, length(radii), ntimes + 1)
    C_plus = zeros(Float64, nv, length(radii), ntimes + 1)
    C_minus = zeros(Float64, nv, length(radii), ntimes + 1)
    log_radius_index = args["log-radius-index"]
    log_time_index = args["log-time-index"] == 0 ? ntimes + 1 : args["log-time-index"]
    1 <= log_radius_index <= length(radii) ||
        throw(ArgumentError("--log-radius-index is out of range"))
    1 <= log_time_index <= length(times) ||
        throw(ArgumentError("--log-time-index is out of range"))

    loginfo("starting baked-equilibrium v sweep",
        :array_id => array_id,
        :array_count => args["array-count"],
        :equilibrium => input,
        :output => output,
        :L => L,
        :gamma => gamma,
        :J => J,
        :dt => dt,
        :dr => dr,
        :r_max => args["r-max"],
        :lag_spacing => lag_spacing,
        :T_max => args["T-max"],
        :ntimes => ntimes,
        :lag_time => lag_time,
        :lag_steps => lag_steps,
        :nv => nv,
        :v_min => minimum(v_values),
        :v_max => maximum(v_values),
        :solver => string(typeof(solver)))

    for (vidx, v) in enumerate(v_values)
        params = ModelParams(; L, Q=gamma, J, v)
        work = LatticeFlockingSDE.DriftWorkspace(params)
        rng = MersenneTwister(args["base-seed"] + 1000 * (array_id - 1) + vidx - 1)
        theta = copy(theta0)
        window = Vector{Vector{Float64}}(undef, 2ntimes + 1)
        window[1] = copy(theta)

        loginfo("starting v window",
            :array_id => array_id,
            :v_index => vidx,
            :nv => nv,
            :v => v)
        for sample_index in 1:(2ntimes)
            theta = advance(theta, schedule.advance_gaps[sample_index], dt, work, solver, rng)
            window[sample_index + 1] = copy(theta)
            if sample_index == 1 || sample_index % args["window-log-every"] == 0 ||
                    sample_index == 2ntimes
                loginfo("v window integration progress",
                    :array_id => array_id,
                    :v_index => vidx,
                    :nv => nv,
                    :v => v,
                    :sample => sample_index,
                    :total_samples => 2ntimes,
                    :time => sum(@view schedule.advance_gaps[1:sample_index]) * dt,
                    :total_time => 2 * schedule.cum_steps[end] * dt)
            end
        end

        loginfo("computing v correlator",
            :array_id => array_id,
            :v_index => vidx,
            :nv => nv,
            :v => v,
            :nradii => length(radii),
            :ntimes => ntimes)
        correlators = spin_aligned_correlators(window, params, radii)
        F[vidx, :, :] .= correlators.F
        C_plus[vidx, :, :] .= correlators.C_plus
        C_minus[vidx, :, :] .= correlators.C_minus
        loginfo("finished v correlator",
            :array_id => array_id,
            :v_index => vidx,
            :nv => nv,
            :v => v,
            :radius => radii[log_radius_index],
            :lag => times[log_time_index],
            :F => F[vidx, log_radius_index, log_time_index],
            :C_plus => C_plus[vidx, log_radius_index, log_time_index],
            :C_minus => C_minus[vidx, log_radius_index, log_time_index])
    end

    config = (;
        L,
        gamma,
        Q=gamma,
        J,
        dt,
        dr,
        r_max=args["r-max"],
        lag_spacing,
        T_max=args["T-max"],
        ntimes,
        lag_time,
        lag_steps,
        array_id,
        array_count=args["array-count"],
        seed_base=args["base-seed"],
        equilibrium=input,
        equilibrium_config,
        solver=string(typeof(solver)),
        interpolation=:periodic_bilinear_spin_vector,
        log_radius_index,
        log_time_index,
        output,
    )
    result = (; config, v_values, radii, times, F, C_plus, C_minus)

    mkpath(dirname(output))
    loginfo("saving baked-equilibrium v sweep", :array_id => array_id, :output => output)
    jldsave(output; result)
    loginfo("saved baked-equilibrium v sweep", :array_id => array_id, :output => output)
end

main()
