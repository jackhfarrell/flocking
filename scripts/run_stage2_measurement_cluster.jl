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
        description="Run one single-core Stage-2 measurement task: load the per-v baked " *
            "state from the Stage-1 library, accumulate a chi-squared-shaped budget of " *
            "spin-aligned correlator windows (no burn-in).",
    )
    @add_arg_table! settings begin
        "--library-dir"
            arg_type = String
            default = "library/J2_Q1_L200"
        "--direction"
            arg_type = String
            default = "up"
        "--traj"
            arg_type = Int
            default = 1
        "--L"
            arg_type = Int
            default = 200
        "--gamma"
            arg_type = Float64
            default = 1.0
        "--J"
            arg_type = Float64
            default = 2.0
        "--vmin"
            arg_type = Float64
            default = 0.1
        "--vmax"
            arg_type = Float64
            default = 10.0
        "--nv"
            arg_type = Int
            default = 30
        "--window-budget"
            arg_type = String
            default = ""
        "--budget-total"
            arg_type = Int
            default = 600
        "--budget-power"
            arg_type = Float64
            default = 1.0
        "--chunks-per-job"
            arg_type = Int
            default = 20
        "--max-jobs"
            arg_type = Int
            default = 1000
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
        "--array-count"
            arg_type = Int
            default = 0
        "--array-id"
            arg_type = Int
            default = 0
        "--base-seed"
            arg_type = Int
            default = 200000
        "--window-log-every"
            arg_type = Int
            default = 4
        "--chunk-log-every"
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
            default = "results/spin_aligned_f_stage2_L200_J2_Q1"
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

# The library is keyed by v exactly as the Stage-1 ladder writes it.
function checkpoint_path(library_dir, vi::Integer, v::Real, direction::AbstractString,
        traj::Integer)
    key = @sprintf("v_%02d_%.6f", vi, v)
    return joinpath(library_dir, key, @sprintf("%s_traj_%03d.jld2", direction, traj))
end

# chi-squared-shaped budget: the per-v window count is heavier at low v
# (statistics-limited, where the band shrinks with samples) and lighter at high v
# (systematics-limited, where extra windows barely move the band). Weighting by
# v^(-power) lands compute where it actually narrows the collapse band.
function chi2_shaped_budget(v_values, total::Integer, power::Real)
    weights = v_values .^ (-power)
    raw = total .* weights ./ sum(weights)
    return max.(1, round.(Int, raw))
end

function parse_budget(args, v_values)
    if !isempty(args["window-budget"])
        budget = [parse(Int, strip(value)) for value in split(args["window-budget"], ',')]
        length(budget) == length(v_values) ||
            throw(ArgumentError("--window-budget must have one entry per v ($(length(v_values)))"))
        all(>(0), budget) || throw(ArgumentError("--window-budget entries must be positive"))
        return budget
    end
    args["budget-total"] > 0 ||
        throw(ArgumentError("--budget-total must be positive when --window-budget is empty"))
    return chi2_shaped_budget(v_values, args["budget-total"], args["budget-power"])
end

# Flatten the per-v budget into single-core array tasks: each task accumulates up to
# chunks_per_job windows for one v, and a v whose budget exceeds chunks_per_job spills
# into several tasks (the last carrying the remainder). The task table is the fan-out.
function job_plan(budget, chunks_per_job::Integer)
    plan = Tuple{Int,Int}[]
    for (vi, b) in enumerate(budget)
        full, rem = divrem(b, chunks_per_job)
        for _ in 1:full
            push!(plan, (vi, chunks_per_job))
        end
        rem > 0 && push!(plan, (vi, rem))
    end
    return plan
end

function validate_args!(args, direction)
    direction in ("up", "down") || throw(ArgumentError("--direction must be up or down"))
    args["traj"] > 0 || throw(ArgumentError("--traj must be positive"))
    args["L"] > 1 || throw(ArgumentError("--L must be greater than 1"))
    args["gamma"] >= 0 || throw(ArgumentError("--gamma must be nonnegative"))
    args["vmin"] > 0 || throw(ArgumentError("--vmin must be positive"))
    args["vmax"] > args["vmin"] || throw(ArgumentError("--vmax must exceed --vmin"))
    args["nv"] > 0 || throw(ArgumentError("--nv must be positive"))
    args["chunks-per-job"] > 0 || throw(ArgumentError("--chunks-per-job must be positive"))
    args["max-jobs"] > 0 || throw(ArgumentError("--max-jobs must be positive"))
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["dr"] > 0 || throw(ArgumentError("--dr must be positive"))
    args["r-max"] > 0 || throw(ArgumentError("--r-max must be positive"))
    args["T-max"] > 0 || throw(ArgumentError("--T-max must be positive"))
    args["ntimes"] > 0 || throw(ArgumentError("--ntimes must be positive"))
    args["window-log-every"] > 0 ||
        throw(ArgumentError("--window-log-every must be positive"))
    args["chunk-log-every"] > 0 ||
        throw(ArgumentError("--chunk-log-every must be positive"))
    return nothing
end

function main()
    args = parse_args()
    direction = args["direction"]
    validate_args!(args, direction)

    L = args["L"]
    gamma = args["gamma"]
    J = args["J"]
    dt = args["dt"]
    dr = args["dr"]
    ntimes = args["ntimes"]
    chunks_per_job = args["chunks-per-job"]
    lag_spacing = Symbol(args["lag-spacing"])
    lag_time = args["T-max"] / ntimes
    lag_steps = steps_for_time(lag_time, dt, "T-max / ntimes")
    schedule = lag_step_schedule(ntimes, lag_steps; spacing=lag_spacing)
    solver = select_solver(args["solver"])

    vgrid = exp.(range(log(args["vmin"]), log(args["vmax"]); length=args["nv"]))
    budget = parse_budget(args, vgrid)
    plan = job_plan(budget, chunks_per_job)
    njobs = length(plan)
    njobs <= args["max-jobs"] ||
        throw(ArgumentError("plan needs $(njobs) tasks, exceeding --max-jobs " *
            "$(args["max-jobs"]); raise --chunks-per-job or lower the budget"))

    array_count = args["array-count"] == 0 ? njobs : args["array-count"]
    array_id = slurm_array_id(args)
    1 <= array_id <= njobs ||
        throw(ArgumentError("array id $(array_id) is out of range 1:$(njobs) " *
            "(total tasks for this budget)"))

    vi, chunks = plan[array_id]
    v = vgrid[vi]
    input = checkpoint_path(args["library-dir"], vi, v, direction, args["traj"])
    isfile(input) || throw(ArgumentError("no baked state for v at $(input)"))
    baked = load(input, "result")
    theta0 = collect(baked.theta)

    if !isempty(args["output"])
        output = args["output"]
    else
        key = @sprintf("v_%02d_%.6f", vi, v)
        output = joinpath(args["output-dir"], key, "sample_$(@sprintf("%04d", array_id)).jld2")
    end

    params = ModelParams(; L, Q=gamma, J, v)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    seed = args["base-seed"] + array_id - 1
    rng = MersenneTwister(seed)

    dr <= L / 2 || throw(ArgumentError("--dr must be at most L / 2"))
    radii = collect(dr:dr:min(args["r-max"], L / 2))
    times = schedule.cum_steps .* dt
    log_radius_index = args["log-radius-index"]
    log_time_index = args["log-time-index"] == 0 ? ntimes + 1 : args["log-time-index"]
    1 <= log_radius_index <= length(radii) ||
        throw(ArgumentError("--log-radius-index is out of range"))
    1 <= log_time_index <= length(times) ||
        throw(ArgumentError("--log-time-index is out of range"))

    loginfo("starting stage-2 measurement",
        :array_id => array_id,
        :array_count => array_count,
        :njobs => njobs,
        :vi => vi,
        :v => v,
        :budget_v => budget[vi],
        :chunks => chunks,
        :direction => direction,
        :traj => args["traj"],
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
        :solver => string(typeof(solver)))

    F_mean = zeros(Float64, length(radii), ntimes + 1)
    F_m2 = zeros(Float64, length(radii), ntimes + 1)
    F_stderr = zeros(Float64, length(radii), ntimes + 1)
    C_plus_mean = zeros(Float64, length(radii), ntimes + 1)
    C_plus_m2 = zeros(Float64, length(radii), ntimes + 1)
    C_plus_stderr = zeros(Float64, length(radii), ntimes + 1)
    C_minus_mean = zeros(Float64, length(radii), ntimes + 1)
    C_minus_m2 = zeros(Float64, length(radii), ntimes + 1)
    C_minus_stderr = zeros(Float64, length(radii), ntimes + 1)

    # No burn-in: the library state is already equilibrated for this v. Windows are
    # consecutive segments of one trajectory seeded directly from the baked field.
    theta = copy(theta0)
    for chunk in 1:chunks
        window = Vector{Vector{Float64}}(undef, 2ntimes + 1)
        window[1] = copy(theta)

        for sample_index in 1:(2ntimes)
            theta = advance(theta, schedule.advance_gaps[sample_index], dt, work, solver, rng)
            window[sample_index + 1] = copy(theta)
            if sample_index == 1 || sample_index % args["window-log-every"] == 0 ||
                    sample_index == 2ntimes
                elapsed_steps = sum(@view schedule.advance_gaps[1:sample_index])
                loginfo("window integration progress",
                    :array_id => array_id,
                    :chunk => chunk,
                    :chunks => chunks,
                    :sample => sample_index,
                    :total_samples => 2ntimes,
                    :time => elapsed_steps * dt,
                    :total_time => 2 * schedule.cum_steps[end] * dt)
            end
        end

        correlators = spin_aligned_correlators(window, params, radii)
        F_stderr = online_mean_stderr!(F_mean, F_m2, correlators.F, chunk)
        C_plus_stderr = online_mean_stderr!(C_plus_mean, C_plus_m2,
            correlators.C_plus, chunk)
        C_minus_stderr = online_mean_stderr!(C_minus_mean, C_minus_m2,
            correlators.C_minus, chunk)
        if chunk == 1 || chunk % args["chunk-log-every"] == 0 || chunk == chunks
            loginfo("rolling stage-2 correlators",
                :array_id => array_id,
                :chunk => chunk,
                :chunks => chunks,
                :v => v,
                :radius => radii[log_radius_index],
                :lag => times[log_time_index],
                :F => F_mean[log_radius_index, log_time_index],
                :stderr_F => F_stderr[log_radius_index, log_time_index])
        end
    end

    # Stack the single measured v on the leading axis so the result schema matches the
    # baked-equilibrium v-sweep output the collapse analysis already consumes.
    v_values = [v]
    F = reshape(F_mean, 1, length(radii), ntimes + 1)
    C_plus = reshape(C_plus_mean, 1, length(radii), ntimes + 1)
    C_minus = reshape(C_minus_mean, 1, length(radii), ntimes + 1)

    config = (;
        L,
        gamma,
        Q=gamma,
        J,
        v,
        vi,
        dt,
        dr,
        r_max=args["r-max"],
        lag_spacing,
        T_max=args["T-max"],
        ntimes,
        lag_time,
        lag_steps,
        direction,
        traj=args["traj"],
        budget_v=budget[vi],
        chunks,
        nwindows=chunks,
        chunks_per_job,
        array_id,
        array_count,
        njobs,
        seed,
        base_seed=args["base-seed"],
        equilibrium=input,
        equilibrium_config=baked.config,
        solver=string(typeof(solver)),
        interpolation=:periodic_bilinear_spin_vector,
        log_radius_index,
        log_time_index,
        output,
    )
    result = (; config, v_values, radii, times, F, C_plus, C_minus,
        F_stderr=reshape(F_stderr, 1, length(radii), ntimes + 1),
        C_plus_stderr=reshape(C_plus_stderr, 1, length(radii), ntimes + 1),
        C_minus_stderr=reshape(C_minus_stderr, 1, length(radii), ntimes + 1))

    mkpath(dirname(output))
    loginfo("saving stage-2 measurement", :array_id => array_id, :output => output)
    jldsave(output; result)
    loginfo("saved stage-2 measurement", :array_id => array_id, :output => output)
end

main()
