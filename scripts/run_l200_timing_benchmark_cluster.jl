#!/usr/bin/env julia

# L=200 timing benchmark (issue 07). Measures the real production cost of one Stage-2
# measurement window at the three anchor velocities under each candidate (solver, dt)
# pair, on the production code path: capped radii, geometric lag spacing, SRA solver.
#
# The window (integration of 2*ntimes geometric-lag advances, then one spin-aligned
# correlator evaluation) is the Stage-2 primitive. Its integration cost also sizes
# Stage-1 equilibration, which is pure integration. So integration and correlator are
# timed separately, and a per-step integration time is reported for Stage-1 sizing.
#
# Timing is field-independent: the SDE step count is fixed by (dt, T_max, ntimes) and
# the drift/correlator do no data-dependent work, so the ordered initial state is only
# a stand-in and its magnetization does not affect the measured wall-time.
#
# Runs as ONE sequential single-core job so every combo is timed on the same node under
# identical conditions (array tasks can land on heterogeneous nodes, muddying the
# comparison). One job is trivially within the <= 1000-job cap; the 24h wall bounds the
# combo count, which the operator sizes.

using ArgParse
using Dates
using JLD2
using Printf
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(
        description="Time one Stage-2 measurement window per (v, solver, dt) on L=200 " *
            "production code, producing a budget table to size Stage-1 and Stage-2.",
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
        "--velocities"
            arg_type = String
            default = "0.1,1.0,10.0"
        "--solvers"
            arg_type = String
            default = "SRA1,SRA2"
        "--dts"
            arg_type = String
            default = "0.001,0.0005"
        "--dr"
            arg_type = Float64
            default = 0.25
        "--r-max"
            arg_type = Float64
            default = 45.0
        "--lag-spacing"
            arg_type = String
            default = "geometric"
        "--T-max"
            arg_type = Float64
            default = 16.0
        "--ntimes"
            arg_type = Int
            default = 8
        "--reps"
            arg_type = Int
            default = 1
        "--base-seed"
            arg_type = Int
            default = 200000
        "--output-dir"
            arg_type = String
            default = "results/l200_timing_benchmark"
        "--output"
            arg_type = String
            default = ""
        "--csv"
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

# SRA1/SRA2/SRA3 are purpose-built for the model's additive, state-independent noise;
# SRIW1/EM remain selectable for cross-checks.
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

# Build one measurement window: the midpoint state plus 2*ntimes advances along the
# symmetric geometric lag schedule. Identical construction to the Stage-2 script.
function integrate_window(theta0, schedule, dt::Real, work, solver, rng)
    nadvances = length(schedule.advance_gaps)
    window = Vector{Vector{Float64}}(undef, nadvances + 1)
    theta = copy(theta0)
    window[1] = copy(theta)
    for i in 1:nadvances
        theta = advance(theta, schedule.advance_gaps[i], dt, work, solver, rng)
        window[i + 1] = copy(theta)
    end
    return window
end

function interpolated_spin_dot(cos_field, sin_field, L::Integer, px::Real, py::Real,
        ref_cos::Real, ref_sin::Real)
    x0 = floor(Int, px)
    y0 = floor(Int, py)
    fx = px - x0
    fy = py - y0
    ix0 = mod1(x0, L)
    ix1 = mod1(x0 + 1, L)
    iy0 = mod1(y0, L)
    iy1 = mod1(y0 + 1, L)

    c = 0.0
    s = 0.0
    @inbounds for (ix, wx) in ((ix0, 1 - fx), (ix1, fx))
        for (iy, wy) in ((iy0, 1 - fy), (iy1, fy))
            w = wx * wy
            idx = site_index(ix, iy, L)
            c += w * cos_field[idx]
            s += w * sin_field[idx]
        end
    end
    return ref_cos * c + ref_sin * s
end

# The full spin-aligned correlator (F, C_plus, C_minus) as accumulated in Stage-2; the
# timing must reflect this whole kernel, not F alone.
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

# The benchmark plan is the full (velocity, solver, dt) cross product, ordered v-major
# then solver then dt. Each entry is one timed window. Kept pure so the fan-out is a
# plain table the operator can read against the cluster limits.
function benchmark_plan(velocities, solvers, dts)
    plan = NamedTuple{(:v, :solver, :dt),Tuple{Float64,String,Float64}}[]
    for v in velocities, solver in solvers, dt in dts
        push!(plan, (; v, solver, dt))
    end
    return plan
end

function main()
    args = parse_args()
    L = args["L"]
    gamma = args["gamma"]
    J = args["J"]
    dr = args["dr"]
    ntimes = args["ntimes"]
    T_max = args["T-max"]
    reps = args["reps"]
    lag_spacing = Symbol(args["lag-spacing"])

    L > 1 || throw(ArgumentError("--L must be greater than 1"))
    dr > 0 || throw(ArgumentError("--dr must be positive"))
    dr <= L / 2 || throw(ArgumentError("--dr must be at most L / 2"))
    ntimes > 0 || throw(ArgumentError("--ntimes must be positive"))
    T_max > 0 || throw(ArgumentError("--T-max must be positive"))
    reps > 0 || throw(ArgumentError("--reps must be positive"))

    velocities = parse.(Float64, split(args["velocities"], ","))
    solvers = String.(strip.(split(args["solvers"], ",")))
    dts = parse.(Float64, split(args["dts"], ","))
    all(>(0), velocities) || throw(ArgumentError("velocities must be positive"))
    all(>(0), dts) || throw(ArgumentError("dts must be positive"))
    foreach(select_solver, solvers)  # fail fast on an unknown solver name

    radii = collect(dr:dr:min(args["r-max"], L / 2))
    plan = benchmark_plan(velocities, solvers, dts)

    output = isempty(args["output"]) ? joinpath(args["output-dir"], "benchmark.jld2") :
        args["output"]
    csv = isempty(args["csv"]) ? joinpath(args["output-dir"], "benchmark.csv") : args["csv"]

    loginfo("starting L=$(L) timing benchmark",
        :L => L, :gamma => gamma, :J => J,
        :velocities => velocities, :solvers => solvers, :dts => dts,
        :dr => dr, :r_max => args["r-max"], :n_radii => length(radii),
        :lag_spacing => lag_spacing, :T_max => T_max, :ntimes => ntimes,
        :reps => reps, :combos => length(plan), :output => output, :csv => csv)

    @printf(stderr, "# L=%d  gamma=%.3g  J=%.3g  dr=%.3g  r_max=%.3g  n_radii=%d  T_max=%.3g  ntimes=%d  reps=%d\n",
        L, gamma, J, dr, args["r-max"], length(radii), T_max, ntimes, reps)
    @printf(stderr, "# %-8s %-6s %-10s %-11s %-12s %-13s %-11s %-16s\n",
        "v", "solver", "dt", "steps/win", "integrate_s", "correlator_s", "window_s",
        "integrate_us/step")

    # Warm up each solver's compiled integration path and the correlator kernel once, on
    # a tiny 2-step schedule at the real L, so the timed windows carry no JIT cost.
    warm_schedule = lag_step_schedule(ntimes, 2; spacing=lag_spacing)
    warm_params = ModelParams(; L, Q=gamma, J, v=first(velocities))
    warm_work = LatticeFlockingSDE.DriftWorkspace(warm_params)
    warm_theta = initial_angles(MersenneTwister(args["base-seed"]), L, :ordered)
    for name in unique(solvers)
        solver = select_solver(name)
        window = integrate_window(warm_theta, warm_schedule, 0.001, warm_work, solver,
            MersenneTwister(args["base-seed"]))
        spin_aligned_correlators(window, warm_params, radii[1:1])
    end

    rows = NamedTuple[]
    for (i, combo) in enumerate(plan)
        v = combo.v
        dt = combo.dt
        solver = select_solver(combo.solver)
        lag_steps = steps_for_time(T_max / ntimes, dt, "T-max / ntimes")
        schedule = lag_step_schedule(ntimes, lag_steps; spacing=lag_spacing)
        window_steps = sum(schedule.advance_gaps)

        params = ModelParams(; L, Q=gamma, J, v)
        work = LatticeFlockingSDE.DriftWorkspace(params)
        theta0 = initial_angles(MersenneTwister(args["base-seed"] + i), L, :ordered)

        # Report the min over reps: the fastest run is the one least contaminated by
        # scheduler/other-process interference, so it best isolates the compute cost.
        t_integrate = Inf
        t_correlator = Inf
        for _ in 1:reps
            local window
            t_integrate = min(t_integrate,
                @elapsed window = integrate_window(theta0, schedule, dt, work, solver,
                    MersenneTwister(args["base-seed"] + i)))
            t_correlator = min(t_correlator,
                @elapsed spin_aligned_correlators(window, params, radii))
        end
        t_window = t_integrate + t_correlator
        integrate_us_per_step = 1e6 * t_integrate / window_steps

        row = (; v, solver=combo.solver, dt, window_steps, t_integrate, t_correlator,
            t_window, integrate_us_per_step)
        push!(rows, row)

        @printf(stderr, "  %-8.3g %-6s %-10.3g %-11d %-12.3f %-13.3f %-11.3f %-16.4f\n",
            v, combo.solver, dt, window_steps, t_integrate, t_correlator, t_window,
            integrate_us_per_step)
        loginfo("timed window",
            :combo => i, :combos => length(plan),
            :v => v, :solver => combo.solver, :dt => dt,
            :window_steps => window_steps, :t_window => t_window)
    end

    config = (;
        L, gamma, Q=gamma, J, dr, r_max=args["r-max"], n_radii=length(radii),
        lag_spacing, T_max, ntimes, reps, velocities, solvers, dts,
        base_seed=args["base-seed"], init=:ordered)
    result = (; config, radii, rows)

    mkpath(dirname(output))
    jldsave(output; result)

    mkpath(dirname(csv))
    open(csv, "w") do io
        println(io, "v,solver,dt,window_steps,t_integrate_s,t_correlator_s,t_window_s,integrate_us_per_step")
        for row in rows
            @printf(io, "%.6g,%s,%.6g,%d,%.6f,%.6f,%.6f,%.6f\n",
                row.v, row.solver, row.dt, row.window_steps, row.t_integrate,
                row.t_correlator, row.t_window, row.integrate_us_per_step)
        end
    end

    loginfo("saved timing benchmark", :output => output, :csv => csv, :rows => length(rows))
end

main()
