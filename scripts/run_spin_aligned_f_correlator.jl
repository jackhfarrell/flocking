#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(
        description="Run a spin-aligned continuous-position F-correlator calculation.",
    )
    @add_arg_table! settings begin
        "--L"
            arg_type = Int
            default = 32
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
        "--dr"
            arg_type = Float64
            default = 0.25
        "--burnin-time"
            arg_type = Float64
            default = 20.0
        "--T-max"
            arg_type = Float64
            default = 16.0
        "--ntimes"
            arg_type = Int
            default = 8
        "--nwindows"
            arg_type = Int
            default = 500
        "--seed"
            arg_type = Int
            default = 1
        "--burnin-log-time"
            arg_type = Float64
            default = 2.0
        "--window-log-every"
            arg_type = Int
            default = 1
        "--log-radius-index"
            arg_type = Int
            default = 1
        "--log-time-index"
            arg_type = Int
            default = 0
        "--output"
            arg_type = String
            default = "results/spin_aligned_f_correlator.jld2"
        "--figure"
            arg_type = String
            default = "figures/spin_aligned_f_correlator.png"
    end
    return ArgParse.parse_args(settings)
end

function steps_for_time(time::Real, dt::Real, name::String)
    steps = round(Int, time / dt)
    isapprox(steps * dt, time; atol=100eps(max(abs(time), abs(dt))), rtol=0) ||
        throw(ArgumentError("$name must be an integer multiple of dt"))
    return steps
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

                forward_plus = interpolated_spin_dot(cos_plus, sin_plus, L, x + dx, y + dy, cx, sy)
                backward_minus = interpolated_spin_dot(cos_minus, sin_minus, L, x - dx, y - dy, cx, sy)
                forward_minus = interpolated_spin_dot(cos_minus, sin_minus, L, x + dx, y + dy, cx, sy)
                backward_plus = interpolated_spin_dot(cos_plus, sin_plus, L, x - dx, y - dy, cx, sy)
                accum += 0.25 * (forward_plus + backward_minus - forward_minus - backward_plus)
            end

            F[ridx, lag + 1] = accum / nsites
        end
    end

    return F
end

function spin_aligned_correlators(window::AbstractVector, params::ModelParams,
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

function main()
    args = parse_args()
    args["dt"] > 0 || throw(ArgumentError("--dt must be positive"))
    args["dr"] > 0 || throw(ArgumentError("--dr must be positive"))
    args["burnin-time"] >= 0 || throw(ArgumentError("--burnin-time must be nonnegative"))
    args["T-max"] > 0 || throw(ArgumentError("--T-max must be positive"))
    args["ntimes"] > 0 || throw(ArgumentError("--ntimes must be positive"))
    args["nwindows"] > 0 || throw(ArgumentError("--nwindows must be positive"))
    args["burnin-log-time"] > 0 ||
        throw(ArgumentError("--burnin-log-time must be positive"))
    args["window-log-every"] > 0 ||
        throw(ArgumentError("--window-log-every must be positive"))

    L = args["L"]
    gamma = args["gamma"]
    dt = args["dt"]
    dr = args["dr"]
    ntimes = args["ntimes"]
    nwindows = args["nwindows"]
    lag_time = args["T-max"] / ntimes
    lag_steps = steps_for_time(lag_time, dt, "T-max / ntimes")
    burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
    burnin_log_steps = steps_for_time(args["burnin-log-time"], dt, "burnin-log-time")

    params = ModelParams(; L, Q=gamma, J=args["J"], v=args["v"])
    rng = MersenneTwister(args["seed"])
    theta = initial_angles(rng, L, :ordered)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    solver = SRIW1()

    dr <= L / 2 || throw(ArgumentError("--dr must be at most L / 2"))
    radii = collect(dr:dr:(L / 2))
    times = collect(0:ntimes) .* lag_steps .* dt
    log_radius_index = args["log-radius-index"]
    log_time_index = args["log-time-index"] == 0 ? ntimes + 1 : args["log-time-index"]
    1 <= log_radius_index <= length(radii) ||
        throw(ArgumentError("--log-radius-index is out of range"))
    1 <= log_time_index <= length(times) ||
        throw(ArgumentError("--log-time-index is out of range"))

    @info "starting spin-aligned F correlator" L gamma J=params.J v=params.v dt dr burnin_time=args["burnin-time"] burnin_steps T_max=args["T-max"] ntimes lag_time lag_steps nwindows seed=args["seed"] solver=string(typeof(solver))

    completed_burnin = 0
    while completed_burnin < burnin_steps
        segment_steps = min(burnin_log_steps, burnin_steps - completed_burnin)
        theta = advance(theta, segment_steps, dt, work, solver, rng)
        completed_burnin += segment_steps
        @info "burn-in progress" steps=completed_burnin total_steps=burnin_steps time=completed_burnin * dt total_time=args["burnin-time"]
    end

    F_mean = zeros(Float64, length(radii), ntimes + 1)
    F_m2 = zeros(Float64, length(radii), ntimes + 1)
    F_stderr = zeros(Float64, length(radii), ntimes + 1)
    C_plus_mean = zeros(Float64, length(radii), ntimes + 1)
    C_plus_m2 = zeros(Float64, length(radii), ntimes + 1)
    C_plus_stderr = zeros(Float64, length(radii), ntimes + 1)
    C_minus_mean = zeros(Float64, length(radii), ntimes + 1)
    C_minus_m2 = zeros(Float64, length(radii), ntimes + 1)
    C_minus_stderr = zeros(Float64, length(radii), ntimes + 1)

    for window_index in 1:nwindows
        window = Vector{Vector{Float64}}(undef, 2ntimes + 1)
        window[1] = copy(theta)

        for sample_index in 1:(2ntimes)
            theta = advance(theta, lag_steps, dt, work, solver, rng)
            window[sample_index + 1] = copy(theta)
            if sample_index == 1 || sample_index % args["window-log-every"] == 0 ||
                    sample_index == 2ntimes
                @info "window integration progress" window=window_index nwindows sample=sample_index total_samples=2ntimes steps=sample_index * lag_steps total_steps=2ntimes * lag_steps time=sample_index * lag_time total_time=2 * args["T-max"]
            end
        end

        correlators = spin_aligned_correlators(window, params, radii)
        F_stderr = online_mean_stderr!(F_mean, F_m2, correlators.F, window_index)
        C_plus_stderr = online_mean_stderr!(C_plus_mean, C_plus_m2,
            correlators.C_plus, window_index)
        C_minus_stderr = online_mean_stderr!(C_minus_mean, C_minus_m2,
            correlators.C_minus, window_index)
        @info "rolling spin-aligned correlators" window=window_index nwindows radius=radii[log_radius_index] lag=times[log_time_index] F=F_mean[log_radius_index, log_time_index] C_plus=C_plus_mean[log_radius_index, log_time_index] C_minus=C_minus_mean[log_radius_index, log_time_index] stderr_F=F_stderr[log_radius_index, log_time_index] stderr_C_plus=C_plus_stderr[log_radius_index, log_time_index] stderr_C_minus=C_minus_stderr[log_radius_index, log_time_index]
    end

    config = (;
        L=params.L,
        gamma,
        Q=gamma,
        J=params.J,
        v=params.v,
        dt,
        dr,
        burnin_time=args["burnin-time"],
        burnin_steps,
        T_max=args["T-max"],
        ntimes,
        lag_time,
        lag_steps,
        nwindows,
        seed=args["seed"],
        initial_condition=:ordered,
        solver=string(typeof(solver)),
        interpolation=:periodic_bilinear_spin_vector,
        log_radius_index,
        log_time_index,
    )
    result = (; config, radii, times, F_mean, F_stderr, C_plus_mean, C_plus_stderr,
        C_minus_mean, C_minus_stderr)

    mkpath(dirname(args["output"]))
    jldsave(args["output"]; result)

    mkpath(dirname(args["figure"]))
    fig = Figure(size=(900, 620))
    ax = Axis(fig[1, 1], xlabel="r", ylabel="F(r, t)",
        title="Spin-aligned F correlator")
    palette = [:blue, :orange, :green, :red, :purple, :brown, :black, :gray]
    for i in 1:ntimes
        color = palette[mod1(i, length(palette))]
        y = F_mean[:, i + 1]
        err = F_stderr[:, i + 1]
        band!(ax, radii, y .- err, y .+ err; color=(color, 0.18))
        lines!(ax, radii, y; color, linewidth=2, label="t=$(round(times[i + 1]; digits=4))")
    end
    axislegend(ax, position=:rb)
    save(args["figure"], fig)

    @info "saved spin-aligned F correlator" output=args["output"] figure=args["figure"]
end

main()
