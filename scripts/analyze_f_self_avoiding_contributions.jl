#!/usr/bin/env julia

using ArgParse
using CairoMakie
using DifferentialEquations
using JLD2
using Random
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function steps_for_time(time::Real, dt::Real, name::String)
    steps = round(Int, time / dt)
    isapprox(steps * dt, time; atol=100eps(max(abs(time), abs(dt))), rtol=0) ||
        throw(ArgumentError("$name must be an integer multiple of dt"))
    return steps
end

function advance(theta, steps::Integer, dt::Real, work, solver, rng)
    problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta,
        (0.0, steps * dt), work)
    solution = solve(problem, solver; dt, adaptive=false, save_everystep=false,
        save_start=false, rng)
    return wrap_angles!(collect(solution.u[end]))
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
            weight = wx * wy
            idx = site_index(ix, iy, L)
            c += weight * cos_field[idx]
            s += weight * sin_field[idx]
        end
    end
    return ref_cos * c + ref_sin * s
end

function angle_at(theta_grid, x, y)
    L = size(theta_grid, 1)
    xw = mod(x - 1, L) + 1
    yw = mod(y - 1, L) + 1
    x0 = floor(Int, xw)
    y0 = floor(Int, yw)
    fx = xw - x0
    fy = yw - y0
    x1 = mod1(x0 + 1, L)
    y1 = mod1(y0 + 1, L)
    x0 = mod1(x0, L)
    y0 = mod1(y0, L)

    c = (1 - fx) * (1 - fy) * cos(theta_grid[x0, y0]) +
        fx * (1 - fy) * cos(theta_grid[x1, y0]) +
        (1 - fx) * fy * cos(theta_grid[x0, y1]) +
        fx * fy * cos(theta_grid[x1, y1])
    s = (1 - fx) * (1 - fy) * sin(theta_grid[x0, y0]) +
        fx * (1 - fy) * sin(theta_grid[x1, y0]) +
        (1 - fx) * fy * sin(theta_grid[x0, y1]) +
        fx * fy * sin(theta_grid[x1, y1])
    return atan(s, c)
end

function rk4_step(theta_grid, x, y, ds, direction_sign)
    θ1 = angle_at(theta_grid, x, y)
    k1x, k1y = cos(θ1), sin(θ1)
    θ2 = angle_at(theta_grid, x + 0.5ds * direction_sign * k1x,
        y + 0.5ds * direction_sign * k1y)
    k2x, k2y = cos(θ2), sin(θ2)
    θ3 = angle_at(theta_grid, x + 0.5ds * direction_sign * k2x,
        y + 0.5ds * direction_sign * k2y)
    k3x, k3y = cos(θ3), sin(θ3)
    θ4 = angle_at(theta_grid, x + ds * direction_sign * k3x,
        y + ds * direction_sign * k3y)
    k4x, k4y = cos(θ4), sin(θ4)
    return (
        x + direction_sign * ds * (k1x + 2k2x + 2k3x + k4x) / 6,
        y + direction_sign * ds * (k1y + 2k2y + 2k3y + k4y) / 6,
    )
end

function arm_stats(theta_grid, x, y, arclength, ds, direction_sign, box_size)
    nsteps = round(Int, arclength / ds)
    stride = max(1, round(Int, box_size / ds))
    boxes = Set{Tuple{Int, Int}}()
    repeats = 0
    sampled = 0
    x0 = x
    y0 = y

    for step in 0:nsteps
        if step % stride == 0
            sampled += 1
            box = (floor(Int, x / box_size), floor(Int, y / box_size))
            if box in boxes
                repeats += 1
            else
                push!(boxes, box)
            end
        end
        step == nsteps && break
        x, y = rk4_step(theta_grid, x, y, ds, direction_sign)
    end

    endpoint_distance = sqrt((x - x0)^2 + (y - y0)^2)
    recurrence_fraction = repeats / sampled
    return (; recurrence_fraction, endpoint_distance)
end

function update_f_class_sums!(sums, window, params, radii, ds, box_size, max_recurrence,
        min_endpoint_ratio)
    L = params.L
    nsites = L * L
    theta_minus, theta_mid, theta_plus = window
    cos_minus = cos.(theta_minus)
    sin_minus = sin.(theta_minus)
    cos_mid = cos.(theta_mid)
    sin_mid = sin.(theta_mid)
    cos_plus = cos.(theta_plus)
    sin_plus = sin.(theta_plus)
    theta_grid = reshape(theta_mid, L, L)

    @inbounds for (ridx, r) in enumerate(radii)
        min_endpoint = min_endpoint_ratio * sqrt(r)
        for y in 1:L, x in 1:L
            center = site_index(x, y, L)
            cx = cos_mid[center]
            sy = sin_mid[center]
            dx = r * cx
            dy = r * sy

            forward_plus = interpolated_spin_dot(cos_plus, sin_plus, L,
                x + dx, y + dy, cx, sy)
            forward_minus = interpolated_spin_dot(cos_minus, sin_minus, L,
                x + dx, y + dy, cx, sy)
            backward_minus = interpolated_spin_dot(cos_minus, sin_minus, L,
                x - dx, y - dy, cx, sy)
            backward_plus = interpolated_spin_dot(cos_plus, sin_plus, L,
                x - dx, y - dy, cx, sy)

            forward_f = 0.25 * (forward_plus - forward_minus) / nsites
            backward_f = 0.25 * (backward_minus - backward_plus) / nsites

            forward_stats = arm_stats(theta_grid, x, y, r, ds, 1.0, box_size)
            backward_stats = arm_stats(theta_grid, x, y, r, ds, -1.0, box_size)
            forward_selected = forward_stats.recurrence_fraction <= max_recurrence &&
                forward_stats.endpoint_distance >= min_endpoint
            backward_selected = backward_stats.recurrence_fraction <= max_recurrence &&
                backward_stats.endpoint_distance >= min_endpoint

            sums.total_signed[ridx] += forward_f + backward_f
            sums.total_abs[ridx] += abs(forward_f) + abs(backward_f)
            sums.total_count[ridx] += 2
            if forward_selected
                sums.selected_signed[ridx] += forward_f
                sums.selected_abs[ridx] += abs(forward_f)
                sums.selected_count[ridx] += 1
            end
            if backward_selected
                sums.selected_signed[ridx] += backward_f
                sums.selected_abs[ridx] += abs(backward_f)
                sums.selected_count[ridx] += 1
            end
        end
    end
    return sums
end

settings = ArgParseSettings(
    description="Measure whether self-avoiding active streamline arms dominate spin-aligned time-antisymmetric F contributions.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 48
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
    "--lag-time"
        arg_type = Float64
        default = 0.5
    "--burnin-time"
        arg_type = Float64
        default = 10.0
    "--nwindows"
        arg_type = Int
        default = 8
    "--seed"
        arg_type = Int
        default = 1
    "--dr"
        arg_type = Float64
        default = 4.0
    "--rmax"
        arg_type = Float64
        default = 24.0
    "--streamline-ds"
        arg_type = Float64
        default = 0.5
    "--avoid-box-size"
        arg_type = Float64
        default = 2.0
    "--max-recurrence"
        arg_type = Float64
        default = 0.0
    "--min-endpoint-ratio"
        arg_type = Float64
        default = 1.0
    "--output-prefix"
        arg_type = String
        default = "f_self_avoiding_contributions"
end
args = ArgParse.parse_args(settings)

params = ModelParams(; L=args["L"], Q=args["gamma"], J=args["J"], v=args["v"])
dt = args["dt"]
lag_steps = steps_for_time(args["lag-time"], dt, "lag-time")
burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
radii = collect(args["dr"]:args["dr"]:args["rmax"])

rng = MersenneTwister(args["seed"])
theta = initial_angles(rng, params.L, :ordered)
work = LatticeFlockingSDE.DriftWorkspace(params)
solver = SRIW1()

if burnin_steps > 0
    theta = advance(theta, burnin_steps, dt, work, solver, rng)
end

sums = (;
    total_signed=zeros(Float64, length(radii)),
    selected_signed=zeros(Float64, length(radii)),
    total_abs=zeros(Float64, length(radii)),
    selected_abs=zeros(Float64, length(radii)),
    total_count=zeros(Int, length(radii)),
    selected_count=zeros(Int, length(radii)),
)

for window_index in 1:args["nwindows"]
    theta_minus = copy(theta)
    theta_mid = advance(theta_minus, lag_steps, dt, work, solver, rng)
    theta_plus = advance(theta_mid, lag_steps, dt, work, solver, rng)
    update_f_class_sums!(sums, (theta_minus, theta_mid, theta_plus), params, radii,
        args["streamline-ds"], args["avoid-box-size"], args["max-recurrence"],
        args["min-endpoint-ratio"])
    global theta = theta_plus
    @info "processed F window" window_index total=args["nwindows"]
end

signed_fraction = sums.selected_signed ./ sums.total_signed
abs_fraction = sums.selected_abs ./ sums.total_abs
count_fraction = sums.selected_count ./ sums.total_count
F_total = sums.total_signed
F_selected = sums.selected_signed

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(980, 460), backgroundcolor=:white)
ax1 = Axis(fig[1, 1], xlabel="r", ylabel="fraction",
    title="self-avoiding share of spin-aligned F")
scatterlines!(ax1, radii, abs_fraction; color=:black, label="absolute F weight")
scatterlines!(ax1, radii, count_fraction; color=:gray40, label="arm count")
hlines!(ax1, [0.5]; color=:gray70, linestyle=:dash)
ylims!(ax1, 0, 1)
axislegend(ax1; position=:rb)

ax2 = Axis(fig[1, 2], xlabel="r", ylabel="signed F",
    title="signed decomposition")
scatterlines!(ax2, radii, F_total; color=:black, label="total")
scatterlines!(ax2, radii, F_selected; color=:red, label="selected")
axislegend(ax2; position=:rb)

save(figure_output, fig)
jldsave(data_output; params, radii, sums, signed_fraction, abs_fraction, count_fraction,
    F_total, F_selected, args)

open(summary_output, "w") do io
    println(io, "# Self-avoiding contribution to spin-aligned F")
    println(io)
    println(io, "- Observable: spin-aligned `F(r,t) = (C(r,t) - C(r,-t))/2` contribution split by midpoint active-streamline arm class.")
    println(io, "- Parameters: `L=$(params.L)`, `gamma=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`, `dt=$(dt)`, `lag_time=$(args["lag-time"])`")
    println(io, "- Windows: `$(args["nwindows"])`, burn-in `$(args["burnin-time"])`")
    println(io, "- Self-avoiding arm rule: recurrence <= `$(args["max-recurrence"])` using box size `$(args["avoid-box-size"])`; endpoint distance >= `$(args["min-endpoint-ratio"]) * sqrt(r)`")
    println(io)
    println(io, "| r | selected arms | abs F fraction | signed selected / total | total F | selected F |")
    println(io, "|---:|---:|---:|---:|---:|---:|")
    for i in eachindex(radii)
        println(io, "| $(radii[i]) | $(round(count_fraction[i]; digits=4)) | $(round(abs_fraction[i]; digits=4)) | $(round(signed_fraction[i]; digits=4)) | $(round(F_total[i]; sigdigits=5)) | $(round(F_selected[i]; sigdigits=5)) |")
    end
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("mean selected arm count fraction: ", mean(count_fraction))
println("mean selected absolute F fraction: ", mean(abs_fraction))
