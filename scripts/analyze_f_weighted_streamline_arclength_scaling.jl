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

function parse_solver(name::AbstractString)
    solver_name = lowercase(name)
    solver_name in ("auto", "em", "sriw1") ||
        throw(ArgumentError("solver must be auto, em, or sriw1"))
    return solver_name
end

function logspace_values(lo, hi, n)
    return exp10.(range(log10(lo), log10(hi); length=n))
end

function linear_fit(x, y)
    mx = mean(x)
    my = mean(y)
    slope = sum((x .- mx) .* (y .- my)) / sum((x .- mx).^2)
    intercept = my - slope * mx
    return slope, intercept
end

function weighted_mean_columns(values, weights)
    total = sum(weights)
    total > 0 || return vec(mean(values; dims=2))
    return vec(values * (weights ./ total))
end

function evolve_state(theta0, duration, params, solver_name, dt, rng)
    iszero(duration) && return wrap_angles!(copy(theta0))

    work = LatticeFlockingSDE.DriftWorkspace(params)
    if iszero(params.Q)
        problem = ODEProblem(LatticeFlockingSDE.drift!, theta0, (0.0, duration), work)
        solution = solve(problem, Tsit5(); save_everystep=false, save_start=false)
    else
        problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta0,
            (0.0, duration), work)
        algorithm = solver_name == "em" ? EM() : SRIW1()
        solution = solve(problem, algorithm; dt, adaptive=false, save_everystep=false,
            save_start=false, rng)
    end
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

function trace_arm(theta_grid, seed_x, seed_y, max_length, ds, direction_sign)
    nsteps = round(Int, max_length / ds)
    path = zeros(Float64, nsteps + 1, 2)
    path[1, :] .= (seed_x, seed_y)
    x = seed_x
    y = seed_y
    for step in 1:nsteps
        x, y = rk4_step(theta_grid, x, y, ds, direction_sign)
        path[step + 1, :] .= (x, y)
    end
    return path
end

function path_r2(path, lag_steps)
    values = zeros(Float64, length(lag_steps))
    @inbounds for (j, lag) in enumerate(lag_steps)
        acc = 0.0
        count = 0
        for i in 1:(size(path, 1) - lag)
            dx = path[i + lag, 1] - path[i, 1]
            dy = path[i + lag, 2] - path[i, 2]
            acc += dx^2 + dy^2
            count += 1
        end
        values[j] = acc / count
    end
    return values
end

function recurrence_fraction(path, lag, box_size, ds)
    stride = max(1, round(Int, box_size / ds))
    sampled_lag = max(1, round(Int, lag / stride))
    repeats = 0
    sampled = 0
    @inbounds for start in 1:stride:(size(path, 1) - lag)
        boxes = Set{Tuple{Int, Int}}()
        for i in start:stride:(start + lag)
            push!(boxes, (floor(Int, path[i, 1] / box_size), floor(Int, path[i, 2] / box_size)))
        end
        repeats += sampled_lag + 1 - length(boxes)
        sampled += sampled_lag + 1
    end
    return repeats / sampled
end

function arm_f_values(theta_minus, theta_mid, theta_plus, params, x, y, r)
    L = params.L
    nsites = L * L
    center = site_index(round(Int, x), round(Int, y), L)
    cx = cos(theta_mid[center])
    sy = sin(theta_mid[center])
    dx = r * cx
    dy = r * sy
    cos_minus = cos.(theta_minus)
    sin_minus = sin.(theta_minus)
    cos_plus = cos.(theta_plus)
    sin_plus = sin.(theta_plus)

    forward_plus = interpolated_spin_dot(cos_plus, sin_plus, L, x + dx, y + dy, cx, sy)
    forward_minus = interpolated_spin_dot(cos_minus, sin_minus, L, x + dx, y + dy, cx, sy)
    backward_minus = interpolated_spin_dot(cos_minus, sin_minus, L, x - dx, y - dy, cx, sy)
    backward_plus = interpolated_spin_dot(cos_plus, sin_plus, L, x - dx, y - dy, cx, sy)

    forward_f = 0.25 * (forward_plus - forward_minus) / nsites
    backward_f = 0.25 * (backward_minus - backward_plus) / nsites
    return forward_f, backward_f
end

settings = ArgParseSettings(
    description="Estimate streamline geometry for arms weighted by their spin-aligned T-odd F contribution.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 200
    "--Q"
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
        default = 2.0^-10
    "--burnin-time"
        arg_type = Float64
        default = 10.0
    "--lag-time"
        arg_type = Float64
        default = 0.5
    "--nwindows"
        arg_type = Int
        default = 4
    "--seed"
        arg_type = Int
        default = 1
    "--init"
        arg_type = String
        default = "ordered"
    "--solver"
        arg_type = String
        default = "auto"
    "--seed-grid"
        arg_type = Int
        default = 16
    "--target-r"
        arg_type = Float64
        default = 24.0
    "--top-f-fraction"
        arg_type = Float64
        default = 0.1
    "--max-length"
        arg_type = Float64
        default = NaN
    "--ds"
        arg_type = Float64
        default = 0.25
    "--n-lags"
        arg_type = Int
        default = 40
    "--min-lag"
        arg_type = Float64
        default = 1.0
    "--max-lag"
        arg_type = Float64
        default = NaN
    "--fit-min"
        arg_type = Float64
        default = 4.0
    "--fit-max"
        arg_type = Float64
        default = NaN
    "--avoid-box-size"
        arg_type = Float64
        default = 2.0
    "--output-prefix"
        arg_type = String
        default = "f_weighted_streamline_arclength_scaling_L200_J2_Q1_v1"
end
args = ArgParse.parse_args(settings)

solver_name = parse_solver(args["solver"])
initial_condition = Symbol(args["init"])
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
L = params.L
ds = args["ds"]
max_length = isnan(args["max-length"]) ? Float64(L) : args["max-length"]
max_lag = isnan(args["max-lag"]) ? L / 3 : min(args["max-lag"], max_length - ds)
fit_max = isnan(args["fit-max"]) ? L / 3 : args["fit-max"]
fit_max = min(fit_max, max_lag)
lag_steps = unique(max.(1, round.(Int, logspace_values(args["min-lag"], max_lag, args["n-lags"]) ./ ds)))
lag_lengths = lag_steps .* ds
fit_mask = lag_lengths .>= args["fit-min"] .&& lag_lengths .<= fit_max
count(fit_mask) >= 3 || throw(ArgumentError("fit window must contain at least 3 lag values"))

rng = MersenneTwister(args["seed"])
theta = initial_angles(rng, L, initial_condition)
theta = evolve_state(theta, args["burnin-time"], params, solver_name, args["dt"], rng)

xs = range(1, L; length=args["seed-grid"] + 2)[2:end - 1]
ys = range(1, L; length=args["seed-grid"] + 2)[2:end - 1]
per_arm_r2_values = Vector{Float64}[]
per_arm_alpha = Float64[]
per_arm_df = Float64[]
f_values = Float64[]
abs_f_values = Float64[]
recurrence_by_lag = zeros(Float64, length(lag_steps))

for window_index in 1:args["nwindows"]
    global theta
    theta_minus = copy(theta)
    theta_mid = evolve_state(theta_minus, args["lag-time"], params, solver_name, args["dt"], rng)
    theta_plus = evolve_state(theta_mid, args["lag-time"], params, solver_name, args["dt"], rng)
    theta_grid = reshape(theta_mid, L, L)

    for y in ys, x in xs
        forward_f, backward_f = arm_f_values(theta_minus, theta_mid, theta_plus, params,
            x, y, args["target-r"])
        for (direction_sign, f) in ((1.0, forward_f), (-1.0, backward_f))
            path = trace_arm(theta_grid, x, y, max_length, ds, direction_sign)
            r2 = path_r2(path, lag_steps)
            alpha_i, _ = linear_fit(log.(lag_lengths[fit_mask]), log.(r2[fit_mask]))
            push!(per_arm_r2_values, r2)
            push!(per_arm_alpha, alpha_i)
            push!(per_arm_df, 2 / alpha_i)
            push!(f_values, f)
            push!(abs_f_values, abs(f))
            recurrence_by_lag .+= [recurrence_fraction(path, lag, args["avoid-box-size"], ds)
                for lag in lag_steps]
        end
    end
    theta = theta_plus
    @info "processed F-weighted window" window_index total=args["nwindows"]
end

narms = length(f_values)
per_arm_r2 = hcat(per_arm_r2_values...)
recurrence_by_lag ./= narms
mean_r2 = vec(mean(per_arm_r2; dims=2))
weighted_r2 = weighted_mean_columns(per_arm_r2, abs_f_values)
top_count = max(1, round(Int, args["top-f-fraction"] * narms))
top_indices = sortperm(abs_f_values; rev=true)[1:top_count]
top_r2 = vec(mean(per_arm_r2[:, top_indices]; dims=2))

alpha_all, intercept_all = linear_fit(log.(lag_lengths[fit_mask]), log.(mean_r2[fit_mask]))
alpha_weighted, intercept_weighted =
    linear_fit(log.(lag_lengths[fit_mask]), log.(weighted_r2[fit_mask]))
alpha_top, intercept_top = linear_fit(log.(lag_lengths[fit_mask]), log.(top_r2[fit_mask]))
df_all = 2 / alpha_all
df_weighted = 2 / alpha_weighted
df_top = 2 / alpha_top

local_lags = lag_lengths[2:end - 1]
local_alpha_weighted = [
    (log(weighted_r2[i + 1]) - log(weighted_r2[i - 1])) /
    (log(lag_lengths[i + 1]) - log(lag_lengths[i - 1]))
    for i in 2:(length(lag_lengths) - 1)
]

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1280, 860), backgroundcolor=:white)
ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="arclength s",
    ylabel="R2(s)", title="F-weighted arms at r = $(args["target-r"])")
scatterlines!(ax1, lag_lengths, mean_r2; color=:gray35, label="all sampled arms")
scatterlines!(ax1, lag_lengths, weighted_r2; color=:red, label="abs(F)-weighted")
scatterlines!(ax1, lag_lengths, top_r2; color=:black, label="top abs(F) arms")
lines!(ax1, lag_lengths, exp.(intercept_weighted .+ alpha_weighted .* log.(lag_lengths));
    color=:red, linewidth=2, label="weighted alpha=$(round(alpha_weighted; digits=3))")
axislegend(ax1; position=:lt)

ax2 = Axis(fig[1, 2], xscale=log10, xlabel="arclength s", ylabel="local alpha",
    title="weighted local slope")
scatterlines!(ax2, local_lags, local_alpha_weighted; color=:black)
hlines!(ax2, [1.5]; color=:dodgerblue, linestyle=:dash, label="SAW alpha=1.5")
axislegend(ax2; position=:rb)

ax3 = Axis(fig[2, 1], xlabel="per-arm d_f", ylabel="count",
    title="top arms over full sampled-arm distribution")
hist!(ax3, per_arm_df[isfinite.(per_arm_df)]; bins=40, color=(:gray50, 0.5),
    label="all")
hist!(ax3, per_arm_df[top_indices]; bins=24, color=(:red, 0.55), label="top abs(F)")
vlines!(ax3, [4 / 3]; color=:dodgerblue, linestyle=:dash, linewidth=2)
vlines!(ax3, [df_weighted]; color=:red, linewidth=2)
axislegend(ax3; position=:rt)

ax4 = Axis(fig[2, 2], xscale=log10, xlabel="arclength s", ylabel="recurrence fraction",
    title="coarse-box recurrence, box size $(args["avoid-box-size"])")
scatterlines!(ax4, lag_lengths, recurrence_by_lag; color=:black)
ylims!(ax4, 0, min(1, maximum(recurrence_by_lag) * 1.15 + 0.02))

save(figure_output, fig)

jldsave(data_output; params, args, lag_lengths, lag_steps, fit_mask, mean_r2,
    weighted_r2, top_r2, alpha_all, alpha_weighted, alpha_top, df_all, df_weighted,
    df_top, per_arm_alpha, per_arm_df, f_values, abs_f_values, top_indices,
    recurrence_by_lag, local_lags, local_alpha_weighted)

open(summary_output, "w") do io
    println(io, "# F-weighted streamline arclength scaling")
    println(io)
    println(io, "- Observable: arclength scaling of streamline arms weighted by spin-aligned T-odd `F` contribution.")
    println(io, "- Parameters: `L=$(params.L)`, `Q=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`")
    println(io, "- Windows: `$(args["nwindows"])`, lag time `$(args["lag-time"])`, target radius `$(args["target-r"])`")
    println(io, "- Arms: `$(narms)`, top abs(F) fraction `$(args["top-f-fraction"])` => `$(top_count)` arms")
    println(io, "- Fit window: `$(args["fit-min"]) <= s <= $(fit_max)`")
    println(io)
    println(io, "## Estimates")
    println(io)
    println(io, "- All sampled arms: `alpha = $(round(alpha_all; digits=5))`, `d_f = $(round(df_all; digits=5))`")
    println(io, "- `abs(F)`-weighted arms: `alpha = $(round(alpha_weighted; digits=5))`, `d_f = $(round(df_weighted; digits=5))`")
    println(io, "- Top `abs(F)` arms: `alpha = $(round(alpha_top; digits=5))`, `d_f = $(round(df_top; digits=5))`")
    println(io, "- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`")
    println(io)
    println(io, "## Diagnostics")
    println(io)
    println(io, "- Mean per-arm `d_f`: `$(round(mean(per_arm_df); digits=5))`")
    println(io, "- Mean top-arm `d_f`: `$(round(mean(per_arm_df[top_indices]); digits=5))`")
    println(io, "- Mean recurrence fraction in fit window: `$(round(mean(recurrence_by_lag[fit_mask]); digits=5))`")
    println(io, "- Total signed sampled F: `$(round(sum(f_values); sigdigits=6))`")
    println(io, "- Total absolute sampled F: `$(round(sum(abs_f_values); sigdigits=6))`")
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("all d_f: ", df_all)
println("weighted d_f: ", df_weighted)
println("top d_f: ", df_top)
