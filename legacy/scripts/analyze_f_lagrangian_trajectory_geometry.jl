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

function evolve_window(theta_mid, Tmax, frame_dt, params, solver_name, dt, rng)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    frame_times = collect(0.0:frame_dt:Tmax)
    problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta_mid,
        (0.0, Tmax), work)
    algorithm = solver_name == "em" ? EM() : SRIW1()
    solution = solve(problem, algorithm; dt, adaptive=false, saveat=frame_times,
        save_start=true, rng)
    return [reshape(wrap_angles!(collect(u)), params.L, params.L) for u in solution.u]
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

function interpolated_spin_dot(theta_grid, x, y, ref_cos, ref_sin)
    θ = angle_at(theta_grid, x, y)
    return ref_cos * cos(θ) + ref_sin * sin(θ)
end

function trace_lagrangian(frames, x0, y0, frame_dt, speed, direction_sign)
    nframes = length(frames)
    path = zeros(Float64, nframes, 2)
    path[1, :] .= (x0, y0)
    x = x0
    y = y0
    for k in 1:(nframes - 1)
        θ = angle_at(frames[k], x, y)
        x += direction_sign * speed * frame_dt * cos(θ)
        y += direction_sign * speed * frame_dt * sin(θ)
        path[k + 1, :] .= (x, y)
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

settings = ArgParseSettings(
    description="Measure time-dependent Lagrangian trajectory geometry weighted by the same time-antisymmetric structure as F.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 64
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
    "--Tmax"
        arg_type = Float64
        default = 8.0
    "--frame-dt"
        arg_type = Float64
        default = 0.25
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
        default = 10
    "--min-lag-time"
        arg_type = Float64
        default = 1.0
    "--fit-min-time"
        arg_type = Float64
        default = 1.0
    "--output-prefix"
        arg_type = String
        default = "f_lagrangian_trajectory_geometry"
end
args = parse_args(settings)

solver_name = parse_solver(args["solver"])
initial_condition = Symbol(args["init"])
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
L = params.L
frame_dt = args["frame-dt"]
lag_steps = collect(max(1, round(Int, args["min-lag-time"] / frame_dt)):
    round(Int, args["Tmax"] / frame_dt))
lag_times = lag_steps .* frame_dt
fit_mask = lag_times .>= args["fit-min-time"]

rng = MersenneTwister(args["seed"])
theta = initial_angles(rng, L, initial_condition)
theta = evolve_state(theta, args["burnin-time"], params, solver_name, args["dt"], rng)

xs = range(1, L; length=args["seed-grid"] + 2)[2:end - 1]
ys = range(1, L; length=args["seed-grid"] + 2)[2:end - 1]
path_r2_values = Vector{Float64}[]
anti_weights = Float64[]
even_weights = Float64[]

for window_index in 1:args["nwindows"]
    global theta
    theta_mid = copy(theta)
    forward_frames = evolve_window(theta_mid, args["Tmax"], frame_dt, params, solver_name,
        args["dt"], rng)
    theta_end = vec(copy(forward_frames[end]))
    backward_frames = evolve_window(theta_mid, args["Tmax"], frame_dt, params, solver_name,
        args["dt"], rng)
    theta = theta_end

    for y in ys, x in xs
        center = site_index(round(Int, x), round(Int, y), L)
        θ0 = theta_mid[center]
        ref_cos = cos(θ0)
        ref_sin = sin(θ0)
        fwd_path = trace_lagrangian(forward_frames, x, y, frame_dt, params.v, 1.0)
        bwd_path = trace_lagrangian(backward_frames, x, y, frame_dt, params.v, -1.0)
        for path in (fwd_path, bwd_path)
            push!(path_r2_values, path_r2(path, lag_steps))
        end

        anti = 0.0
        even = 0.0
        for k in 2:length(forward_frames)
            fwd_dot = interpolated_spin_dot(forward_frames[k], fwd_path[k, 1],
                fwd_path[k, 2], ref_cos, ref_sin)
            bwd_dot = interpolated_spin_dot(backward_frames[k], bwd_path[k, 1],
                bwd_path[k, 2], ref_cos, ref_sin)
            anti += abs(fwd_dot - bwd_dot)
            even += abs(fwd_dot + bwd_dot)
        end
        push!(anti_weights, anti)
        push!(anti_weights, anti)
        push!(even_weights, even)
        push!(even_weights, even)
    end
    @info "processed Lagrangian F-geometry window" window_index total=args["nwindows"]
end

per_path_r2 = hcat(path_r2_values...)
mean_r2 = vec(mean(per_path_r2; dims=2))
anti_weighted_r2 = weighted_mean_columns(per_path_r2, anti_weights)
even_weighted_r2 = weighted_mean_columns(per_path_r2, even_weights)

alpha_all, intercept_all = linear_fit(log.(lag_times[fit_mask]), log.(mean_r2[fit_mask]))
alpha_anti, intercept_anti =
    linear_fit(log.(lag_times[fit_mask]), log.(anti_weighted_r2[fit_mask]))
alpha_even, intercept_even =
    linear_fit(log.(lag_times[fit_mask]), log.(even_weighted_r2[fit_mask]))
df_all = 2 / alpha_all
df_anti = 2 / alpha_anti
df_even = 2 / alpha_even

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1120, 560), backgroundcolor=:white)
ax = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="trajectory time lag",
    ylabel="mean squared displacement along Lagrangian path",
    title="Time-dependent trajectories through evolving spin field")
scatterlines!(ax, lag_times, mean_r2; color=:gray40, label="all paths")
scatterlines!(ax, lag_times, anti_weighted_r2; color=:red,
    label="time-antisymmetric weighted")
scatterlines!(ax, lag_times, even_weighted_r2; color=:black,
    label="time-even weighted")
lines!(ax, lag_times, exp.(intercept_anti .+ alpha_anti .* log.(lag_times));
    color=:red, linewidth=2, label="anti d_f=$(round(df_anti; digits=3))")
lines!(ax, lag_times,
    exp.(mean(log.(anti_weighted_r2[fit_mask]) .- 1.5 .* log.(lag_times[fit_mask])) .+
        1.5 .* log.(lag_times));
    color=:dodgerblue, linestyle=:dash, linewidth=2, label="SAW alpha=1.5")
axislegend(ax; position=:lt)
save(figure_output, fig)

jldsave(data_output; args, params, lag_times, lag_steps, fit_mask, mean_r2,
    anti_weighted_r2, even_weighted_r2, anti_weights, even_weights, alpha_all,
    alpha_anti, alpha_even, df_all, df_anti, df_even)

open(summary_output, "w") do io
    println(io, "# F-weighted Lagrangian trajectory geometry")
    println(io)
    println(io, "- Observable: tracer paths through the time-dependent spin field from the midpoint time.")
    println(io, "- Weight: accumulated absolute time-antisymmetric spin overlap between forward and backward traced paths.")
    println(io, "- Parameters: `L=$(params.L)`, `Q=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`, `Tmax=$(args["Tmax"])`")
    println(io, "- Windows: `$(args["nwindows"])`, seed grid `$(args["seed-grid"])x$(args["seed-grid"])`, paths `$(size(per_path_r2, 2))`")
    println(io, "- Fit: `R2(s) ~ s^alpha`, using lag time `>= $(args["fit-min-time"])`")
    println(io)
    println(io, "## Estimates")
    println(io)
    println(io, "- All paths: `alpha = $(round(alpha_all; digits=5))`, `d_f = $(round(df_all; digits=5))`")
    println(io, "- Time-antisymmetric weighted paths: `alpha = $(round(alpha_anti; digits=5))`, `d_f = $(round(df_anti; digits=5))`")
    println(io, "- Time-even weighted paths: `alpha = $(round(alpha_even; digits=5))`, `d_f = $(round(df_even; digits=5))`")
    println(io, "- Reference SAW: `alpha = 1.5`, `d_f = 1.33333`")
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("all df: ", df_all)
println("anti df: ", df_anti)
println("even df: ", df_even)
