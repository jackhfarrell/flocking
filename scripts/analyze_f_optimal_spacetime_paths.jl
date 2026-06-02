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

function evolve_saved_window(theta0, duration, frame_dt, params, solver_name, dt, rng)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    frame_times = collect(0.0:frame_dt:duration)
    if iszero(params.Q)
        problem = ODEProblem(LatticeFlockingSDE.drift!, theta0, (0.0, duration), work)
        solution = solve(problem, Tsit5(); saveat=frame_times, save_start=true)
    else
        problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta0,
            (0.0, duration), work)
        algorithm = solver_name == "em" ? EM() : SRIW1()
        solution = solve(problem, algorithm; dt, adaptive=false, saveat=frame_times,
            save_start=true, rng)
    end
    return [reshape(wrap_angles!(collect(u)), params.L, params.L) for u in solution.u]
end

function reward_field(theta_plus, theta_minus, ref_cos, ref_sin)
    plus = ref_cos .* cos.(theta_plus) .+ ref_sin .* sin.(theta_plus)
    minus = ref_cos .* cos.(theta_minus) .+ ref_sin .* sin.(theta_minus)
    return plus .- minus
end

function optimal_path(reward_frames, seed_x, seed_y, move_penalty)
    L = size(reward_frames[1], 1)
    nframes = length(reward_frames)
    score = fill(-Inf, L, L)
    score[seed_x, seed_y] = reward_frames[1][seed_x, seed_y]
    parents = Array{CartesianIndex{2}}(undef, L, L, nframes)
    parents[:, :, 1] .= CartesianIndex(seed_x, seed_y)
    moves = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    for k in 2:nframes
        new_score = fill(-Inf, L, L)
        for x in 1:L, y in 1:L
            best = -Inf
            best_parent = CartesianIndex(x, y)
            for (dx, dy) in moves
                px = mod1(x - dx, L)
                py = mod1(y - dy, L)
                step_cost = move_penalty * sqrt(dx^2 + dy^2)
                candidate = score[px, py] - step_cost
                if candidate > best
                    best = candidate
                    best_parent = CartesianIndex(px, py)
                end
            end
            new_score[x, y] = best + reward_frames[k][x, y]
            parents[x, y, k] = best_parent
        end
        score = new_score
    end

    endpoint = argmax(score)
    path = zeros(Float64, nframes, 2)
    current = endpoint
    for k in nframes:-1:1
        path[k, :] .= (current[1], current[2])
        current = parents[current[1], current[2], k]
    end
    return path, score[endpoint]
end

function unwrap_path!(path, L)
    for k in 2:size(path, 1)
        for d in 1:2
            delta = path[k, d] - path[k - 1, d]
            if delta > L / 2
                path[k:end, d] .-= L
            elseif delta < -L / 2
                path[k:end, d] .+= L
            end
        end
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
    description="Find dynamic-programming spacetime paths maximizing the time-antisymmetric F-like reward and measure their geometry.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 48
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
        default = 0.5
    "--nwindows"
        arg_type = Int
        default = 3
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
        default = 8
    "--move-penalty"
        arg_type = Float64
        default = 0.05
    "--fit-min-lag"
        arg_type = Float64
        default = 1.0
    "--output-prefix"
        arg_type = String
        default = "f_optimal_spacetime_paths"
end
args = parse_args(settings)

solver_name = parse_solver(args["solver"])
initial_condition = Symbol(args["init"])
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
L = params.L
rng = MersenneTwister(args["seed"])
theta = initial_angles(rng, L, initial_condition)
theta = evolve_state(theta, args["burnin-time"], params, solver_name, args["dt"], rng)

xs = round.(Int, range(1, L; length=args["seed-grid"] + 2)[2:end - 1])
ys = round.(Int, range(1, L; length=args["seed-grid"] + 2)[2:end - 1])
nframes_half = round(Int, args["Tmax"] / args["frame-dt"]) + 1
lag_steps = collect(1:(nframes_half - 1))
lag_times = lag_steps .* args["frame-dt"]
fit_mask = lag_times .>= args["fit-min-lag"]

r2_values = Vector{Float64}[]
scores = Float64[]
end_displacements = Float64[]

for window_index in 1:args["nwindows"]
    global theta
    frames = evolve_saved_window(theta, 2args["Tmax"], args["frame-dt"], params,
        solver_name, args["dt"], rng)
    mid = nframes_half
    theta_mid = vec(frames[mid])
    for y in ys, x in xs
        center = site_index(x, y, L)
        ref_cos = cos(theta_mid[center])
        ref_sin = sin(theta_mid[center])
        reward_frames = [
            reward_field(frames[mid + k], frames[mid - k], ref_cos, ref_sin)
            for k in 0:(nframes_half - 1)
        ]
        path, score = optimal_path(reward_frames, x, y, args["move-penalty"])
        unwrap_path!(path, L)
        push!(r2_values, path_r2(path, lag_steps))
        push!(scores, score)
        dx = path[end, 1] - path[1, 1]
        dy = path[end, 2] - path[1, 2]
        push!(end_displacements, sqrt(dx^2 + dy^2))
    end
    theta = vec(copy(frames[end]))
    @info "processed optimal spacetime path window" window_index total=args["nwindows"]
end

per_path_r2 = hcat(r2_values...)
mean_r2 = vec(mean(per_path_r2; dims=2))
positive_scores = max.(scores .- minimum(scores) .+ eps(), eps())
weighted_r2 = vec(per_path_r2 * (positive_scores ./ sum(positive_scores)))
top_count = max(1, round(Int, 0.1length(scores)))
top_indices = sortperm(scores; rev=true)[1:top_count]
top_r2 = vec(mean(per_path_r2[:, top_indices]; dims=2))

alpha_all, intercept_all = linear_fit(log.(lag_times[fit_mask]), log.(mean_r2[fit_mask]))
alpha_weighted, intercept_weighted =
    linear_fit(log.(lag_times[fit_mask]), log.(weighted_r2[fit_mask]))
alpha_top, intercept_top = linear_fit(log.(lag_times[fit_mask]), log.(top_r2[fit_mask]))
df_all = 2 / alpha_all
df_weighted = 2 / alpha_weighted
df_top = 2 / alpha_top

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1120, 560), backgroundcolor=:white)
ax = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="path time lag",
    ylabel="R2 along optimal spacetime path",
    title="Time-antisymmetric optimal paths")
scatterlines!(ax, lag_times, mean_r2; color=:gray45, label="all optimal paths")
scatterlines!(ax, lag_times, weighted_r2; color=:red, label="score-weighted")
scatterlines!(ax, lag_times, top_r2; color=:black, label="top 10% score")
lines!(ax, lag_times, exp.(intercept_weighted .+ alpha_weighted .* log.(lag_times));
    color=:red, linewidth=2, label="weighted d_f=$(round(df_weighted; digits=3))")
lines!(ax, lag_times,
    exp.(mean(log.(weighted_r2[fit_mask]) .- 1.5 .* log.(lag_times[fit_mask])) .+
        1.5 .* log.(lag_times));
    color=:dodgerblue, linestyle=:dash, linewidth=2, label="SAW alpha=1.5")
axislegend(ax; position=:lt)
save(figure_output, fig)

jldsave(data_output; args, params, lag_times, lag_steps, fit_mask, per_path_r2,
    mean_r2, weighted_r2, top_r2, scores, end_displacements, alpha_all,
    alpha_weighted, alpha_top, df_all, df_weighted, df_top)

open(summary_output, "w") do io
    println(io, "# F optimal spacetime paths")
    println(io)
    println(io, "- Observable: dynamic-programming paths maximizing a local time-antisymmetric reward `n(x,t0+k) - n(x,t0-k)` projected onto the seed spin.")
    println(io, "- Parameters: `L=$(params.L)`, `Q=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`, `Tmax=$(args["Tmax"])`, `frame_dt=$(args["frame-dt"])`")
    println(io, "- Windows: `$(args["nwindows"])`, seed grid `$(args["seed-grid"])x$(args["seed-grid"])`, paths `$(length(scores))`")
    println(io, "- Move penalty: `$(args["move-penalty"])`")
    println(io)
    println(io, "## Estimates")
    println(io)
    println(io, "- All optimal paths: `alpha = $(round(alpha_all; digits=5))`, `d_f = $(round(df_all; digits=5))`")
    println(io, "- Score-weighted optimal paths: `alpha = $(round(alpha_weighted; digits=5))`, `d_f = $(round(df_weighted; digits=5))`")
    println(io, "- Top 10% score paths: `alpha = $(round(alpha_top; digits=5))`, `d_f = $(round(df_top; digits=5))`")
    println(io, "- Reference SAW: `alpha = 1.5`, `d_f = 1.33333`")
    println(io)
    println(io, "## Diagnostics")
    println(io)
    println(io, "- Mean endpoint displacement: `$(round(mean(end_displacements); digits=5))`")
    println(io, "- Mean top-score endpoint displacement: `$(round(mean(end_displacements[top_indices]); digits=5))`")
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("all df: ", df_all)
println("weighted df: ", df_weighted)
println("top df: ", df_top)
