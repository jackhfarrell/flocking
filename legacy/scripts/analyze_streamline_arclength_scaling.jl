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

function parse_indices(text::AbstractString, n::Integer)
    isempty(text) && return collect(1:n)
    lowercase(text) == "all" && return collect(1:n)
    indices = parse.(Int, split(text, ","))
    return clamp.(indices, 1, n)
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

function evolve_state(theta0, duration, params, solver_name, dt, seed)
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
            save_start=false, seed)
    end
    return wrap_angles!(collect(solution.u[end]))
end

function load_theta_snapshots(input, snapshot_indices_text)
    dataset = load(input, "dataset")
    if hasproperty(dataset, :theta_snapshots)
        snapshots = dataset.theta_snapshots
        indices = parse_indices(snapshot_indices_text, size(snapshots, 3))
        theta_grids = [collect(snapshots[:, :, k]) for k in indices]
        return theta_grids, dataset.params, collect(dataset.times[indices])
    elseif hasproperty(dataset, :final_theta) && dataset.final_theta !== nothing
        params = dataset.config.params
        return [reshape(collect(dataset.final_theta), params.L, params.L)], params,
            [dataset.times[end]]
    else
        throw(ArgumentError("input must contain dataset.theta_snapshots or dataset.final_theta"))
    end
end

function generate_theta_snapshots(args)
    solver_name = parse_solver(args["solver"])
    initial_condition = Symbol(args["init"])
    initial_condition in (:random, :ordered) ||
        throw(ArgumentError("init must be random or ordered"))

    params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
    rng = MersenneTwister(args["seed"])
    theta = initial_angles(rng, params.L, initial_condition)
    theta = evolve_state(theta, args["burnin-time"], params, solver_name, args["dt"],
        args["seed"])

    theta_grids = Matrix{Float64}[]
    times = Float64[]
    elapsed = args["burnin-time"]
    for snapshot_index in 1:args["nsnapshots"]
        if snapshot_index > 1 || args["snapshot-spacing"] > 0
            theta = evolve_state(theta, args["snapshot-spacing"], params, solver_name,
                args["dt"], args["seed"] + snapshot_index)
            elapsed += args["snapshot-spacing"]
        end
        push!(theta_grids, reshape(copy(theta), params.L, params.L))
        push!(times, elapsed)
    end
    return theta_grids, params, times
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

function trace_streamline(theta_grid, seed_x, seed_y, max_length, ds)
    nsteps = round(Int, max_length / ds)
    forward = zeros(Float64, nsteps + 1, 2)
    backward = zeros(Float64, nsteps + 1, 2)
    forward[1, :] .= (seed_x, seed_y)
    backward[1, :] .= (seed_x, seed_y)

    x = seed_x
    y = seed_y
    for step in 1:nsteps
        x, y = rk4_step(theta_grid, x, y, ds, 1.0)
        forward[step + 1, :] .= (x, y)
    end

    x = seed_x
    y = seed_y
    for step in 1:nsteps
        x, y = rk4_step(theta_grid, x, y, ds, -1.0)
        backward[step + 1, :] .= (x, y)
    end

    return vcat(reverse(backward[2:end, :]; dims=1), forward)
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
            box = (floor(Int, path[i, 1] / box_size), floor(Int, path[i, 2] / box_size))
            push!(boxes, box)
        end
        repeats += sampled_lag + 1 - length(boxes)
        sampled += sampled_lag + 1
    end
    return repeats / sampled
end

function box_count_dimension(path, box_sizes)
    counts = zeros(Float64, length(box_sizes))
    @inbounds for (j, eps) in enumerate(box_sizes)
        boxes = Set{Tuple{Int, Int}}()
        for i in axes(path, 1)
            push!(boxes, (floor(Int, path[i, 1] / eps), floor(Int, path[i, 2] / eps)))
        end
        counts[j] = length(boxes)
    end
    slope, _ = linear_fit(log.(1.0 ./ box_sizes), log.(counts))
    return slope, counts
end

settings = ArgParseSettings(
    description="Estimate active-streamline fractal dimension from arclength-displacement scaling.",
)
@add_arg_table! settings begin
    "--input"
        arg_type = String
        default = ""
    "--snapshot-indices"
        arg_type = String
        default = "all"
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
    "--snapshot-spacing"
        arg_type = Float64
        default = 1.0
    "--nsnapshots"
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
    "--seed-jitter"
        arg_type = Float64
        default = 0.0
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
    "--box-sizes"
        arg_type = String
        default = "2,4,8,16,32"
    "--output-prefix"
        arg_type = String
        default = "streamline_arclength_scaling_L200_J2_Q1_v1"
end
args = ArgParse.parse_args(settings)

theta_grids, params, snapshot_times = isempty(args["input"]) ?
    generate_theta_snapshots(args) :
    load_theta_snapshots(args["input"], args["snapshot-indices"])

L = params.L
ds = args["ds"]
max_length = isnan(args["max-length"]) ? Float64(L) : args["max-length"]
max_lag = isnan(args["max-lag"]) ? L / 3 : args["max-lag"]
fit_max = isnan(args["fit-max"]) ? L / 3 : args["fit-max"]
max_lag = min(max_lag, 2max_length - ds)
fit_max = min(fit_max, max_lag)
lag_values = logspace_values(args["min-lag"], max_lag, args["n-lags"])
lag_steps = unique(max.(1, round.(Int, lag_values ./ ds)))
lag_lengths = lag_steps .* ds
fit_mask = lag_lengths .>= args["fit-min"] .&& lag_lengths .<= fit_max
count(fit_mask) >= 3 || throw(ArgumentError("fit window must contain at least 3 lag values"))

seed_grid = args["seed-grid"]
rng = MersenneTwister(args["seed"] + 100_000)
xs = range(1, L; length=seed_grid + 2)[2:end - 1]
ys = range(1, L; length=seed_grid + 2)[2:end - 1]
box_sizes = parse.(Float64, split(args["box-sizes"], ","))

paths = Matrix{Float64}[]
per_path_r2_values = Vector{Float64}[]
per_path_alpha = Float64[]
per_path_df = Float64[]
box_dimensions = Float64[]
box_count_values = Vector{Float64}[]
recurrence_by_lag = zeros(Float64, length(lag_steps))

for (snapshot_index, theta_grid) in enumerate(theta_grids)
    for y0 in ys, x0 in xs
        x = x0 + args["seed-jitter"] * (rand(rng) - 0.5)
        y = y0 + args["seed-jitter"] * (rand(rng) - 0.5)
        path = trace_streamline(theta_grid, x, y, max_length, ds)
        r2 = path_r2(path, lag_steps)
        alpha_i, _ = linear_fit(log.(lag_lengths[fit_mask]), log.(r2[fit_mask]))
        box_dim_i, counts_i = box_count_dimension(path, box_sizes)

        push!(paths, path)
        push!(per_path_r2_values, r2)
        push!(per_path_alpha, alpha_i)
        push!(per_path_df, 2 / alpha_i)
        push!(box_dimensions, box_dim_i)
        push!(box_count_values, counts_i)
        recurrence_by_lag .+= [recurrence_fraction(path, lag, args["avoid-box-size"], ds)
            for lag in lag_steps]
    end
    @info "processed snapshot" snapshot_index total=length(theta_grids)
end

npaths = length(paths)
per_path_r2 = hcat(per_path_r2_values...)
box_counts = hcat(box_count_values...)
mean_r2 = vec(mean(per_path_r2; dims=2))
stderr_r2 = vec(std(per_path_r2; dims=2)) ./ sqrt(npaths)
recurrence_by_lag ./= npaths
alpha, intercept = linear_fit(log.(lag_lengths[fit_mask]), log.(mean_r2[fit_mask]))
df = 2 / alpha

local_lags = lag_lengths[2:end - 1]
local_alpha = [
    (log(mean_r2[i + 1]) - log(mean_r2[i - 1])) /
    (log(lag_lengths[i + 1]) - log(lag_lengths[i - 1]))
    for i in 2:(length(lag_lengths) - 1)
]
local_df = 2.0 ./ local_alpha

center_log_s = mean(log.(lag_lengths[fit_mask]))
center_log_r2 = intercept + alpha * center_log_s
reference_intercept = center_log_r2 - 1.5 * center_log_s
fit_line = exp.(intercept .+ alpha .* log.(lag_lengths))
reference_line = exp.(reference_intercept .+ 1.5 .* log.(lag_lengths))

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1280, 860), backgroundcolor=:white)
ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="arclength s",
    ylabel="mean squared displacement R2(s)",
    title="R2(s): alpha = $(round(alpha; digits=3)), d_f = $(round(df; digits=3))")
scatterlines!(ax1, lag_lengths, mean_r2; color=:black, label="ensemble mean")
band!(ax1, lag_lengths, max.(mean_r2 .- stderr_r2, eps()), mean_r2 .+ stderr_r2;
    color=(:black, 0.15))
lines!(ax1, lag_lengths, fit_line; color=:red, linewidth=2,
    label="fit alpha=$(round(alpha; digits=3))")
lines!(ax1, lag_lengths, reference_line; color=:dodgerblue, linestyle=:dash,
    linewidth=2, label="SAW alpha=1.5")
vspan!(ax1, args["fit-min"], fit_max; color=(:red, 0.08))
axislegend(ax1; position=:lt)

ax2 = Axis(fig[1, 2], xscale=log10, xlabel="arclength s", ylabel="local alpha",
    title="local slope and d_f = 2 / alpha")
scatterlines!(ax2, local_lags, local_alpha; color=:black, label="local alpha")
hlines!(ax2, [1.5]; color=:dodgerblue, linestyle=:dash, label="SAW alpha=1.5")
axislegend(ax2; position=:rb)
ax2r = Axis(fig[1, 2], xscale=log10, yaxisposition=:right, ylabel="local d_f")
hidespines!(ax2r, :l, :b, :t)
hidexdecorations!(ax2r)
lines!(ax2r, local_lags, local_df; color=(:darkorange, 0.75), linewidth=1.5)
ylims!(ax2r, minimum(local_df[isfinite.(local_df)]) * 0.95,
    maximum(local_df[isfinite.(local_df)]) * 1.05)

ax3 = Axis(fig[2, 1], xlabel="per-streamline d_f", ylabel="count",
    title="per-streamline fit distribution")
hist!(ax3, per_path_df[isfinite.(per_path_df)]; bins=32, color=(:gray30, 0.65))
vlines!(ax3, [4 / 3]; color=:dodgerblue, linestyle=:dash, linewidth=2)
vlines!(ax3, [df]; color=:red, linewidth=2)

ax4 = Axis(fig[2, 2], xscale=log10, xlabel="arclength s",
    ylabel="recurrence fraction",
    title="coarse-box recurrence, box size $(args["avoid-box-size"])")
scatterlines!(ax4, lag_lengths, recurrence_by_lag; color=:black)
ylims!(ax4, 0, min(1, maximum(recurrence_by_lag) * 1.15 + 0.02))

save(figure_output, fig)

jldsave(data_output; params, args, snapshot_times, lag_lengths, lag_steps, mean_r2,
    stderr_r2, per_path_r2, alpha, df, per_path_alpha, per_path_df, fit_mask,
    fit_min=args["fit-min"], fit_max, local_lags, local_alpha, local_df,
    recurrence_by_lag, box_sizes, box_counts, box_dimensions)

open(summary_output, "w") do io
    println(io, "# Streamline arclength scaling")
    println(io)
    println(io, "- Observable: `R2(s) = <|x(s0+s)-x(s0)|^2>` on unwrapped active streamlines.")
    println(io, "- Parameters: `L=$(params.L)`, `Q=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`")
    println(io, "- Snapshots: `$(length(theta_grids))`, times `$(join(round.(snapshot_times; digits=4), ", "))`")
    println(io, "- Streamlines: `$(npaths)`, seed grid `$(seed_grid)x$(seed_grid)`, jitter `$(args["seed-jitter"])`")
    println(io, "- Trace arclength each direction: `$(max_length)`, step `$(ds)`")
    println(io, "- Fit window: `$(args["fit-min"]) <= s <= $(fit_max)`")
    println(io)
    println(io, "## Primary estimate")
    println(io)
    println(io, "- Fit `R2(s) ~ s^alpha`: `alpha = $(round(alpha; digits=5))`")
    println(io, "- Fractal dimension `d_f = 2 / alpha`: `$(round(df; digits=5))`")
    println(io, "- Reference SAW values: `alpha = 1.5`, `d_f = 1.33333`")
    println(io)
    println(io, "## Secondary diagnostics")
    println(io)
    println(io, "- Mean per-streamline `d_f`: `$(round(mean(per_path_df); digits=5))`")
    println(io, "- Std per-streamline `d_f`: `$(round(std(per_path_df); digits=5))`")
    println(io, "- Mean box-count dimension proxy: `$(round(mean(box_dimensions); digits=5))`")
    println(io, "- Mean recurrence fraction in fit window: `$(round(mean(recurrence_by_lag[fit_mask]); digits=5))`")
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("alpha: ", alpha)
println("d_f: ", df)
