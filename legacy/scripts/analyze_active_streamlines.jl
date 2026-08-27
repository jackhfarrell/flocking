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

function evolve_state(theta0, tspan, params, solver_name, dt, seed)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    if iszero(params.Q)
        problem = ODEProblem(LatticeFlockingSDE.drift!, theta0, tspan, work)
        solution = solve(problem, Tsit5(); save_everystep=false, save_start=false)
    else
        problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta0, tspan, work)
        algorithm = solver_name == "em" ? EM() : SRIW1()
        solution = solve(problem, algorithm; dt, adaptive=false, save_everystep=false,
            save_start=false, seed)
    end
    return wrap_angles!(collect(solution.u[end]))
end

function load_theta_snapshot(input, snapshot_index)
    dataset = load(input, "dataset")
    if hasproperty(dataset, :theta_snapshots)
        snapshots = dataset.theta_snapshots
        k = min(snapshot_index, size(snapshots, 3))
        return collect(snapshots[:, :, k]), dataset.params, dataset.times[k]
    elseif hasproperty(dataset, :final_theta) && dataset.final_theta !== nothing
        params = dataset.config.params
        return reshape(collect(dataset.final_theta), params.L, params.L), params, dataset.times[end]
    else
        throw(ArgumentError("input must contain dataset.theta_snapshots or dataset.final_theta"))
    end
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

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    c = w00 * cos(theta_grid[x0, y0]) + w10 * cos(theta_grid[x1, y0]) +
        w01 * cos(theta_grid[x0, y1]) + w11 * cos(theta_grid[x1, y1])
    s = w00 * sin(theta_grid[x0, y0]) + w10 * sin(theta_grid[x1, y0]) +
        w01 * sin(theta_grid[x0, y1]) + w11 * sin(theta_grid[x1, y1])
    return atan(s, c)
end

function direction_at(theta_grid, x, y)
    theta = angle_at(theta_grid, x, y)
    return cos(theta), sin(theta)
end

function rk4_step(theta_grid, x, y, ds, direction_sign)
    k1x, k1y = direction_at(theta_grid, x, y)
    k2x, k2y = direction_at(theta_grid, x + 0.5ds * direction_sign * k1x,
        y + 0.5ds * direction_sign * k1y)
    k3x, k3y = direction_at(theta_grid, x + 0.5ds * direction_sign * k2x,
        y + 0.5ds * direction_sign * k2y)
    k4x, k4y = direction_at(theta_grid, x + ds * direction_sign * k3x,
        y + ds * direction_sign * k3y)
    xnew = x + direction_sign * ds * (k1x + 2k2x + 2k3x + k4x) / 6
    ynew = y + direction_sign * ds * (k1y + 2k2y + 2k3y + k4y) / 6
    return xnew, ynew
end

function trace_streamline(theta_grid, seed_x, seed_y, max_length, ds)
    nsteps = round(Int, max_length / ds)
    forward = zeros(Float64, nsteps + 1, 2)
    backward = zeros(Float64, nsteps + 1, 2)
    forward[1, :] .= (seed_x, seed_y)
    backward[1, :] .= (seed_x, seed_y)

    x = seed_x
    y = seed_y
    for i in 1:nsteps
        x, y = rk4_step(theta_grid, x, y, ds, 1.0)
        forward[i + 1, :] .= (x, y)
    end

    x = seed_x
    y = seed_y
    for i in 1:nsteps
        x, y = rk4_step(theta_grid, x, y, ds, -1.0)
        backward[i + 1, :] .= (x, y)
    end

    path = vcat(reverse(backward[2:end, :]; dims=1), forward)
    return path
end

function rg_fit_dimension(path, ds, min_length, max_length)
    center = (size(path, 1) + 1) ÷ 2
    lengths = exp10.(range(log10(min_length), log10(max_length); length=16))
    rg_values = Float64[]
    used_lengths = Float64[]
    for len in lengths
        half_steps = round(Int, len / (2ds))
        lo = max(1, center - half_steps)
        hi = min(size(path, 1), center + half_steps)
        segment = @view path[lo:hi, :]
        mx = mean(@view segment[:, 1])
        my = mean(@view segment[:, 2])
        rg = sqrt(mean((@view(segment[:, 1]) .- mx).^2 .+ (@view(segment[:, 2]) .- my).^2))
        if isfinite(rg) && rg > 0
            push!(used_lengths, (hi - lo) * ds)
            push!(rg_values, rg)
        end
    end
    x = log.(rg_values)
    y = log.(used_lengths)
    slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)
    return slope, used_lengths, rg_values
end

function box_count_dimension(path, box_sizes)
    counts = Float64[]
    for eps in box_sizes
        boxes = Set{Tuple{Int, Int}}()
        @inbounds for i in axes(path, 1)
            push!(boxes, (floor(Int, path[i, 1] / eps), floor(Int, path[i, 2] / eps)))
        end
        push!(counts, length(boxes))
    end
    x = log.(1.0 ./ box_sizes)
    y = log.(counts)
    slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)
    return slope, counts
end

function divider_dimension(path, yardsticks)
    counts = Float64[]
    for eps in yardsticks
        count = 0
        anchor_x = path[1, 1]
        anchor_y = path[1, 2]
        @inbounds for i in 2:size(path, 1)
            dx = path[i, 1] - anchor_x
            dy = path[i, 2] - anchor_y
            if sqrt(dx^2 + dy^2) >= eps
                count += 1
                anchor_x = path[i, 1]
                anchor_y = path[i, 2]
            end
        end
        push!(counts, max(count, 1))
    end
    x = log.(1.0 ./ yardsticks)
    y = log.(counts)
    slope = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)).^2)
    return slope, counts
end

function self_avoidance_stats(path, box_size, ds)
    stride = max(1, round(Int, box_size / ds))
    boxes = Tuple{Int, Int}[]
    @inbounds for i in 1:stride:size(path, 1)
        push!(boxes, (floor(Int, path[i, 1] / box_size), floor(Int, path[i, 2] / box_size)))
    end
    unique_boxes = length(Set(boxes))
    sampled_boxes = length(boxes)
    unique_fraction = unique_boxes / sampled_boxes
    recurrence_fraction = 1 - unique_fraction

    mx = mean(@view path[:, 1])
    my = mean(@view path[:, 2])
    rg = sqrt(mean((@view(path[:, 1]) .- mx).^2 .+ (@view(path[:, 2]) .- my).^2))
    endpoint_distance = sqrt((path[end, 1] - path[1, 1])^2 + (path[end, 2] - path[1, 2])^2)
    return (; unique_fraction, recurrence_fraction, unique_boxes, sampled_boxes, rg,
        endpoint_distance)
end

function wrapped_plot_path(path, L)
    xs = Float64[]
    ys = Float64[]
    last_x = NaN
    last_y = NaN
    @inbounds for i in axes(path, 1)
        x = mod(path[i, 1] - 1, L) + 1
        y = mod(path[i, 2] - 1, L) + 1
        if i > 1 && (abs(x - last_x) > L / 2 || abs(y - last_y) > L / 2)
            push!(xs, NaN)
            push!(ys, NaN)
        end
        push!(xs, x)
        push!(ys, y)
        last_x = x
        last_y = y
    end
    return xs, ys
end

settings = ArgParseSettings(
    description="Trace active streamlines of v n-hat in a theta snapshot and estimate finite-size fractal dimensions.",
)
@add_arg_table! settings begin
    "--input"
        arg_type = String
        default = ""
    "--snapshot-index"
        arg_type = Int
        default = 1
    "--L"
        arg_type = Int
        default = 128
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
        default = 0.001
    "--snapshot-time"
        arg_type = Float64
        default = 0.5
    "--burnin-time"
        arg_type = Float64
        default = 0.0
    "--seed"
        arg_type = Int
        default = 1
    "--init"
        arg_type = String
        default = "random"
    "--solver"
        arg_type = String
        default = "auto"
    "--seed-grid"
        arg_type = Int
        default = 12
    "--max-length"
        arg_type = Float64
        default = 512.0
    "--ds"
        arg_type = Float64
        default = 0.25
    "--fit-min-length"
        arg_type = Float64
        default = 8.0
    "--fit-max-length"
        arg_type = Float64
        default = 256.0
    "--box-sizes"
        arg_type = String
        default = "2,4,8,16,32"
    "--yardsticks"
        arg_type = String
        default = "2,4,8,16,32"
    "--avoid-box-size"
        arg_type = Float64
        default = 2.0
    "--max-recurrence"
        arg_type = Float64
        default = 0.05
    "--min-rg"
        arg_type = Float64
        default = 12.0
    "--min-endpoint-distance"
        arg_type = Float64
        default = 12.0
    "--output-prefix"
        arg_type = String
        default = "active_streamlines_T0p5"
end
args = ArgParse.parse_args(settings)

solver_name = parse_solver(args["solver"])
initial_condition = Symbol(args["init"])
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

theta_grid, params, actual_time = if isempty(args["input"])
    params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
    rng = MersenneTwister(args["seed"])
    theta = initial_angles(rng, params.L, initial_condition)
    if args["burnin-time"] > 0
        theta = evolve_state(theta, (0.0, args["burnin-time"]), params, solver_name,
            args["dt"], args["seed"])
    end
    theta = evolve_state(theta, (0.0, args["snapshot-time"]), params, solver_name,
        args["dt"], args["seed"] + 1)
    reshape(theta, params.L, params.L), params, args["snapshot-time"]
else
    load_theta_snapshot(args["input"], args["snapshot-index"])
end

L = params.L
seed_grid = args["seed-grid"]
max_length = args["max-length"]
ds = args["ds"]
fit_min_length = args["fit-min-length"]
fit_max_length = min(args["fit-max-length"], 2max_length)
box_sizes = parse.(Float64, split(args["box-sizes"], ","))
yardsticks = parse.(Float64, split(args["yardsticks"], ","))
avoid_box_size = args["avoid-box-size"]
max_recurrence = args["max-recurrence"]
min_rg = args["min-rg"]
min_endpoint_distance = args["min-endpoint-distance"]

xs = range(1, L; length=seed_grid + 2)[2:end - 1]
ys = range(1, L; length=seed_grid + 2)[2:end - 1]
paths = Matrix{Float64}[]
rg_dimensions = Float64[]
box_dimensions = Float64[]
divider_dimensions = Float64[]
selected = Bool[]
self_avoidance = NamedTuple[]
box_counts = zeros(Float64, length(box_sizes), seed_grid * seed_grid)
divider_counts = zeros(Float64, length(yardsticks), seed_grid * seed_grid)

streamline_index = 0
for (streamline_index, (y, x)) in enumerate(Iterators.product(ys, xs))
    path = trace_streamline(theta_grid, x, y, max_length, ds)
    push!(paths, path)
    dim_rg, _, _ = rg_fit_dimension(path, ds, fit_min_length, fit_max_length)
    dim_box, counts = box_count_dimension(path, box_sizes)
    dim_divider, div_counts = divider_dimension(path, yardsticks)
    stats = self_avoidance_stats(path, avoid_box_size, ds)
    keep = stats.recurrence_fraction <= max_recurrence &&
        stats.rg >= min_rg &&
        stats.endpoint_distance >= min_endpoint_distance
    push!(rg_dimensions, dim_rg)
    push!(box_dimensions, dim_box)
    push!(divider_dimensions, dim_divider)
    push!(self_avoidance, stats)
    push!(selected, keep)
    box_counts[:, streamline_index] .= counts
    divider_counts[:, streamline_index] .= div_counts
end

selected_indices = findall(selected)
selected_count = length(selected_indices)
selected_rg_dimensions = rg_dimensions[selected_indices]
selected_box_dimensions = box_dimensions[selected_indices]
selected_divider_dimensions = divider_dimensions[selected_indices]
selected_box_counts = box_counts[:, selected_indices]
selected_divider_counts = divider_counts[:, selected_indices]

mean_box_counts = vec(mean(box_counts; dims=2))
box_x = log.(1.0 ./ box_sizes)
box_y = log.(mean_box_counts)
box_dimension_mean_count =
    sum((box_x .- mean(box_x)) .* (box_y .- mean(box_y))) / sum((box_x .- mean(box_x)).^2)
mean_divider_counts = vec(mean(divider_counts; dims=2))
divider_x = log.(1.0 ./ yardsticks)
divider_y = log.(mean_divider_counts)
divider_dimension_mean_count =
    sum((divider_x .- mean(divider_x)) .* (divider_y .- mean(divider_y))) /
    sum((divider_x .- mean(divider_x)).^2)
selected_box_dimension_mean_count = NaN
selected_divider_dimension_mean_count = NaN
if selected_count > 0
    selected_mean_box_counts = vec(mean(selected_box_counts; dims=2))
    selected_box_y = log.(selected_mean_box_counts)
    selected_box_dimension_mean_count =
        sum((box_x .- mean(box_x)) .* (selected_box_y .- mean(selected_box_y))) /
        sum((box_x .- mean(box_x)).^2)
    selected_mean_divider_counts = vec(mean(selected_divider_counts; dims=2))
    selected_divider_y = log.(selected_mean_divider_counts)
    selected_divider_dimension_mean_count =
        sum((divider_x .- mean(divider_x)) .* (selected_divider_y .- mean(selected_divider_y))) /
        sum((divider_x .- mean(divider_x)).^2)
end

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")

mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1320, 620), backgroundcolor=:white)
ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y",
    title="active streamlines of v n-hat at t = $(round(actual_time; digits=3))")
heatmap!(ax, 1:L, 1:L, theta_grid; colormap=:hsv, colorrange=(0, 2π))
for (i, path) in enumerate(paths)
    plot_x, plot_y = wrapped_plot_path(path, L)
    color = selected[i] ? (:red, 0.85) : (:black, 0.18)
    linewidth = selected[i] ? 1.6 : 0.7
    lines!(ax, plot_x, plot_y; color, linewidth)
end
xlims!(ax, 1, L)
ylims!(ax, 1, L)
Colorbar(fig[2, 1], limits=(0, 2π), colormap=:hsv, vertical=false, label="theta")

ax2 = Axis(fig[1, 2], xlabel="dimension estimate", ylabel="count",
    title="streamline dimension proxies")
hist!(ax2, rg_dimensions; bins=24, color=(:steelblue, 0.7), label="length vs Rg")
hist!(ax2, divider_dimensions; bins=24, color=(:darkorange, 0.45), label="divider")
hist!(ax2, box_dimensions; bins=24, color=(:seagreen, 0.35), label="box count")
if selected_count > 0
    scatter!(ax2, selected_rg_dimensions, fill(-0.4, selected_count); color=:red,
        markersize=8, label="selected")
end
vlines!(ax2, [4 / 3]; color=:black, linestyle=:dash, linewidth=2, label="4/3")
axislegend(ax2; position=:rt)

ax3 = Axis(fig[2, 2], xlabel="log(1 / scale)", ylabel="log count",
    title="mean counts: divider D = $(round(divider_dimension_mean_count; digits=3)), box D = $(round(box_dimension_mean_count; digits=3))")
scatterlines!(ax3, divider_x, divider_y; color=:darkorange, label="divider")
scatterlines!(ax3, box_x, box_y; color=:seagreen, label="box")
if selected_count > 0
    scatterlines!(ax3, divider_x, log.(vec(mean(selected_divider_counts; dims=2)));
        color=:red, label="selected divider")
end
axislegend(ax3; position=:lt)

save(figure_output, fig)
jldsave(data_output; theta_grid, params, actual_time, paths, rg_dimensions, box_dimensions,
    divider_dimensions, box_sizes, box_counts, box_dimension_mean_count, yardsticks,
    divider_counts, divider_dimension_mean_count, self_avoidance, selected,
    selected_rg_dimensions, selected_box_dimensions, selected_divider_dimensions,
    selected_box_dimension_mean_count, selected_divider_dimension_mean_count)

open(summary_output, "w") do io
    println(io, "# Active streamline dimension check")
    println(io)
    println(io, "- Field: `v (cos(theta), sin(theta))`; streamline geometry is independent of the speed scale `v`.")
    println(io, "- Snapshot time: `$(actual_time)`")
    println(io, "- Parameters: `L=$(params.L)`, `Q=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`")
    println(io, "- Streamlines: `$(length(paths))`, arclength each direction `$(max_length)`, step `$(ds)`")
    println(io, "- Fit arclength window: `$(fit_min_length)` to `$(fit_max_length)`")
    println(io, "- Box sizes: `$(join(box_sizes, ", "))`")
    println(io, "- Divider yardsticks: `$(join(yardsticks, ", "))`")
    println(io, "- Self-avoiding selection: coarse box `$(avoid_box_size)`, recurrence <= `$(max_recurrence)`, `Rg >= $(min_rg)`, endpoint distance >= `$(min_endpoint_distance)`")
    println(io)
    println(io, "## Estimates")
    println(io)
    println(io, "- Mean `length ~ Rg^D`: `$(round(mean(rg_dimensions); digits=4))`")
    println(io, "- Std `length ~ Rg^D`: `$(round(std(rg_dimensions); digits=4))`")
    println(io, "- Mean per-streamline divider dimension: `$(round(mean(divider_dimensions); digits=4))`")
    println(io, "- Std per-streamline divider dimension: `$(round(std(divider_dimensions); digits=4))`")
    println(io, "- Dimension from ensemble-mean divider counts: `$(round(divider_dimension_mean_count; digits=4))`")
    println(io, "- Mean per-streamline box dimension: `$(round(mean(box_dimensions); digits=4))`")
    println(io, "- Std per-streamline box dimension: `$(round(std(box_dimensions); digits=4))`")
    println(io, "- Dimension from ensemble-mean box counts: `$(round(box_dimension_mean_count; digits=4))`")
    println(io, "- Reference SAW value `4/3`: `$(round(4 / 3; digits=4))`")
    println(io)
    println(io, "## Self-avoiding/open subset")
    println(io)
    println(io, "- Selected streamlines: `$(selected_count)` of `$(length(paths))`")
    if selected_count > 0
        println(io, "- Mean `length ~ Rg^D`: `$(round(mean(selected_rg_dimensions); digits=4))`")
        println(io, "- Std `length ~ Rg^D`: `$(round(std(selected_rg_dimensions); digits=4))`")
        println(io, "- Mean per-streamline divider dimension: `$(round(mean(selected_divider_dimensions); digits=4))`")
        println(io, "- Dimension from selected ensemble-mean divider counts: `$(round(selected_divider_dimension_mean_count; digits=4))`")
        println(io, "- Mean per-streamline box dimension: `$(round(mean(selected_box_dimensions); digits=4))`")
        println(io, "- Dimension from selected ensemble-mean box counts: `$(round(selected_box_dimension_mean_count; digits=4))`")
    end
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("mean length-vs-Rg dimension: ", mean(rg_dimensions))
println("mean divider dimension: ", mean(divider_dimensions))
println("mean box dimension: ", mean(box_dimensions))
println("ensemble mean divider-count dimension: ", divider_dimension_mean_count)
println("ensemble mean box-count dimension: ", box_dimension_mean_count)
println("selected streamlines: ", selected_count, " / ", length(paths))
if selected_count > 0
    println("selected mean length-vs-Rg dimension: ", mean(selected_rg_dimensions))
    println("selected mean divider dimension: ", mean(selected_divider_dimensions))
    println("selected mean box dimension: ", mean(selected_box_dimensions))
    println("selected ensemble mean divider-count dimension: ", selected_divider_dimension_mean_count)
    println("selected ensemble mean box-count dimension: ", selected_box_dimension_mean_count)
end
