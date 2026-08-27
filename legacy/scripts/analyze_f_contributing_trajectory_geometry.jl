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

function weighted_mean(values, weights)
    total = sum(weights)
    total > 0 || return mean(values)
    return sum(values .* weights) / total
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

function closest_streamline_arclength(theta_grid, x, y, target_x, target_y, ds,
        max_arclength, direction_sign)
    nsteps = round(Int, max_arclength / ds)
    best_s = 0.0
    best_d2 = (x - target_x)^2 + (y - target_y)^2
    for step in 1:nsteps
        x, y = rk4_step(theta_grid, x, y, ds, direction_sign)
        d2 = (x - target_x)^2 + (y - target_y)^2
        if d2 < best_d2
            best_d2 = d2
            best_s = step * ds
        end
    end
    return best_s, sqrt(best_d2)
end

function arm_f(theta_minus, theta_mid, theta_plus, params, x, y, r, direction_sign)
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

    if direction_sign > 0
        plus_value = interpolated_spin_dot(cos_plus, sin_plus, L, x + dx, y + dy, cx, sy)
        minus_value = interpolated_spin_dot(cos_minus, sin_minus, L, x + dx, y + dy, cx, sy)
        return 0.25 * (plus_value - minus_value) / nsites, x + dx, y + dy
    end
    minus_value = interpolated_spin_dot(cos_minus, sin_minus, L, x - dx, y - dy, cx, sy)
    plus_value = interpolated_spin_dot(cos_plus, sin_plus, L, x - dx, y - dy, cx, sy)
    return 0.25 * (minus_value - plus_value) / nsites, x - dx, y - dy
end

settings = ArgParseSettings(
    description="Measure the streamline arclength geometry of source-endpoint pairs that contribute to spin-aligned F.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 96
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
        default = 12
    "--dr"
        arg_type = Float64
        default = 4.0
    "--rmax"
        arg_type = Float64
        default = 32.0
    "--ds"
        arg_type = Float64
        default = 0.5
    "--max-arc-factor"
        arg_type = Float64
        default = 4.0
    "--hit-tolerance"
        arg_type = Float64
        default = 2.0
    "--top-f-fraction"
        arg_type = Float64
        default = 0.1
    "--fit-min-r"
        arg_type = Float64
        default = 8.0
    "--output-prefix"
        arg_type = String
        default = "f_contributing_trajectory_geometry"
end
args = parse_args(settings)

solver_name = parse_solver(args["solver"])
initial_condition = Symbol(args["init"])
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

params = ModelParams(; L=args["L"], Q=args["Q"], J=args["J"], v=args["v"])
L = params.L
radii = collect(args["dr"]:args["dr"]:args["rmax"])
rng = MersenneTwister(args["seed"])
theta = initial_angles(rng, L, initial_condition)
theta = evolve_state(theta, args["burnin-time"], params, solver_name, args["dt"], rng)

xs = range(1, L; length=args["seed-grid"] + 2)[2:end - 1]
ys = range(1, L; length=args["seed-grid"] + 2)[2:end - 1]
all_s = [Float64[] for _ in radii]
all_d = [Float64[] for _ in radii]
all_f = [Float64[] for _ in radii]
all_signed_f = [Float64[] for _ in radii]
hit_s = [Float64[] for _ in radii]
hit_f = [Float64[] for _ in radii]

for window_index in 1:args["nwindows"]
    global theta
    theta_minus = copy(theta)
    theta_mid = evolve_state(theta_minus, args["lag-time"], params, solver_name, args["dt"], rng)
    theta_plus = evolve_state(theta_mid, args["lag-time"], params, solver_name, args["dt"], rng)
    theta_grid = reshape(theta_mid, L, L)

    for (ridx, r) in enumerate(radii)
        max_arc = args["max-arc-factor"] * r^(4 / 3)
        for y in ys, x in xs
            for direction_sign in (1.0, -1.0)
                f, target_x, target_y = arm_f(theta_minus, theta_mid, theta_plus, params,
                    x, y, r, direction_sign)
                s, distance = closest_streamline_arclength(theta_grid, x, y, target_x,
                    target_y, args["ds"], max_arc, direction_sign)
                push!(all_s[ridx], s)
                push!(all_d[ridx], distance)
                push!(all_f[ridx], abs(f))
                push!(all_signed_f[ridx], f)
                if distance <= args["hit-tolerance"]
                    push!(hit_s[ridx], s)
                    push!(hit_f[ridx], abs(f))
                end
            end
        end
    end
    theta = theta_plus
    @info "processed F-geometry window" window_index total=args["nwindows"]
end

mean_s = [mean(values) for values in all_s]
weighted_s = [weighted_mean(all_s[i], all_f[i]) for i in eachindex(radii)]
signed_f = [Float64[] for _ in radii]
for i in eachindex(radii)
    total_signed = sum(all_signed_f[i])
    coherent_sign = iszero(total_signed) ? 1.0 : sign(total_signed)
    signed_f[i] = max.(coherent_sign .* all_signed_f[i], 0.0)
end
coherent_s = [weighted_mean(all_s[i], signed_f[i]) for i in eachindex(radii)]
hit_weighted_s = [
    isempty(hit_s[i]) ? NaN : weighted_mean(hit_s[i], hit_f[i])
    for i in eachindex(radii)
]
hit_fraction = [count(<=(args["hit-tolerance"]), all_d[i]) / length(all_d[i])
    for i in eachindex(radii)]
top_s = Float64[]
for i in eachindex(radii)
    n = length(all_f[i])
    top_count = max(1, round(Int, args["top-f-fraction"] * n))
    top_indices = sortperm(all_f[i]; rev=true)[1:top_count]
    push!(top_s, mean(all_s[i][top_indices]))
end

fit_mask = radii .>= args["fit-min-r"]
df_all, intercept_all = linear_fit(log.(radii[fit_mask]), log.(mean_s[fit_mask]))
df_weighted, intercept_weighted =
    linear_fit(log.(radii[fit_mask]), log.(weighted_s[fit_mask]))
df_coherent, intercept_coherent =
    linear_fit(log.(radii[fit_mask]), log.(coherent_s[fit_mask]))
df_top, intercept_top = linear_fit(log.(radii[fit_mask]), log.(top_s[fit_mask]))
hit_fit_mask = fit_mask .& isfinite.(hit_weighted_s)
df_hit_weighted, intercept_hit_weighted =
    linear_fit(log.(radii[hit_fit_mask]), log.(hit_weighted_s[hit_fit_mask]))

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1280, 620), backgroundcolor=:white)
ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="F endpoint Euclidean r",
    ylabel="closest streamline arclength s_hit",
    title="Geometry of F-contributing source-endpoint pairs")
scatterlines!(ax1, radii, mean_s; color=:gray45, label="all pairs")
scatterlines!(ax1, radii, weighted_s; color=:red, label="abs(F)-weighted")
scatterlines!(ax1, radii, coherent_s; color=:purple, label="sign-coherent F-weighted")
scatterlines!(ax1, radii, top_s; color=:black, label="top abs(F) pairs")
scatterlines!(ax1, radii[isfinite.(hit_weighted_s)], hit_weighted_s[isfinite.(hit_weighted_s)];
    color=:darkorange, label="hit pairs, abs(F)-weighted")
lines!(ax1, radii, exp.(intercept_weighted .+ df_weighted .* log.(radii));
    color=:red, linewidth=2, label="weighted d_f=$(round(df_weighted; digits=3))")
lines!(ax1, radii,
    exp.(mean(log.(weighted_s[fit_mask]) .- (4 / 3) .* log.(radii[fit_mask])) .+
        (4 / 3) .* log.(radii));
    color=:dodgerblue, linestyle=:dash, linewidth=2, label="SAW d_f=4/3")
axislegend(ax1; position=:lt)

ax2 = Axis(fig[1, 2], xlabel="r", ylabel="fraction",
    title="Endpoint lies within $(args["hit-tolerance"]) of traced streamline")
scatterlines!(ax2, radii, hit_fraction; color=:black)
ylims!(ax2, 0, min(1, maximum(hit_fraction) * 1.15 + 0.02))

save(figure_output, fig)

jldsave(data_output; args, params, radii, all_s, all_d, all_f, all_signed_f, hit_s, hit_f,
    mean_s, weighted_s, top_s, hit_weighted_s, hit_fraction, fit_mask,
    coherent_s, signed_f, df_all, df_weighted, df_coherent, df_top, df_hit_weighted)

open(summary_output, "w") do io
    println(io, "# F-contributing trajectory geometry")
    println(io)
    println(io, "- Observable: midpoint spin streamline arclength needed to approach the Euclidean endpoint sampled by spin-aligned `F`.")
    println(io, "- Parameters: `L=$(params.L)`, `Q=$(params.Q)`, `J=$(params.J)`, `v=$(params.v)`, lag time `$(args["lag-time"])`")
    println(io, "- Windows: `$(args["nwindows"])`, seed grid `$(args["seed-grid"])x$(args["seed-grid"])`, arms per radius `$(length(all_s[1]))`")
    println(io, "- Fit: `s_hit(r) ~ r^d_f`, using `r >= $(args["fit-min-r"])`")
    println(io, "- Hit rule: closest streamline distance <= `$(args["hit-tolerance"])`")
    println(io)
    println(io, "## Estimates")
    println(io)
    println(io, "- All pairs: `d_f = $(round(df_all; digits=5))`")
    println(io, "- `abs(F)`-weighted pairs: `d_f = $(round(df_weighted; digits=5))`")
    println(io, "- Sign-coherent `F`-weighted pairs: `d_f = $(round(df_coherent; digits=5))`")
    println(io, "- Top `abs(F)` fraction $(args["top-f-fraction"]): `d_f = $(round(df_top; digits=5))`")
    println(io, "- Hit-only `abs(F)`-weighted pairs: `d_f = $(round(df_hit_weighted; digits=5))`")
    println(io, "- Reference SAW: `d_f = 1.33333`")
    println(io)
    println(io, "| r | mean s_hit | abs(F)-weighted s_hit | coherent F-weighted s_hit | top abs(F) s_hit | hit abs(F)-weighted s_hit | hit fraction |")
    println(io, "|---:|---:|---:|---:|---:|---:|---:|")
    for i in eachindex(radii)
        hit_text = isfinite(hit_weighted_s[i]) ? string(round(hit_weighted_s[i]; digits=4)) : "NaN"
        println(io, "| $(radii[i]) | $(round(mean_s[i]; digits=4)) | $(round(weighted_s[i]; digits=4)) | $(round(coherent_s[i]; digits=4)) | $(round(top_s[i]; digits=4)) | $(hit_text) | $(round(hit_fraction[i]; digits=4)) |")
    end
end

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("all df: ", df_all)
println("weighted df: ", df_weighted)
println("coherent df: ", df_coherent)
println("top df: ", df_top)
println("hit weighted df: ", df_hit_weighted)
