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

# Active snapshot demos need a smaller fixed step than the passive scripts.
const DEFAULT_ACTIVE_SNAPSHOT_DT = 2.0^-10
const DEFAULT_SNAPSHOT_DISPLACEMENTS = (0.0, 0.25, 0.5, 0.75)
const DEFAULT_CALIBRATION_TIME = 2.0

function parse_solver(name::AbstractString)
    solver_name = lowercase(name)
    solver_name in ("auto", "em", "sriw1") ||
        throw(ArgumentError("solver must be auto, em, or sriw1"))
    return solver_name
end

function solve_snapshot(theta0, tspan, params, work, solver_name, dt, saveat, save_start, seed)
    if iszero(params.Q)
        problem = ODEProblem(LatticeFlockingSDE.drift!, theta0, tspan, work)
        return solve(problem, Tsit5(); saveat, save_start)
    end

    problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta0, tspan, work)
    if solver_name == "em"
        return solve(problem, EM(); dt, adaptive=false, saveat, save_start, seed)
    end
    return solve(problem, SRIW1(); dt, adaptive=false, saveat, save_start, seed)
end

function calibrated_snapshot_times(theta0, params, work, dt, radius)
    calibration_solution = solve_snapshot(theta0, (0.0, DEFAULT_CALIBRATION_TIME), params, work,
        "auto", dt, [DEFAULT_CALIBRATION_TIME], false, 0)
    x0 = positive_sin_marker_x(theta0, params.L)
    x1 = positive_sin_marker_x(calibration_solution.u[end], params.L)
    drift_speed = (x1 - x0) / DEFAULT_CALIBRATION_TIME
    drift_speed > 0 || throw(ErrorException("calibrated drift speed must be positive, got $(drift_speed)"))
    times = [displacement * radius / drift_speed for displacement in DEFAULT_SNAPSHOT_DISPLACEMENTS]
    return times, drift_speed
end

settings = ArgParseSettings(
    description="Save four low-temperature advection snapshots with a localized upward bump on a right-pointing background.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 200
    "--Q"
        arg_type = Float64
        default = 1e-4
    "--J"
        arg_type = Float64
        default = 8.0
    "--v"
        arg_type = Float64
        default = 2.0
    "--dt"
        arg_type = Float64
        default = DEFAULT_ACTIVE_SNAPSHOT_DT
    "--seed"
        arg_type = Int
        default = 1
    "--solver"
        arg_type = String
        default = "auto"
    "--radius"
        arg_type = Float64
        default = NaN
    "--transition-width"
        arg_type = Float64
        default = NaN
    "--center-x"
        arg_type = Float64
        default = NaN
    "--center-y"
        arg_type = Float64
        default = NaN
    "--block-size"
        arg_type = Int
        default = 10
    "--times"
        arg_type = String
        default = ""
    "--output"
        arg_type = String
        default = "results/snapshot_dataset_L200_v2.jld2"
    "--figure"
        arg_type = String
        default = ""
    "--arrow-scale"
        arg_type = Float64
        default = 7.0
    "--arrow-size"
        arg_type = Float64
        default = 4.0
end
args = ArgParse.parse_args(settings)

L = args["L"]
Q = args["Q"]
J = args["J"]
v = args["v"]
dt = args["dt"]
seed = args["seed"]
block_size = args["block-size"]
radius = isnan(args["radius"]) ? L / 10 : args["radius"]
transition_width = isnan(args["transition-width"]) ? L / 40 : args["transition-width"]
center_x = isnan(args["center-x"]) ? L / 4 : args["center-x"]
center_y = isnan(args["center-y"]) ? L / 2 : args["center-y"]
output = args["output"]
solver_name = parse_solver(args["solver"])
figure_output = args["figure"]
arrow_scale = args["arrow-scale"]
arrow_size = args["arrow-size"]

dt > 0 || throw(ArgumentError("dt must be positive"))
block_size > 0 || throw(ArgumentError("block-size must be positive"))
L % block_size == 0 || throw(ArgumentError("L must be divisible by block-size"))
radius > 0 || throw(ArgumentError("radius must be positive"))
transition_width > 0 || throw(ArgumentError("transition-width must be positive"))
arrow_scale > 0 || throw(ArgumentError("arrow-scale must be positive"))
arrow_size > 0 || throw(ArgumentError("arrow-size must be positive"))

params = ModelParams(; L, Q, J, v)
theta0 = zeros(Float64, L * L)  # background: all spins pointing right (θ = 0)
seed_upward_bump!(theta0, L; center_x, center_y, radius, transition_width)

work = LatticeFlockingSDE.DriftWorkspace(params)
times, drift_speed = if isempty(args["times"])
    calibrated_snapshot_times(theta0, params, work, dt, radius)
else
    parsed = parse.(Float64, split(args["times"], ","))
    length(parsed) == 4 || throw(ArgumentError("--times must contain exactly four comma-separated values"))
    sort(parsed), NaN
end
all(t -> t >= 0, times) || throw(ArgumentError("snapshot times must be nonnegative"))

solution = solve_snapshot(theta0, (0.0, maximum(times)), params, work, solver_name, dt,
    times, iszero(first(times)), seed)

nsnapshots = length(times)
length(solution.u) == nsnapshots ||
    throw(ErrorException("solver returned $(length(solution.u)) snapshots for $(nsnapshots) requested times"))
theta_snapshots = zeros(Float64, L, L, nsnapshots)
spin_x_snapshots = zeros(Float64, L, L, nsnapshots)
spin_y_snapshots = zeros(Float64, L, L, nsnapshots)
x_centroids = zeros(Float64, nsnapshots)

for (k, state) in enumerate(solution.u)
    theta = wrap_angles!(collect(state))
    theta_grid = reshape(theta, L, L)
    theta_snapshots[:, :, k] .= theta_grid
    spin_x_snapshots[:, :, k] .= cos.(theta_grid)
    spin_y_snapshots[:, :, k] .= sin.(theta_grid)
    x_centroids[k] = positive_sin_marker_x(theta, L)
end

nblocks = L ÷ block_size
block_x = zeros(Float64, nblocks, nblocks)
block_y = zeros(Float64, nblocks, nblocks)
block_u = zeros(Float64, nblocks, nblocks, nsnapshots)
block_v = zeros(Float64, nblocks, nblocks, nsnapshots)

@inbounds for by in 1:nblocks, bx in 1:nblocks
    xlo = (bx - 1) * block_size + 1
    xhi = bx * block_size
    ylo = (by - 1) * block_size + 1
    yhi = by * block_size
    block_x[bx, by] = (xlo + xhi) / 2
    block_y[bx, by] = (ylo + yhi) / 2

    for k in 1:nsnapshots
        block_u[bx, by, k] = mean(@view spin_x_snapshots[xlo:xhi, ylo:yhi, k])
        block_v[bx, by, k] = mean(@view spin_y_snapshots[xlo:xhi, ylo:yhi, k])
    end
end

metadata = (;
    center=(center_x, center_y),
    radius,
    transition_width,
    block_size,
    solver=solver_name,
    default_dt=DEFAULT_ACTIVE_SNAPSHOT_DT,
    default_displacements=collect(DEFAULT_SNAPSHOT_DISPLACEMENTS),
    calibrated_drift_speed=drift_speed,
    initialization=:right_background_upward_circular_perturbation,
)
dataset = (;
    params,
    seed,
    times=collect(solution.t),
    x_centroids,
    theta_snapshots,
    spin_x_snapshots,
    spin_y_snapshots,
    block_x,
    block_y,
    block_u,
    block_v,
    metadata,
)

mkpath(dirname(output))
jldsave(output; dataset)

figure_output = isempty(figure_output) ?
    replace(output, r"^results/" => "figures/") |> x -> replace(x, r"\.jld2$" => ".png") :
    figure_output

fig = Figure(size=(1500, 430), backgroundcolor=:white)
for k in 1:length(times)
    ax = Axis(
        fig[1, k],
        aspect=DataAspect(),
        title="t = $(round(times[k]; digits=2))\nx = $(round(x_centroids[k]; digits=2))",
        xlabel="x",
        ylabel=k == 1 ? "y" : "",
    )
    heatmap!(ax, 1:L, 1:L, theta_snapshots[:, :, k];
        colormap=:hsv, colorrange=(0, 2π))
    arrows2d!(
        ax,
        vec(block_x),
        vec(block_y),
        arrow_scale .* vec(block_u[:, :, k]),
        arrow_scale .* vec(block_v[:, :, k]);
        color=:black,
        tipwidth=arrow_size,
        tiplength=1.5arrow_size,
        shaftwidth=1.2,
        lengthscale=1,
    )
    xlims!(ax, 1, L)
    ylims!(ax, 1, L)
end

Colorbar(fig[2, 1:4], limits=(0, 2π), colormap=:hsv, vertical=false,
    label="spin angle θ")
mkpath(dirname(figure_output))
save(figure_output, fig)

println("saved: ", output)
println("saved figure: ", figure_output)
println("times: ", collect(solution.t))
println("x centroids: ", x_centroids)
println("theta_snapshots size: ", size(theta_snapshots))
println("block arrow size: ", size(block_u))
