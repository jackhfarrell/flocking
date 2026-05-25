#!/usr/bin/env julia

using ArgParse
using DifferentialEquations
using JLD2
using Random
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

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

function evolve_state(theta0, tspan, params, work, solver_name, dt, seed)
    if iszero(params.Q)
        problem = ODEProblem(LatticeFlockingSDE.drift!, theta0, tspan, work)
        solution = solve(problem, Tsit5(); save_everystep=false, save_start=false)
        return wrap_angles!(collect(solution.u[end]))
    end

    problem = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!, theta0, tspan, work)
    if solver_name == "em"
        solution = solve(problem, EM(); dt, adaptive=false, save_everystep=false,
            save_start=false, seed)
    else
        solution = solve(problem, SRIW1(); dt, adaptive=false, save_everystep=false,
            save_start=false, seed)
    end
    return wrap_angles!(collect(solution.u[end]))
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
    description="Equilibrate a background, insert a localized upward bump, and save snapshots in the harder fluctuating-background regime.",
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
    "--equilibration-time"
        arg_type = Float64
        default = 50.0
    "--seed"
        arg_type = Int
        default = 1
    "--init"
        arg_type = String
        default = "random"
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
        default = "results/equilibrated_local_bump_L200_v2.jld2"
end
args = ArgParse.parse_args(settings)

L = args["L"]
Q = args["Q"]
J = args["J"]
v = args["v"]
dt = args["dt"]
equilibration_time = args["equilibration-time"]
seed = args["seed"]
initial_condition = Symbol(args["init"])
solver_name = parse_solver(args["solver"])
radius = isnan(args["radius"]) ? L / 10 : args["radius"]
transition_width = isnan(args["transition-width"]) ? L / 40 : args["transition-width"]
center_x = isnan(args["center-x"]) ? L / 4 : args["center-x"]
center_y = isnan(args["center-y"]) ? L / 2 : args["center-y"]
block_size = args["block-size"]
output = args["output"]

dt > 0 || throw(ArgumentError("dt must be positive"))
equilibration_time >= 0 || throw(ArgumentError("equilibration-time must be nonnegative"))
block_size > 0 || throw(ArgumentError("block-size must be positive"))
L % block_size == 0 || throw(ArgumentError("L must be divisible by block-size"))
radius > 0 || throw(ArgumentError("radius must be positive"))
transition_width > 0 || throw(ArgumentError("transition-width must be positive"))
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

params = ModelParams(; L, Q, J, v)
rng = MersenneTwister(seed)
theta = initial_angles(rng, L, initial_condition)
work = LatticeFlockingSDE.DriftWorkspace(params)

if equilibration_time > 0
    theta = evolve_state(theta, (0.0, equilibration_time), params, work, solver_name, dt, seed)
end

theta_equilibrated = copy(theta)
seed_upward_bump!(theta, L; center_x, center_y, radius, transition_width)
theta_kicked = copy(theta)

times, drift_speed = if isempty(args["times"])
    calibrated_snapshot_times(theta, params, work, dt, radius)
else
    parsed = parse.(Float64, split(args["times"], ","))
    length(parsed) == 4 || throw(ArgumentError("--times must contain exactly four comma-separated values"))
    sort(parsed), NaN
end
all(t -> t >= 0, times) || throw(ArgumentError("snapshot times must be nonnegative"))

snapshot_solution = solve_snapshot(theta, (0.0, maximum(times)), params, work, solver_name, dt,
    times, iszero(first(times)), seed + 1)

nsnapshots = length(times)
length(snapshot_solution.u) == nsnapshots ||
    throw(ErrorException("solver returned $(length(snapshot_solution.u)) snapshots for $(nsnapshots) requested times"))
theta_snapshots = zeros(Float64, L, L, nsnapshots)
spin_x_snapshots = zeros(Float64, L, L, nsnapshots)
spin_y_snapshots = zeros(Float64, L, L, nsnapshots)
x_centroids = zeros(Float64, nsnapshots)

for (k, state) in enumerate(snapshot_solution.u)
    theta_state = wrap_angles!(collect(state))
    theta_grid = reshape(theta_state, L, L)
    theta_snapshots[:, :, k] .= theta_grid
    spin_x_snapshots[:, :, k] .= cos.(theta_grid)
    spin_y_snapshots[:, :, k] .= sin.(theta_grid)
    x_centroids[k] = positive_sin_marker_x(theta_state, L)
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
    initialization=:equilibrated_background_localized_upward_bump,
    initial_condition,
    equilibration_time,
    center=(center_x, center_y),
    radius,
    transition_width,
    block_size,
    solver=solver_name,
    default_dt=DEFAULT_ACTIVE_SNAPSHOT_DT,
    default_displacements=collect(DEFAULT_SNAPSHOT_DISPLACEMENTS),
    calibrated_drift_speed=drift_speed,
)
dataset = (;
    params,
    seed,
    times=collect(snapshot_solution.t),
    theta_equilibrated=reshape(theta_equilibrated, L, L),
    theta_kicked=reshape(theta_kicked, L, L),
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

println("saved: ", output)
println("times: ", collect(snapshot_solution.t))
println("x centroids: ", x_centroids)
println("theta_snapshots size: ", size(theta_snapshots))
println("block arrow size: ", size(block_u))
println("mean cos before kick: ", mean(cos, theta_equilibrated))
println("mean cos after kick: ", mean(cos, theta_kicked))
