#!/usr/bin/env julia

using ArgParse
using JLD2
using Random
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(
    description="Save four angle-field snapshots with right-pointing background and an upward circular perturbation.", 
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
        default = 2.0
    "--dt"
        arg_type = Float64
        default = 0.01
    "--seed"
        arg_type = Int
        default = 1
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

dt > 0 || throw(ArgumentError("dt must be positive"))
block_size > 0 || throw(ArgumentError("block-size must be positive"))
L % block_size == 0 || throw(ArgumentError("L must be divisible by block-size"))
radius > 0 || throw(ArgumentError("radius must be positive"))
transition_width > 0 || throw(ArgumentError("transition-width must be positive"))

times = if isempty(args["times"])
    v > 0 || throw(ArgumentError("v must be positive when --times is omitted"))
    [0.0, L / (4v), L / (2v), 3L / (4v)]
else
    parsed = parse.(Float64, split(args["times"], ","))
    length(parsed) == 4 || throw(ArgumentError("--times must contain exactly four comma-separated values"))
    sort(parsed)
end
all(t -> t >= 0, times) || throw(ArgumentError("snapshot times must be nonnegative"))

params = ModelParams(; L, Q, J, v)
rng = MersenneTwister(seed)
theta0 = zeros(Float64, L * L)  # background: all spins pointing right (θ = 0)
twoπ = 2π

@inbounds for y in 1:L, x in 1:L
    idx = site_index(x, y, L)
    dx = x - center_x
    dy = y - center_y
    r = sqrt(dx^2 + dy^2)
    weight = 0.5 * (1 - tanh((r - radius) / transition_width))
    # blend from background (θ=0) toward upward pointing (θ=π/2) inside the circle
    c = (1 - weight) * cos(theta0[idx]) + weight * cos(pi/2)
    s = (1 - weight) * sin(theta0[idx]) + weight * sin(pi/2)
    theta0[idx] = mod(atan(s, c), twoπ)
end

work = LatticeFlockingSDE.DriftWorkspace(params)
problem = SDEProblem(
    LatticeFlockingSDE.drift!,
    LatticeFlockingSDE.noise!,
    theta0,
    (0.0, maximum(times)),
    work,
)
solution = solve(problem, EM(); dt, adaptive=false, saveat=times,
    save_start=iszero(first(times)), seed)

nsnapshots = length(times)
length(solution.u) == nsnapshots ||
    throw(ErrorException("solver returned $(length(solution.u)) snapshots for $(nsnapshots) requested times"))
theta_snapshots = zeros(Float64, L, L, nsnapshots)
spin_x_snapshots = zeros(Float64, L, L, nsnapshots)
spin_y_snapshots = zeros(Float64, L, L, nsnapshots)

for (k, state) in enumerate(solution.u)
    theta = wrap_angles!(collect(state))
    theta_grid = reshape(theta, L, L)
    theta_snapshots[:, :, k] .= theta_grid
    spin_x_snapshots[:, :, k] .= cos.(theta_grid)
    spin_y_snapshots[:, :, k] .= sin.(theta_grid)
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
    initialization=:right_background_upward_circular_perturbation,
)
dataset = (;
    params,
    seed,
    times=collect(solution.t),
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
println("times: ", collect(solution.t))
println("theta_snapshots size: ", size(theta_snapshots))
println("block arrow size: ", size(block_u))
