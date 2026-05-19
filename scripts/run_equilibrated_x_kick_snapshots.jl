#!/usr/bin/env julia

using ArgParse
using JLD2
using Random
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(
    description="Equilibrate a state, rotate every rotor toward theta=0, and save snapshots.",
)
@add_arg_table! settings begin
    "--L"
        arg_type = Int
        default = 200
    "--Q"
        arg_type = Float64
        default = 0.05
    "--J"
        arg_type = Float64
        default = 2.0
    "--v"
        arg_type = Float64
        default = 10.0
    "--dt"
        arg_type = Float64
        default = 0.01
    "--equilibration-time"
        arg_type = Float64
        default = 50.0
    "--rotation-fraction"
        arg_type = Float64
        default = 0.5
    "--seed"
        arg_type = Int
        default = 1
    "--init"
        arg_type = String
        default = "random"
    "--block-size"
        arg_type = Int
        default = 10
    "--times"
        arg_type = String
        default = "0,4,8,16"
    "--output"
        arg_type = String
        default = "results/equilibrated_right_rotation_L200_J2_Q1_v1.jld2"
end
args = ArgParse.parse_args(settings)

L = args["L"]
Q = args["Q"]
J = args["J"]
v = args["v"]
dt = args["dt"]
equilibration_time = args["equilibration-time"]
rotation_fraction = args["rotation-fraction"]
seed = args["seed"]
initial_condition = Symbol(args["init"])
block_size = args["block-size"]
times = parse.(Float64, split(args["times"], ","))
output = args["output"]

dt > 0 || throw(ArgumentError("dt must be positive"))
equilibration_time >= 0 || throw(ArgumentError("equilibration-time must be nonnegative"))
block_size > 0 || throw(ArgumentError("block-size must be positive"))
L % block_size == 0 || throw(ArgumentError("L must be divisible by block-size"))
length(times) == 4 || throw(ArgumentError("--times must contain exactly four comma-separated values"))
times = sort(times)
all(t -> t >= 0, times) || throw(ArgumentError("snapshot times must be nonnegative"))
0 <= rotation_fraction <= 1 ||
    throw(ArgumentError("rotation-fraction must be between 0 and 1"))
initial_condition in (:random, :ordered) ||
    throw(ArgumentError("init must be random or ordered"))

params = ModelParams(; L, Q, J, v)
rng = MersenneTwister(seed)
theta = initial_angles(rng, L, initial_condition)
work = LatticeFlockingSDE.DriftWorkspace(params)

if equilibration_time > 0
    equilibration_problem = SDEProblem(
        LatticeFlockingSDE.drift!,
        LatticeFlockingSDE.noise!,
        theta,
        (0.0, equilibration_time),
        work,
    )
    equilibration_solution = solve(equilibration_problem, EM(); dt, adaptive=false,
        save_everystep=false, save_start=false, seed)
    theta = wrap_angles!(collect(equilibration_solution.u[end]))
end

theta_equilibrated = copy(theta)
twoπ = 2π
@inbounds for i in eachindex(theta)
    delta = atan(-sin(theta[i]), cos(theta[i]))
    theta[i] = mod(theta[i] + rotation_fraction * delta, twoπ)
end
theta_rotated = copy(theta)

snapshot_problem = SDEProblem(
    LatticeFlockingSDE.drift!,
    LatticeFlockingSDE.noise!,
    theta,
    (0.0, maximum(times)),
    work,
)
snapshot_solution = solve(snapshot_problem, EM(); dt, adaptive=false, saveat=times,
    save_start=iszero(first(times)), seed=seed + 1)

nsnapshots = length(times)
length(snapshot_solution.u) == nsnapshots ||
    throw(ErrorException("solver returned $(length(snapshot_solution.u)) snapshots for $(nsnapshots) requested times"))
theta_snapshots = zeros(Float64, L, L, nsnapshots)
spin_x_snapshots = zeros(Float64, L, L, nsnapshots)
spin_y_snapshots = zeros(Float64, L, L, nsnapshots)

for (k, state) in enumerate(snapshot_solution.u)
    theta_state = wrap_angles!(collect(state))
    theta_grid = reshape(theta_state, L, L)
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
    initialization=:equilibrated_uniform_rotation_toward_right,
    initial_condition,
    equilibration_time,
    rotation_fraction,
    block_size,
)
dataset = (;
    params,
    seed,
    times=collect(snapshot_solution.t),
    theta_equilibrated=reshape(theta_equilibrated, L, L),
    theta_rotated=reshape(theta_rotated, L, L),
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
println("theta_snapshots size: ", size(theta_snapshots))
println("block arrow size: ", size(block_u))
println("mean cos before rotation: ", mean(cos, theta_equilibrated))
println("mean cos after rotation: ", mean(cos, theta_rotated))
