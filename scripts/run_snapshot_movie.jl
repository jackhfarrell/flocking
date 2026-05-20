#!/usr/bin/env julia

using ArgParse
using CairoMakie
using DifferentialEquations
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

struct MovieWorkspace
    drift::LatticeFlockingSDE.DriftWorkspace
    noise_scale::Float64
end

# Active snapshot demos need a smaller fixed step than the passive scripts.
const DEFAULT_ACTIVE_SNAPSHOT_DT = 2.0^-9

function movie_drift!(du, theta, work::MovieWorkspace, t)
    return LatticeFlockingSDE.drift!(du, theta, work.drift, t)
end

function movie_noise!(du, theta, work::MovieWorkspace, t)
    fill!(du, sqrt(work.noise_scale * work.drift.params.Q))
    return du
end

signed_angle(theta) = atan(sin(theta), cos(theta))

settings = ArgParseSettings(
    description="Seed a circular perturbation on a uniform or pre-equilibrated background and save an animation of the angle field.",
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
        default = 1.1
    "--v"
        arg_type = Float64
        default = 10.0
    "--dt"
        arg_type = Float64
        default = DEFAULT_ACTIVE_SNAPSHOT_DT
    "--tmax"
        arg_type = Float64
        default = 75.0
    "--equilibration-time"
        arg_type = Float64
        default = 0.0
    "--nframes"
        arg_type = Int
        default = 151
    "--seed"
        arg_type = Int
        default = 1
    "--noise-scale"
        arg_type = Float64
        default = 1.0
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
    "--output"
        arg_type = String
        default = "figures/snapshot_dataset/snapshot_movie.mp4"
    "--log-every"
        arg_type = Int
        default = 25
end
args = ArgParse.parse_args(settings)

L = args["L"]
Q = args["Q"]
J = args["J"]
v = args["v"]
dt = args["dt"]
tmax = args["tmax"]
equilibration_time = args["equilibration-time"]
nframes = args["nframes"]
seed = args["seed"]
noise_scale = args["noise-scale"]
radius = isnan(args["radius"]) ? L / 10 : args["radius"]
transition_width = isnan(args["transition-width"]) ? L / 40 : args["transition-width"]
center_x = isnan(args["center-x"]) ? L / 4 : args["center-x"]
center_y = isnan(args["center-y"]) ? L / 2 : args["center-y"]
output = args["output"]
log_every = args["log-every"]

dt > 0 || throw(ArgumentError("dt must be positive"))
tmax > 0 || throw(ArgumentError("tmax must be positive"))
equilibration_time >= 0 || throw(ArgumentError("equilibration-time must be nonnegative"))
nframes > 1 || throw(ArgumentError("nframes must be greater than 1"))
noise_scale >= 0 || throw(ArgumentError("noise-scale must be nonnegative"))
radius > 0 || throw(ArgumentError("radius must be positive"))
transition_width > 0 || throw(ArgumentError("transition-width must be positive"))
log_every > 0 || throw(ArgumentError("log-every must be positive"))

params = ModelParams(; L, Q, J, v)
twoπ = 2π
movie_work = MovieWorkspace(LatticeFlockingSDE.DriftWorkspace(params), noise_scale)

theta0 = zeros(Float64, L * L)

base_theta = if equilibration_time > 0
    if iszero(noise_scale)
        equilibration_problem = ODEProblem(
            movie_drift!,
            theta0,
            (0.0, equilibration_time),
            movie_work,
        )
        equilibration_solution = solve(equilibration_problem, Tsit5();
            save_everystep=false, save_start=false)
    else
        equilibration_problem = SDEProblem(
            movie_drift!,
            movie_noise!,
            theta0,
            (0.0, equilibration_time),
            movie_work,
        )
        equilibration_solution = solve(equilibration_problem, EM(); dt, adaptive=false,
            save_everystep=false, save_start=false, seed=seed)
    end
    wrap_angles!(collect(equilibration_solution.u[end]))
else
    zeros(Float64, L * L)
end

theta0 = copy(base_theta)
@inbounds for y in 1:L, x in 1:L
    idx = site_index(x, y, L)
    dx = x - center_x
    dy = y - center_y
    r = sqrt(dx^2 + dy^2)
    weight = 0.5 * (1 - tanh((r - radius) / transition_width))
    c = (1 - weight) * cos(base_theta[idx]) + weight * cos(pi / 2)
    s = (1 - weight) * sin(base_theta[idx]) + weight * sin(pi / 2)
    theta0[idx] = mod(atan(s, c), twoπ)
end

times = collect(range(0.0, tmax; length=nframes))
println("starting solve: tmax=", tmax, " dt=", dt, " nframes=", nframes,
    " noise_scale=", noise_scale)
solution = if iszero(noise_scale)
    problem = ODEProblem(
        movie_drift!,
        theta0,
        (0.0, tmax),
        movie_work,
    )
    solve(problem, Tsit5(); saveat=times, save_start=true)
else
    problem = SDEProblem(
        movie_drift!,
        movie_noise!,
        theta0,
        (0.0, tmax),
        movie_work,
    )
    solve(problem, SRIW1(); dt, adaptive=false, saveat=times, save_start=true,
        seed=seed + 1)
end
println("finished solve")

length(solution.u) == length(times) ||
    throw(ErrorException("solver returned $(length(solution.u)) snapshots for $(length(times)) requested times"))

theta_snapshots = Array{Float64}(undef, L, L, length(times))
for (k, state) in enumerate(solution.u)
    theta_snapshots[:, :, k] .= reshape(signed_angle.(state), L, L)
end

fig = Figure(size=(900, 900), backgroundcolor=:white)
ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
theta_obs = Observable(theta_snapshots[:, :, 1])
heatmap!(ax, 1:L, 1:L, theta_obs; colormap=:viridis, colorrange=(-pi, pi))
ax.title = "t = $(round(times[1]; digits=2))"
Colorbar(fig[2, 1], limits=(-pi, pi), colormap=:viridis, vertical=false,
    label="signed angle θ")

mkpath(dirname(output))
record(fig, output, 1:length(times); framerate=24) do k
    theta_obs[] = theta_snapshots[:, :, k]
    ax.title = "t = $(round(times[k]; digits=2))"
    if k == 1 || k == length(times) || (k % log_every == 0)
        println("render frame ", k, "/", length(times), " t=", round(times[k]; digits=3))
    end
end

println("saved: ", output)
