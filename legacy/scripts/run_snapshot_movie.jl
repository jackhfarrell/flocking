#!/usr/bin/env julia

using ArgParse
using CairoMakie
using DifferentialEquations
using JLD2
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

struct MovieWorkspace
    drift::LatticeFlockingSDE.DriftWorkspace
    noise_scale::Float64
end

# Active snapshot demos need a smaller fixed step than the passive scripts.
const DEFAULT_ACTIVE_SNAPSHOT_DT = 2.0^-10
const DEFAULT_CALIBRATION_TIME = 2.0
const DEFAULT_MOVIE_DISPLACEMENT = 1.0

function parse_solver(name::AbstractString)
    solver_name = lowercase(name)
    solver_name in ("auto", "em", "sriw1") ||
        throw(ArgumentError("solver must be auto, em, or sriw1"))
    return solver_name
end

function solve_snapshot(theta0, tspan, params, work, solver_name, dt, saveat, save_start, seed)
    if iszero(work.noise_scale) || iszero(params.Q)
        problem = ODEProblem(movie_drift!, theta0, tspan, work)
        return solve(problem, Tsit5(); saveat, save_start)
    end

    problem = SDEProblem(movie_drift!, movie_noise!, theta0, tspan, work)
    if solver_name == "em"
        return solve(problem, EM(); dt, adaptive=false, saveat, save_start, seed)
    end
    return solve(problem, SRIW1(); dt, adaptive=false, saveat, save_start, seed)
end

function evolve_state(theta0, tspan, params, work, solver_name, dt, seed)
    if iszero(work.noise_scale) || iszero(params.Q)
        problem = ODEProblem(movie_drift!, theta0, tspan, work)
        solution = solve(problem, Tsit5(); save_everystep=false, save_start=false)
        return wrap_angles!(collect(solution.u[end]))
    end

    problem = SDEProblem(movie_drift!, movie_noise!, theta0, tspan, work)
    if solver_name == "em"
        solution = solve(problem, EM(); dt, adaptive=false, save_everystep=false,
            save_start=false, seed)
    else
        solution = solve(problem, SRIW1(); dt, adaptive=false, save_everystep=false,
            save_start=false, seed)
    end
    return wrap_angles!(collect(solution.u[end]))
end

function calibrated_tmax(theta0, params, work, dt, radius)
    calibration_solution = solve_snapshot(theta0, (0.0, DEFAULT_CALIBRATION_TIME), params, work,
        "auto", dt, [DEFAULT_CALIBRATION_TIME], false, 0)
    x0 = positive_sin_marker_x(theta0, params.L)
    x1 = positive_sin_marker_x(calibration_solution.u[end], params.L)
    drift_speed = (x1 - x0) / DEFAULT_CALIBRATION_TIME
    drift_speed > 0 || throw(ErrorException("calibrated drift speed must be positive, got $(drift_speed)"))
    return DEFAULT_MOVIE_DISPLACEMENT * radius / drift_speed, drift_speed
end

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
        default = 8.0
    "--v"
        arg_type = Float64
        default = 2.0
    "--dt"
        arg_type = Float64
        default = DEFAULT_ACTIVE_SNAPSHOT_DT
    "--tmax"
        arg_type = Float64
        default = NaN
    "--equilibration-time"
        arg_type = Float64
        default = 0.0
    "--nframes"
        arg_type = Int
        default = 151
    "--seed"
        arg_type = Int
        default = 1
    "--solver"
        arg_type = String
        default = "auto"
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
    "--diagnostics-output"
        arg_type = String
        default = ""
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
solver_name = parse_solver(args["solver"])
noise_scale = args["noise-scale"]
radius = isnan(args["radius"]) ? L / 10 : args["radius"]
transition_width = isnan(args["transition-width"]) ? L / 40 : args["transition-width"]
center_x = isnan(args["center-x"]) ? L / 4 : args["center-x"]
center_y = isnan(args["center-y"]) ? L / 2 : args["center-y"]
output = args["output"]
diagnostics_output = args["diagnostics-output"]
log_every = args["log-every"]

dt > 0 || throw(ArgumentError("dt must be positive"))
equilibration_time >= 0 || throw(ArgumentError("equilibration-time must be nonnegative"))
nframes > 1 || throw(ArgumentError("nframes must be greater than 1"))
noise_scale >= 0 || throw(ArgumentError("noise-scale must be nonnegative"))
radius > 0 || throw(ArgumentError("radius must be positive"))
transition_width > 0 || throw(ArgumentError("transition-width must be positive"))
log_every > 0 || throw(ArgumentError("log-every must be positive"))

params = ModelParams(; L, Q, J, v)
movie_work = MovieWorkspace(LatticeFlockingSDE.DriftWorkspace(params), noise_scale)

theta0 = zeros(Float64, L * L)

base_theta = if equilibration_time > 0
    evolve_state(theta0, (0.0, equilibration_time), params, movie_work, solver_name, dt, seed)
else
    zeros(Float64, L * L)
end

theta0 = copy(base_theta)
seed_upward_bump!(theta0, L; center_x, center_y, radius, transition_width)

tmax, drift_speed = if isnan(tmax)
    calibrated_tmax(theta0, params, movie_work, dt, radius)
else
    tmax > 0 || throw(ArgumentError("tmax must be positive"))
    tmax, NaN
end
times = collect(range(0.0, tmax; length=nframes))
println("starting solve: tmax=", tmax, " dt=", dt, " nframes=", nframes,
    " noise_scale=", noise_scale, " solver=", solver_name)
solution = solve_snapshot(theta0, (0.0, tmax), params, movie_work, solver_name, dt,
    times, true, seed + 1)
println("finished solve")

length(solution.u) == length(times) ||
    throw(ErrorException("solver returned $(length(solution.u)) snapshots for $(length(times)) requested times"))

theta_snapshots = Array{Float64}(undef, L, L, length(times))
x_centroids = zeros(Float64, length(times))
for (k, state) in enumerate(solution.u)
    theta_state = wrap_angles!(collect(state))
    theta_snapshots[:, :, k] .= reshape(signed_angle.(theta_state), L, L)
    x_centroids[k] = positive_sin_marker_x(theta_state, L)
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
    ax.title = "t = $(round(times[k]; digits=2))  x = $(round(x_centroids[k]; digits=2))"
    if k == 1 || k == length(times) || (k % log_every == 0)
        println("render frame ", k, "/", length(times), " t=", round(times[k]; digits=3))
    end
end

diagnostics_output = isempty(diagnostics_output) ?
    replace(output, r"\.[^.]+$" => ".jld2") : diagnostics_output
mkpath(dirname(diagnostics_output))
jldsave(diagnostics_output;
    diagnostics=(;
        params,
        seed,
        solver=solver_name,
        noise_scale,
        equilibration_time,
        times,
        x_centroids,
        drift_speed,
        center=(center_x, center_y),
        radius,
        transition_width,
    ),
)

println("saved: ", output)
println("saved diagnostics: ", diagnostics_output)
println("x centroids: ", x_centroids)
