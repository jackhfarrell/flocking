#!/usr/bin/env julia

using JLD2
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

L = 32
gamma = 1.0
J = 2.0
v = 1.0
dt = 0.005
sample_stride = 10
burnin_steps = 10_000
T_max = 10
nchunks = 100
seed = 1
initial_condition = :random
solver = SRIW1()

log_radius_index = 1
log_time_index = T_max + 1
output = "results/chunked_correlator.jld2"

params = ModelParams(; L, Q=gamma, J, v)
rng = MersenneTwister(seed)
theta0 = initial_angles(rng, L, initial_condition)
work = LatticeFlockingSDE.DriftWorkspace(params)

burnin_problem = SDEProblem(
    LatticeFlockingSDE.drift!,
    LatticeFlockingSDE.noise!,
    theta0,
    (0.0, burnin_steps * dt),
    work,
)
burnin_solution = solve(burnin_problem, solver; dt, adaptive=false,
    save_everystep=false, save_start=false, rng)
theta = wrap_angles!(collect(burnin_solution.u[end]))

shell_data = radial_displacement_shells(L; oriented=true)
radii = shell_data.radii
times = collect(0:T_max) .* sample_stride .* dt
F_mean = zeros(Float64, length(radii), T_max + 1)
F_m2 = zeros(Float64, length(radii), T_max + 1)
F_stderr = zeros(Float64, length(radii), T_max + 1)

chunk_advance_steps = (2 * T_max + 1) * sample_stride
saveat = collect(0:sample_stride:chunk_advance_steps) .* dt

@info "starting chunked correlator" L gamma J v dt sample_stride burnin_steps T_max nchunks

for chunk in 1:nchunks
    global theta, F_stderr
    chunk_problem = SDEProblem(
        LatticeFlockingSDE.drift!,
        LatticeFlockingSDE.noise!,
        theta,
        (0.0, chunk_advance_steps * dt),
        work,
    )
    chunk_solution = solve(chunk_problem, solver; dt, adaptive=false,
        saveat, save_start=true, rng)

    window = [wrap_angles!(collect(state)) for state in chunk_solution.u[1:(end - 1)]]
    F_chunk = chunk_correlator(window, params, shell_data)
    F_stderr = online_mean_stderr!(F_mean, F_m2, F_chunk, chunk)
    theta = wrap_angles!(collect(chunk_solution.u[end]))

    @info "rolling correlator" chunk nchunks radius=radii[log_radius_index] lag=times[log_time_index] F=F_mean[log_radius_index, log_time_index] stderr=F_stderr[log_radius_index, log_time_index]
end

config = (;
    L, gamma, Q=gamma, J, v, dt, sample_stride, burnin_steps, T_max, nchunks,
    seed, initial_condition, solver=string(typeof(solver)), log_radius_index, log_time_index,
)
result = (; config, radii, times, F_mean, F_stderr)

mkpath(dirname(output))
jldsave(output; result)
@info "saved chunked correlator" output
