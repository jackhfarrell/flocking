#!/usr/bin/env julia

# Measure F(r,t) for one temperature, direction, and trajectory. Cluster tasks split over
# trajectories and use Julia threads for the velocity points, which keeps the submitted job
# count below Alpine's per-user limit.

using ArgParse
using JLD2
using Printf
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(description="Measure one trajectory of the exponent sweep.")
@add_arg_table! settings begin
    "--temperature"
        arg_type = Float64
        required = true
    "--direction"
        arg_type = String
        required = true
    "--ntrajectories"
        arg_type = Int
        default = 20
    "--array-id"
        arg_type = Int
        default = 0
    "--trajectory"
        arg_type = Int
        default = 0
    "--L"
        arg_type = Int
        default = 200
    "--Q"
        arg_type = Float64
        default = 1.0
    "--dt"
        arg_type = Float64
        default = 2.0^-10
    "--v-min"
        arg_type = Float64
        default = 0.1
    "--v-max"
        arg_type = Float64
        default = 10.0
    "--nv"
        arg_type = Int
        default = 30
    "--dr"
        arg_type = Float64
        default = 0.5
    "--r-max"
        arg_type = Float64
        default = 60.0
    "--T-max"
        arg_type = Float64
        default = 16.0
    "--ntimes"
        arg_type = Int
        default = 8
    "--windows-low-v"
        arg_type = Int
        default = 10
    "--windows-high-v"
        arg_type = Int
        default = 2
    "--budget-power"
        arg_type = Float64
        default = 1.0
    "--base-seed"
        arg_type = Int
        default = 1_000_000
    "--library-dir"
        arg_type = String
        default = "library/exponent_sweep"
    "--output-dir"
        arg_type = String
        default = "results/exponent_sweep"
end
args = ArgParse.parse_args(settings)

temperature = args["temperature"]
J = inv(temperature)
direction = args["direction"]
direction in ("up", "down") || error("direction must be up or down")
nv = args["nv"]
if args["trajectory"] == 0
    array_id = args["array-id"] == 0 ?
        parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1")) : args["array-id"]
    trajectory = (array_id - 1) ÷ nv + 1
    velocity_indices = mod1(array_id, nv):mod1(array_id, nv)
else
    trajectory = args["trajectory"]
    velocity_indices = 1:nv
end
trajectory <= args["ntrajectories"] || error("trajectory exceeds the sweep size")

L = args["L"]
Q = args["Q"]
dt = args["dt"]
v_values = exp.(range(log(args["v-min"]), log(args["v-max"]); length=nv))
lag_time = args["T-max"] / args["ntimes"]
lag_steps = round(Int, lag_time / dt)
schedule = lag_step_schedule(args["ntimes"], lag_steps; spacing=:geometric)
times = schedule.cum_steps .* dt
radii = collect(args["dr"]:args["dr"]:min(args["r-max"], L / 2))
solver = SRA1()

Threads.@threads for vi in velocity_indices
    v = v_values[vi]
    fraction = (log(args["v-max"]) - log(v)) /
        (log(args["v-max"]) - log(args["v-min"]))
    nwindows = round(Int, args["windows-high-v"] +
        (args["windows-low-v"] - args["windows-high-v"]) *
        fraction^args["budget-power"])

    key = @sprintf("v_%02d_%.8f", vi, v)
    input = joinpath(args["library-dir"], key,
        @sprintf("%s_traj_%03d.jld2", direction, trajectory))
    baked = load(input, "result")
    params = ModelParams(; L, Q, J, v)
    work = LatticeFlockingSDE.DriftWorkspace(params)
    seed = args["base-seed"] + 100_000 * (direction == "down") +
        nv * (trajectory - 1) + vi
    rng = MersenneTwister(seed)
    F_mean = zeros(Float64, length(radii), length(times))
    F_m2 = zeros(Float64, size(F_mean))
    F_stderr = similar(F_mean)

    @info "starting measurement" temperature direction trajectory vi v nwindows seed
    let theta = copy(baked.theta)
        for window_index in 1:nwindows
            window = Vector{Vector{Float64}}(undef, 2args["ntimes"] + 1)
            window[1] = copy(theta)
            for sample_index in eachindex(schedule.advance_gaps)
                theta = LatticeFlockingSDE.advance_fixed(theta,
                    schedule.advance_gaps[sample_index], dt, work, solver, rng)
                window[sample_index + 1] = copy(theta)
            end
            sample = LatticeFlockingSDE.spin_aligned_correlators(window, L, radii)
            F_stderr .= online_mean_stderr!(F_mean, F_m2, sample, window_index)
            @info(
                "measured window",
                temperature,
                direction,
                trajectory,
                vi,
                v,
                window_index,
                nwindows,
            )
        end
    end

    config = (;
        temperature, J, L, Q, v, vi, direction, trajectory, dt,
        radii_step=args["dr"], r_max=args["r-max"], T_max=args["T-max"],
        ntimes=args["ntimes"], nwindows, seed, solver=string(typeof(solver)),
        equilibrium=input,
    )
    result = (; config, radii, times, F_mean, F_stderr)
    output = joinpath(args["output-dir"], @sprintf("traj_%03d", trajectory), key,
        "measurement.jld2")
    mkpath(dirname(output))
    jldsave(output; result)
    @info "saved measurement" temperature direction trajectory vi v output
end
