#!/usr/bin/env julia

# Walk one independent state through the velocity ladder and save every equilibrated rung.
# Running the ladder in both directions exposes hysteresis instead of hiding it in a single
# preparation path.

using ArgParse
using JLD2
using Printf
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(description="Bake one velocity ladder for the exponent sweep.")
@add_arg_table! settings begin
    "--temperature"
        arg_type = Float64
        required = true
    "--direction"
        arg_type = String
        required = true
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
    "--block-steps"
        arg_type = Int
        default = 2048
    "--max-blocks"
        arg_type = Int
        default = 10_000
    "--window-time"
        arg_type = Float64
        default = 50.0
    "--window-blocks"
        arg_type = Int
        default = 5
    "--energy-threshold"
        arg_type = Float64
        default = 0.02
    "--magnetization-threshold"
        arg_type = Float64
        default = 0.02
    "--base-seed"
        arg_type = Int
        default = 100_000
    "--output-dir"
        arg_type = String
        default = "library/exponent_sweep"
end
args = ArgParse.parse_args(settings)

temperature = args["temperature"]
J = inv(temperature)
direction = args["direction"]
direction in ("up", "down") || error("direction must be up or down")
trajectory = args["trajectory"] == 0 ?
    parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "1")) : args["trajectory"]
L = args["L"]
Q = args["Q"]
dt = args["dt"]
v_values = exp.(range(log(args["v-min"]), log(args["v-max"]); length=args["nv"]))
order = direction == "up" ? eachindex(v_values) : reverse(eachindex(v_values))
block_time = args["block-steps"] * dt
window_blocks = equilibrium_window_blocks(
    block_time, args["window-time"], args["window-blocks"])
solver = SRA1()

let theta = nothing
    for (rung, vi) in enumerate(order)
        v = v_values[vi]
        key = @sprintf("v_%02d_%.8f", vi, v)
        output = joinpath(args["output-dir"], key,
            @sprintf("%s_traj_%03d.jld2", direction, trajectory))

        if isfile(output)
            theta = load(output, "result").theta
            @info(
                "loaded equilibrated rung",
                temperature,
                direction,
                trajectory,
                rung,
                vi,
                v,
                output,
            )
            continue
        end

        if theta === nothing
            seed = args["base-seed"] + 100_000 * (direction == "down") + trajectory
            theta = initial_angles(MersenneTwister(seed), L, :ordered)
        end

        params = ModelParams(; L, Q, J, v)
        work = LatticeFlockingSDE.DriftWorkspace(params)
        seed = args["base-seed"] + 10_000 * (direction == "down") +
            args["nv"] * (trajectory - 1) + vi
        rng = MersenneTwister(seed)
        energies = Float64[]
        magnetizations = Float64[]
        reached = false

        for block in 1:args["max-blocks"]
            theta = LatticeFlockingSDE.advance_fixed(
                theta, args["block-steps"], dt, work, solver, rng)
            push!(energies, xy_energy(theta, params) / L^2)
            push!(magnetizations, magnetization(theta))
            energy_range = observable_window_range(energies, window_blocks)
            magnetization_range = observable_window_range(magnetizations, window_blocks)
            reached = energy_range <= args["energy-threshold"] &&
                magnetization_range <= args["magnetization-threshold"]

            if block == 1 || block % window_blocks == 0 || reached
                @info(
                    "equilibrating rung",
                    temperature,
                    direction,
                    trajectory,
                    rung,
                    vi,
                    v,
                    block,
                    energy_density=energies[end],
                    magnetization=magnetizations[end],
                    energy_range,
                    magnetization_range,
                )
            end
            reached && break
        end

        config = (;
            temperature, J, L, Q, v, vi, rung, direction, trajectory, dt,
            block_steps=args["block-steps"], blocks=length(energies), window_blocks,
            energy_threshold=args["energy-threshold"],
            magnetization_threshold=args["magnetization-threshold"], reached, seed,
            solver=string(typeof(solver)), energy_density=energies[end],
            magnetization=magnetizations[end],
        )
        result = (; config, theta)
        mkpath(dirname(output))
        jldsave(output; result)
        @info(
            "saved equilibrated rung",
            temperature,
            direction,
            trajectory,
            rung,
            vi,
            v,
            reached,
            output,
        )
    end
end
