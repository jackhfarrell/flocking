#!/usr/bin/env julia

using ArgParse
using JLD2

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(description="Run the 2D XY-lattice flocking SDE.")
    @add_arg_table! settings begin
        "--L"
            arg_type = Int
            default = 32
        "--Q"
            arg_type = Float64
            default = 1.0
        "--J"
            arg_type = Float64
            default = 2.0
        "--v"
            arg_type = Float64
            default = 0.0
        "--dt"
            arg_type = Float64
            default = 0.01
        "--burnin-steps"
            arg_type = Int
            default = 1000
        "--sample-stride"
            arg_type = Int
            default = 10
        "--nsamples"
            arg_type = Int
            default = 100
        "--ntrajectories"
            arg_type = Int
            default = 1
        "--seed"
            arg_type = Int
            default = 1
        "--fit-rmin"
            arg_type = Float64
            default = 2.0
        "--fit-rmax"
            arg_type = Float64
            default = Inf
        "--save-final-theta"
            action = :store_true
        "--init"
            arg_type = String
            default = "random"
        "--output"
            arg_type = String
            default = "results/run.jld2"
    end
    return ArgParse.parse_args(settings)
end

function main()
    args = parse_args()
    config = SimulationConfig(;
        L=args["L"], Q=args["Q"], J=args["J"], v=args["v"], dt=args["dt"],
        burnin_steps=args["burnin-steps"], sample_stride=args["sample-stride"],
        nsamples=args["nsamples"], ntrajectories=args["ntrajectories"],
        seed=args["seed"], save_final_theta=args["save-final-theta"],
        fit_rmin=args["fit-rmin"], fit_rmax=args["fit-rmax"],
        initial_condition=Symbol(args["init"]),
    )

    mkpath(dirname(args["output"]))
    result = run_ensemble(config)
    jldsave(args["output"]; result)

    println("saved: ", args["output"])
    println("eta_fit: ", result.fit.eta)
    println("eta_lowT_spin_wave: ", expected_eta(config.params))
end

main()
