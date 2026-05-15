#!/usr/bin/env julia

using ArgParse
using JLD2
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(description="Track block observables to choose burn-in for the 2D XY-lattice SDE.")
    @add_arg_table! settings begin
        "--L"
            arg_type = Int
            default = 24
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
        "--block-steps"
            arg_type = Int
            default = 1000
        "--nblocks"
            arg_type = Int
            default = 20
        "--ntrajectories"
            arg_type = Int
            default = 4
        "--seed"
            arg_type = Int
            default = 1
        "--fit-rmin"
            arg_type = Float64
            default = 2.0
        "--fit-rmax"
            arg_type = Float64
            default = Inf
        "--init"
            arg_type = String
            default = "random"
        "--output"
            arg_type = String
            default = "results/equilibration_L24.jld2"
    end
    return ArgParse.parse_args(settings)
end

function main()
    args = parse_args()
    config = SimulationConfig(;
        L=args["L"], Q=args["Q"], J=args["J"], v=args["v"], dt=args["dt"],
        burnin_steps=0, sample_stride=args["block-steps"], nsamples=args["nblocks"],
        ntrajectories=1, seed=args["seed"], fit_rmin=args["fit-rmin"],
        fit_rmax=args["fit-rmax"], initial_condition=Symbol(args["init"]),
    )
    params = config.params
    L = params.L
    nblocks = args["nblocks"]
    ntr = args["ntrajectories"]
    seeds = collect(args["seed"]:(args["seed"] + ntr - 1))
    fit_rmax = isfinite(config.fit_rmax) ? config.fit_rmax : L / 3

    times = collect(1:nblocks) .* args["block-steps"] .* config.dt
    energies = zeros(Float64, nblocks, ntr)
    mags = zeros(Float64, nblocks, ntr)
    etas = fill(NaN, nblocks, ntr)
    radii = Float64[]
    correlations = nothing

    for (j, seed) in enumerate(seeds)
        rng = MersenneTwister(seed)
        theta0 = initial_angles(rng, L, config.initial_condition)
        sol = LatticeFlockingSDE.solve_one(theta0, config; seed)

        for (i, state) in enumerate(sol.u)
            theta = wrap_angles!(collect(state))
            energies[i, j] = xy_energy(theta, params) / L^2
            mags[i, j] = magnetization(theta)
            r, c, _ = radial_correlation(theta, L)
            fit = fit_power_law(r, c; rmin=config.fit_rmin, rmax=fit_rmax)
            etas[i, j] = fit.eta

            if correlations === nothing
                radii = r
                correlations = zeros(Float64, length(c), nblocks, ntr)
            end
            correlations[:, i, j] .= c
        end
    end

    energy_mean = vec(mean(energies; dims=2))
    mag_mean = vec(mean(mags; dims=2))
    eta_mean = vec(mean(etas; dims=2))
    eta_stderr = ntr == 1 ? zeros(nblocks) : vec(std(etas; dims=2)) ./ sqrt(ntr)
    correlation_mean = dropdims(mean(correlations; dims=3); dims=3)

    diagnostic = (;
        config, seeds, times, radii, energies, mags, etas,
        energy_mean, mag_mean, eta_mean, eta_stderr, correlation_mean,
        eta_spin_wave=expected_eta(params),
    )

    mkpath(dirname(args["output"]))
    jldsave(args["output"]; diagnostic)

    println("saved: ", args["output"])
    println("eta_spin_wave: ", diagnostic.eta_spin_wave)
    println("last eta_mean: ", eta_mean[end], " +/- ", eta_stderr[end])
    println("last energy density: ", energy_mean[end])
end

main()
