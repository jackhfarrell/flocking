#!/usr/bin/env julia

using ArgParse
using GLMakie
using JLD2
using Random
using Statistics

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(description="Track block observables with a live energy plot for the 2D XY-lattice SDE.")
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
            default = 1000000
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
        "--live-energy-window-time"
            arg_type = Float64
            default = 50.0
        "--output"
            arg_type = String
            default = "results/equilibration_L24_live.jld2"
    end
    return ArgParse.parse_args(settings)
end

function start_live_energy_plot(window_blocks::Integer)
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1], xlabel="t", ylabel="energy density",
        title="live energy trajectory")
    full_times = Observable(Float64[])
    full_energies = Observable(Float64[])
    window_times = Observable(Float64[])
    window_energies = Observable(Float64[])
    lines!(ax, full_times, full_energies, color=(:gray, 0.35), linewidth=1.5)
    lines!(ax, window_times, window_energies, color=:dodgerblue, linewidth=3)
    display(fig)
    return (; full_times, full_energies, window_times, window_energies, window_blocks)
end

function update_live_energy_plot!(plot, times::AbstractVector{<:Real},
        energies::AbstractVector{<:Real})
    plot.full_times[] = collect(times)
    plot.full_energies[] = collect(energies)
    start = max(1, length(times) - plot.window_blocks + 1)
    plot.window_times[] = collect(times[start:end])
    plot.window_energies[] = collect(energies[start:end])
    yield()
    return nothing
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
    block_time = args["block-steps"] * config.dt
    live_window_blocks = equilibrium_window_blocks(
        block_time, args["live-energy-window-time"], 1)
    live_plot = start_live_energy_plot(live_window_blocks)

    times = collect(1:nblocks) .* block_time
    energies = zeros(Float64, nblocks, ntr)
    mags = zeros(Float64, nblocks, ntr)
    etas = fill(NaN, nblocks, ntr)
    radii = Float64[]
    correlations = nothing

    for (j, seed) in enumerate(seeds)
        rng = MersenneTwister(seed)
        theta0 = initial_angles(rng, L, config.initial_condition)
        sol = LatticeFlockingSDE.solve_one(theta0, config; seed)
        live_times = Float64[]
        live_energies = Float64[]

        @info "live energy trajectory" trajectory=j ntrajectories=ntr

        for (i, state) in enumerate(sol.u)
            theta = wrap_angles!(collect(state))
            energy_density = xy_energy(theta, params) / L^2
            energies[i, j] = energy_density
            mags[i, j] = magnetization(theta)
            r, c, _ = radial_correlation(theta, L)
            fit = fit_power_law(r, c; rmin=config.fit_rmin, rmax=fit_rmax)
            etas[i, j] = fit.eta

            push!(live_times, times[i])
            push!(live_energies, energy_density)
            update_live_energy_plot!(live_plot, live_times, live_energies)

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
