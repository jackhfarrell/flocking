#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(description="Plot block equilibration diagnostics.")
    @add_arg_table! settings begin
        "input"
            arg_type = String
        "--output-dir"
            arg_type = String
            default = "figures/equilibration"
    end
    return ArgParse.parse_args(settings)
end

function main()
    args = parse_args()
    mkpath(args["output-dir"])
    diagnostic = load(args["input"], "diagnostic")

    fig = Figure(size=(800, 700))
    ax1 = Axis(fig[1, 1], xlabel="t", ylabel="eta fit")
    valid_eta = isfinite.(diagnostic.eta_mean)
    scatterlines!(ax1, diagnostic.times[valid_eta], diagnostic.eta_mean[valid_eta], label="block mean")
    if any(valid_eta) && any(isfinite.(diagnostic.eta_stderr[valid_eta]))
        errorbars!(ax1, diagnostic.times[valid_eta], diagnostic.eta_mean[valid_eta],
            diagnostic.eta_stderr[valid_eta])
    end
    hlines!(ax1, [diagnostic.eta_spin_wave], linestyle=:dash, label="spin-wave")
    axislegend(ax1)

    ax2 = Axis(fig[2, 1], xlabel="t", ylabel="energy density")
    lines!(ax2, diagnostic.times, diagnostic.energy_mean)

    ax3 = Axis(fig[3, 1], xlabel="t", ylabel="|m|")
    lines!(ax3, diagnostic.times, diagnostic.mag_mean)
    save(joinpath(args["output-dir"], "equilibration.png"), fig)

    last_corr = diagnostic.correlation_mean[:, end]
    positive = isfinite.(diagnostic.radii) .& isfinite.(last_corr) .&
        (diagnostic.radii .> 0) .& (last_corr .> 0)
    fig2 = Figure(size=(700, 500))
    ax4 = Axis(fig2[1, 1], xlabel="r", ylabel="C(r)", xscale=log10, yscale=log10)
    lines!(ax4, diagnostic.radii[positive], last_corr[positive])
    save(joinpath(args["output-dir"], "last_correlation.png"), fig2)

    println("saved figures to ", args["output-dir"])
end

main()
