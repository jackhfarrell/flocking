#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function parse_args()
    settings = ArgParseSettings(description="Plot reduced outputs from run_sim.jl.")
    @add_arg_table! settings begin
        "input"
            arg_type = String
        "--output-dir"
            arg_type = String
            default = "figures"
    end
    return ArgParse.parse_args(settings)
end

function main()
    args = parse_args()
    mkpath(args["output-dir"])
    result = load(args["input"], "result")

    fig = Figure(size=(700, 500))
    ax = Axis(fig[1, 1], xlabel="r", ylabel="C(r)", xscale=log10, yscale=log10)
    positive = isfinite.(result.radii) .& isfinite.(result.correlation_mean) .&
        (result.radii .> 0) .& (result.correlation_mean .> 0)
    lines!(ax, result.radii[positive], result.correlation_mean[positive], label="simulation")
    fit = result.fit
    if isfinite(fit.eta) && any(positive)
        fit_line = exp(fit.intercept) .* result.radii[positive] .^ fit.slope
        lines!(ax, result.radii[positive], fit_line, linestyle=:dash,
            label="fit eta=$(round(fit.eta; digits=3))")
    end
    axislegend(ax)
    save(joinpath(args["output-dir"], "correlation.png"), fig)

    fig2 = Figure(size=(800, 500))
    ax1 = Axis(fig2[1, 1], xlabel="t", ylabel="energy density")
    lines!(ax1, result.times, result.energy_density_mean)
    ax2 = Axis(fig2[2, 1], xlabel="t", ylabel="|m|")
    lines!(ax2, result.times, result.magnetization_mean)
    save(joinpath(args["output-dir"], "diagnostics.png"), fig2)

    if result.final_theta !== nothing
        L = result.config.params.L
        fig3 = Figure(size=(600, 500))
        ax3 = Axis(fig3[1, 1], aspect=DataAspect(), xlabel="x", ylabel="y")
        heatmap!(ax3, reshape(result.final_theta, L, L), colormap=:hsv)
        save(joinpath(args["output-dir"], "final_theta.png"), fig3)
    end

    println("saved figures to ", args["output-dir"])
end

main()
