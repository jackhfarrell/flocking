#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_J2_v1_gamma1")
const DEFAULT_OUTPUT_DIR = joinpath(REPO_ROOT, "figures", "spin_aligned_f_correlator_L200_J2_v1_gamma1")
const DEFAULT_OUTPUT_PREFIX = "spin_aligned_f_correlator_ensemble"

function parse_args()
    settings = ArgParseSettings(
        description="Plot ensemble-averaged spin-aligned correlator traces.",
    )
    @add_arg_table! settings begin
        "--input-dir"
            arg_type = String
            default = DEFAULT_INPUT_DIR
        "--output-dir"
            arg_type = String
            default = DEFAULT_OUTPUT_DIR
        "--output-prefix"
            arg_type = String
            default = DEFAULT_OUTPUT_PREFIX
        "--radius-max"
            arg_type = Float64
            default = 80.0
    end
    return ArgParse.parse_args(settings)
end

function plot_trace_family(path::String, radii::AbstractVector, times::AbstractVector,
        mean_field, stderr_field, ylabel::String, title::String, radius_max::Real)
    fig = Figure(size=(1100, 700))
    Label(fig[0, :], title, fontsize=22)
    ax = Axis(fig[1, 1], xlabel="radius", ylabel=ylabel)
    radius_mask = radii .<= radius_max
    plot_radii = radii[radius_mask]
    palette = Makie.wong_colors()
    for (k, tidx) in enumerate(2:length(times))
        color = palette[mod1(k, length(palette))]
        mean = mean_field[radius_mask, tidx]
        err = stderr_field[radius_mask, tidx]
        band!(ax, plot_radii, mean .- err, mean .+ err, color=(color, 0.18))
        lines!(ax, plot_radii, mean, color=color, linewidth=3,
            label="t = $(round(times[tidx]; digits=3))")
    end
    xlims!(ax, 0, radius_max)
    axislegend(ax, position=:rb, nbanks=2)
    save(path, fig)
end

function main()
    args = parse_args()
    args["radius-max"] > 0 || throw(ArgumentError("--radius-max must be positive"))

    files = collect_job_files(args["input-dir"])
    ensemble = load_ensemble(files)
    mkpath(args["output-dir"])

    result = (; config=ensemble.config, radii=ensemble.radii, times=ensemble.times,
        F_mean=ensemble.F_mean, F_stderr=ensemble.F_stderr)
    if hasproperty(ensemble, :C_plus_mean)
        result = merge(result, (; C_plus_mean=ensemble.C_plus_mean,
            C_plus_stderr=ensemble.C_plus_stderr, C_minus_mean=ensemble.C_minus_mean,
            C_minus_stderr=ensemble.C_minus_stderr))
    end
    jldsave(joinpath(args["output-dir"], args["output-prefix"] * ".jld2"); result)

    nruns = ensemble.nruns
    radius_max = args["radius-max"]
    prefix = joinpath(args["output-dir"], args["output-prefix"])
    plot_trace_family(prefix * "_traces.png", ensemble.radii, ensemble.times,
        ensemble.F_mean, ensemble.F_stderr, "F(r, t)",
        "Spin-aligned F correlator across $(nruns) runs", radius_max)

    if hasproperty(ensemble, :C_plus_mean)
        plot_trace_family(prefix * "_c_plus_traces.png", ensemble.radii, ensemble.times,
            ensemble.C_plus_mean, ensemble.C_plus_stderr, "C+(r, t)",
            "Spin-aligned C+(r, t) across $(nruns) runs", radius_max)
        plot_trace_family(prefix * "_c_minus_traces.png", ensemble.radii, ensemble.times,
            ensemble.C_minus_mean, ensemble.C_minus_stderr, "C-(r, t)",
            "Spin-aligned C-(r, t) across $(nruns) runs", radius_max)
    end

    println("saved ", nruns, " runs from ", args["input-dir"])
end

main()
