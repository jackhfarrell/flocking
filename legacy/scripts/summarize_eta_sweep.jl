#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

const FIT_WINDOWS = [(3.0, 16.0), (4.0, 16.0), (4.0, 20.0),
    (5.0, 20.0), (6.0, 24.0), (8.0, 24.0)]

function parse_args()
    settings = ArgParseSettings(description="Summarize an eta(T) baseline sweep.")
    @add_arg_table! settings begin
        "input_dir"
            arg_type = String
        "--output-dir"
            arg_type = String
            default = "figures/baseline_xy/eta_vs_T_highJ_L64"
        "--summary-csv"
            arg_type = String
            default = "results/baseline_xy/eta_vs_T_highJ_L64/summary.csv"
        "--summary-md"
            arg_type = String
            default = "results/baseline_xy/eta_vs_T_highJ_L64/summary.md"
    end
    return ArgParse.parse_args(settings)
end

function load_results(input_dir::String)
    files = filter(endswith(".jld2"), readdir(input_dir; join=true))
    isempty(files) && error("no .jld2 files found in $input_dir")
    results = [(file=file, result=load(file, "result")) for file in files]
    sort!(results; by=x -> x.result.config.params.J)
    return results
end

function linear_fit(x::AbstractVector, y::AbstractVector)
    xbar = mean(x)
    ybar = mean(y)
    slope = sum((x .- xbar) .* (y .- ybar)) / sum((x .- xbar).^2)
    intercept = ybar - slope * xbar
    return slope, intercept
end

function through_origin_slope(x::AbstractVector, y::AbstractVector)
    return sum(x .* y) / sum(x .* x)
end

function csv_escape(s)
    text = string(s)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_summary_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        header = ["L", "J", "T", "Q", "v", "dt", "init", "burnin_time",
            "sample_spacing", "nsamples_total", "fit_rmin", "fit_rmax",
            "eta_fit", "eta_spin_wave", "eta_minus_spin_wave", "output_file"]
        println(io, join(header, ","))
        for row in rows
            values = [row.L, row.J, row.T, row.Q, row.v, row.dt, row.init,
                row.burnin_time, row.sample_spacing, row.nsamples_total,
                row.fit_rmin, row.fit_rmax, row.eta_fit, row.eta_spin_wave,
                row.eta_minus_spin_wave, row.output_file]
            println(io, join(csv_escape.(values), ","))
        end
    end
end

function write_markdown(path::String, rows, window_rows, slope, intercept, origin_slope)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# L=64 high-J eta(T) baseline")
        println(io)
        println(io, "Reference slope: `1/(2π) = $(@sprintf("%.8f", 1 / (2π)))`.")
        println(io, "Default fit: `r = 4..20`.")
        println(io)
        println(io, "- Free linear fit: `eta = $(@sprintf("%.6f", slope)) T + $(@sprintf("%.6f", intercept))`")
        println(io, "- Through-origin fit: `eta = $(@sprintf("%.6f", origin_slope)) T`")
        println(io)
        println(io, "| L | J | T | eta_fit | eta_spin_wave | difference |")
        println(io, "|---:|---:|---:|---:|---:|---:|")
        for row in rows
            println(io, "| $(row.L) | $(@sprintf("%.6g", row.J)) | $(@sprintf("%.6g", row.T)) | ",
                "$(@sprintf("%.8f", row.eta_fit)) | $(@sprintf("%.8f", row.eta_spin_wave)) | ",
                "$(@sprintf("%+.8f", row.eta_minus_spin_wave)) |")
        end
        println(io)
        println(io, "## Fit-window sensitivity")
        println(io)
        println(io, "| J | T | window | eta | npoints |")
        println(io, "|---:|---:|:---|---:|---:|")
        for row in window_rows
            println(io, "| $(@sprintf("%.6g", row.J)) | $(@sprintf("%.6g", row.T)) | ",
                "`$(@sprintf("%.0f", row.rmin))..$(@sprintf("%.0f", row.rmax))` | ",
                "$(@sprintf("%.8f", row.eta)) | $(row.npoints) |")
        end
    end
end

function plot_eta(path::String, rows, slope, intercept, origin_slope)
    mkpath(dirname(path))
    T = [row.T for row in rows]
    eta = [row.eta_fit for row in rows]
    tref = range(minimum(T) * 0.95, maximum(T) * 1.05; length=200)

    fig = Figure(size=(760, 560))
    ax = Axis(fig[1, 1], xlabel="T = 1/J", ylabel="eta",
        title="Low-temperature XY baseline")
    scatter!(ax, T, eta, markersize=14, label="SDE fit")
    lines!(ax, tref, tref ./ (2π), linestyle=:dash, label="spin-wave T/(2π)")
    lines!(ax, tref, slope .* tref .+ intercept, label="linear fit")
    lines!(ax, tref, origin_slope .* tref, linestyle=:dot, label="fit through origin")
    axislegend(ax; position=:lt)
    save(path, fig)
end

function main()
    args = parse_args()
    loaded = load_results(args["input_dir"])

    rows = NamedTuple[]
    window_rows = NamedTuple[]
    for item in loaded
        result = item.result
        config = result.config
        params = config.params
        T = 1 / params.J
        eta_spin_wave = expected_eta(params)
        push!(rows, (;
            L=params.L,
            J=params.J,
            T,
            Q=params.Q,
            v=params.v,
            dt=config.dt,
            init=config.initial_condition,
            burnin_time=config.dt * config.burnin_steps,
            sample_spacing=config.dt * config.sample_stride,
            nsamples_total=config.nsamples * config.ntrajectories,
            fit_rmin=result.fit.rmin,
            fit_rmax=result.fit.rmax,
            eta_fit=result.fit.eta,
            eta_spin_wave,
            eta_minus_spin_wave=result.fit.eta - eta_spin_wave,
            output_file=item.file,
        ))

        for (rmin, rmax) in FIT_WINDOWS
            fit = fit_power_law(result.radii, result.correlation_mean; rmin, rmax)
            push!(window_rows, (;
                J=params.J,
                T,
                rmin,
                rmax,
                eta=fit.eta,
                npoints=fit.npoints,
            ))
        end
    end

    T = [row.T for row in rows]
    eta = [row.eta_fit for row in rows]
    slope, intercept = linear_fit(T, eta)
    origin_slope = through_origin_slope(T, eta)

    write_summary_csv(args["summary-csv"], rows)
    write_markdown(args["summary-md"], rows, window_rows, slope, intercept, origin_slope)
    plot_eta(joinpath(args["output-dir"], "eta_vs_T.png"), rows, slope, intercept, origin_slope)

    println("saved summary csv: ", args["summary-csv"])
    println("saved summary markdown: ", args["summary-md"])
    println("saved aggregate figure: ", joinpath(args["output-dir"], "eta_vs_T.png"))
    println("linear slope: ", slope, " intercept: ", intercept)
    println("through-origin slope: ", origin_slope)
end

main()
