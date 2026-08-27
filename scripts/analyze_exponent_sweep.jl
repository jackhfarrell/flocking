#!/usr/bin/env julia

# Fit the spreading exponent for every temperature and velocity. The final band keeps the
# largest of the local collapse sensitivity, fit-window drift, and up-down hysteresis.

using ArgParse
using CairoMakie
using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function temperature_tag(temperature)
    return "T_" * replace(@sprintf("%.6g", temperature), "." => "p")
end

function measurement_files(path)
    files = String[]
    for (dir, _, names) in walkdir(path), name in names
        name == "measurement.jld2" && push!(files, joinpath(dir, name))
    end
    sort!(files)
    return files
end

settings = ArgParseSettings(description="Fit and plot the temperature exponent sweep.")
@add_arg_table! settings begin
    "--temperatures"
        arg_type = String
        default = "0.25,0.5,0.8"
    "--input-dir"
        arg_type = String
        default = "results/exponent_sweep"
    "--output-dir"
        arg_type = String
        default = "analysis/exponent_sweep"
end
args = ArgParse.parse_args(settings)

temperatures = parse.(Float64, split(args["temperatures"], ","))
datasets = NamedTuple[]
for temperature in temperatures, direction in ("up", "down")
    input = joinpath(args["input-dir"], temperature_tag(temperature), direction)
    files = measurement_files(input)
    isempty(files) && error("no measurements below $input")
    runs = [load(file, "result") for file in files]
    vi_values = sort!(unique(run.config.vi for run in runs))

    for vi in vi_values
        selected = filter(run -> run.config.vi == vi, runs)
        trajectories = sort!(unique(run.config.trajectory for run in selected))
        length(selected) == length(trajectories) ||
            error(
                "duplicate or missing trajectories for T=$temperature, $direction, vi=$vi")
        radii = selected[1].radii
        times = selected[1].times
        fields = cat((run.F_mean for run in selected)...; dims=3)
        mean_F = dropdims(mean(fields; dims=3), dims=3)
        stderr_F = length(selected) == 1 ? zeros(size(mean_F)) :
            dropdims(std(fields; dims=3), dims=3) ./ sqrt(length(selected))
        push!(datasets, (;
            temperature, J=inv(temperature), direction, vi,
            v=selected[1].config.v, trajectories, radii, times, mean_F, stderr_F,
        ))
    end
end

fits = Vector{Any}(undef, length(datasets))
Threads.@threads for index in eachindex(datasets)
    data = datasets[index]
    robustness = fit_window_robustness(
        data.radii, data.times, data.mean_F, data.stderr_F)
    fits[index] = merge(data, (; robustness))
end

mkpath(args["output-dir"])
direction_csv = joinpath(args["output-dir"], "zeta_by_direction.csv")
open(direction_csv, "w") do io
    println(io, "temperature,J,direction,v_index,v,log10_v,zeta,zeta_low,zeta_high," *
        "sensitivity_halfwidth,fit_window_halfspread,ntrajectories,reduced_chi2")
    for row in sort(fits; by=row -> (row.temperature, row.direction, row.vi))
        reference = row.robustness.reference
        println(io, join((
            row.temperature, row.J, row.direction, row.vi, row.v, log10(row.v),
            reference.best.zeta, reference.band.zeta_min, reference.band.zeta_max,
            reference.halfwidth, row.robustness.halfspread, length(row.trajectories),
            reference.best.reduced_chi2,
        ), ","))
    end
end

combined = NamedTuple[]
for temperature in temperatures
    temperature_rows = filter(row -> row.temperature == temperature, fits)
    for vi in sort!(unique(row.vi for row in temperature_rows))
        up = only(filter(row -> row.vi == vi && row.direction == "up", temperature_rows))
        down = only(filter(
            row -> row.vi == vi && row.direction == "down", temperature_rows))
        zeta_up = up.robustness.reference.best.zeta
        zeta_down = down.robustness.reference.best.zeta
        zeta = (zeta_up + zeta_down) / 2
        hysteresis_halfwidth = abs(zeta_up - zeta_down) / 2
        sensitivity_halfwidth = max(
            up.robustness.reference.halfwidth, down.robustness.reference.halfwidth)
        fit_window_halfwidth = max(
            up.robustness.halfspread, down.robustness.halfspread)
        uncertainty = max(
            hysteresis_halfwidth, sensitivity_halfwidth, fit_window_halfwidth)
        push!(combined, (;
            temperature, J=inv(temperature), vi, v=up.v, zeta, zeta_up, zeta_down,
            hysteresis_halfwidth, sensitivity_halfwidth, fit_window_halfwidth,
            uncertainty,
        ))
    end
end

combined_csv = joinpath(args["output-dir"], "zeta_vs_v_temperature.csv")
open(combined_csv, "w") do io
    println(io, "temperature,J,v_index,v,log10_v,zeta,zeta_up,zeta_down," *
        "hysteresis_halfwidth,sensitivity_halfwidth,fit_window_halfwidth,uncertainty")
    for row in sort(combined; by=row -> (row.temperature, row.vi))
        println(io, join((
            row.temperature, row.J, row.vi, row.v, log10(row.v), row.zeta,
            row.zeta_up, row.zeta_down, row.hysteresis_halfwidth,
            row.sensitivity_halfwidth, row.fit_window_halfwidth, row.uncertainty,
        ), ","))
    end
end

figure = Figure(size=(900, 600))
axis = Axis(figure[1, 1], xlabel="log₁₀ v", ylabel="ζ")
colors = Makie.wong_colors()
for (index, temperature) in enumerate(temperatures)
    rows = sort(filter(row -> row.temperature == temperature, combined); by=row -> row.vi)
    x = log10.([row.v for row in rows])
    zeta = [row.zeta for row in rows]
    uncertainty = [row.uncertainty for row in rows]
    color = colors[mod1(index, length(colors))]
    band!(axis, x, zeta .- uncertainty, zeta .+ uncertainty; color=(color, 0.18))
    lines!(axis, x, zeta; color, linewidth=2.5,
        label=@sprintf("T = %.2g", temperature))
    scatter!(axis, x, [row.zeta_up for row in rows]; color, marker=:circle,
        markersize=7)
    scatter!(axis, x, [row.zeta_down for row in rows]; color=:white,
        strokecolor=color, strokewidth=1.5, marker=:circle, markersize=7)
end
scatter!(axis, [NaN], [NaN]; color=:black, marker=:circle, markersize=7,
    label="up sweep")
scatter!(axis, [NaN], [NaN]; color=:white, strokecolor=:black, strokewidth=1.5,
    marker=:circle, markersize=7, label="down sweep")
axislegend(axis; position=:lb)
png = joinpath(args["output-dir"], "zeta_vs_v_temperature.png")
pdf = joinpath(args["output-dir"], "zeta_vs_v_temperature.pdf")
save(png, figure)
save(pdf, figure)

summary = joinpath(args["output-dir"], "summary.md")
open(summary, "w") do io
    println(io, "# Spreading exponent across velocity and temperature")
    println(io)
    println(io, "The filled and open points are the up and down velocity ladders. ",
        "The line is their midpoint. The band is the largest of the collapse sensitivity, ",
        "fit-window half-spread, and up-down half-difference.")
    println(io)
    println(io, "- Temperatures: `", join(temperatures, ", "), "`")
    println(io, "- Inverse temperatures J: `", join(inv.(temperatures), ", "), "`")
    println(io, "- Directional fits: `", direction_csv, "`")
    println(io, "- Combined fits: `", combined_csv, "`")
end

println("directional fits: ", direction_csv)
println("combined fits: ", combined_csv)
println("figure: ", png)
println("summary: ", summary)
