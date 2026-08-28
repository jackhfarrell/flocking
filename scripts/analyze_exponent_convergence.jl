#!/usr/bin/env julia

# Fit late-time collapses after averaging the two velocity ladders at the field level.
# Whole trajectories are resampled within each ladder, so the bootstrap keeps the
# independent preparation histories and the correlations between radii and times.

using ArgParse
using JLD2
using Printf
using Random
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

function stratified_mean(up_fields, down_fields, rng)
    mean_field = zeros(Float64, size(up_fields, 1), size(up_fields, 2))
    for _ in axes(up_fields, 3)
        mean_field .+= @view up_fields[:, :, rand(rng, axes(up_fields, 3))]
    end
    for _ in axes(down_fields, 3)
        mean_field .+= @view down_fields[:, :, rand(rng, axes(down_fields, 3))]
    end
    return mean_field ./ (size(up_fields, 3) + size(down_fields, 3))
end

function local_bootstrap_fit(radii, times, field, time_indices, center;
        rmax, grid_points)
    eta_values = collect((center.eta - 0.06):0.01:(center.eta + 0.06))
    zeta_values = collect((center.zeta - 0.06):0.005:(center.zeta + 0.06))
    fit = LatticeFlockingSDE.scan_common_grid_collapse(
        radii, times, field, time_indices, eta_values, zeta_values;
        rmax, grid_points).best
    at_edge = abs(fit.eta - center.eta) >= 0.059 ||
        abs(fit.zeta - center.zeta) >= 0.059
    at_edge || return fit
    return best_common_grid_collapse(
        radii, times, field, time_indices; rmax, grid_points).best
end

settings = ArgParseSettings(
    description="Fit late-time exponent convergence with a trajectory bootstrap.")
@add_arg_table! settings begin
    "--temperatures"
        arg_type = String
        default = "0.25,0.5,0.8"
    "--input-dir"
        arg_type = String
        default = "results/exponent_sweep"
    "--output-dir"
        arg_type = String
        default = "analysis/exponent_convergence"
    "--bootstrap"
        arg_type = Int
        default = 500
    "--reference-points"
        arg_type = Int
        default = 5
    "--minimum-points"
        arg_type = Int
        default = 4
    "--rmax"
        arg_type = Float64
        default = 40.0
    "--grid-points"
        arg_type = Int
        default = 48
    "--seed"
        arg_type = Int
        default = 8_120_427
end
args = ArgParse.parse_args(settings)

temperatures = parse.(Float64, split(args["temperatures"], ","))
conditions = NamedTuple[]
for temperature in temperatures
    directional_runs = Dict{String, Any}()
    for direction in ("up", "down")
        input = joinpath(args["input-dir"], temperature_tag(temperature), direction)
        files = measurement_files(input)
        isempty(files) && error("no measurements below $input")
        directional_runs[direction] = [load(file, "result") for file in files]
    end

    vi_values = sort!(unique(run.config.vi for run in directional_runs["up"]))
    for vi in vi_values
        up = filter(run -> run.config.vi == vi, directional_runs["up"])
        down = filter(run -> run.config.vi == vi, directional_runs["down"])
        up_fields = cat((run.F_mean for run in up)...; dims=3)
        down_fields = cat((run.F_mean for run in down)...; dims=3)
        push!(conditions, (;
            temperature,
            J=inv(temperature),
            vi,
            v=up[1].config.v,
            radii=up[1].radii,
            times=up[1].times,
            up_fields,
            down_fields,
        ))
    end
end

fits = Vector{Any}(undef, length(conditions))
convergence = Vector{Any}(undef, length(conditions))
Threads.@threads for index in eachindex(conditions)
    data = conditions[index]
    positive = findall(>(0), data.times)
    reference_count = min(args["reference-points"], length(positive))
    reference_indices = positive[(end - reference_count + 1):end]
    mean_field = dropdims(mean(
        cat(data.up_fields, data.down_fields; dims=3); dims=3), dims=3)

    reference = best_common_grid_collapse(
        data.radii, data.times, mean_field, reference_indices;
        rmax=args["rmax"], grid_points=args["grid-points"])

    rows = NamedTuple[]
    for count in args["minimum-points"]:length(positive)
        time_indices = positive[(end - count + 1):end]
        fit = best_common_grid_collapse(
            data.radii, data.times, mean_field, time_indices;
            rmax=args["rmax"], grid_points=args["grid-points"])
        push!(rows, (;
            data.temperature, data.J, data.vi, data.v,
            log10_v=log10(data.v), ntime=count,
            t_min=data.times[first(time_indices)], t_max=data.times[last(time_indices)],
            fit.best.eta, fit.best.zeta, fit.best.score,
        ))
    end
    convergence[index] = rows

    time_counts = unique(clamp.((reference_count - 1):reference_count + 1,
        args["minimum-points"], length(positive)))
    time_zetas = [
        best_common_grid_collapse(data.radii, data.times, mean_field,
            positive[(end - count + 1):end];
            rmax=args["rmax"], grid_points=args["grid-points"]).best.zeta
        for count in time_counts
    ]
    radius_maxima = unique(min.((
        0.75 * args["rmax"], args["rmax"], 1.25 * args["rmax"]),
        last(data.radii)))
    radius_zetas = [
        best_common_grid_collapse(
            data.radii, data.times, mean_field, reference_indices;
            rmax, grid_points=args["grid-points"]).best.zeta
        for rmax in radius_maxima
    ]

    up_mean = dropdims(mean(data.up_fields; dims=3), dims=3)
    down_mean = dropdims(mean(data.down_fields; dims=3), dims=3)
    up = best_common_grid_collapse(
        data.radii, data.times, up_mean, reference_indices;
        rmax=args["rmax"], grid_points=args["grid-points"]).best
    down = best_common_grid_collapse(
        data.radii, data.times, down_mean, reference_indices;
        rmax=args["rmax"], grid_points=args["grid-points"]).best

    fixed_half = best_fixed_common_grid_collapse(
        data.radii, data.times, mean_field, reference_indices, 0.5;
        rmax=args["rmax"], grid_points=args["grid-points"])
    fixed_three_eighths = best_fixed_common_grid_collapse(
        data.radii, data.times, mean_field, reference_indices, 3 / 8;
        rmax=args["rmax"], grid_points=args["grid-points"])

    bootstrap_zetas = Float64[]
    rng = MersenneTwister(args["seed"] + index)
    for _ in 1:args["bootstrap"]
        field = stratified_mean(data.up_fields, data.down_fields, rng)
        fit = local_bootstrap_fit(
            data.radii, data.times, field, reference_indices, reference.best;
            rmax=args["rmax"], grid_points=args["grid-points"])
        push!(bootstrap_zetas, fit.zeta)
    end
    bootstrap_low = isempty(bootstrap_zetas) ? reference.best.zeta :
        quantile(bootstrap_zetas, 0.16)
    bootstrap_high = isempty(bootstrap_zetas) ? reference.best.zeta :
        quantile(bootstrap_zetas, 0.84)
    bootstrap_halfwidth = max(
        abs(reference.best.zeta - bootstrap_low),
        abs(bootstrap_high - reference.best.zeta),
    )
    time_halfspread = (maximum(time_zetas) - minimum(time_zetas)) / 2
    radius_halfspread = (maximum(radius_zetas) - minimum(radius_zetas)) / 2
    hysteresis_halfwidth = abs(up.zeta - down.zeta) / 2
    uncertainty = max(
        bootstrap_halfwidth, time_halfspread, radius_halfspread, hysteresis_halfwidth)

    fits[index] = (;
        data.temperature, data.J, data.vi, data.v, log10_v=log10(data.v),
        reference.best.eta, reference.best.zeta, reference.best.score,
        bootstrap_low, bootstrap_high, bootstrap_halfwidth,
        time_halfspread, radius_halfspread, hysteresis_halfwidth, uncertainty,
        zeta_up=up.zeta, zeta_down=down.zeta,
        score_up=up.score, score_down=down.score,
        fixed_half_ratio=fixed_half.score / reference.best.score,
        fixed_three_eighths_ratio=fixed_three_eighths.score / reference.best.score,
        ntrajectories=size(data.up_fields, 3) + size(data.down_fields, 3),
        reference_points=reference_count,
        t_min=data.times[first(reference_indices)],
        t_max=data.times[last(reference_indices)],
    )
    @info(
        "fitted late-time collapse",
        temperature=data.temperature,
        velocity_index=data.vi,
        velocity=data.v,
        zeta=reference.best.zeta,
        uncertainty,
    )
end

mkpath(args["output-dir"])
summary_csv = joinpath(args["output-dir"], "zeta_late_time.csv")
open(summary_csv, "w") do io
    println(io, "temperature,J,v_index,v,log10_v,eta,zeta,score,bootstrap_low," *
        "bootstrap_high,bootstrap_halfwidth,time_halfspread,radius_halfspread," *
        "hysteresis_halfwidth,uncertainty,zeta_up,zeta_down,score_up,score_down," *
        "fixed_half_ratio,fixed_three_eighths_ratio,ntrajectories," *
        "reference_points,t_min,t_max")
    for row in sort(fits; by=row -> (row.temperature, row.vi))
        println(io, join(values(row), ","))
    end
end

convergence_csv = joinpath(args["output-dir"], "zeta_time_convergence.csv")
open(convergence_csv, "w") do io
    println(io, "temperature,J,v_index,v,log10_v,ntime,t_min,t_max,eta,zeta,score")
    rows = sort(vcat(convergence...);
        by=row -> (row.temperature, row.vi, row.t_min))
    for row in rows
        println(io, join(values(row), ","))
    end
end

summary_md = joinpath(args["output-dir"], "summary.md")
open(summary_md, "w") do io
    println(io, "# Late-time exponent convergence")
    println(io)
    println(io, "The primary fit uses the last $(args["reference-points"]) positive lags. ",
        "The common-grid objective compares complete time traces without treating ",
        "neighboring radii as independent samples.")
    println(io)
    println(io, "- Temperatures: `", join(temperatures, ", "), "`")
    println(io, "- Stratified bootstrap draws: `", args["bootstrap"], "`")
    println(io, "- Reference radius cutoff: `", args["rmax"], "`")
    println(io, "- Main fits: `", summary_csv, "`")
    println(io, "- Time convergence: `", convergence_csv, "`")
end

println("main fits: ", summary_csv)
println("time convergence: ", convergence_csv)
println("summary: ", summary_md)
