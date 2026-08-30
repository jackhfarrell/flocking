#!/usr/bin/env julia

# Collect the latest checkpoint from every single-chain velocity job.

using ArgParse
using CairoMakie
using Dates
using JLD2

settings = ArgParseSettings(description="Collect a reblocked single-chain velocity sweep.")
@add_arg_table! settings begin
    "--input-dir"
        arg_type = String
        default = "results/reblocked_v_sweep/T_0p8/L_256"
    "--output-dir"
        arg_type = String
        default = "analysis/reblocked_v_sweep/T_0p8_L256"
    "--expected-points"
        arg_type = Int
        default = 41
    "--tolerance"
        arg_type = Float64
        default = 0.005
end
args = ArgParse.parse_args(settings)

files = isdir(args["input-dir"]) ? sort([
    joinpath(directory, name)
    for (directory, _, names) in walkdir(args["input-dir"])
    for name in names if endswith(name, ".jld2")
]) : String[]

rows = NamedTuple[]
for file in files
    run = load(file, "run")
    latest = isempty(run.history) ? nothing : run.history[end]
    preliminary = !hasproperty(run, :estimates) || isempty(run.estimates) ?
        nothing : run.estimates[end]
    value = (name, fallback=NaN) -> latest === nothing || !hasproperty(latest, name) ?
        fallback : getproperty(latest, name)
    preliminary_value = (name, fallback=NaN) ->
        preliminary === nothing || !hasproperty(preliminary, name) ?
        fallback : getproperty(preliminary, name)
    phase = if !run.equilibrium_reached
        "equilibrating"
    elseif latest !== nothing && value(:converged, false)
        "converged"
    else
        "measuring"
    end
    push!(rows, (;
        temperature=run.config.temperature,
        v=run.config.v,
        log10_v=log10(run.config.v),
        L=run.config.L,
        dt=run.config.dt,
        phase,
        equilibrium_blocks=length(run.equilibrium_energies),
        nwindows=run.nwindows,
        nblocks=size(run.F_blocks, 3),
        checkpoints=length(run.history),
        measurement_time=value(:measurement_time, 0.0),
        eta=value(:eta),
        zeta=value(:zeta),
        preliminary_eta=preliminary_value(:eta),
        preliminary_zeta=preliminary_value(:zeta),
        uncertainty=value(:estimated_error, Inf),
        statistical_error=value(:statistical_error, Inf),
        quarter_halfspread=value(:quarter_halfspread, Inf),
        time_halfspread=value(:time_halfspread, Inf),
        radius_halfspread=value(:radius_halfspread, Inf),
        stability_error=value(:stability_error, Inf),
        reblock_plateau=value(:reblock_plateau, false),
        maximum_edge_ratio=value(:maximum_edge_ratio, Inf),
        profile_at_boundary=value(:profile_at_boundary, true),
        fit_at_boundary=value(:fit_at_boundary, true),
        converged=value(:converged, false),
        file,
    ))
end
sort!(rows; by=row -> row.v)

mkpath(args["output-dir"])
csv = joinpath(args["output-dir"], "zeta_vs_v.csv")
temporary_csv = csv * ".tmp"
open(temporary_csv, "w") do io
    if !isempty(rows)
        println(io, join(keys(rows[1]), ","))
        for row in rows
            println(io, join(values(row), ","))
        end
    end
end
mv(temporary_csv, csv; force=true)

converged_count = count(row -> row.converged, rows)
within_tolerance = count(row ->
    isfinite(row.uncertainty) && row.uncertainty <= args["tolerance"], rows)
summary = joinpath(args["output-dir"], "summary.md")
temporary_summary = summary * ".tmp"
open(temporary_summary, "w") do io
    println(io, "# Reblocked velocity sweep")
    println(io)
    println(io, "- Expected velocity points: `", args["expected-points"], "`")
    println(io, "- Checkpoints found: `", length(rows), "`")
    println(io, "- Formally converged: `", converged_count, "`")
    println(io, "- Latest uncertainty at or below `", args["tolerance"], "`: `",
        within_tolerance, "`")
    println(io, "- Missing points: `", max(0, args["expected-points"] - length(rows)), "`")
    println(io)
    println(io, "| v | current zeta | uncertainty | blocks | estimate | phase |")
    println(io, "|---:|---:|---:|---:|:---|:---|")
    for row in rows
        reported_zeta = isfinite(row.zeta) ? row.zeta : row.preliminary_zeta
        estimate_kind = isfinite(row.zeta) ? "reblocked" :
            isfinite(row.preliminary_zeta) ? "preliminary" : "waiting"
        println(io, "| ", row.v, " | ", reported_zeta, " | ", row.uncertainty,
            " | ", row.nblocks, " | ", estimate_kind, " | ", row.phase, " |")
    end
end
mv(temporary_summary, summary; force=true)

display_zeta = [isfinite(row.zeta) ? row.zeta : row.preliminary_zeta for row in rows]
available = findall(isfinite, display_zeta)
robust = findall(index -> isfinite(rows[index].zeta), eachindex(rows))
with_error = findall(index -> isfinite(rows[index].zeta) &&
    isfinite(rows[index].uncertainty), eachindex(rows))
preliminary_only = findall(index -> !isfinite(rows[index].zeta) &&
    isfinite(rows[index].preliminary_zeta), eachindex(rows))
converged = findall(index -> rows[index].converged, eachindex(rows))

figure = Figure(size=(900, 560))
axis = Axis(
    figure[1, 1],
    xlabel="log₁₀ v",
    ylabel="ζ",
    title="Spreading exponent during the velocity sweep",
    xticks=-1:0.25:1,
    limits=(-1, 1, 0, 1),
)
if isempty(available)
    text!(axis, 0, 0.5;
        text="Waiting for the first measured block", align=(:center, :center))
else
    lines!(axis, Float64[rows[index].log10_v for index in available],
        Float64[display_zeta[index] for index in available];
        color=(:gray40, 0.6), linewidth=1.5)
    if !isempty(robust)
        scatter!(axis, Float64[rows[index].log10_v for index in robust],
            Float64[rows[index].zeta for index in robust];
            color=:steelblue, markersize=9, label="reblocked checkpoint")
    end
    if !isempty(with_error)
        errorbars!(axis, Float64[rows[index].log10_v for index in with_error],
            Float64[rows[index].zeta for index in with_error],
            Float64[rows[index].uncertainty for index in with_error];
            color=:steelblue, whiskerwidth=7)
    end
    if !isempty(preliminary_only)
        scatter!(axis, Float64[rows[index].log10_v for index in preliminary_only],
            Float64[rows[index].preliminary_zeta for index in preliminary_only];
            color=:darkorange, marker=:diamond, markersize=9, label="preliminary")
    end
    if !isempty(converged)
        scatter!(axis, Float64[rows[index].log10_v for index in converged],
            Float64[rows[index].zeta for index in converged];
            color=:seagreen, markersize=10, label="converged")
    end
    axislegend(axis; position=:rt)
end
Label(
    figure[2, 1],
    "Updated $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))  •  " *
    "$(length(available))/$(args["expected-points"]) estimates  •  " *
    "$(converged_count) converged",
    fontsize=13,
)
rowsize!(figure.layout, 2, Auto(0.08))

png = joinpath(args["output-dir"], "zeta_vs_v_live.png")
pdf = joinpath(args["output-dir"], "zeta_vs_v_live.pdf")
temporary_png = joinpath(args["output-dir"], "zeta_vs_v_live.tmp.png")
temporary_pdf = joinpath(args["output-dir"], "zeta_vs_v_live.tmp.pdf")
save(temporary_png, figure; px_per_unit=2)
save(temporary_pdf, figure)
mv(temporary_png, png; force=true)
mv(temporary_pdf, pdf; force=true)

@info(
    "collected reblocked velocity sweep",
    expected=args["expected-points"],
    found=length(rows),
    converged=converged_count,
    within_tolerance,
    csv,
    summary,
    png,
)
flush(stderr)
