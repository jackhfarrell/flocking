#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

settings = ArgParseSettings(description="Collect threaded convergence sweep checkpoints.")
@add_arg_table! settings begin
    "--input-dir"
        arg_type = String
        default = "results/threaded_v_sweep"
    "--output-dir"
        arg_type = String
        default = "analysis/threaded_v_sweep"
    "--expected-points"
        arg_type = Int
        default = 0
end
args = ArgParse.parse_args(settings)

files = sort([
    joinpath(directory, name)
    for (directory, _, names) in walkdir(args["input-dir"])
    for name in names if endswith(name, ".jld2")
])

rows = NamedTuple[]
for file in files
    run = load(file, "run")
    latest = isempty(run.history) ? nothing : run.history[end]
    value = (name, fallback=NaN) -> latest === nothing || !hasproperty(latest, name) ?
        fallback : getproperty(latest, name)
    push!(rows, (;
        temperature=run.config.temperature,
        J=run.config.J,
        v=run.config.v,
        log10_v=log10(run.config.v),
        L=run.config.L,
        chains=run.config.nchains,
        rounds=run.nrounds,
        samples=run.nrounds * run.config.nchains,
        blocks=size(run.F_blocks, 3),
        eta=value(:eta),
        zeta=value(:zeta),
        uncertainty=value(:estimated_error),
        jackknife_error=value(:jackknife_error),
        between_chain_error=value(:between_chain_error),
        chain_range_halfwidth=value(:cross_chain_halfwidth),
        half_run_halfwidth=value(:half_run_halfwidth),
        time_halfspread=value(:time_halfspread),
        radius_halfspread=value(:radius_halfspread),
        stability_error=value(:stability_error),
        converged=value(:converged, false),
        file,
    ))
end
sort!(rows; by=row -> (row.temperature, row.v))

mkpath(args["output-dir"])
csv = joinpath(args["output-dir"], "zeta_vs_v_temperature.csv")
open(csv, "w") do io
    if !isempty(rows)
        println(io, join(keys(rows[1]), ","))
        for row in rows
            println(io, join(values(row), ","))
        end
    end
end

converged_count = count(row -> row.converged, rows)
summary = joinpath(args["output-dir"], "summary.md")
open(summary, "w") do io
    println(io, "# Threaded velocity sweep")
    println(io)
    println(io, "- Expected points: `", args["expected-points"], "`")
    println(io, "- Checkpoints found: `", length(rows), "`")
    println(io, "- Converged to requested tolerance: `", converged_count, "`")
    println(io, "- Incomplete or unconverged: `", length(rows) - converged_count, "`")
end

@info "collected threaded sweep" expected=args["expected-points"] found=length(rows) converged=converged_count csv summary
