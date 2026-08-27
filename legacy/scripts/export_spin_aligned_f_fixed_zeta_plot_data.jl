#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf
using TOML

function parse_args()
    settings = ArgParseSettings(
        description="Export fixed-zeta collapse comparison data for matplotlib plotting.",
    )
    @add_arg_table! settings begin
        "--input-archive"
            arg_type = String
            required = true
        "--output-dir"
            arg_type = String
            required = true
        "--label"
            arg_type = String
            required = true
    end
    return ArgParse.parse_args(settings)
end

function panel_rows(io, panel_name, collapse, times)
    for t in times
        inds = findall(==(t), collapse.time)
        local_order = sortperm(collapse.x[inds])
        for idx in inds[local_order]
            println(io, join((
                panel_name,
                @sprintf("%.8f", t),
                @sprintf("%.8f", collapse.x[idx]),
                @sprintf("%.8f", collapse.y[idx]),
            ), '\t'))
        end
    end
end

function main()
    args = parse_args()
    archive = load(args["input-archive"])
    analysis = archive["analysis"]
    fixed_zeta = archive["fixed_zeta"]
    mkpath(args["output-dir"])

    data_path = joinpath(args["output-dir"], "collapsed_curves.tsv")
    open(data_path, "w") do io
        println(io, "panel\ttime\tx\ty")
        panel_rows(io, "free", analysis.fine_best, analysis.feature.times)
        panel_rows(io, "fixed", analysis.fixed_best, analysis.feature.times)
    end

    metadata = Dict(
        "label" => args["label"],
        "times" => collect(Float64.(analysis.feature.times)),
        "free_title" => @sprintf("Best free fit: η_F = %.3f, ζ = %.3f, χ² = %.2f",
            analysis.fine_best.eta, analysis.fine_best.zeta, analysis.fine_best.reduced_chi2),
        "fixed_title" => @sprintf("Fixed ζ = %.3f: η_F = %.3f, χ² = %.2f",
            fixed_zeta, analysis.fixed_best.eta, analysis.fixed_best.reduced_chi2),
        "free_overlap_min" => Float64(analysis.fine_best.overlap_min),
        "free_overlap_max" => Float64(analysis.fine_best.overlap_max),
        "fixed_overlap_min" => Float64(analysis.fixed_best.overlap_min),
        "fixed_overlap_max" => Float64(analysis.fixed_best.overlap_max),
    )
    metadata_path = joinpath(args["output-dir"], "metadata.toml")
    open(metadata_path, "w") do io
        TOML.print(io, metadata)
    end

    println("saved plot data: ", data_path)
    println("saved metadata: ", metadata_path)
end

main()
