#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))
include(joinpath(@__DIR__, "stage2_sweep_ensemble.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_stage2_L200_J2_Q1")
const DEFAULT_RESULTS_DIR = joinpath(DEFAULT_INPUT_DIR, "zeta_vs_logv_two_direction")
const DEFAULT_FIGURES_DIR = joinpath(REPO_ROOT, "figures", "spin_aligned_f_stage2_L200_J2_Q1")

const RADIUS_MAX = 40.0
const TIME_MIN = 0.0
const POLY_ORDER = 3
const COLLAPSE_BINS = 60

function parse_args()
    settings = ArgParseSettings(description="Analyze production Stage-2 up/down velocity sweeps.")
    @add_arg_table! settings begin
        "--up-dir"
            arg_type = String
            default = joinpath(DEFAULT_INPUT_DIR, "up")
        "--down-dir"
            arg_type = String
            default = joinpath(DEFAULT_INPUT_DIR, "down")
        "--results-dir"
            arg_type = String
            default = DEFAULT_RESULTS_DIR
        "--figures-dir"
            arg_type = String
            default = DEFAULT_FIGURES_DIR
    end
    return ArgParse.parse_args(settings)
end

function trace_collapse_for_v(data, vidx, radius_mask, time_indices)
    mean_field = data.mean_F[vidx, :, :]
    stderr_field = data.stderr_F[vidx, :, :]

    coarse_eta_values = collect(-0.2:0.02:1.6)
    coarse_zeta_values = collect(0.0:0.02:0.8)
    _, coarse_best = scan_grid(data.radii, data.times, mean_field, stderr_field,
        radius_mask, time_indices, coarse_eta_values, coarse_zeta_values, POLY_ORDER,
        COLLAPSE_BINS)

    fine_eta_values = collect((coarse_best.eta - 0.06):0.005:(coarse_best.eta + 0.06))
    fine_zeta_values = collect((coarse_best.zeta - 0.06):0.005:(coarse_best.zeta + 0.06))
    fine_objective, fine_best = scan_grid(data.radii, data.times, mean_field, stderr_field,
        radius_mask, time_indices, fine_eta_values, fine_zeta_values, POLY_ORDER,
        COLLAPSE_BINS)

    band = sensitivity_band(fine_objective, fine_eta_values, fine_zeta_values, 1.05)
    band_unc = band_summary(band, fine_eta_values, fine_zeta_values)
    return (; v_index=vidx, v=data.v_values[vidx], coarse_best, fine_best, band, band_unc)
end

function collapse_rows(data)
    radius_mask = data.radii .<= RADIUS_MAX
    time_indices = findall(t -> t > 0 && t >= TIME_MIN, data.times)
    return [trace_collapse_for_v(data, vidx, radius_mask, time_indices)
        for vidx in eachindex(data.v_values)]
end

function write_csv(path, direction_rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "direction,v_index,v,log10_v,zeta,zeta_low,zeta_high," *
            "zeta_band_center,zeta_band_halfwidth,eta_F,reduced_chi2")
        for (direction, rows) in direction_rows
            for row in rows
                println(io, join((
                    direction,
                    row.v_index,
                    @sprintf("%.12g", row.v),
                    @sprintf("%.12g", log10(row.v)),
                    @sprintf("%.8f", row.fine_best.zeta),
                    @sprintf("%.8f", row.band.zeta_min),
                    @sprintf("%.8f", row.band.zeta_max),
                    @sprintf("%.8f", row.band_unc.zeta_center),
                    @sprintf("%.8f", row.band_unc.zeta_halfwidth),
                    @sprintf("%.8f", row.fine_best.eta),
                    @sprintf("%.8f", row.fine_best.reduced_chi2),
                ), ","))
            end
        end
    end
end

function write_summary(path, direction_data, direction_rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Trace-collapse zeta versus log10(v), up versus down sweep")
        println(io)
        for (direction, data) in direction_data
            println(io, "- `", direction, "` samples: ", data.nsamples,
                " independent trajectories (", data.input_dir, ")")
        end
        println(io, "- Radius cutoff: `r <= ", @sprintf("%.1f", RADIUS_MAX), "`")
        println(io, "- Error envelope: sensitivity band with `objective <= 1.05 * min`.")
        println(io, "- Agreement between directions certifies equilibration; divergence",
            " localizes a transition (hysteresis).")
        println(io)
        println(io, "| direction | v | log10(v) | zeta | band low | band high | eta_F | chi2 |")
        println(io, "|:--|---:|---:|---:|---:|---:|---:|---:|")
        for (direction, rows) in direction_rows
            for row in rows
                println(io, "| ", direction, " | ", @sprintf("%.5g", row.v), " | ",
                    @sprintf("%.4f", log10(row.v)), " | ",
                    @sprintf("%.4f", row.fine_best.zeta), " | ",
                    @sprintf("%.4f", row.band.zeta_min), " | ",
                    @sprintf("%.4f", row.band.zeta_max), " | ",
                    @sprintf("%.4f", row.fine_best.eta), " | ",
                    @sprintf("%.4f", row.fine_best.reduced_chi2), " |")
            end
        end
    end
end

function plot_two_direction(path, direction_rows)
    mkpath(dirname(path))
    styles = Dict(
        "up" => (line=:dodgerblue4, band=:dodgerblue3),
        "down" => (line=:firebrick4, band=:firebrick2),
    )
    fig = Figure(size=(1000, 650))
    ax = Axis(fig[1, 1], xlabel="log10(v)", ylabel="ζ",
        title="Trace-collapse ζ(v): up- versus down-sweep")
    for (direction, rows) in direction_rows
        x = log10.([row.v for row in rows])
        zeta = [row.fine_best.zeta for row in rows]
        zeta_low = [row.band.zeta_min for row in rows]
        zeta_high = [row.band.zeta_max for row in rows]
        style = get(styles, direction, (line=:black, band=:gray60))
        band!(ax, x, zeta_low, zeta_high, color=(style.band, 0.22))
        lines!(ax, x, zeta, color=style.line, linewidth=3, label="$direction sweep")
        scatter!(ax, x, zeta, color=style.line, markersize=9)
    end
    axislegend(ax, position=:lt)
    save(path, fig)
end

args = parse_args()
groups = ["up" => args["up-dir"], "down" => args["down-dir"]]
direction_data = [direction => merge(load_stage2_sweep(input_dir), (; input_dir))
    for (direction, input_dir) in groups]
direction_rows = [direction => collapse_rows(data) for (direction, data) in direction_data]

mkpath(args["results-dir"])
summary_path = joinpath(args["results-dir"], "zeta_vs_logv_two_direction_summary.md")
csv_path = joinpath(args["results-dir"], "zeta_vs_logv_two_direction.csv")
archive_path = joinpath(args["results-dir"], "zeta_vs_logv_two_direction.jld2")
plot_path = joinpath(args["figures-dir"], "zeta_vs_logv_two_direction_trace_collapse.png")

write_csv(csv_path, direction_rows)
write_summary(summary_path, direction_data, direction_rows)
plot_two_direction(plot_path, direction_rows)
jldsave(archive_path; direction_data, direction_rows)

println("summary: ", summary_path)
println("csv: ", csv_path)
println("plot: ", plot_path)
