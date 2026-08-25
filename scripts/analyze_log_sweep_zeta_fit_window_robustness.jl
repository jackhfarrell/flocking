#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))
include(joinpath(@__DIR__, "stage2_sweep_ensemble.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT,
    "results", "spin_aligned_f_stage2_L200_J2_Q1", "up")
const DEFAULT_RESULTS_DIR = joinpath(REPO_ROOT,
    "results", "spin_aligned_f_stage2_L200_J2_Q1", "zeta_fit_window_robustness", "up")
const DEFAULT_FIGURES_DIR = joinpath(REPO_ROOT,
    "figures", "spin_aligned_f_stage2_L200_J2_Q1", "zeta_fit_window_robustness", "up")

const RADIUS_MAX = 40.0
const POLY_ORDER = 3
const COLLAPSE_BINS = 60

function parse_args()
    settings = ArgParseSettings(description="Run the production zeta fit-window robustness scan.")
    @add_arg_table! settings begin
        "--input-dir"
            arg_type = String
            default = DEFAULT_INPUT_DIR
        "--results-dir"
            arg_type = String
            default = DEFAULT_RESULTS_DIR
        "--figures-dir"
            arg_type = String
            default = DEFAULT_FIGURES_DIR
    end
    return ArgParse.parse_args(settings)
end

# Best-fit zeta for one (v, fit window) via the established coarse-then-fine collapse scan.
function best_zeta_for_window(data, vidx, radius_mask, time_indices)
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
    return (; zeta=fine_best.zeta, eta=fine_best.eta, reduced_chi2=fine_best.reduced_chi2,
        band, band_unc)
end

function robustness_for_v(data, vidx, base_time_indices)
    # The r <= 40, all-lags fit is the reference; its sensitivity band is the published bar.
    mask40 = data.radii .<= 40.0
    reference = best_zeta_for_window(data, vidx, mask40, base_time_indices)

    # Fit-window variants: three radius cutoffs plus dropping the first / last lag at r <= 40.
    mask20 = data.radii .<= 20.0
    mask60 = data.radii .<= 60.0
    drop_first = base_time_indices[2:end]
    drop_last = base_time_indices[1:end-1]

    zeta_r20 = best_zeta_for_window(data, vidx, mask20, base_time_indices).zeta
    zeta_r40 = reference.zeta
    zeta_r60 = best_zeta_for_window(data, vidx, mask60, base_time_indices).zeta
    zeta_drop_first = best_zeta_for_window(data, vidx, mask40, drop_first).zeta
    zeta_drop_last = best_zeta_for_window(data, vidx, mask40, drop_last).zeta

    zetas = [zeta_r20, zeta_r40, zeta_r60, zeta_drop_first, zeta_drop_last]
    robust_min = minimum(zetas)
    robust_max = maximum(zetas)
    robust_spread = robust_max - robust_min
    robust_halfspread = 0.5 * robust_spread
    robust_std = std(zetas)
    band_halfwidth = reference.band_unc.zeta_halfwidth

    return (; v_index=vidx, v=data.v_values[vidx], reference,
        zeta_r20, zeta_r40, zeta_r60, zeta_drop_first, zeta_drop_last,
        robust_min, robust_max, robust_spread, robust_halfspread, robust_std,
        band_halfwidth, band_understates=robust_halfspread > band_halfwidth)
end

function write_csv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "v_index,v,log10_v,zeta_r20,zeta_r40,zeta_r60,zeta_drop_first,",
            "zeta_drop_last,zeta_robust_min,zeta_robust_max,zeta_robust_spread,",
            "zeta_robust_halfspread,zeta_robust_std,zeta_band_halfwidth,",
            "reduced_chi2,band_understates")
        for row in rows
            println(io, join((
                row.v_index,
                @sprintf("%.12g", row.v),
                @sprintf("%.12g", log10(row.v)),
                @sprintf("%.8f", row.zeta_r20),
                @sprintf("%.8f", row.zeta_r40),
                @sprintf("%.8f", row.zeta_r60),
                @sprintf("%.8f", row.zeta_drop_first),
                @sprintf("%.8f", row.zeta_drop_last),
                @sprintf("%.8f", row.robust_min),
                @sprintf("%.8f", row.robust_max),
                @sprintf("%.8f", row.robust_spread),
                @sprintf("%.8f", row.robust_halfspread),
                @sprintf("%.8f", row.robust_std),
                @sprintf("%.8f", row.band_halfwidth),
                @sprintf("%.8f", row.reference.reduced_chi2),
                row.band_understates ? 1 : 0,
            ), ","))
        end
    end
end

function write_summary(path, data, rows, input_dir)
    mkpath(dirname(path))
    understated = [row for row in rows if row.band_understates]
    open(path, "w") do io
        println(io, "# Fit-window robustness of trace-collapse zeta")
        println(io)
        println(io, "- Input directory: `", input_dir, "`")
        println(io, "- Independent trajectories: ", data.nsamples)
        println(io, "- Reference window: `r <= 40`, all positive lags (its 1.05x ",
            "sensitivity band is the published bar).")
        println(io, "- Fit-window variants: `r <= 20`, `r <= 40`, `r <= 60`, ",
            "drop-first-lag, drop-last-lag.")
        println(io, "- Robustness uncertainty: half-spread `0.5*(max - min)` of zeta ",
            "across the five variants.")
        println(io, "- The robustness half-spread exceeds the sensitivity-band halfwidth ",
            "at ", length(understated), " of ", length(rows), " velocities ",
            "(where the band understates the true uncertainty; expected at high v).")
        println(io)
        println(io, "| v | log10(v) | zeta(r40) | robust ± | band ± | chi2 | band understates |")
        println(io, "|---:|---:|---:|---:|---:|---:|:--:|")
        for row in rows
            println(io, "| ", @sprintf("%.5g", row.v), " | ",
                @sprintf("%.4f", log10(row.v)), " | ",
                @sprintf("%.4f", row.zeta_r40), " | ",
                @sprintf("%.4f", row.robust_halfspread), " | ",
                @sprintf("%.4f", row.band_halfwidth), " | ",
                @sprintf("%.4f", row.reference.reduced_chi2), " | ",
                row.band_understates ? "yes" : "no", " |")
        end
    end
end

function plot_robustness(path, rows)
    mkpath(dirname(path))
    x = log10.([row.v for row in rows])
    zeta = [row.zeta_r40 for row in rows]
    robust_low = [row.zeta_r40 - row.robust_halfspread for row in rows]
    robust_high = [row.zeta_r40 + row.robust_halfspread for row in rows]
    band_half = [row.band_halfwidth for row in rows]
    robust_half = [row.robust_halfspread for row in rows]

    fig = Figure(size=(1000, 850))
    ax1 = Axis(fig[1, 1], xlabel="log10(v)", ylabel="ζ",
        title="Trace-collapse ζ with fit-window robustness band")
    band!(ax1, x, robust_low, robust_high, color=(:darkorange3, 0.22),
        label="fit-window robustness half-spread")
    lines!(ax1, x, zeta, color=:dodgerblue4, linewidth=3)
    scatter!(ax1, x, zeta, color=:dodgerblue4, markersize=9)
    axislegend(ax1, position=:lt)

    ax2 = Axis(fig[2, 1], xlabel="log10(v)", ylabel="ζ uncertainty (halfwidth)",
        title="Robustness half-spread vs sensitivity-band halfwidth")
    lines!(ax2, x, band_half, color=:dodgerblue4, linewidth=3,
        label="sensitivity band ±")
    scatter!(ax2, x, band_half, color=:dodgerblue4, markersize=9)
    lines!(ax2, x, robust_half, color=:darkorange3, linewidth=3,
        label="fit-window robustness ±")
    scatter!(ax2, x, robust_half, color=:darkorange3, markersize=9)
    axislegend(ax2, position=:lt)
    save(path, fig)
end

args = parse_args()
data = load_stage2_sweep(args["input-dir"])
base_time_indices = findall(t -> t > 0, data.times)

rows = [robustness_for_v(data, vidx, base_time_indices)
    for vidx in eachindex(data.v_values)]

mkpath(args["results-dir"])
summary_path = joinpath(args["results-dir"], "summary.md")
csv_path = joinpath(args["results-dir"], "zeta_fit_window_robustness.csv")
archive_path = joinpath(args["results-dir"], "zeta_fit_window_robustness.jld2")
plot_path = joinpath(args["figures-dir"], "zeta_fit_window_robustness.png")

write_csv(csv_path, rows)
write_summary(summary_path, data, rows, args["input-dir"])
plot_robustness(plot_path, rows)
jldsave(archive_path; data, rows)

println("summary: ", summary_path)
println("csv: ", csv_path)
println("plot: ", plot_path)
