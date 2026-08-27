#!/usr/bin/env julia

using ArgParse
using CairoMakie
using DelimitedFiles
using JLD2
using Printf
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results",
    "ordinary_c_correlator_L200_J2_v0_v1_gamma1_20260525_142415")
const DEFAULT_OUTPUT_DIR = joinpath(REPO_ROOT, "figures",
    "ordinary_c_correlator_L200_J2_v0_v1_gamma1_20260525_142415")

settings = ArgParseSettings(
    description="Plot a split heatmap of ordinary C(r,t): v=0 on the left, v=1 on the right.",
)
@add_arg_table! settings begin
    "--input-dir"
        arg_type = String
        default = DEFAULT_INPUT_DIR
    "--output-dir"
        arg_type = String
        default = DEFAULT_OUTPUT_DIR
    "--prefix"
        arg_type = String
        default = "ordinary_c_split"
    "--r-max"
        arg_type = Float64
        default = Inf
    "--time-upsampling"
        arg_type = Int
        default = 1
    "--connected"
        action = :store_true
    "--background-r-min"
        arg_type = Float64
        default = 80.0
    "--color-scale"
        arg_type = String
        default = "linear"
end
args = parse_args(settings)

files = sort(filter(path -> startswith(basename(path), "job_") && endswith(path, ".jld2"),
    readdir(args["input-dir"]; join=true)))
isempty(files) && error("no job_*.jld2 files found in $(args["input-dir"])")

first_result = load(files[1], "result")
radii = first_result.passive.radii
times = first_result.passive.times_signed
passive_mean = zeros(Float64, size(first_result.passive.C_mean))
active_mean = zeros(Float64, size(first_result.active.C_mean))

for file in files
    result = load(file, "result")
    result.passive.radii == radii || error("radii mismatch in $file")
    result.active.radii == radii || error("active radii mismatch in $file")
    result.passive.times_signed == times || error("time mismatch in $file")
    result.active.times_signed == times || error("active time mismatch in $file")
    passive_mean .+= result.passive.C_mean
    active_mean .+= result.active.C_mean
end
passive_mean ./= length(files)
active_mean ./= length(files)

if args["connected"]
    background_mask = radii .>= args["background-r-min"]
    any(background_mask) ||
        error("no radii satisfy --background-r-min=$(args["background-r-min"])")
    passive_background = vec(mean(passive_mean[background_mask, :]; dims=1))
    active_background = vec(mean(active_mean[background_mask, :]; dims=1))
    passive_mean .-= reshape(passive_background, 1, :)
    active_mean .-= reshape(active_background, 1, :)
end

radius_mask = radii .<= args["r-max"]
any(radius_mask) || error("no radii satisfy --r-max=$(args["r-max"])")
radii_plot = radii[radius_mask]
passive_plot = passive_mean[radius_mask, :]
active_plot = active_mean[radius_mask, :]

args["time-upsampling"] >= 1 || error("--time-upsampling must be at least 1")
if args["time-upsampling"] == 1
    times_plot = times
else
    nsegments = length(times) - 1
    times_plot = range(first(times), last(times);
        length=nsegments * args["time-upsampling"] + 1) |> collect

    function interpolate_time(C, times, times_plot)
        C_plot = Matrix{Float64}(undef, size(C, 1), length(times_plot))
        right_index = 2
        @inbounds for (j, t) in enumerate(times_plot)
            while right_index < length(times) && t > times[right_index]
                right_index += 1
            end
            left_index = max(1, right_index - 1)
            if t <= first(times)
                C_plot[:, j] .= C[:, 1]
            elseif t >= last(times)
                C_plot[:, j] .= C[:, end]
            else
                weight = (t - times[left_index]) / (times[right_index] - times[left_index])
                C_plot[:, j] .= (1 - weight) .* C[:, left_index] .+
                    weight .* C[:, right_index]
            end
        end
        return C_plot
    end

    passive_plot = interpolate_time(passive_plot, times, times_plot)
    active_plot = interpolate_time(active_plot, times, times_plot)
end

x = vcat(-reverse(radii_plot), radii_plot)
split_C = vcat(reverse(passive_plot; dims=1), active_plot)
colorrange = extrema(split_C)
color_scale = if args["color-scale"] == "linear"
    identity
elseif args["color-scale"] == "sqrt"
    minimum(split_C) >= 0 || error("--color-scale sqrt requires nonnegative values")
    sqrt
elseif args["color-scale"] == "log10"
    minimum(split_C) > 0 || error("--color-scale log10 requires positive values")
    log10
else
    error("--color-scale must be one of: linear, sqrt, log10")
end
tick_step = maximum(radii_plot) <= 30 ? 10 : 25
xticks = collect(-floor(Int, maximum(radii_plot) / tick_step) * tick_step:tick_step:
    floor(Int, maximum(radii_plot) / tick_step) * tick_step)
xticklabels = [@sprintf("%g", abs(tick)) for tick in xticks]

mkpath(args["output-dir"])
archive_path = joinpath(args["output-dir"], args["prefix"] * ".jld2")
jldsave(archive_path; radii, times, passive_mean, active_mean, split_C,
    radii_plot, times_plot, passive_plot, active_plot, input_dir=args["input-dir"], files,
    colorrange, time_upsampling=args["time-upsampling"], connected=args["connected"],
    background_r_min=args["background-r-min"], color_scale=args["color-scale"])
csv_prefix = joinpath(args["output-dir"], args["prefix"])
writedlm(csv_prefix * "_x.csv", x, ',')
writedlm(csv_prefix * "_t.csv", times_plot, ',')
writedlm(csv_prefix * "_C.csv", split_C', ',')
open(csv_prefix * "_metadata.txt", "w") do io
    println(io, "connected=", args["connected"])
    println(io, "background_r_min=", args["background-r-min"])
    println(io, "time_upsampling=", args["time-upsampling"])
    println(io, "color_scale=", args["color-scale"])
    println(io, "input_dir=", args["input-dir"])
    println(io, "jobs=", length(files))
end

set_theme!(theme_latexfonts())
fig = Figure(size=(980, 620), fontsize=22)
ax = Axis(fig[1, 1];
    xlabel="r",
    ylabel="t",
    title=args["connected"] ? "Connected C(r,t)" : "Ordinary C(r,t)",
    xticks=(xticks, xticklabels),
    limits=((-maximum(radii_plot), maximum(radii_plot)), (minimum(times_plot), maximum(times_plot))),
)
hm = heatmap!(ax, x, times_plot, split_C; colormap=:viridis, colorrange,
    colorscale=color_scale)
vlines!(ax, [0.0]; color=:white, linewidth=2)
text!(ax, -0.5maximum(radii_plot), maximum(times_plot); text="v = 0", align=(:center, :top),
    color=:white, fontsize=22)
text!(ax, 0.5maximum(radii_plot), maximum(times_plot); text="v = 1", align=(:center, :top),
    color=:white, fontsize=22)
Colorbar(fig[1, 2], hm; label=args["connected"] ? "C_conn(r,t)" : "C(r,t)",
    scale=color_scale)

png_path = joinpath(args["output-dir"], args["prefix"] * ".png")
pdf_path = joinpath(args["output-dir"], args["prefix"] * ".pdf")
save(png_path, fig; px_per_unit=2)
save(pdf_path, fig)

println("averaged jobs: ", length(files))
println("saved data: ", archive_path)
println("saved csv prefix: ", csv_prefix)
println("saved figure: ", png_path)
println("saved figure: ", pdf_path)
