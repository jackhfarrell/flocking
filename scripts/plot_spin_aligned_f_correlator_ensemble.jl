#!/usr/bin/env julia

using CairoMakie
using JLD2
using Statistics

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const INPUT_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_J2_v1_gamma1")
const OUTPUT_DIR = joinpath(REPO_ROOT, "figures", "spin_aligned_f_correlator_L200_J2_v1_gamma1")
const OUTPUT_PREFIX = "spin_aligned_f_correlator_ensemble"

function main()
    isdir(INPUT_DIR) || error("missing input directory: $INPUT_DIR")
    mkpath(OUTPUT_DIR)

    files = String[]
    for (dir, _, names) in walkdir(INPUT_DIR)
        for name in names
            startswith(name, "job_") && endswith(name, ".jld2") || continue
            push!(files, joinpath(dir, name))
        end
    end
    isempty(files) && error("no job_*.jld2 files found in $INPUT_DIR")
    sort!(files)

    runs = [load(file, "result") for file in files]
    radii = runs[1].radii
    times = runs[1].times
    shape = size(runs[1].F_mean)
    all(run -> run.radii == radii && run.times == times && size(run.F_mean) == shape, runs) ||
        error("all runs must share the same radii, times, and F_mean shape")

    stack = cat((run.F_mean for run in runs)...; dims=3)
    F_mean = dropdims(mean(stack; dims=3), dims=3)
    F_stderr = if length(runs) == 1
        zeros(size(F_mean))
    else
        dropdims(std(stack; dims=3), dims=3) ./ sqrt(length(runs))
    end

    result = (; config=runs[1].config, radii, times, F_mean, F_stderr)
    jldsave(joinpath(OUTPUT_DIR, OUTPUT_PREFIX * ".jld2"); result)

    fig = Figure(size=(1100, 700))
    Label(fig[0, :], "Spin-aligned F correlator across $(length(runs)) runs",
        fontsize=22)
    ax = Axis(fig[1, 1], xlabel="radius", ylabel="F(r, t)")
    radius_mask = radii .<= 40
    plot_radii = radii[radius_mask]
    palette = Makie.wong_colors()
    for (k, tidx) in enumerate(2:length(times))
        color = palette[mod1(k, length(palette))]
        mean = F_mean[radius_mask, tidx]
        err = F_stderr[radius_mask, tidx]
        band!(ax, plot_radii, mean .- err, mean .+ err, color=(color, 0.18))
        lines!(ax, plot_radii, mean, color=color, linewidth=3,
            label="t = $(round(times[tidx]; digits=3))")
    end
    xlims!(ax, 0, 40)
    axislegend(ax, position=:rb, nbanks=2)
    save(joinpath(OUTPUT_DIR, OUTPUT_PREFIX * "_traces.png"), fig)

    println("saved ", length(runs), " runs from ", INPUT_DIR)
end

main()
