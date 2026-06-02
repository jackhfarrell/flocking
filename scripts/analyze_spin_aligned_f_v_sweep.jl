#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT,
    "results", "spin_aligned_f_correlator_L200_J2_Q1_v_log_sweep")
const DEFAULT_RESULTS_DIR = DEFAULT_INPUT_DIR
const DEFAULT_FIGURES_DIR = joinpath(REPO_ROOT,
    "figures", "spin_aligned_f_correlator_L200_J2_Q1_v_log_sweep")

function parse_args()
    settings = ArgParseSettings(
        description="Analyze a baked-equilibrium spin-aligned F-correlator v sweep.",
    )
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
        "--radius-max"
            arg_type = Float64
            default = 40.0
        "--time-min"
            arg_type = Float64
            default = 0.0
        "--poly-order"
            arg_type = Int
            default = 3
        "--collapse-bins"
            arg_type = Int
            default = 60
        "--eta-min"
            arg_type = Float64
            default = -0.2
        "--eta-max"
            arg_type = Float64
            default = 0.8
        "--eta-step"
            arg_type = Float64
            default = 0.02
        "--zeta-min"
            arg_type = Float64
            default = 0.0
        "--zeta-max"
            arg_type = Float64
            default = 0.8
        "--zeta-step"
            arg_type = Float64
            default = 0.02
        "--fine-window"
            arg_type = Float64
            default = 0.06
        "--fine-step"
            arg_type = Float64
            default = 0.005
        "--bootstrap-samples"
            arg_type = Int
            default = 200
        "--bootstrap-seed"
            arg_type = Int
            default = 12345
        "--selected-v-indices"
            arg_type = String
            default = ""
    end
    return ArgParse.parse_args(settings)
end

function collect_sample_files(input_dir::String)
    isdir(input_dir) || error("missing input directory: $input_dir")
    files = String[]
    for (dir, _, names) in walkdir(input_dir)
        for name in names
            startswith(name, "sample_") && endswith(name, ".jld2") || continue
            push!(files, joinpath(dir, name))
        end
    end
    isempty(files) && error("no sample_*.jld2 files found in $input_dir")
    sort!(files)
    return files
end

function load_v_sweep(files::AbstractVector{<:AbstractString})
    runs = [load(file, "result") for file in files]
    v_values = runs[1].v_values
    radii = runs[1].radii
    times = runs[1].times
    shape = size(runs[1].F)
    all(run -> run.v_values == v_values && run.radii == radii &&
        run.times == times && size(run.F) == shape, runs) ||
        error("all sample files must share v_values, radii, times, and F shape")
    F_samples = cat((run.F for run in runs)...; dims=4)
    C_plus_samples = cat((run.C_plus for run in runs)...; dims=4)
    C_minus_samples = cat((run.C_minus for run in runs)...; dims=4)
    return (; config=runs[1].config, files, v_values, radii, times,
        F_samples, C_plus_samples, C_minus_samples, nsamples=length(runs))
end

function sample_mean_stderr(samples, vidx::Integer)
    stack = samples[vidx, :, :, :]
    mean_field = dropdims(mean(stack; dims=3), dims=3)
    stderr_field = if size(stack, 3) == 1
        zeros(size(mean_field))
    else
        dropdims(std(stack; dims=3), dims=3) ./ sqrt(size(stack, 3))
    end
    return mean_field, stderr_field
end

function finite_stderr(values::AbstractVector{<:Real})
    finite_values = filter(isfinite, values)
    isempty(finite_values) && return NaN
    length(finite_values) == 1 && return NaN
    return std(finite_values)
end

function finite_mean(values::AbstractVector{<:Real})
    finite_values = filter(isfinite, values)
    isempty(finite_values) && return NaN
    return mean(finite_values)
end

function fit_curve_collapse(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        args)
    eta_values = collect(args["eta-min"]:args["eta-step"]:args["eta-max"])
    zeta_values = collect(args["zeta-min"]:args["zeta-step"]:args["zeta-max"])
    _, coarse_best = scan_grid(radii, times, F_mean, F_stderr, radius_mask,
        time_indices, eta_values, zeta_values, args["poly-order"], args["collapse-bins"])
    fine_eta_values = collect((coarse_best.eta - args["fine-window"]):
        args["fine-step"]:(coarse_best.eta + args["fine-window"]))
    fine_zeta_values = collect((coarse_best.zeta - args["fine-window"]):
        args["fine-step"]:(coarse_best.zeta + args["fine-window"]))
    _, fine_best = scan_grid(radii, times, F_mean, F_stderr, radius_mask,
        time_indices, fine_eta_values, fine_zeta_values, args["poly-order"],
        args["collapse-bins"])
    return fine_best
end

function fit_curve_collapse_near(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        center, args)
    eta_values = collect((center.eta - args["fine-window"]):
        args["fine-step"]:(center.eta + args["fine-window"]))
    zeta_values = collect((center.zeta - args["fine-window"]):
        args["fine-step"]:(center.zeta + args["fine-window"]))
    _, best = scan_grid(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        eta_values, zeta_values, args["poly-order"], args["collapse-bins"])
    return best
end

function bootstrap_curve_collapse(rng, samples, vidx::Integer, radii, times, radius_mask,
        time_indices, center, args)
    nsamples = size(samples, 4)
    zetas = Float64[]
    etas = Float64[]
    chi2s = Float64[]
    for _ in 1:args["bootstrap-samples"]
        inds = rand(rng, 1:nsamples, nsamples)
        stack = samples[vidx, :, :, inds]
        mean_field = dropdims(mean(stack; dims=3), dims=3)
        stderr_field = if nsamples == 1
            zeros(size(mean_field))
        else
            dropdims(std(stack; dims=3), dims=3) ./ sqrt(nsamples)
        end
        best = fit_curve_collapse_near(radii, times, mean_field, stderr_field,
            radius_mask, time_indices, center, args)
        push!(zetas, best.zeta)
        push!(etas, best.eta)
        push!(chi2s, best.reduced_chi2)
    end
    return (; zeta_mean=finite_mean(zetas), zeta_stderr=finite_stderr(zetas),
        eta_mean=finite_mean(etas), eta_stderr=finite_stderr(etas),
        chi2_mean=finite_mean(chi2s), chi2_stderr=finite_stderr(chi2s))
end

function selected_indices(spec::AbstractString, nv::Integer)
    if isempty(spec)
        return unique([1, cld(nv, 2), nv])
    end
    inds = [parse(Int, strip(value)) for value in split(spec, ',')]
    all(i -> 1 <= i <= nv, inds) || error("--selected-v-indices contains an out-of-range index")
    return unique(inds)
end

function write_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(("v_index", "v", "nsamples", "collapse_eta",
            "collapse_eta_bootstrap_stderr", "collapse_zeta", "collapse_zeta_bootstrap_stderr",
            "collapse_reduced_chi2", "collapse_chi2_bootstrap_stderr",
            "collapse_overlap_min", "collapse_overlap_max"), ','))
        for row in rows
            println(io, join((
                row.v_index,
                @sprintf("%.12g", row.v),
                row.nsamples,
                @sprintf("%.8f", row.collapse.eta),
                @sprintf("%.8f", row.bootstrap.eta_stderr),
                @sprintf("%.8f", row.collapse.zeta),
                @sprintf("%.8f", row.bootstrap.zeta_stderr),
                @sprintf("%.8f", row.collapse.reduced_chi2),
                @sprintf("%.8f", row.bootstrap.chi2_stderr),
                @sprintf("%.8f", row.collapse.overlap_min),
                @sprintf("%.8f", row.collapse.overlap_max),
            ), ','))
        end
    end
end

function write_summary(path::String, sweep, rows, args)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Spin-aligned F v-sweep summary")
        println(io)
        println(io, "- Input directory: `", args["input-dir"], "`")
        println(io, "- Number of samples: ", sweep.nsamples)
        println(io, "- Number of v values: ", length(sweep.v_values))
        println(io, "- Radius cutoff: `r <= ", @sprintf("%.1f", args["radius-max"]), "`")
        println(io, "- Time cutoff: `t >= ", @sprintf("%.1f", args["time-min"]), "`")
        println(io, "- Bootstrap resamples: ", args["bootstrap-samples"])
        println(io)
        println(io, "| v | collapse zeta | bootstrap stderr | collapse eta_F | eta stderr | chi2 |")
        println(io, "|---:|---:|---:|---:|---:|---:|")
        for row in rows
            println(io, "| ", @sprintf("%.5g", row.v), " | ",
                @sprintf("%.4f", row.collapse.zeta), " | ",
                @sprintf("%.4f", row.bootstrap.zeta_stderr), " | ",
                @sprintf("%.4f", row.collapse.eta), " | ",
                @sprintf("%.4f", row.bootstrap.eta_stderr), " | ",
                @sprintf("%.4f", row.collapse.reduced_chi2), " |")
        end
    end
end

function plot_zeta_vs_v(path::String, rows)
    mkpath(dirname(path))
    v = [row.v for row in rows]
    collapse_zeta = [row.collapse.zeta for row in rows]
    collapse_err = [row.bootstrap.zeta_stderr for row in rows]

    fig = Figure(size=(900, 650))
    ax = Axis(fig[1, 1], xlabel="v", ylabel="ζ", xscale=log10,
        title="Spin-aligned F exponent sweep at J = 2")
    errorbars!(ax, v, collapse_zeta, collapse_err, linewidth=2)
    scatterlines!(ax, v, collapse_zeta, markersize=10, linewidth=3,
        label="curve collapse")
    hlines!(ax, [0.390], color=:gray45, linestyle=:dash, linewidth=2,
        label="paper ζ = 0.390")
    axislegend(ax, position=:rt)
    save(path, fig)
end

function main()
    args = parse_args()
    args["radius-max"] > 0 || throw(ArgumentError("--radius-max must be positive"))
    args["bootstrap-samples"] > 0 ||
        throw(ArgumentError("--bootstrap-samples must be positive"))

    files = collect_sample_files(args["input-dir"])
    sweep = load_v_sweep(files)
    radius_mask = sweep.radii .<= args["radius-max"]
    any(radius_mask) || error("no radii satisfy r <= $(args["radius-max"])")
    time_indices = findall(t -> t > 0 && t >= args["time-min"], sweep.times)
    length(time_indices) >= 3 || error("need at least 3 positive times for zeta fits")

    rng = MersenneTwister(args["bootstrap-seed"])
    rows = []

    for vidx in eachindex(sweep.v_values)
        F_mean, F_stderr = sample_mean_stderr(sweep.F_samples, vidx)
        fine_best = fit_curve_collapse(sweep.radii, sweep.times, F_mean, F_stderr,
            radius_mask, time_indices, args)
        bootstrap = bootstrap_curve_collapse(rng, sweep.F_samples, vidx, sweep.radii,
            sweep.times, radius_mask, time_indices, fine_best, args)
        push!(rows, (; v_index=vidx, v=sweep.v_values[vidx], nsamples=sweep.nsamples,
            bootstrap, collapse=fine_best))
        println(@sprintf("v[%02d] = %.6g: collapse zeta = %.4f ± %.4f, eta_F = %.4f ± %.4f",
            vidx, sweep.v_values[vidx], fine_best.zeta, bootstrap.zeta_stderr,
            fine_best.eta, bootstrap.eta_stderr))
    end

    mkpath(args["results-dir"])
    mkpath(args["figures-dir"])
    csv_path = joinpath(args["results-dir"], "collapse_summary.csv")
    summary_path = joinpath(args["results-dir"], "collapse_summary.md")
    archive_path = joinpath(args["results-dir"], "collapse_summary.jld2")
    zeta_plot_path = joinpath(args["figures-dir"], "zeta_vs_v.png")

    write_csv(csv_path, rows)
    write_summary(summary_path, sweep, rows, args)
    plot_zeta_vs_v(zeta_plot_path, rows)

    for vidx in selected_indices(args["selected-v-indices"], length(sweep.v_values))
        F_mean, F_stderr = sample_mean_stderr(sweep.F_samples, vidx)
        row = rows[vidx]
        prefix = @sprintf("v%02d_%s", vidx,
            replace(@sprintf("%.6g", sweep.v_values[vidx]), "." => "p"))
        raw_path = joinpath(args["figures-dir"], prefix * "_raw_traces.png")
        collapse_path = joinpath(args["figures-dir"], prefix * "_collapsed.png")
        curve_times = (; times=sweep.times[time_indices])
        plot_raw_traces(raw_path, sweep.radii, sweep.times, F_mean, F_stderr,
            radius_mask, time_indices;
            title=@sprintf("Spin-aligned F(r,t), v = %.5g", sweep.v_values[vidx]))
        plot_collapsed_traces(collapse_path, row.collapse, curve_times;
            title=@sprintf("v = %.5g, η_F = %.3f, ζ = %.3f",
            sweep.v_values[vidx], row.collapse.eta, row.collapse.zeta))
    end

    jldsave(archive_path; sweep, rows, args)
    println("saved summary csv: ", csv_path)
    println("saved summary markdown: ", summary_path)
    println("saved archive: ", archive_path)
    println("saved zeta plot: ", zeta_plot_path)
end

main()
