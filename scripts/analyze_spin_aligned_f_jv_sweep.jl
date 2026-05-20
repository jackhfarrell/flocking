#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_ROOT = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_gamma1_jv_sweep")
const DEFAULT_RESULTS_ROOT = DEFAULT_INPUT_ROOT
const DEFAULT_FIGURES_ROOT = joinpath(REPO_ROOT, "figures", "spin_aligned_f_correlator_L200_gamma1_jv_sweep")
const DATASET_REGEX = r"^J([0-9p]+)_v([0-9p]+)$"

function parse_args()
    settings = ArgParseSettings(
        description="Batch collapse analysis for the spin-aligned F(r,t) J-v sweep.",
    )
    @add_arg_table! settings begin
        "--input-root"
            arg_type = String
            default = DEFAULT_INPUT_ROOT
        "--results-root"
            arg_type = String
            default = DEFAULT_RESULTS_ROOT
        "--figures-root"
            arg_type = String
            default = DEFAULT_FIGURES_ROOT
        "--radius-max"
            arg_type = Float64
            default = 80.0
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
        "--sensitivity-factor"
            arg_type = Float64
            default = 1.05
        "--collapse-chi2-max"
            arg_type = Float64
            default = 1.5
        "--eta-feature-tol"
            arg_type = Float64
            default = 0.15
        "--zeta-feature-tol"
            arg_type = Float64
            default = 0.12
        "--boundary-tol"
            arg_type = Float64
            default = 0.02
        "--zeta-min-physical"
            arg_type = Float64
            default = 0.05
        "--phase-j-min"
            arg_type = Float64
            default = 1 / 0.9
    end
    return ArgParse.parse_args(settings)
end

parse_tag_value(tag::AbstractString) = parse(Float64, replace(tag, 'p' => '.'))

function parse_dataset_name(name::AbstractString)
    match_result = match(DATASET_REGEX, name)
    match_result === nothing && error("invalid dataset directory name: $name")
    return (; J=parse_tag_value(match_result.captures[1]),
        v=parse_tag_value(match_result.captures[2]))
end

function validate_args(args)
    args["radius-max"] > 0 || throw(ArgumentError("--radius-max must be positive"))
    args["poly-order"] >= 0 || throw(ArgumentError("--poly-order must be nonnegative"))
    args["collapse-bins"] >= 2 || throw(ArgumentError("--collapse-bins must be at least 2"))
    args["eta-step"] > 0 || throw(ArgumentError("--eta-step must be positive"))
    args["zeta-step"] > 0 || throw(ArgumentError("--zeta-step must be positive"))
    args["fine-window"] > 0 || throw(ArgumentError("--fine-window must be positive"))
    args["fine-step"] > 0 || throw(ArgumentError("--fine-step must be positive"))
    args["sensitivity-factor"] >= 1 ||
        throw(ArgumentError("--sensitivity-factor must be at least 1"))
    args["collapse-chi2-max"] > 0 ||
        throw(ArgumentError("--collapse-chi2-max must be positive"))
    args["eta-feature-tol"] >= 0 ||
        throw(ArgumentError("--eta-feature-tol must be nonnegative"))
    args["zeta-feature-tol"] >= 0 ||
        throw(ArgumentError("--zeta-feature-tol must be nonnegative"))
    args["boundary-tol"] >= 0 ||
        throw(ArgumentError("--boundary-tol must be nonnegative"))
    args["zeta-min-physical"] >= 0 ||
        throw(ArgumentError("--zeta-min-physical must be nonnegative"))
    args["phase-j-min"] > 0 ||
        throw(ArgumentError("--phase-j-min must be positive"))
end

function collect_dataset_dirs(input_root::String)
    isdir(input_root) || error("missing input root: $input_root")
    dirs = String[]
    for entry in sort(readdir(input_root; join=true))
        isdir(entry) || continue
        occursin(DATASET_REGEX, basename(entry)) || continue
        push!(dirs, entry)
    end
    isempty(dirs) && error("no J*_v* dataset directories found in $input_root")
    return dirs
end

function analyze_dataset(dataset_dir::String, args)
    files = collect_job_files(dataset_dir)
    ensemble = load_ensemble(files)
    radius_mask = ensemble.radii .<= args["radius-max"]
    any(radius_mask) || error("no radii satisfy r <= $(args["radius-max"]) in $dataset_dir")
    time_indices = findall(>(0), ensemble.times)
    isempty(time_indices) && error("no positive times available in $dataset_dir")

    eta_values = collect(args["eta-min"]:args["eta-step"]:args["eta-max"])
    zeta_values = collect(args["zeta-min"]:args["zeta-step"]:args["zeta-max"])
    coarse_objective, coarse_best = scan_grid(ensemble.radii, ensemble.times,
        ensemble.F_mean, ensemble.F_stderr, radius_mask, time_indices, eta_values,
        zeta_values, args["poly-order"], args["collapse-bins"])

    fine_eta_values = collect((coarse_best.eta - args["fine-window"]):
        args["fine-step"]:(coarse_best.eta + args["fine-window"]))
    fine_zeta_values = collect((coarse_best.zeta - args["fine-window"]):
        args["fine-step"]:(coarse_best.zeta + args["fine-window"]))
    fine_objective, fine_best = scan_grid(ensemble.radii, ensemble.times,
        ensemble.F_mean, ensemble.F_stderr, radius_mask, time_indices, fine_eta_values,
        fine_zeta_values, args["poly-order"], args["collapse-bins"])

    band = sensitivity_band(fine_objective, fine_eta_values, fine_zeta_values,
        args["sensitivity-factor"])
    feature = feature_estimate(ensemble.radii, ensemble.times, ensemble.F_mean,
        radius_mask, time_indices)

    return (;
        files,
        ensemble,
        radius_mask,
        time_indices,
        coarse_eta_values=eta_values,
        coarse_zeta_values=zeta_values,
        coarse_objective,
        coarse_best,
        fine_eta_values,
        fine_zeta_values,
        fine_objective,
        fine_best,
        band,
        feature,
    )
end

function assess_collapse_quality(params, analysis, args)
    reasons = String[]
    warnings = String[]
    chi2 = analysis.fine_best.reduced_chi2
    eta_delta = abs(analysis.fine_best.eta - analysis.feature.eta)
    zeta_delta = abs(analysis.fine_best.zeta - analysis.feature.zeta)
    band = analysis.band
    tol = args["boundary-tol"]

    params.J >= args["phase-j-min"] || push!(reasons,
        @sprintf("J %.3f < J_KT proxy %.3f", params.J, args["phase-j-min"]))

    chi2 <= args["collapse-chi2-max"] || push!(reasons,
        @sprintf("reduced chi^2 %.3f > %.3f", chi2, args["collapse-chi2-max"]))
    eta_delta <= args["eta-feature-tol"] || push!(reasons,
        @sprintf("|eta_F - eta_feature| %.3f > %.3f", eta_delta, args["eta-feature-tol"]))
    zeta_delta <= args["zeta-feature-tol"] || push!(reasons,
        @sprintf("|zeta - zeta_feature| %.3f > %.3f", zeta_delta, args["zeta-feature-tol"]))
    analysis.fine_best.zeta >= args["zeta-min-physical"] || push!(reasons,
        @sprintf("zeta %.3f < %.3f", analysis.fine_best.zeta, args["zeta-min-physical"]))

    min(abs(analysis.fine_best.eta - band.eta_min),
        abs(analysis.fine_best.eta - band.eta_max)) <= tol &&
        push!(warnings, "eta_F optimum pinned to sensitivity-band edge")
    min(abs(analysis.fine_best.zeta - band.zeta_min),
        abs(analysis.fine_best.zeta - band.zeta_max)) <= tol &&
        push!(warnings, "zeta optimum pinned to sensitivity-band edge")

    accepted = isempty(reasons)
    label = accepted ? "collapse accepted" : "no clear scaling phase"
    return (; accepted, label, reasons, warnings, chi2, eta_delta, zeta_delta)
end

function write_dataset_summary(path::String, input_dir::String, name::String, params, analysis,
        quality, args)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Spin-aligned F sweep dataset summary")
        println(io)
        println(io, "- Dataset: `", name, "`")
        println(io, "- Input directory: `", input_dir, "`")
        println(io, "- J: `", @sprintf("%.9g", params.J), "`")
        println(io, "- v: `", @sprintf("%.9g", params.v), "`")
        println(io, "- Number of runs: ", analysis.ensemble.nruns)
        println(io, "- Radius cutoff: `r <= ", @sprintf("%.1f", args["radius-max"]), "`")
        included_times = [@sprintf("%.3g", analysis.ensemble.times[i]) for i in analysis.time_indices]
        println(io, "- Included times: `", join(included_times, ", "), "`")
        println(io, "- Classification: `", quality.label, "`")
        println(io)
        println(io, "## Collapse fit")
        println(io)
        println(io, "- Coarse best: `eta_F = ", @sprintf("%.4f", analysis.coarse_best.eta),
            "`, `zeta = ", @sprintf("%.4f", analysis.coarse_best.zeta),
            "`, reduced `chi^2 = ", @sprintf("%.4f", analysis.coarse_best.reduced_chi2), "`")
        println(io, "- Refined best: `eta_F = ", @sprintf("%.4f", analysis.fine_best.eta),
            "`, `zeta = ", @sprintf("%.4f", analysis.fine_best.zeta),
            "`, reduced `chi^2 = ", @sprintf("%.4f", analysis.fine_best.reduced_chi2), "`")
        println(io, "- Shared collapsed window: `x in [",
            @sprintf("%.4f", analysis.fine_best.overlap_min), ", ",
            @sprintf("%.4f", analysis.fine_best.overlap_max), "]`")
        println(io, "- Sensitivity band: `eta_F in [", @sprintf("%.4f", analysis.band.eta_min),
            ", ", @sprintf("%.4f", analysis.band.eta_max), "]`, `zeta in [",
            @sprintf("%.4f", analysis.band.zeta_min), ", ",
            @sprintf("%.4f", analysis.band.zeta_max), "]`")
        println(io)
        println(io, "## Feature check")
        println(io)
        println(io, "- Trough-position `zeta = ", @sprintf("%.4f", analysis.feature.zeta), "`")
        println(io, "- Trough-amplitude `eta_F = ", @sprintf("%.4f", analysis.feature.eta), "`")
        println(io, "- Feature disagreement: `Δeta_F = ", @sprintf("%.4f", quality.eta_delta),
            "`, `Δzeta = ", @sprintf("%.4f", quality.zeta_delta), "`")
        println(io)
        println(io, "## Assessment")
        println(io)
        if quality.accepted
            println(io, "- Collapse passes current acceptance cuts.")
        else
            println(io, "- Collapse rejected as `", quality.label, "`.")
            for reason in quality.reasons
                println(io, "- ", reason)
            end
        end
        if !isempty(quality.warnings)
            println(io)
            println(io, "## Warnings")
            println(io)
            for warning in quality.warnings
                println(io, "- ", warning)
            end
        end
    end
end

function save_dataset_outputs(dataset_name::String, params, analysis, args)
    dataset_results_dir = joinpath(args["results-root"], dataset_name, "collapse")
    dataset_figures_dir = joinpath(args["figures-root"], dataset_name)
    mkpath(dataset_results_dir)
    mkpath(dataset_figures_dir)
    quality = assess_collapse_quality(params, analysis, args)

    raw_plot = joinpath(dataset_figures_dir, "spin_aligned_f_correlator_raw_traces.png")
    collapse_plot = joinpath(dataset_figures_dir, "spin_aligned_f_correlator_collapsed.png")
    summary_path = joinpath(dataset_results_dir, "spin_aligned_f_correlator_summary.md")
    archive_path = joinpath(dataset_results_dir, "spin_aligned_f_correlator_collapse.jld2")

    plot_raw_traces(raw_plot, analysis.ensemble.radii, analysis.ensemble.times,
        analysis.ensemble.F_mean, analysis.ensemble.F_stderr, analysis.radius_mask,
        analysis.time_indices;
        title=@sprintf("%s raw traces (J = %.3g, v = %.3g)", dataset_name, params.J, params.v))
    plot_collapsed_traces(collapse_plot, analysis.fine_best, analysis.feature;
        title=@sprintf("%s %s (η_F = %.3f, ζ = %.3f)", dataset_name, quality.label,
            analysis.fine_best.eta, analysis.fine_best.zeta))
    write_dataset_summary(summary_path, joinpath(args["input-root"], dataset_name),
        dataset_name, params, analysis, quality, args)

    result = (;
        dataset_name,
        J=params.J,
        v=params.v,
        quality,
        config=analysis.ensemble.config,
        input_dir=joinpath(args["input-root"], dataset_name),
        files=analysis.files,
        nruns=analysis.ensemble.nruns,
        radius_max=args["radius-max"],
        time_indices=analysis.time_indices,
        selected_times=analysis.ensemble.times[analysis.time_indices],
        radii=analysis.ensemble.radii[analysis.radius_mask],
        F_mean=analysis.ensemble.F_mean[analysis.radius_mask, :],
        F_stderr=analysis.ensemble.F_stderr[analysis.radius_mask, :],
        coarse_eta_values=analysis.coarse_eta_values,
        coarse_zeta_values=analysis.coarse_zeta_values,
        coarse_objective=analysis.coarse_objective,
        fine_eta_values=analysis.fine_eta_values,
        fine_zeta_values=analysis.fine_zeta_values,
        fine_objective=analysis.fine_objective,
        coarse_best=analysis.coarse_best,
        fine_best=analysis.fine_best,
        sensitivity_band=analysis.band,
        feature=analysis.feature,
    )
    jldsave(archive_path; result)

    return (; raw_plot, collapse_plot, summary_path, archive_path, quality)
end

function write_sweep_csv(path::String, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join([
            "dataset",
            "J",
            "v",
            "nruns",
            "eta_F",
            "zeta",
            "reduced_chi2",
            "accepted",
            "classification",
            "eta_min",
            "eta_max",
            "zeta_min",
            "zeta_max",
        ], ","))
        for row in rows
            println(io, join([
                row.dataset_name,
                @sprintf("%.9g", row.J),
                @sprintf("%.9g", row.v),
                string(row.nruns),
                @sprintf("%.6f", row.eta),
                @sprintf("%.6f", row.zeta),
                @sprintf("%.6f", row.reduced_chi2),
                row.accepted ? "true" : "false",
                row.classification,
                @sprintf("%.6f", row.eta_min),
                @sprintf("%.6f", row.eta_max),
                @sprintf("%.6f", row.zeta_min),
                @sprintf("%.6f", row.zeta_max),
            ], ","))
        end
    end
end

function write_sweep_summary(path::String, rows, skipped)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Spin-aligned F J-v sweep summary")
        println(io)
        println(io, "- Successful datasets: ", length(rows))
        println(io, "- Accepted collapses: ", count(row -> row.accepted, rows))
        println(io, "- Rejected collapses: ", count(row -> !row.accepted, rows))
        println(io, "- Skipped datasets: ", length(skipped))
        println(io)
        println(io, "| Dataset | J | v | nruns | eta_F | zeta | reduced chi^2 | accepted |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|:---:|")
        for row in rows
            println(io, "| ", row.dataset_name, " | ",
                @sprintf("%.6g", row.J), " | ",
                @sprintf("%.6g", row.v), " | ",
                row.nruns, " | ",
                @sprintf("%.4f", row.eta), " | ",
                @sprintf("%.4f", row.zeta), " | ",
                @sprintf("%.4f", row.reduced_chi2), " | ",
                row.accepted ? "yes" : "no", " |")
        end
        rejected_rows = filter(row -> !row.accepted, rows)
        if !isempty(rejected_rows)
            println(io)
            println(io, "## Rejected Collapses")
            println(io)
            for row in rejected_rows
                println(io, "- `", row.dataset_name, "`: ", row.classification)
                for reason in row.reasons
                    println(io, "  - ", reason)
                end
            end
        end
        warned_rows = filter(row -> !isempty(row.warnings), rows)
        if !isempty(warned_rows)
            println(io)
            println(io, "## Warnings")
            println(io)
            for row in warned_rows
                println(io, "- `", row.dataset_name, "`")
                for warning in row.warnings
                    println(io, "  - ", warning)
                end
            end
        end
        if !isempty(skipped)
            println(io)
            println(io, "## Skipped datasets")
            println(io)
            for item in skipped
                println(io, "- `", item.dataset_name, "`: ", item.reason)
            end
        end
    end
end

function plot_exponent_heatmap(path::String, values, j_values, v_values, label::String,
        title::String)
    mkpath(dirname(path))
    fig = Figure(size=(900, 700))
    Label(fig[0, :], title, fontsize=22)
    ax = Axis(fig[1, 1], xlabel="v", ylabel="J",
        xticks=(1:length(v_values), [@sprintf("%.3g", v) for v in v_values]),
        yticks=(1:length(j_values), [@sprintf("%.3g", j) for j in j_values]))
    finite_points = [(jidx, vidx, values[jidx, vidx]) for jidx in eachindex(j_values),
        vidx in eachindex(v_values) if isfinite(values[jidx, vidx])]
    if isempty(finite_points)
        text!(ax, 0.5, 0.5, space=:relative,
            text="no accepted collapses", align=(:center, :center))
    else
        xs = [point[2] for point in finite_points]
        ys = [point[1] for point in finite_points]
        cs = [point[3] for point in finite_points]
        hm = scatter!(ax, xs, ys; color=cs, marker=:rect, markersize=95,
            strokecolor=:white, strokewidth=1)
        Colorbar(fig[1, 2], hm, label=label)
    end
    xlims!(ax, 0.5, length(v_values) + 0.5)
    ylims!(ax, 0.5, length(j_values) + 0.5)
    save(path, fig)
end

function main()
    args = parse_args()
    validate_args(args)

    dataset_dirs = collect_dataset_dirs(args["input-root"])
    results_rows = NamedTuple[]
    skipped = NamedTuple[]

    for dataset_dir in dataset_dirs
        dataset_name = basename(dataset_dir)
        params = parse_dataset_name(dataset_name)
        try
            analysis = analyze_dataset(dataset_dir, args)
            outputs = save_dataset_outputs(dataset_name, params, analysis, args)
            push!(results_rows, (;
                dataset_name,
                J=params.J,
                v=params.v,
                nruns=analysis.ensemble.nruns,
                eta=analysis.fine_best.eta,
                zeta=analysis.fine_best.zeta,
                reduced_chi2=analysis.fine_best.reduced_chi2,
                accepted=outputs.quality.accepted,
                classification=outputs.quality.label,
                reasons=outputs.quality.reasons,
                warnings=outputs.quality.warnings,
                eta_min=analysis.band.eta_min,
                eta_max=analysis.band.eta_max,
                zeta_min=analysis.band.zeta_min,
                zeta_max=analysis.band.zeta_max,
                raw_plot=outputs.raw_plot,
                collapse_plot=outputs.collapse_plot,
                summary_path=outputs.summary_path,
            ))
            println(@sprintf("processed %s: eta_F = %.4f, zeta = %.4f, nruns = %d",
                dataset_name, analysis.fine_best.eta, analysis.fine_best.zeta,
                analysis.ensemble.nruns))
        catch err
            @warn "skipping dataset" dataset_name exception=(err, catch_backtrace())
            push!(skipped, (; dataset_name, reason=sprint(showerror, err)))
        end
    end

    isempty(results_rows) && error("no datasets were successfully analyzed")
    sort!(results_rows; by=row -> (row.J, row.v))

    j_values = sort(unique(row.J for row in results_rows))
    v_values = sort(unique(row.v for row in results_rows))
    eta_grid = fill(NaN, length(j_values), length(v_values))
    zeta_grid = fill(NaN, length(j_values), length(v_values))
    for row in results_rows
        jidx = findfirst(==(row.J), j_values)
        vidx = findfirst(==(row.v), v_values)
        if row.accepted
            eta_grid[jidx, vidx] = row.eta
            zeta_grid[jidx, vidx] = row.zeta
        end
    end

    summary_csv = joinpath(args["results-root"], "collapse_summary.csv")
    summary_md = joinpath(args["results-root"], "collapse_summary.md")
    eta_heatmap = joinpath(args["figures-root"], "eta_heatmap.png")
    zeta_heatmap = joinpath(args["figures-root"], "zeta_heatmap.png")

    write_sweep_csv(summary_csv, results_rows)
    write_sweep_summary(summary_md, results_rows, skipped)
    plot_exponent_heatmap(eta_heatmap, eta_grid, j_values, v_values, "eta_F",
        "Spin-aligned F sweep: eta_F")
    plot_exponent_heatmap(zeta_heatmap, zeta_grid, j_values, v_values, "zeta",
        "Spin-aligned F sweep: zeta")

    println("saved sweep summary: ", summary_md)
    println("saved sweep csv: ", summary_csv)
    println("saved eta heatmap: ", eta_heatmap)
    println("saved zeta heatmap: ", zeta_heatmap)
end

main()
