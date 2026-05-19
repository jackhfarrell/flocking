#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_J2_v1_gamma1")
const DEFAULT_RESULTS_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_J2_v1_gamma1_collapse")
const DEFAULT_FIGURES_DIR = joinpath(REPO_ROOT, "figures", "spin_aligned_f_correlator_L200_J2_v1_gamma1_collapse")
const DEFAULT_PREFIX = "spin_aligned_f_correlator_collapse"

function parse_args()
    settings = ArgParseSettings(
        description="Estimate scaling exponents for the spin-aligned F(r,t) ensemble.",
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
        "--output-prefix"
            arg_type = String
            default = DEFAULT_PREFIX
        "--radius-max"
            arg_type = Float64
            default = 40.0
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
    end
    return ArgParse.parse_args(settings)
end

function write_summary(path::String, ensemble, radius_max::Real, time_indices::AbstractVector{Int},
        coarse_best, fine_best, band, feature, args)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Spin-aligned F(r,t) collapse analysis")
        println(io)
        println(io, "- Input directory: `", args["input-dir"], "`")
        println(io, "- Number of runs: ", ensemble.nruns)
        println(io, "- Radius cutoff: `r <= ", @sprintf("%.1f", radius_max), "`")
        included_times = [@sprintf("%.3g", ensemble.times[i]) for i in time_indices]
        println(io, "- Included times: `", join(included_times, ", "), "`")
        println(io, "- Polynomial order: `", args["poly-order"], "`")
        println(io, "- Collapse bins: `", args["collapse-bins"], "`")
        println(io)
        println(io, "## Primary collapse fit")
        println(io)
        println(io, "- Coarse best fit: `eta_F = ", @sprintf("%.4f", coarse_best.eta),
            "`, `zeta = ", @sprintf("%.4f", coarse_best.zeta),
            "`, reduced `chi^2 = ", @sprintf("%.4f", coarse_best.reduced_chi2), "`")
        println(io, "- Refined best fit: `eta_F = ", @sprintf("%.4f", fine_best.eta),
            "`, `zeta = ", @sprintf("%.4f", fine_best.zeta),
            "`, reduced `chi^2 = ", @sprintf("%.4f", fine_best.reduced_chi2), "`")
        println(io, "- Smooth master curve: weighted polynomial of order `",
            args["poly-order"], "` fit after exponent selection")
        println(io, "- Shared collapsed window: `x in [", @sprintf("%.4f", fine_best.overlap_min),
            ", ", @sprintf("%.4f", fine_best.overlap_max), "]`")
        println(io, "- Sensitivity band (`objective <= ",
            @sprintf("%.3f", args["sensitivity-factor"]), " * min`): ",
            "`eta_F in [", @sprintf("%.4f", band.eta_min), ", ", @sprintf("%.4f", band.eta_max),
            "]`, `zeta in [", @sprintf("%.4f", band.zeta_min), ", ",
            @sprintf("%.4f", band.zeta_max), "]`")
        println(io)
        println(io, "## Feature-based sanity check")
        println(io)
        println(io, "- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = ",
            @sprintf("%.4f", feature.zeta), "`")
        println(io, "- Trough-amplitude scaling: `-F_min(t) ~ t^{-eta_F}`, fitted `eta_F = ",
            @sprintf("%.4f", feature.eta), "`")
        println(io)
        println(io, "## Comparison")
        println(io)
        println(io, "- Collapse minus feature estimate: `Δeta_F = ",
            @sprintf("%+.4f", fine_best.eta - feature.eta), "`, `Δzeta = ",
            @sprintf("%+.4f", fine_best.zeta - feature.zeta), "`")
        println(io)
        println(io, "## Trough data")
        println(io)
        println(io, "| t | r_trough | -F_min |")
        println(io, "|---:|---:|---:|")
        for i in eachindex(feature.times)
            println(io, "| ", @sprintf("%.3f", feature.times[i]), " | ",
                @sprintf("%.3f", feature.trough_radii[i]), " | ",
                @sprintf("%.6f", feature.trough_amplitudes[i]), " |")
        end
    end
end

function main()
    args = parse_args()
    args["radius-max"] > 0 || throw(ArgumentError("--radius-max must be positive"))
    args["poly-order"] >= 0 || throw(ArgumentError("--poly-order must be nonnegative"))
    args["collapse-bins"] >= 2 || throw(ArgumentError("--collapse-bins must be at least 2"))
    args["eta-step"] > 0 || throw(ArgumentError("--eta-step must be positive"))
    args["zeta-step"] > 0 || throw(ArgumentError("--zeta-step must be positive"))
    args["fine-window"] > 0 || throw(ArgumentError("--fine-window must be positive"))
    args["fine-step"] > 0 || throw(ArgumentError("--fine-step must be positive"))
    args["sensitivity-factor"] >= 1 ||
        throw(ArgumentError("--sensitivity-factor must be at least 1"))

    files = collect_job_files(args["input-dir"])
    ensemble = load_ensemble(files)
    radius_mask = ensemble.radii .<= args["radius-max"]
    any(radius_mask) || error("no radii satisfy r <= $(args["radius-max"])")
    time_indices = findall(>(0), ensemble.times)
    isempty(time_indices) && error("no positive times available for collapse analysis")

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

    mkpath(args["results-dir"])
    mkpath(args["figures-dir"])
    output_prefix = args["output-prefix"]
    raw_plot = joinpath(args["figures-dir"], output_prefix * "_raw_traces.png")
    collapse_plot = joinpath(args["figures-dir"], output_prefix * "_collapsed.png")
    heatmap_plot = joinpath(args["figures-dir"], output_prefix * "_objective.png")
    feature_plot = joinpath(args["figures-dir"], output_prefix * "_features.png")
    summary_path = joinpath(args["results-dir"], output_prefix * "_summary.md")
    archive_path = joinpath(args["results-dir"], output_prefix * ".jld2")

    plot_raw_traces(raw_plot, ensemble.radii, ensemble.times, ensemble.F_mean,
        ensemble.F_stderr, radius_mask, time_indices)
    plot_collapsed_traces(collapse_plot, fine_best, feature)
    plot_objective_heatmap(heatmap_plot, fine_eta_values, fine_zeta_values,
        fine_objective, fine_best, band)
    plot_feature_diagnostics(feature_plot, feature)
    write_summary(summary_path, ensemble, args["radius-max"], time_indices,
        coarse_best, fine_best, band, feature, args)

    result = (;
        config=ensemble.config,
        input_dir=args["input-dir"],
        files,
        nruns=ensemble.nruns,
        radius_max=args["radius-max"],
        poly_order=args["poly-order"],
        collapse_bins=args["collapse-bins"],
        time_indices,
        selected_times=ensemble.times[time_indices],
        radii=ensemble.radii[radius_mask],
        F_mean=ensemble.F_mean[radius_mask, :],
        F_stderr=ensemble.F_stderr[radius_mask, :],
        coarse_eta_values=eta_values,
        coarse_zeta_values=zeta_values,
        coarse_objective,
        fine_eta_values,
        fine_zeta_values,
        fine_objective,
        coarse_best,
        fine_best,
        sensitivity_band=band,
        feature,
    )
    jldsave(archive_path; result)

    println("saved summary: ", summary_path)
    println("saved archive: ", archive_path)
    println("saved raw traces: ", raw_plot)
    println("saved collapse plot: ", collapse_plot)
    println("saved objective heatmap: ", heatmap_plot)
    println("saved feature diagnostics: ", feature_plot)
    println(@sprintf("coarse best: eta_F = %.4f, zeta = %.4f, reduced chi^2 = %.4f",
        coarse_best.eta, coarse_best.zeta, coarse_best.reduced_chi2))
    println(@sprintf("refined best: eta_F = %.4f, zeta = %.4f, reduced chi^2 = %.4f",
        fine_best.eta, fine_best.zeta, fine_best.reduced_chi2))
    println(@sprintf("feature estimates: eta_F = %.4f, zeta = %.4f",
        feature.eta, feature.zeta))
end

main()
