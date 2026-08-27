#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results",
    "spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353")
const DEFAULT_RESULTS_DIR = joinpath(REPO_ROOT, "results",
    "spin_aligned_c_asymmetry_L200_J2_v1_gamma1_20260520_153353")
const DEFAULT_FIGURES_DIR = joinpath(REPO_ROOT, "figures",
    "spin_aligned_c_asymmetry_L200_J2_v1_gamma1_20260520_153353")
const DEFAULT_PREFIX = "spin_aligned_c_asymmetry"

function parse_args()
    settings = ArgParseSettings(
        description="Analyze F-convention ΔC(r,t) = C(r,t) - C(r,-t) from spin-aligned correlator jobs.",
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
        "--radius-min"
            arg_type = Float64
            default = 0.0
        "--radius-max"
            arg_type = Float64
            default = 40.0
        "--time-min"
            arg_type = Float64
            default = 2.0
        "--poly-order"
            arg_type = Int
            default = 3
        "--collapse-bins"
            arg_type = Int
            default = 60
        "--eta-min"
            arg_type = Float64
            default = -0.8
        "--eta-max"
            arg_type = Float64
            default = 0.8
        "--eta-step"
            arg_type = Float64
            default = 0.02
        "--z-min"
            arg_type = Float64
            default = 1.2
        "--z-max"
            arg_type = Float64
            default = 5.0
        "--z-step"
            arg_type = Float64
            default = 0.05
        "--fine-window"
            arg_type = Float64
            default = 0.06
        "--fine-z-window"
            arg_type = Float64
            default = 0.4
        "--fine-step"
            arg_type = Float64
            default = 0.005
        "--fine-z-step"
            arg_type = Float64
            default = 0.01
        "--sensitivity-factor"
            arg_type = Float64
            default = 1.05
    end
    return ArgParse.parse_args(settings)
end

function c_asymmetry_ensemble(input_dir::String)
    files = collect_job_files(input_dir)
    ensemble = load_ensemble(files)

    C_asym_mean = 2 .* ensemble.F_mean
    C_asym_stderr = 2 .* ensemble.F_stderr
    return merge(ensemble, (; files, C_asym_mean, C_asym_stderr,
        convention=:f_correlator_symmetrized_difference))
end

function evaluate_collapse_z(radii::AbstractVector, times::AbstractVector, mean_field,
        stderr_field, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int},
        eta::Real, z::Real, poly_order::Integer, nbins::Integer)
    z > 0 || return nothing
    result = evaluate_collapse(radii, times, mean_field, stderr_field, radius_mask,
        time_indices, eta, inv(z), poly_order, nbins)
    result === nothing && return nothing
    return merge(result, (; z))
end

function scan_grid_z(radii::AbstractVector, times::AbstractVector, mean_field,
        stderr_field, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int},
        eta_values::AbstractVector, z_values::AbstractVector, poly_order::Integer,
        nbins::Integer)
    objective = fill(Inf, length(eta_values), length(z_values))
    best = nothing
    for (i, eta) in enumerate(eta_values), (j, z) in enumerate(z_values)
        result = evaluate_collapse_z(radii, times, mean_field, stderr_field,
            radius_mask, time_indices, eta, z, poly_order, nbins)
        result === nothing && continue
        isfinite(result.reduced_chi2) || continue
        objective[i, j] = result.reduced_chi2
        if best === nothing || result.reduced_chi2 < best.reduced_chi2
            best = result
        end
    end
    best === nothing && error("collapse scan found no valid overlap region")
    return objective, best
end

function sensitivity_band_z(objective, eta_values::AbstractVector, z_values::AbstractVector,
        factor::Real)
    finite_values = filter(isfinite, vec(objective))
    isempty(finite_values) && error("no finite objective values in sensitivity scan")
    min_value = minimum(finite_values)
    threshold = factor * min_value
    eta_hits = Float64[]
    z_hits = Float64[]
    for i in eachindex(eta_values), j in eachindex(z_values)
        isfinite(objective[i, j]) || continue
        objective[i, j] <= threshold || continue
        push!(eta_hits, eta_values[i])
        push!(z_hits, z_values[j])
    end
    isempty(eta_hits) && error("empty sensitivity region")
    return (; threshold, eta_min=minimum(eta_hits), eta_max=maximum(eta_hits),
        z_min=minimum(z_hits), z_max=maximum(z_hits))
end

function band_summary_z(band, eta_values::AbstractVector, z_values::AbstractVector)
    eta_step = length(eta_values) >= 2 ? abs(eta_values[2] - eta_values[1]) : 0.0
    z_step = length(z_values) >= 2 ? abs(z_values[2] - z_values[1]) : 0.0
    eta_center = 0.5 * (band.eta_min + band.eta_max)
    eta_halfwidth = max(0.5 * (band.eta_max - band.eta_min), 0.5 * eta_step)
    z_center = 0.5 * (band.z_min + band.z_max)
    z_halfwidth = max(0.5 * (band.z_max - band.z_min), 0.5 * z_step)
    return (; eta_center, eta_halfwidth, z_center, z_halfwidth)
end

function plot_objective_heatmap_z(path::String, eta_values::AbstractVector,
        z_values::AbstractVector, objective, best, band)
    mkpath(dirname(path))
    finite_mask = isfinite.(objective)
    finite_values = objective[finite_mask]
    isempty(finite_values) && error("no finite objective values available for heatmap")
    zmin = minimum(finite_values)
    zmax = min(maximum(finite_values), band.threshold)
    plotted_objective = map(x -> isfinite(x) ? x : NaN, objective)
    colorrange_valid = Float32(zmax) > Float32(zmin)

    fig = Figure(size=(900, 700))
    Label(fig[0, :], "Collapse objective over (η, z)", fontsize=22)
    ax = Axis(fig[1, 1], xlabel="z", ylabel="η")
    if colorrange_valid
        hm = heatmap!(ax, z_values, eta_values, plotted_objective;
            colorrange=(zmin, zmax))
        Colorbar(fig[1, 2], hm, label="reduced χ²")
    else
        text!(ax, 0.5, 0.5;
            space=:relative,
            text=@sprintf("Flat finite objective surface\nχ² = %.4f", zmin),
            align=(:center, :center))
    end
    scatter!(ax, [best.z], [best.eta], color=:white, marker=:star5, markersize=22)
    text!(ax, best.z, best.eta,
        text=@sprintf(" best = (%.3f, %.3f)", best.eta, best.z),
        color=:white, align=(:left, :bottom))
    save(path, fig)
end

function plot_c_asym_raw(path::String, radii::AbstractVector, times::AbstractVector,
        C_asym_mean, C_asym_stderr, radius_mask::AbstractVector{Bool},
        time_indices::AbstractVector{Int})
    mkpath(dirname(path))
    fig = Figure(size=(1100, 700))
    Label(fig[0, :], "Spin-aligned ΔC_F(r,t) = C(r,t) - C(r,-t)", fontsize=22)
    ax = Axis(fig[1, 1], xlabel="r", ylabel="ΔC_F(r,t) = 2F(r,t)")
    palette = Makie.wong_colors()
    x = radii[radius_mask]
    for (k, tidx) in enumerate(time_indices)
        color = palette[mod1(k, length(palette))]
        mean = C_asym_mean[radius_mask, tidx]
        err = C_asym_stderr[radius_mask, tidx]
        band!(ax, x, mean .- err, mean .+ err, color=(color, 0.18))
        lines!(ax, x, mean, color=color, linewidth=3,
            label="t = $(round(times[tidx]; digits=3))")
    end
    xlims!(ax, 0, maximum(x))
    axislegend(ax, position=:rb, nbanks=2)
    save(path, fig)
end

function plot_c_asym_collapsed(path::String, collapse, feature)
    mkpath(dirname(path))
    fig = Figure(size=(1100, 700))
    Label(fig[0, :],
        @sprintf("Collapsed ΔC_F traces, η = %.3f, z = %.3f",
            collapse.eta, collapse.z),
        fontsize=22)
    ax = Axis(fig[1, 1], xlabel="r / t^(1/z)", ylabel="t^η ΔC_F(r,t)")
    palette = Makie.wong_colors()

    for (k, t) in enumerate(feature.times)
        inds = findall(==(t), collapse.time)
        local_order = sortperm(collapse.x[inds])
        ordered_inds = inds[local_order]
        band!(ax, collapse.x[ordered_inds],
            collapse.y[ordered_inds] .- collapse.sigma[ordered_inds],
            collapse.y[ordered_inds] .+ collapse.sigma[ordered_inds],
            color=(palette[mod1(k, length(palette))], 0.18))
        lines!(ax, collapse.x[ordered_inds], collapse.y[ordered_inds],
            color=palette[mod1(k, length(palette))], linewidth=3,
            label="t = $(round(t; digits=3))")
    end
    vlines!(ax, [collapse.overlap_min, collapse.overlap_max], color=:gray60,
        linestyle=:dash, linewidth=2, label="shared x window")
    axislegend(ax, position=:rb, nbanks=2)
    save(path, fig)
end

function plot_c_asym_feature_diagnostics(path::String, feature)
    mkpath(dirname(path))
    t = feature.times
    log_t = log.(t)
    fit_r = exp.(feature.zeta_intercept .+ feature.zeta .* log_t)
    fit_a = exp.(feature.amp_intercept .- feature.eta .* log_t)

    fig = Figure(size=(1000, 450))
    ax1 = Axis(fig[1, 1], xlabel="t", ylabel="r_trough", xscale=log10, yscale=log10,
        title=@sprintf("Trough position fit: ζ = %.3f ± %.3f",
            feature.zeta, feature.zeta_stderr))
    scatter!(ax1, t, feature.trough_radii, markersize=12)
    lines!(ax1, t, fit_r, linewidth=3)

    ax2 = Axis(fig[1, 2], xlabel="t", ylabel="-ΔC_F,min", xscale=log10, yscale=log10,
        title=@sprintf("Trough amplitude fit: η = %.3f ± %.3f",
            feature.eta, feature.eta_stderr))
    scatter!(ax2, t, feature.trough_amplitudes, markersize=12)
    lines!(ax2, t, fit_a, linewidth=3)
    save(path, fig)
end

function write_summary(path::String, ensemble, radius_mask::AbstractVector{Bool},
        time_indices::AbstractVector{Int}, coarse_best, fine_best, band,
        collapse_unc, feature, args)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Spin-aligned C asymmetry analysis")
        println(io)
        println(io, "- Quantity: `ΔC_F(r,t) = C(r,t) - C(r,-t) = 2F(r,t)`")
        println(io, "- Convention: same space-time symmetrization used by the stored `F_mean`")
        if hasproperty(ensemble, :C_plus_mean)
            println(io, "- Note: stored `C_plus_mean - C_minus_mean` is not used here; those fields pair ",
                "`forward_plus/backward_plus` and `forward_minus/backward_minus`, which differs ",
                "from the F-correlator subtraction.")
        end
        println(io, "- Input directory: `", args["input-dir"], "`")
        println(io, "- Number of runs: ", ensemble.nruns)
        println(io, "- Radius window: `", @sprintf("%.1f", minimum(ensemble.radii[radius_mask])),
            " <= r <= ", @sprintf("%.1f", maximum(ensemble.radii[radius_mask])), "`")
        println(io, "- Included times: `",
            join([@sprintf("%.3g", ensemble.times[i]) for i in time_indices], ", "), "`")
        println(io)
        println(io, "## Collapse fit")
        println(io)
        println(io, "- Coarse best fit: `eta = ", @sprintf("%.4f", coarse_best.eta),
            "`, `z = ", @sprintf("%.4f", coarse_best.z),
            "`, `zeta = 1/z = ", @sprintf("%.4f", coarse_best.zeta),
            "`, reduced `chi^2 = ", @sprintf("%.4f", coarse_best.reduced_chi2), "`")
        println(io, "- Refined best fit: `eta = ", @sprintf("%.4f", fine_best.eta),
            "`, `z = ", @sprintf("%.4f", fine_best.z),
            "`, `zeta = 1/z = ", @sprintf("%.4f", fine_best.zeta),
            "`, reduced `chi^2 = ", @sprintf("%.4f", fine_best.reduced_chi2), "`")
        println(io, "- Sensitivity band: `eta = ",
            @sprintf("%.4f ± %.4f", collapse_unc.eta_center, collapse_unc.eta_halfwidth),
            "`, `z = ", @sprintf("%.4f ± %.4f",
            collapse_unc.z_center, collapse_unc.z_halfwidth), "`")
        println(io, "- Shared collapsed window: `x in [", @sprintf("%.4f", fine_best.overlap_min),
            ", ", @sprintf("%.4f", fine_best.overlap_max), "]`")
        println(io)
        println(io, "## Feature check")
        println(io)
        println(io, "- Trough-position scaling: `r_min(t) ~ t^zeta`, fitted `zeta = ",
            @sprintf("%.4f ± %.4f", feature.zeta, feature.zeta_stderr),
            "`, equivalently `z = ", @sprintf("%.4f ± %.4f",
            inv(feature.zeta), feature.zeta_stderr / feature.zeta^2), "`")
        println(io, "- Trough-amplitude scaling: `-ΔC_F,min(t) ~ t^{-eta}`, fitted `eta = ",
            @sprintf("%.4f ± %.4f", feature.eta, feature.eta_stderr), "`")
    end
end

function main()
    args = parse_args()
    args["radius-max"] > args["radius-min"] ||
        throw(ArgumentError("--radius-max must exceed --radius-min"))
    args["time-min"] >= 0 || throw(ArgumentError("--time-min must be nonnegative"))
    args["poly-order"] >= 0 || throw(ArgumentError("--poly-order must be nonnegative"))
    args["collapse-bins"] >= 2 || throw(ArgumentError("--collapse-bins must be at least 2"))
    args["eta-step"] > 0 || throw(ArgumentError("--eta-step must be positive"))
    args["z-min"] > 0 || throw(ArgumentError("--z-min must be positive"))
    args["z-max"] > args["z-min"] || throw(ArgumentError("--z-max must exceed --z-min"))
    args["z-step"] > 0 || throw(ArgumentError("--z-step must be positive"))
    args["fine-window"] > 0 || throw(ArgumentError("--fine-window must be positive"))
    args["fine-z-window"] > 0 || throw(ArgumentError("--fine-z-window must be positive"))
    args["fine-step"] > 0 || throw(ArgumentError("--fine-step must be positive"))
    args["fine-z-step"] > 0 || throw(ArgumentError("--fine-z-step must be positive"))
    args["sensitivity-factor"] >= 1 ||
        throw(ArgumentError("--sensitivity-factor must be at least 1"))

    ensemble = c_asymmetry_ensemble(args["input-dir"])
    radius_mask = (ensemble.radii .>= args["radius-min"]) .&
        (ensemble.radii .<= args["radius-max"])
    any(radius_mask) || error("no radii satisfy requested window")
    time_indices = findall(t -> t > 0 && t >= args["time-min"], ensemble.times)
    isempty(time_indices) && error("no positive times available for collapse analysis")

    eta_values = collect(args["eta-min"]:args["eta-step"]:args["eta-max"])
    z_values = collect(args["z-min"]:args["z-step"]:args["z-max"])
    coarse_objective, coarse_best = scan_grid_z(ensemble.radii, ensemble.times,
        ensemble.C_asym_mean, ensemble.C_asym_stderr, radius_mask, time_indices,
        eta_values, z_values, args["poly-order"], args["collapse-bins"])

    fine_eta_values = collect((coarse_best.eta - args["fine-window"]):
        args["fine-step"]:(coarse_best.eta + args["fine-window"]))
    fine_z_values = collect((coarse_best.z - args["fine-z-window"]):
        args["fine-z-step"]:(coarse_best.z + args["fine-z-window"]))
    fine_objective, fine_best = scan_grid_z(ensemble.radii, ensemble.times,
        ensemble.C_asym_mean, ensemble.C_asym_stderr, radius_mask, time_indices,
        fine_eta_values, fine_z_values, args["poly-order"], args["collapse-bins"])

    band = sensitivity_band_z(fine_objective, fine_eta_values, fine_z_values,
        args["sensitivity-factor"])
    collapse_unc = band_summary_z(band, fine_eta_values, fine_z_values)
    feature = feature_estimate(ensemble.radii, ensemble.times, ensemble.C_asym_mean,
        radius_mask, time_indices)

    mkpath(args["results-dir"])
    mkpath(args["figures-dir"])
    prefix = args["output-prefix"]
    raw_plot = joinpath(args["figures-dir"], prefix * "_raw_traces.png")
    collapse_plot = joinpath(args["figures-dir"], prefix * "_collapsed.png")
    heatmap_plot = joinpath(args["figures-dir"], prefix * "_objective.png")
    feature_plot = joinpath(args["figures-dir"], prefix * "_features.png")
    summary_path = joinpath(args["results-dir"], prefix * "_summary.md")
    archive_path = joinpath(args["results-dir"], prefix * ".jld2")

    plot_c_asym_raw(raw_plot, ensemble.radii, ensemble.times, ensemble.C_asym_mean,
        ensemble.C_asym_stderr, radius_mask, time_indices)
    plot_c_asym_collapsed(collapse_plot, fine_best, feature)
    plot_objective_heatmap_z(heatmap_plot, fine_eta_values, fine_z_values,
        fine_objective, fine_best, band)
    plot_c_asym_feature_diagnostics(feature_plot, feature)
    write_summary(summary_path, ensemble, radius_mask, time_indices, coarse_best,
        fine_best, band, collapse_unc, feature, args)

    result = (;
        config=ensemble.config,
        input_dir=args["input-dir"],
        files=ensemble.files,
        nruns=ensemble.nruns,
        radius_min=args["radius-min"],
        radius_max=args["radius-max"],
        time_indices,
        selected_times=ensemble.times[time_indices],
        radii=ensemble.radii[radius_mask],
        C_asym_mean=ensemble.C_asym_mean[radius_mask, :],
        C_asym_stderr=ensemble.C_asym_stderr[radius_mask, :],
        convention=ensemble.convention,
        coarse_eta_values=eta_values,
        coarse_z_values=z_values,
        coarse_zeta_values=inv.(z_values),
        coarse_objective,
        fine_eta_values,
        fine_z_values,
        fine_zeta_values=inv.(fine_z_values),
        fine_objective,
        coarse_best,
        fine_best,
        band,
        collapse_unc,
        feature,
        raw_plot,
        collapse_plot,
        heatmap_plot,
        feature_plot,
        summary_path,
    )
    jldsave(archive_path; result)
    @info "saved spin-aligned C asymmetry analysis" raw_plot collapse_plot heatmap_plot feature_plot summary_path archive_path
end

main()
