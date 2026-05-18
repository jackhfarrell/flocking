#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using LinearAlgebra
using Printf
using Statistics

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

function collect_job_files(input_dir::String)
    isdir(input_dir) || error("missing input directory: $input_dir")
    files = String[]
    for (dir, _, names) in walkdir(input_dir)
        for name in names
            startswith(name, "job_") && endswith(name, ".jld2") || continue
            push!(files, joinpath(dir, name))
        end
    end
    isempty(files) && error("no job_*.jld2 files found in $input_dir")
    sort!(files)
    return files
end

function load_ensemble(files::AbstractVector{<:AbstractString})
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
    return (; config=runs[1].config, radii, times, F_mean, F_stderr, nruns=length(runs))
end

function linear_fit(x::AbstractVector, y::AbstractVector)
    xbar = mean(x)
    ybar = mean(y)
    slope = sum((x .- xbar) .* (y .- ybar)) / sum((x .- xbar).^2)
    intercept = ybar - slope * xbar
    return slope, intercept
end

function polynomial_design(x::AbstractVector, order::Integer)
    X = Matrix{Float64}(undef, length(x), order + 1)
    X[:, 1] .= 1.0
    for k in 1:order
        X[:, k + 1] .= x .^ k
    end
    return X
end

function evaluate_polynomial(coeffs::AbstractVector, x::AbstractVector)
    y = zeros(Float64, length(x))
    for (k, coeff) in enumerate(coeffs)
        y .+= coeff .* (x .^ (k - 1))
    end
    return y
end

function weighted_polynomial_fit(x::AbstractVector, y::AbstractVector,
        sigma::AbstractVector, order::Integer)
    length(x) > order + 1 || error("need more samples than polynomial coefficients")
    X = polynomial_design(x, order)
    w = 1.0 ./ (sigma .^ 2)
    Wsqrt = sqrt.(w)
    Xw = X .* Wsqrt
    yw = y .* Wsqrt
    coeffs = Xw \ yw
    fitted = X * coeffs
    dof = length(y) - length(coeffs)
    chi2 = sum(((y .- fitted) ./ sigma).^2)
    return (; coeffs, fitted, chi2, reduced_chi2=chi2 / dof, dof)
end

function collapse_points(radii::AbstractVector, times::AbstractVector, F_mean,
        F_stderr, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int},
        eta::Real, zeta::Real)
    xmins = Float64[]
    xmaxs = Float64[]
    for tidx in time_indices
        t = times[tidx]
        x = radii[radius_mask] ./ (t ^ zeta)
        push!(xmins, minimum(x))
        push!(xmaxs, maximum(x))
    end
    overlap_min = maximum(xmins)
    overlap_max = minimum(xmaxs)
    overlap_min < overlap_max || return nothing

    x_all = Float64[]
    y_all = Float64[]
    sigma_all = Float64[]
    time_all = Float64[]
    radius_all = Float64[]
    for tidx in time_indices
        t = times[tidx]
        x = radii[radius_mask] ./ (t ^ zeta)
        y = F_mean[radius_mask, tidx] .* (t ^ eta)
        sigma = F_stderr[radius_mask, tidx] .* (t ^ eta)
        keep = (x .>= overlap_min) .& (x .<= overlap_max)
        append!(x_all, x[keep])
        append!(y_all, y[keep])
        append!(sigma_all, max.(sigma[keep], 1e-12))
        append!(time_all, fill(t, count(keep)))
        append!(radius_all, radii[radius_mask][keep])
    end

    isempty(x_all) && return nothing
    return (; x=x_all, y=y_all, sigma=sigma_all, time=time_all, radius=radius_all,
        overlap_min, overlap_max)
end

function evaluate_collapse(radii::AbstractVector, times::AbstractVector, F_mean,
        F_stderr, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int},
        eta::Real, zeta::Real, poly_order::Integer, nbins::Integer)
    collapsed = collapse_points(radii, times, F_mean, F_stderr, radius_mask,
        time_indices, eta, zeta)
    collapsed === nothing && return nothing
    bin_score = weighted_bin_scatter(collapsed, nbins)
    bin_score === nothing && return nothing
    fit = weighted_polynomial_fit(collapsed.x, collapsed.y, collapsed.sigma, poly_order)
    return merge(collapsed, (eta=eta, zeta=zeta, poly_order, coeffs=fit.coeffs,
        fitted=fit.fitted, chi2=fit.chi2, reduced_chi2=bin_score.score, dof=bin_score.dof,
        bin_edges=bin_score.edges, nbins=bin_score.nbins))
end

function scan_grid(radii::AbstractVector, times::AbstractVector, F_mean, F_stderr,
        radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int},
        eta_values::AbstractVector, zeta_values::AbstractVector, poly_order::Integer,
        nbins::Integer)
    objective = fill(Inf, length(eta_values), length(zeta_values))
    best = nothing
    for (i, eta) in enumerate(eta_values), (j, zeta) in enumerate(zeta_values)
        result = evaluate_collapse(radii, times, F_mean, F_stderr, radius_mask,
            time_indices, eta, zeta, poly_order, nbins)
        result === nothing && continue
        objective[i, j] = result.reduced_chi2
        if best === nothing || result.reduced_chi2 < best.reduced_chi2
            best = result
        end
    end
    best === nothing && error("collapse scan found no valid overlap region")
    return objective, best
end

function feature_estimate(radii::AbstractVector, times::AbstractVector, F_mean,
        radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int})
    trough_radii = Float64[]
    trough_amplitudes = Float64[]
    fit_times = Float64[]
    for tidx in time_indices
        trace = F_mean[radius_mask, tidx]
        local_radii = radii[radius_mask]
        min_index = argmin(trace)
        push!(trough_radii, local_radii[min_index])
        push!(trough_amplitudes, -trace[min_index])
        push!(fit_times, times[tidx])
    end
    zeta, zeta_intercept = linear_fit(log.(fit_times), log.(trough_radii))
    amp_slope, amp_intercept = linear_fit(log.(fit_times), log.(trough_amplitudes))
    eta = -amp_slope
    return (; times=fit_times, trough_radii, trough_amplitudes, zeta,
        zeta_intercept, eta, amp_intercept)
end

function weighted_bin_scatter(collapsed, nbins::Integer)
    edges = collect(range(collapsed.overlap_min, collapsed.overlap_max; length=nbins + 1))
    score = 0.0
    dof = 0
    for b in 1:nbins
        if b == nbins
            inds = findall(i -> collapsed.x[i] >= edges[b] && collapsed.x[i] <= edges[b + 1],
                eachindex(collapsed.x))
        else
            inds = findall(i -> collapsed.x[i] >= edges[b] && collapsed.x[i] < edges[b + 1],
                eachindex(collapsed.x))
        end
        length(inds) >= 2 || continue
        length(unique(collapsed.time[inds])) >= 2 || continue
        y = collapsed.y[inds]
        sigma = collapsed.sigma[inds]
        w = 1.0 ./ (sigma .^ 2)
        mu = sum(w .* y) / sum(w)
        score += sum(w .* (y .- mu).^2)
        dof += length(inds) - 1
    end
    dof > 0 || return nothing
    return (; score=score / dof, dof, edges, nbins)
end

function sensitivity_band(objective, eta_values::AbstractVector, zeta_values::AbstractVector,
        factor::Real)
    min_value = minimum(objective)
    threshold = factor * min_value
    eta_hits = Float64[]
    zeta_hits = Float64[]
    for i in eachindex(eta_values), j in eachindex(zeta_values)
        objective[i, j] <= threshold || continue
        push!(eta_hits, eta_values[i])
        push!(zeta_hits, zeta_values[j])
    end
    isempty(eta_hits) && error("empty sensitivity region")
    return (; threshold, eta_min=minimum(eta_hits), eta_max=maximum(eta_hits),
        zeta_min=minimum(zeta_hits), zeta_max=maximum(zeta_hits))
end

function plot_raw_traces(path::String, radii::AbstractVector, times::AbstractVector,
        F_mean, F_stderr, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int})
    mkpath(dirname(path))
    fig = Figure(size=(1100, 700))
    Label(fig[0, :], "Spin-aligned F(r,t) ensemble mean, r ≤ 40", fontsize=22)
    ax = Axis(fig[1, 1], xlabel="r", ylabel="F(r, t)")
    palette = Makie.wong_colors()
    x = radii[radius_mask]
    for (k, tidx) in enumerate(time_indices)
        color = palette[mod1(k, length(palette))]
        mean = F_mean[radius_mask, tidx]
        err = F_stderr[radius_mask, tidx]
        band!(ax, x, mean .- err, mean .+ err, color=(color, 0.18))
        lines!(ax, x, mean, color=color, linewidth=3,
            label="t = $(round(times[tidx]; digits=3))")
    end
    xlims!(ax, 0, maximum(x))
    axislegend(ax, position=:rb, nbanks=2)
    save(path, fig)
end

function plot_collapsed_traces(path::String, collapse, feature)
    mkpath(dirname(path))
    fig = Figure(size=(1100, 700))
    Label(fig[0, :], @sprintf("Collapsed traces, η_F = %.3f, ζ = %.3f",
        collapse.eta, collapse.zeta), fontsize=22)
    ax = Axis(fig[1, 1], xlabel="r / t^ζ", ylabel="t^η F(r,t)")
    palette = Makie.wong_colors()

    for (k, t) in enumerate(feature.times)
        inds = findall(==(t), collapse.time)
        local_order = sortperm(collapse.x[inds])
        ordered_inds = inds[local_order]
        lines!(ax, collapse.x[ordered_inds], collapse.y[ordered_inds],
            color=palette[mod1(k, length(palette))], linewidth=3,
            label="t = $(round(t; digits=3))")
    end
    vlines!(ax, [collapse.overlap_min, collapse.overlap_max], color=:gray60,
        linestyle=:dash, linewidth=2, label="shared x window")
    axislegend(ax, position=:rb, nbanks=2)
    save(path, fig)
end

function plot_objective_heatmap(path::String, eta_values::AbstractVector,
        zeta_values::AbstractVector, objective, best, band)
    mkpath(dirname(path))
    finite_values = objective[isfinite.(objective)]
    zmax = isempty(finite_values) ? 1.0 : minimum([maximum(finite_values), band.threshold])

    fig = Figure(size=(900, 700))
    Label(fig[0, :], "Collapse objective over (η_F, ζ)", fontsize=22)
    ax = Axis(fig[1, 1], xlabel="ζ", ylabel="η_F")
    hm = heatmap!(ax, zeta_values, eta_values, objective;
        colorrange=(minimum(finite_values), zmax))
    scatter!(ax, [best.zeta], [best.eta], color=:white, marker=:star5, markersize=22)
    Colorbar(fig[1, 2], hm, label="reduced χ²")
    text!(ax, best.zeta, best.eta,
        text=@sprintf(" best = (%.3f, %.3f)", best.eta, best.zeta),
        color=:white, align=(:left, :bottom))
    save(path, fig)
end

function plot_feature_diagnostics(path::String, feature)
    mkpath(dirname(path))
    t = feature.times
    log_t = log.(t)
    fit_r = exp.(feature.zeta_intercept .+ feature.zeta .* log_t)
    fit_a = exp.(feature.amp_intercept .- feature.eta .* log_t)

    fig = Figure(size=(1000, 450))
    ax1 = Axis(fig[1, 1], xlabel="t", ylabel="r_trough", xscale=log10, yscale=log10,
        title=@sprintf("Trough position fit: ζ = %.3f", feature.zeta))
    scatter!(ax1, t, feature.trough_radii, markersize=12)
    lines!(ax1, t, fit_r, linewidth=3)

    ax2 = Axis(fig[1, 2], xlabel="t", ylabel="-F_min", xscale=log10, yscale=log10,
        title=@sprintf("Trough amplitude fit: η_F = %.3f", feature.eta))
    scatter!(ax2, t, feature.trough_amplitudes, markersize=12)
    lines!(ax2, t, fit_a, linewidth=3)
    save(path, fig)
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
