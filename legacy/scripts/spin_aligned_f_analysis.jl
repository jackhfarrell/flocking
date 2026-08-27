using CairoMakie
using JLD2
using Printf
using Statistics

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

    function aggregate_field(field::Symbol)
        stack = cat((getproperty(run, field) for run in runs)...; dims=3)
        mean_field = dropdims(mean(stack; dims=3), dims=3)
        stderr_field = if length(runs) == 1
            zeros(size(mean_field))
        else
            dropdims(std(stack; dims=3), dims=3) ./ sqrt(length(runs))
        end
        return mean_field, stderr_field
    end

    F_mean, F_stderr = aggregate_field(:F_mean)
    ensemble = (; config=runs[1].config, radii, times, F_mean, F_stderr, nruns=length(runs))

    has_c_plus = all(run -> hasproperty(run, :C_plus_mean), runs)
    has_c_minus = all(run -> hasproperty(run, :C_minus_mean), runs)
    has_c_plus == has_c_minus || error("C_plus and C_minus fields must appear together")
    if has_c_plus
        C_plus_mean, C_plus_stderr = aggregate_field(:C_plus_mean)
        C_minus_mean, C_minus_stderr = aggregate_field(:C_minus_mean)
        ensemble = merge(ensemble, (; C_plus_mean, C_plus_stderr, C_minus_mean,
            C_minus_stderr))
    end

    return ensemble
end

function linear_fit(x::AbstractVector, y::AbstractVector)
    xbar = mean(x)
    ybar = mean(y)
    slope = sum((x .- xbar) .* (y .- ybar)) / sum((x .- xbar).^2)
    intercept = ybar - slope * xbar
    return slope, intercept
end

function linear_fit_with_stderr(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n >= 3 || error("need at least 3 samples for linear-fit uncertainty")
    slope, intercept = linear_fit(x, y)
    residuals = y .- (intercept .+ slope .* x)
    sxx = sum((x .- mean(x)).^2)
    sigma2 = sum(residuals .^ 2) / (n - 2)
    slope_stderr = sqrt(sigma2 / sxx)
    intercept_stderr = sqrt(sigma2 * (1 / n + mean(x)^2 / sxx))
    return (; slope, intercept, slope_stderr, intercept_stderr, residuals, sigma2)
end

function polynomial_design(x::AbstractVector, order::Integer)
    X = Matrix{Float64}(undef, length(x), order + 1)
    X[:, 1] .= 1.0
    for k in 1:order
        X[:, k + 1] .= x .^ k
    end
    return X
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
    masked_radii = radii[radius_mask]
    for tidx in time_indices
        t = times[tidx]
        x = masked_radii ./ (t ^ zeta)
        y = F_mean[radius_mask, tidx] .* (t ^ eta)
        sigma = F_stderr[radius_mask, tidx] .* (t ^ eta)
        keep = (x .>= overlap_min) .& (x .<= overlap_max)
        append!(x_all, x[keep])
        append!(y_all, y[keep])
        append!(sigma_all, max.(sigma[keep], 1e-12))
        append!(time_all, fill(t, count(keep)))
        append!(radius_all, masked_radii[keep])
    end

    isempty(x_all) && return nothing
    return (; x=x_all, y=y_all, sigma=sigma_all, time=time_all, radius=radius_all,
        overlap_min, overlap_max)
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
        isfinite(result.reduced_chi2) || continue
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
    zeta_fit = linear_fit_with_stderr(log.(fit_times), log.(trough_radii))
    amp_fit = linear_fit_with_stderr(log.(fit_times), log.(trough_amplitudes))
    zeta = zeta_fit.slope
    eta = -amp_fit.slope
    return (; times=fit_times, trough_radii, trough_amplitudes, zeta,
        zeta_stderr=zeta_fit.slope_stderr, zeta_intercept=zeta_fit.intercept,
        zeta_intercept_stderr=zeta_fit.intercept_stderr, eta,
        eta_stderr=amp_fit.slope_stderr, amp_intercept=amp_fit.intercept,
        amp_intercept_stderr=amp_fit.intercept_stderr)
end

function sensitivity_band(objective, eta_values::AbstractVector, zeta_values::AbstractVector,
        factor::Real)
    finite_values = filter(isfinite, vec(objective))
    isempty(finite_values) && error("no finite objective values in sensitivity scan")
    min_value = minimum(finite_values)
    threshold = factor * min_value
    eta_hits = Float64[]
    zeta_hits = Float64[]
    for i in eachindex(eta_values), j in eachindex(zeta_values)
        isfinite(objective[i, j]) || continue
        objective[i, j] <= threshold || continue
        push!(eta_hits, eta_values[i])
        push!(zeta_hits, zeta_values[j])
    end
    isempty(eta_hits) && error("empty sensitivity region")
    return (; threshold, eta_min=minimum(eta_hits), eta_max=maximum(eta_hits),
        zeta_min=minimum(zeta_hits), zeta_max=maximum(zeta_hits))
end

function band_summary(band, eta_values::AbstractVector, zeta_values::AbstractVector)
    eta_step = length(eta_values) >= 2 ? abs(eta_values[2] - eta_values[1]) : 0.0
    zeta_step = length(zeta_values) >= 2 ? abs(zeta_values[2] - zeta_values[1]) : 0.0
    eta_center = 0.5 * (band.eta_min + band.eta_max)
    eta_halfwidth = max(0.5 * (band.eta_max - band.eta_min), 0.5 * eta_step)
    zeta_center = 0.5 * (band.zeta_min + band.zeta_max)
    zeta_halfwidth = max(0.5 * (band.zeta_max - band.zeta_min), 0.5 * zeta_step)
    return (; eta_center, eta_halfwidth, zeta_center, zeta_halfwidth)
end

function plot_raw_traces(path::String, radii::AbstractVector, times::AbstractVector,
        F_mean, F_stderr, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int};
        title::AbstractString="Spin-aligned F(r,t) ensemble mean, r ≤ 80")
    mkpath(dirname(path))
    fig = Figure(size=(1100, 700))
    Label(fig[0, :], title, fontsize=22)
    ax = Axis(fig[1, 1], xlabel="r", ylabel="F(r, t)")
    palette = Makie.wong_colors()
    x = radii[radius_mask]
    for (k, tidx) in enumerate(time_indices)
        color = palette[mod1(k, length(palette))]
        mean = F_mean[radius_mask, tidx]
        err = F_stderr[radius_mask, tidx]
        band!(ax, x, mean .- err, mean .+ err, color=(color, 0.35))
        lines!(ax, x, mean, color=color, linewidth=3,
            label="t = $(round(times[tidx]; digits=3))")
    end
    xlims!(ax, 0, maximum(x))
    axislegend(ax, position=:rb, nbanks=2)
    save(path, fig)
end

function plot_collapsed_traces(path::String, collapse, feature;
        title::Union{Nothing, AbstractString}=nothing)
    mkpath(dirname(path))
    fig = Figure(size=(1100, 700))
    plot_title = isnothing(title) ?
        @sprintf("Collapsed traces, η_F = %.3f, ζ = %.3f", collapse.eta, collapse.zeta) :
        title
    Label(fig[0, :], plot_title, fontsize=22)
    ax = Axis(fig[1, 1], xlabel="r / t^ζ", ylabel="t^η F(r,t)")
    palette = Makie.wong_colors()

    for (k, t) in enumerate(feature.times)
        inds = findall(==(t), collapse.time)
        local_order = sortperm(collapse.x[inds])
        ordered_inds = inds[local_order]
        band!(ax, collapse.x[ordered_inds],
            collapse.y[ordered_inds] .- collapse.sigma[ordered_inds],
            collapse.y[ordered_inds] .+ collapse.sigma[ordered_inds],
            color=(palette[mod1(k, length(palette))], 0.30))
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
    finite_mask = isfinite.(objective)
    finite_values = objective[finite_mask]
    isempty(finite_values) && error("no finite objective values available for heatmap")
    zmin = minimum(finite_values)
    zmax = min(maximum(finite_values), band.threshold)
    plotted_objective = map(x -> isfinite(x) ? x : NaN, objective)
    colorrange_valid = Float32(zmax) > Float32(zmin)

    fig = Figure(size=(900, 700))
    Label(fig[0, :], "Collapse objective over (η_F, ζ)", fontsize=22)
    ax = Axis(fig[1, 1], xlabel="ζ", ylabel="η_F")
    if colorrange_valid
        hm = heatmap!(ax, zeta_values, eta_values, plotted_objective;
            colorrange=(zmin, zmax))
        Colorbar(fig[1, 2], hm, label="reduced χ²")
    else
        text!(ax, 0.5, 0.5;
            space=:relative,
            text=@sprintf("Flat finite objective surface\nχ² = %.4f", zmin),
            align=(:center, :center))
    end
    scatter!(ax, [best.zeta], [best.eta], color=:white, marker=:star5, markersize=22)
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
        title=@sprintf("Trough position fit: ζ = %.3f ± %.3f",
            feature.zeta, feature.zeta_stderr))
    scatter!(ax1, t, feature.trough_radii, markersize=12)
    lines!(ax1, t, fit_r, linewidth=3)

    ax2 = Axis(fig[1, 2], xlabel="t", ylabel="-F_min", xscale=log10, yscale=log10,
        title=@sprintf("Trough amplitude fit: η_F = %.3f ± %.3f",
            feature.eta, feature.eta_stderr))
    scatter!(ax2, t, feature.trough_amplitudes, markersize=12)
    lines!(ax2, t, fit_a, linewidth=3)
    save(path, fig)
end
