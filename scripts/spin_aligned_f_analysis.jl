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
        F_mean, F_stderr, radius_mask::AbstractVector{Bool}, time_indices::AbstractVector{Int};
        title::AbstractString="Spin-aligned F(r,t) ensemble mean, r ≤ 40")
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
