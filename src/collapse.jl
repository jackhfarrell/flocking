# Fit F(r,t) = t^(-eta) f(r/t^zeta) by minimizing the weighted scatter between
# collapsed time traces. A coarse scan locates the minimum and a fine scan fixes the
# reported exponent and its local sensitivity band.

function polynomial_design(x::AbstractVector, order::Integer)
    design = Matrix{Float64}(undef, length(x), order + 1)
    design[:, 1] .= 1.0
    for degree in 1:order
        design[:, degree + 1] .= x .^ degree
    end
    return design
end

function weighted_polynomial_fit(x, y, sigma, order::Integer)
    design = polynomial_design(x, order)
    weights = 1.0 ./ sigma
    coefficients = (design .* weights) \ (y .* weights)
    return design * coefficients
end

function collapse_points(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        eta::Real, zeta::Real)
    x_min = maximum(minimum(radii[radius_mask] ./ times[t]^zeta) for t in time_indices)
    x_max = minimum(maximum(radii[radius_mask] ./ times[t]^zeta) for t in time_indices)
    x_min < x_max || return nothing

    x_all = Float64[]
    y_all = Float64[]
    sigma_all = Float64[]
    time_all = Float64[]
    masked_radii = radii[radius_mask]
    for tidx in time_indices
        time = times[tidx]
        x = masked_radii ./ time^zeta
        keep = (x .>= x_min) .& (x .<= x_max)
        append!(x_all, x[keep])
        append!(y_all, F_mean[radius_mask, tidx][keep] .* time^eta)
        append!(sigma_all, max.(F_stderr[radius_mask, tidx][keep] .* time^eta, 1e-12))
        append!(time_all, fill(time, count(keep)))
    end
    return (; x=x_all, y=y_all, sigma=sigma_all, time=time_all, x_min, x_max)
end

function binned_scatter(collapsed, nbins::Integer)
    edges = range(collapsed.x_min, collapsed.x_max; length=nbins + 1)
    score = 0.0
    dof = 0
    for bin in 1:nbins
        right = bin == nbins ? collapsed.x .<= edges[bin + 1] :
            collapsed.x .< edges[bin + 1]
        indices = findall((collapsed.x .>= edges[bin]) .& right)
        length(indices) >= 2 || continue
        length(unique(collapsed.time[indices])) >= 2 || continue
        values = collapsed.y[indices]
        sigma = collapsed.sigma[indices]
        weights = 1.0 ./ sigma.^2
        average = sum(weights .* values) / sum(weights)
        score += sum(weights .* (values .- average).^2)
        dof += length(indices) - 1
    end
    return dof == 0 ? Inf : score / dof
end

function evaluate_collapse(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        eta::Real, zeta::Real, poly_order::Integer, nbins::Integer)
    collapsed = collapse_points(radii, times, F_mean, F_stderr, radius_mask,
        time_indices, eta, zeta)
    collapsed === nothing && return nothing
    fitted = weighted_polynomial_fit(
        collapsed.x, collapsed.y, collapsed.sigma, poly_order)
    reduced_chi2 = binned_scatter(collapsed, nbins)
    return merge(collapsed, (; eta, zeta, fitted, reduced_chi2))
end

function scan_collapse(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        eta_values, zeta_values, poly_order::Integer, nbins::Integer)
    objective = fill(Inf, length(eta_values), length(zeta_values))
    best = nothing
    for (i, eta) in enumerate(eta_values), (j, zeta) in enumerate(zeta_values)
        fit = evaluate_collapse(radii, times, F_mean, F_stderr, radius_mask,
            time_indices, eta, zeta, poly_order, nbins)
        fit === nothing && continue
        objective[i, j] = fit.reduced_chi2
        if best === nothing || fit.reduced_chi2 < best.reduced_chi2
            best = fit
        end
    end
    return objective, best
end

function sensitivity_band(objective, eta_values, zeta_values, factor::Real)
    threshold = factor * minimum(filter(isfinite, objective))
    hits = findall(objective .<= threshold)
    etas = [eta_values[index[1]] for index in hits]
    zetas = [zeta_values[index[2]] for index in hits]
    return (; threshold, eta_min=minimum(etas), eta_max=maximum(etas),
        zeta_min=minimum(zetas), zeta_max=maximum(zetas))
end

"""
    best_collapse(radii, times, mean_F, stderr_F; rmax=40, time_indices)

Fit the spin-aligned correlator by a coarse and fine trace-collapse scan. The returned
band contains points whose objective is at most 1.05 times the minimum.
"""
function best_collapse(radii, times, F_mean, F_stderr; rmax::Real=40.0,
        time_indices=findall(>(0), times), poly_order::Integer=3, nbins::Integer=60)
    radius_mask = radii .<= rmax
    coarse_eta = collect(-0.2:0.02:1.6)
    coarse_zeta = collect(0.0:0.02:0.8)
    _, coarse = scan_collapse(radii, times, F_mean, F_stderr, radius_mask,
        time_indices, coarse_eta, coarse_zeta, poly_order, nbins)

    fine_eta = collect((coarse.eta - 0.06):0.005:(coarse.eta + 0.06))
    fine_zeta = collect((coarse.zeta - 0.06):0.005:(coarse.zeta + 0.06))
    objective, best = scan_collapse(radii, times, F_mean, F_stderr, radius_mask,
        time_indices, fine_eta, fine_zeta, poly_order, nbins)
    band = sensitivity_band(objective, fine_eta, fine_zeta, 1.05)
    halfwidth = max((band.zeta_max - band.zeta_min) / 2,
        abs(fine_zeta[2] - fine_zeta[1]) / 2)
    return (; best, band, halfwidth)
end

"""
    fit_window_robustness(radii, times, mean_F, stderr_F)

Compare the reference collapse with three radius cutoffs and with the first or last
positive lag removed. The half-spread is the fit-window uncertainty.
"""
function fit_window_robustness(radii, times, F_mean, F_stderr)
    time_indices = findall(>(0), times)
    reference = best_collapse(radii, times, F_mean, F_stderr;
        rmax=40.0, time_indices)
    zetas = [
        best_collapse(radii, times, F_mean, F_stderr; rmax=20.0, time_indices).best.zeta,
        reference.best.zeta,
        best_collapse(radii, times, F_mean, F_stderr; rmax=60.0, time_indices).best.zeta,
        best_collapse(radii, times, F_mean, F_stderr;
            rmax=40.0, time_indices=time_indices[2:end]).best.zeta,
        best_collapse(radii, times, F_mean, F_stderr;
            rmax=40.0, time_indices=time_indices[1:(end - 1)]).best.zeta,
    ]
    return (; reference, zetas, halfspread=(maximum(zetas) - minimum(zetas)) / 2)
end

# Compare every time trace on the same scaled-radius grid. This avoids changing the
# membership of spatial bins as ζ moves through the fit grid.
function interpolate_radius(radii, values, radius::Real, last_index::Integer)
    index = searchsortedlast(radii, radius)
    index <= 0 && return values[1]
    index >= last_index && return values[last_index]
    fraction = (radius - radii[index]) / (radii[index + 1] - radii[index])
    return (1 - fraction) * values[index] + fraction * values[index + 1]
end

"""
    common_grid_collapse_score(radii, times, F, time_indices, η, ζ;
        rmax=40, grid_points=48)

Measure the scatter of `t^η F(r,t)` after interpolation onto one common
`r/t^ζ` grid. The score is the squared between-trace scatter divided by the total
collapsed signal power. Radii and times receive equal weight because their errors are
correlated within each trajectory.
"""
function common_grid_collapse_score(radii, times, F, time_indices,
        eta::Real, zeta::Real; rmax::Real=40.0, grid_points::Integer=48)
    last_index = searchsortedlast(radii, rmax)
    last_index >= 2 || return Inf
    length(time_indices) >= 3 || return Inf

    x_scale = [times[index]^zeta for index in time_indices]
    y_scale = [times[index]^eta for index in time_indices]
    x_min = maximum(radii[1] ./ x_scale)
    x_max = minimum(radii[last_index] ./ x_scale)
    x_min < x_max || return Inf

    values = zeros(Float64, length(time_indices))
    residual = 0.0
    power = 0.0
    for x in range(x_min, x_max; length=grid_points)
        for (slot, index) in enumerate(time_indices)
            radius = x * x_scale[slot]
            values[slot] = y_scale[slot] * interpolate_radius(
                radii, @view(F[:, index]), radius, last_index)
        end
        average = mean(values)
        residual += sum(value -> abs2(value - average), values)
        power += sum(abs2, values)
    end
    power > eps(Float64) || return Inf
    return residual / power
end

function scan_common_grid_collapse(radii, times, F, time_indices,
        eta_values, zeta_values; rmax::Real=40.0, grid_points::Integer=48)
    objective = fill(Inf, length(eta_values), length(zeta_values))
    best = (; eta=NaN, zeta=NaN, score=Inf)
    for (i, eta) in enumerate(eta_values), (j, zeta) in enumerate(zeta_values)
        score = common_grid_collapse_score(radii, times, F, time_indices, eta, zeta;
            rmax, grid_points)
        objective[i, j] = score
        score < best.score && (best = (; eta, zeta, score))
    end
    return (; objective, best)
end

"""
    best_common_grid_collapse(radii, times, F, time_indices;
        rmax=40, grid_points=48, zeta_min=0.2, zeta_max=0.65,
        fine_eta_step=0.005, fine_zeta_step=0.005)

Fit `η` and `ζ` with a coarse scan followed by a local scan. The common-grid score
does not assign independent statistical meaning to neighboring radii or times.
"""
function best_common_grid_collapse(radii, times, F, time_indices;
        rmax::Real=40.0, grid_points::Integer=48,
        zeta_min::Real=0.2, zeta_max::Real=0.65,
        fine_eta_step::Real=0.005, fine_zeta_step::Real=0.005)
    coarse_eta = collect(-0.2:0.025:1.4)
    coarse_zeta = collect(zeta_min:0.025:zeta_max)
    coarse = scan_common_grid_collapse(radii, times, F, time_indices,
        coarse_eta, coarse_zeta; rmax, grid_points)

    fine_eta = collect((coarse.best.eta - 0.05):fine_eta_step:(coarse.best.eta + 0.05))
    fine_zeta = collect(
        max(zeta_min, coarse.best.zeta - 0.05):fine_zeta_step:
        min(zeta_max, coarse.best.zeta + 0.05))
    fine = scan_common_grid_collapse(radii, times, F, time_indices,
        fine_eta, fine_zeta; rmax, grid_points)
    return (; best=fine.best, coarse_objective=coarse.objective,
        fine_objective=fine.objective, fine_eta, fine_zeta)
end

"""
    best_fixed_common_grid_collapse(radii, times, F, time_indices, ζ;
        rmax=40, grid_points=48)

Fit the amplitude exponent `η` while keeping `ζ` fixed. The returned score can be
compared with the free-fit score on the same data and fit window.
"""
function best_fixed_common_grid_collapse(radii, times, F, time_indices, zeta::Real;
        rmax::Real=40.0, grid_points::Integer=48)
    coarse_eta = collect(-0.2:0.025:1.4)
    coarse = scan_common_grid_collapse(radii, times, F, time_indices,
        coarse_eta, [zeta]; rmax, grid_points)
    fine_eta = collect((coarse.best.eta - 0.05):0.005:(coarse.best.eta + 0.05))
    fine = scan_common_grid_collapse(radii, times, F, time_indices,
        fine_eta, [zeta]; rmax, grid_points)
    return fine.best
end
