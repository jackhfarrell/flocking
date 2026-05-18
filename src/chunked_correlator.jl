function radial_displacement_shells(L::Integer; oriented::Bool=false)
    L > 1 || throw(ArgumentError("L must be greater than 1"))
    radii, _ = radial_bins(L)
    shells = [Tuple{Int, Int}[] for _ in eachindex(radii)]

    for dy in 0:(L - 1), dx in 0:(L - 1)
        if oriented
            invdx = mod(-dx, L)
            invdy = mod(-dy, L)
            (dx == invdx && dy == invdy) && continue
            (dy, dx) > (invdy, invdx) && continue
        end

        sx = min(dx, L - dx)
        sy = min(dy, L - dy)
        bin = round(Int, sqrt(sx^2 + sy^2))
        if 1 <= bin <= length(shells)
            push!(shells[bin], (dx, dy))
        end
    end

    return (; radii, shells)
end

function chunk_correlator(window::AbstractVector, params::ModelParams, shell_data)
    L = params.L
    nsites = L * L
    isodd(length(window)) || throw(ArgumentError("window length must be odd"))
    all(length(state) == nsites for state in window) ||
        throw(DimensionMismatch("each window state must have length L^2"))

    shells = shell_data.shells
    tmax = length(window) ÷ 2
    mid = tmax + 1
    F = fill(NaN, length(shells), tmax + 1)

    @inbounds for (ridx, shell) in enumerate(shells)
        isempty(shell) && continue
        norm = inv(nsites * length(shell))
        for t in 0:tmax
            theta_minus = window[mid - t]
            theta_mid = window[mid]
            theta_plus = window[mid + t]
            accum = 0.0

            for (dx, dy) in shell, y in 1:L, x in 1:L
                center = site_index(x, y, L)
                forward = site_index(x + dx, y + dy, L)
                backward = site_index(x - dx, y - dy, L)
                theta0 = theta_mid[center]

                accum += 0.25 * (
                    cos(theta_plus[forward] - theta0) +
                    cos(theta_minus[backward] - theta0) -
                    cos(theta_minus[forward] - theta0) -
                    cos(theta_plus[backward] - theta0)
                )
            end

            F[ridx, t + 1] = accum * norm
        end
    end

    return F
end

function online_mean_stderr!(mean::AbstractArray{<:Real}, m2::AbstractArray{<:Real},
        sample::AbstractArray{<:Real}, n::Integer)
    size(mean) == size(m2) == size(sample) ||
        throw(DimensionMismatch("mean, m2, and sample must have matching sizes"))
    n > 0 || throw(ArgumentError("n must be positive"))

    @inbounds for i in eachindex(mean, m2, sample)
        delta = sample[i] - mean[i]
        mean[i] += delta / n
        m2[i] += delta * (sample[i] - mean[i])
    end

    stderr = similar(mean, Float64)
    if n == 1
        fill!(stderr, 0.0)
    else
        @inbounds for i in eachindex(stderr, m2)
            stderr[i] = sqrt(m2[i] / (n - 1)) / sqrt(n)
        end
    end
    return stderr
end

function observable_window_range(values::AbstractVector{<:Real}, window::Integer)
    window > 0 || throw(ArgumentError("window must be positive"))
    length(values) >= window || return Inf
    tail = @view values[(end - window + 1):end]
    all(isfinite, tail) || return Inf
    return maximum(tail) - minimum(tail)
end

function eta_window_range(etas::AbstractVector{<:Real}, window::Integer)
    return observable_window_range(etas, window)
end

function eta_equilibrium_reached(etas::AbstractVector{<:Real}, window::Integer, threshold::Real)
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    return eta_window_range(etas, window) <= threshold
end

function equilibrium_window_blocks(block_time::Real, window_time::Real, min_blocks::Integer)
    block_time > 0 || throw(ArgumentError("block_time must be positive"))
    window_time >= 0 || throw(ArgumentError("window_time must be nonnegative"))
    min_blocks > 0 || throw(ArgumentError("min_blocks must be positive"))
    return max(Int(min_blocks), ceil(Int, window_time / block_time))
end

function equilibrium_stationarity_reached(etas::AbstractVector{<:Real},
        energies::AbstractVector{<:Real}, magnetizations::AbstractVector{<:Real},
        window::Integer, eta_threshold::Real, energy_threshold::Real,
        magnetization_threshold::Real)
    eta_range = observable_window_range(etas, window)
    energy_range = observable_window_range(energies, window)
    magnetization_range = observable_window_range(magnetizations, window)
    reached = eta_range <= eta_threshold &&
        energy_range <= energy_threshold &&
        magnetization_range <= magnetization_threshold
    return (; reached, eta_range, energy_range, magnetization_range, window)
end
