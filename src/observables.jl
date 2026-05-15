function magnetization(theta::AbstractVector{<:Real})
    mx = mean(cos, theta)
    my = mean(sin, theta)
    return sqrt(mx^2 + my^2)
end

function radial_bins(L::Integer)
    maxr = L ÷ 2
    bins = collect(1:maxr)
    counts = zeros(Int, maxr)
    radii = zeros(Float64, maxr)

    for dy in 0:(L - 1), dx in 0:(L - 1)
        sx = min(dx, L - dx)
        sy = min(dy, L - dy)
        r = sqrt(sx^2 + sy^2)
        bin = round(Int, r)
        if 1 <= bin <= maxr
            counts[bin] += 1
            radii[bin] += r
        end
    end

    for i in eachindex(radii)
        radii[i] = counts[i] == 0 ? bins[i] : radii[i] / counts[i]
    end
    return radii, counts
end

function radial_correlation(theta::AbstractVector{<:Real}, L::Integer)
    length(theta) == L * L || throw(DimensionMismatch("theta length must be L^2"))
    z = reshape(exp.(im .* theta), L, L)
    corr = real.(ifft(abs2.(fft(z)))) ./ (L * L)

    radii, counts = radial_bins(L)
    radial = zeros(Float64, length(radii))

    for dy in 0:(L - 1), dx in 0:(L - 1)
        sx = min(dx, L - dx)
        sy = min(dy, L - dy)
        bin = round(Int, sqrt(sx^2 + sy^2))
        if 1 <= bin <= length(radial)
            radial[bin] += corr[dx + 1, dy + 1]
        end
    end

    for i in eachindex(radial)
        radial[i] = counts[i] == 0 ? NaN : radial[i] / counts[i]
    end
    return radii, radial, counts
end

function fit_power_law(r::AbstractVector{<:Real}, c::AbstractVector{<:Real}; rmin::Real=2, rmax::Real=Inf)
    mask = map(eachindex(r, c)) do i
        isfinite(r[i]) && isfinite(c[i]) && r[i] >= rmin && r[i] <= rmax && c[i] > 0
    end
    xs = log.(Float64.(r[mask]))
    ys = log.(Float64.(c[mask]))
    if length(xs) < 2
        return (; eta=NaN, slope=NaN, intercept=NaN, rmin=Float64(rmin),
            rmax=Float64(rmax), npoints=length(xs))
    end

    xbar = mean(xs)
    ybar = mean(ys)
    slope = sum((xs .- xbar) .* (ys .- ybar)) / sum((xs .- xbar).^2)
    intercept = ybar - slope * xbar
    eta = -slope
    return (; eta, slope, intercept, rmin=Float64(rmin), rmax=Float64(rmax), npoints=length(xs))
end
