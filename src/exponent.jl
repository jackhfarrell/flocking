# Build the spin-aligned antisymmetric correlator used to measure the spreading exponent.
# Displacements follow the spin at the middle of the time window, with periodic bilinear
# interpolation at the displaced point.

# Advance a fixed-step SDE segment without retaining the path.
function advance_fixed(theta, steps::Integer, dt::Real, work, solver, rng)
    problem = SDEProblem(
        drift!,
        noise!,
        theta,
        (0.0, steps * dt),
        work,
    )
    solution = solve(problem, solver; dt, adaptive=false, save_everystep=false,
        save_start=false, rng)
    return wrap_angles!(collect(solution.u[end]))
end

# Advance an entire correlator window in one solve and retain only the requested states.
function sample_fixed_window(theta, advance_gaps, dt::Real, work, solver, rng)
    sample_steps = cumsum(advance_gaps)
    problem = SDEProblem(
        drift!,
        noise!,
        theta,
        (0.0, sample_steps[end] * dt),
        work,
    )
    solution = solve(problem, solver; dt, adaptive=false,
        saveat=sample_steps .* dt, save_everystep=false, save_start=false, rng)
    window = Vector{Vector{Float64}}(undef, length(sample_steps) + 1)
    window[1] = copy(theta)
    for index in eachindex(solution.u)
        window[index + 1] = wrap_angles!(collect(solution.u[index]))
    end
    return window
end

# Interpolate the spin vector before taking its dot product with the reference spin.
function interpolated_spin_dot(cos_field, sin_field, L::Integer, px::Real, py::Real,
        ref_cos::Real, ref_sin::Real)
    x0 = floor(Int, px)
    y0 = floor(Int, py)
    fx = px - x0
    fy = py - y0

    c = 0.0
    s = 0.0
    @inbounds for (ix, wx) in ((mod1(x0, L), 1 - fx), (mod1(x0 + 1, L), fx))
        for (iy, wy) in ((mod1(y0, L), 1 - fy), (mod1(y0 + 1, L), fy))
            weight = wx * wy
            index = site_index(ix, iy, L)
            c += weight * cos_field[index]
            s += weight * sin_field[index]
        end
    end
    return ref_cos * c + ref_sin * s
end

function spin_aligned_correlators(window::AbstractVector, L::Integer,
        radii::AbstractVector; origins=nothing)
    isodd(length(window)) || throw(ArgumentError("window length must be odd"))

    ntimes = length(window) ÷ 2
    mid = ntimes + 1
    cos_window = [cos.(state) for state in window]
    sin_window = [sin.(state) for state in window]
    cos_mid = cos_window[mid]
    sin_mid = sin_window[mid]
    F = zeros(Float64, length(radii), ntimes + 1)
    origin_points = isnothing(origins) ? Iterators.product(1:L, 1:L) : origins
    norigins = length(origin_points)

    @inbounds for (ridx, r) in enumerate(radii), lag in 0:ntimes
        cos_minus = cos_window[mid - lag]
        sin_minus = sin_window[mid - lag]
        cos_plus = cos_window[mid + lag]
        sin_plus = sin_window[mid + lag]
        accum = 0.0

        for (x, y) in origin_points
            center = x + (y - 1) * L
            cx = cos_mid[center]
            sy = sin_mid[center]
            dx = r * cx
            dy = r * sy

            forward_plus = interpolated_spin_dot(
                cos_plus, sin_plus, L, x + dx, y + dy, cx, sy)
            backward_minus = interpolated_spin_dot(
                cos_minus, sin_minus, L, x - dx, y - dy, cx, sy)
            forward_minus = interpolated_spin_dot(
                cos_minus, sin_minus, L, x + dx, y + dy, cx, sy)
            backward_plus = interpolated_spin_dot(
                cos_plus, sin_plus, L, x - dx, y - dy, cx, sy)

            accum += 0.25 * (forward_plus + backward_minus -
                forward_minus - backward_plus)
        end
        F[ridx, lag + 1] = accum / norigins
    end

    return F
end
