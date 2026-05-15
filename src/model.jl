struct ModelParams
    L::Int
    Q::Float64
    J::Float64
    v::Float64
end

function ModelParams(; L::Integer, Q::Real, J::Real, v::Real)
    L > 1 || throw(ArgumentError("L must be greater than 1"))
    Q >= 0 || throw(ArgumentError("Q must be nonnegative"))
    return ModelParams(Int(L), Float64(Q), Float64(J), Float64(v))
end

struct DriftWorkspace
    params::ModelParams
    mu::Vector{Float64}
end

DriftWorkspace(params::ModelParams) = DriftWorkspace(params, zeros(params.L * params.L))

site_index(x::Integer, y::Integer, L::Integer) = mod1(x, L) + (mod1(y, L) - 1) * L

function wrap_angles!(theta::AbstractVector{<:Real})
    twoπ = 2π
    @inbounds for i in eachindex(theta)
        theta[i] = mod(theta[i], twoπ)
    end
    return theta
end

function random_angles(rng::AbstractRNG, L::Integer)
    return 2π .* rand(rng, Int(L) * Int(L))
end

function initial_angles(rng::AbstractRNG, L::Integer, initial_condition::Symbol)
    if initial_condition === :random
        return random_angles(rng, L)
    elseif initial_condition === :ordered
        return zeros(Int(L) * Int(L))
    else
        throw(ArgumentError("initial_condition must be :random or :ordered"))
    end
end

function compute_mu!(mu::AbstractVector{Float64}, theta::AbstractVector{<:Real}, params::ModelParams)
    L = params.L
    length(theta) == L * L || throw(DimensionMismatch("theta length must be L^2"))
    length(mu) == L * L || throw(DimensionMismatch("mu length must be L^2"))
    J = params.J

    @inbounds for y in 1:L, x in 1:L
        c = site_index(x, y, L)
        xp = site_index(x + 1, y, L)
        xm = site_index(x - 1, y, L)
        yp = site_index(x, y + 1, L)
        ym = site_index(x, y - 1, L)
        θc = theta[c]
        mu[c] = J * (
            -sin(theta[xp] - θc) + sin(θc - theta[xm]) -
            sin(theta[yp] - θc) + sin(θc - theta[ym])
        )
    end
    return mu
end

function xy_energy(theta::AbstractVector{<:Real}, params::ModelParams)
    L = params.L
    length(theta) == L * L || throw(DimensionMismatch("theta length must be L^2"))
    energy = 0.0
    J = params.J

    @inbounds for y in 1:L, x in 1:L
        c = site_index(x, y, L)
        xp = site_index(x + 1, y, L)
        yp = site_index(x, y + 1, L)
        energy -= J * (cos(theta[c] - theta[xp]) + cos(theta[c] - theta[yp]))
    end
    return energy
end

function drift!(du::AbstractVector{Float64}, theta::AbstractVector{<:Real}, work::DriftWorkspace, t)
    params = work.params
    L = params.L
    length(du) == L * L || throw(DimensionMismatch("du length must be L^2"))
    compute_mu!(work.mu, theta, params)

    Q = params.Q
    vhalf = params.v / 2
    mu = work.mu

    @inbounds for y in 1:L, x in 1:L
        c = site_index(x, y, L)
        xp = site_index(x + 1, y, L)
        xm = site_index(x - 1, y, L)
        yp = site_index(x, y + 1, L)
        ym = site_index(x, y - 1, L)

        θc = theta[c]
        θxp = theta[xp]
        θxm = theta[xm]
        θyp = theta[yp]
        θym = theta[ym]

        passive = -(Q / 2) * mu[c]
        active_gradient = sin(θxp) - sin(θxm) - cos(θyp) + cos(θym)
        active_x = cos(θc) * (mu[xp] - mu[xm]) + cos(θxp) * mu[xp] - cos(θxm) * mu[xm]
        active_y = sin(θc) * (mu[yp] - mu[ym]) + sin(θyp) * mu[yp] - sin(θym) * mu[ym]

        du[c] = passive + vhalf * (active_gradient + active_x + active_y)
    end
    return du
end

function noise!(du::AbstractVector{Float64}, theta, work::DriftWorkspace, t)
    fill!(du, sqrt(work.params.Q))
    return du
end

expected_eta(params::ModelParams) = 1 / (2π * params.J)
