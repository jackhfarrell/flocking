struct SimulationConfig
    params::ModelParams
    dt::Float64
    burnin_steps::Int
    sample_stride::Int
    nsamples::Int
    ntrajectories::Int
    seed::Int
    save_final_theta::Bool
    fit_rmin::Float64
    fit_rmax::Float64
    initial_condition::Symbol
end

function SimulationConfig(; L::Integer=32, Q::Real=1.0, J::Real=2.0, v::Real=0.0,
        dt::Real=0.01, burnin_steps::Integer=1_000, sample_stride::Integer=10,
        nsamples::Integer=100, ntrajectories::Integer=1, seed::Integer=1,
        save_final_theta::Bool=false, fit_rmin::Real=2.0, fit_rmax::Real=Inf,
        initial_condition::Symbol=:random)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    burnin_steps >= 0 || throw(ArgumentError("burnin_steps must be nonnegative"))
    sample_stride > 0 || throw(ArgumentError("sample_stride must be positive"))
    nsamples > 0 || throw(ArgumentError("nsamples must be positive"))
    ntrajectories > 0 || throw(ArgumentError("ntrajectories must be positive"))
    initial_condition in (:random, :ordered) ||
        throw(ArgumentError("initial_condition must be :random or :ordered"))
    params = ModelParams(; L, Q, J, v)
    return SimulationConfig(params, Float64(dt), Int(burnin_steps), Int(sample_stride),
        Int(nsamples), Int(ntrajectories), Int(seed), save_final_theta,
        Float64(fit_rmin), Float64(fit_rmax), initial_condition)
end

struct RunResult
    config::SimulationConfig
    seeds::Vector{Int}
    times::Vector{Float64}
    energy_density_mean::Vector{Float64}
    magnetization_mean::Vector{Float64}
    radii::Vector{Float64}
    correlation_mean::Vector{Float64}
    correlation_stderr::Vector{Float64}
    fit::NamedTuple
    final_theta::Union{Nothing, Vector{Float64}}
end

function solve_one(theta0::Vector{Float64}, config::SimulationConfig; seed::Union{Nothing, Integer}=nothing)
    params = config.params
    work = DriftWorkspace(params)
    total_steps = config.burnin_steps + config.sample_stride * config.nsamples
    tspan = (0.0, total_steps * config.dt)
    sample_steps = config.burnin_steps .+ config.sample_stride .* collect(1:config.nsamples)
    saveat = sample_steps .* config.dt
    prob = SDEProblem(drift!, noise!, theta0, tspan, work)
    kwargs = (; dt=config.dt, adaptive=false, saveat=saveat, save_start=false)
    return seed === nothing ? solve(prob, EM(); kwargs...) :
        solve(prob, EM(); kwargs..., seed=Int(seed))
end

function run_trajectory(config::SimulationConfig, seed::Integer)
    rng = MersenneTwister(seed)
    theta0 = initial_angles(rng, config.params.L, config.initial_condition)
    sol = solve_one(theta0, config; seed)

    ns = length(sol.u)
    L = config.params.L
    energies = zeros(Float64, ns)
    mags = zeros(Float64, ns)
    radii = Float64[]
    corr_accum = nothing

    for (i, state) in enumerate(sol.u)
        theta = collect(state)
        wrap_angles!(theta)
        energies[i] = xy_energy(theta, config.params) / (L * L)
        mags[i] = magnetization(theta)
        r, c, _ = radial_correlation(theta, L)
        if corr_accum === nothing
            radii = r
            corr_accum = zeros(Float64, length(c))
        end
        corr_accum .+= c
    end

    corr_mean = corr_accum ./ ns
    fit_rmax = isfinite(config.fit_rmax) ? config.fit_rmax : L / 3
    fit = fit_power_law(radii, corr_mean; rmin=config.fit_rmin, rmax=fit_rmax)
    final_theta = config.save_final_theta ? wrap_angles!(collect(sol.u[end])) : nothing
    return (; seed=Int(seed), times=collect(sol.t), energies, mags, radii,
        correlation=corr_mean, fit, final_theta)
end

function run_ensemble(config::SimulationConfig)
    seeds = collect(config.seed:(config.seed + config.ntrajectories - 1))
    trajectories = Vector{Any}(undef, config.ntrajectories)
    Threads.@threads for i in eachindex(seeds)
        trajectories[i] = run_trajectory(config, seeds[i])
    end

    times = trajectories[1].times
    radii = trajectories[1].radii
    energy_mat = hcat((tr.energies for tr in trajectories)...)
    mag_mat = hcat((tr.mags for tr in trajectories)...)
    corr_mat = hcat((tr.correlation for tr in trajectories)...)

    energy_mean = vec(mean(energy_mat; dims=2))
    mag_mean = vec(mean(mag_mat; dims=2))
    corr_mean = vec(mean(corr_mat; dims=2))
    corr_stderr = config.ntrajectories == 1 ? zeros(length(corr_mean)) :
        vec(std(corr_mat; dims=2)) ./ sqrt(config.ntrajectories)

    fit_rmax = isfinite(config.fit_rmax) ? config.fit_rmax : config.params.L / 3
    fit = fit_power_law(radii, corr_mean; rmin=config.fit_rmin, rmax=fit_rmax)
    final_theta = config.save_final_theta ? trajectories[end].final_theta : nothing

    return RunResult(config, seeds, times, energy_mean, mag_mean, radii,
        corr_mean, corr_stderr, fit, final_theta)
end
