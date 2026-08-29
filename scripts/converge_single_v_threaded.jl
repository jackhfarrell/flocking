#!/usr/bin/env julia

# Average four continuing trajectories and report exponent convergence while they run.

using ArgParse
using JLD2
using Printf
using Random
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(description="Threaded streaming convergence at one velocity.")
@add_arg_table! settings begin
    "--v"
        arg_type = Float64
        default = 0.1
    "--temperature"
        arg_type = Float64
        default = 0.5
    "--L"
        arg_type = Int
        default = 96
    "--Q"
        arg_type = Float64
        default = 1.0
    "--dt"
        arg_type = Float64
        default = 2.0^-9
    "--chains"
        arg_type = Int
        default = 4
    "--tolerance"
        arg_type = Float64
        default = 0.03
    "--maximum-rounds"
        arg_type = Int
        default = 1000
    "--block-windows"
        arg_type = Int
        default = 4
    "--minimum-blocks"
        arg_type = Int
        default = 2
    "--stable-checks"
        arg_type = Int
        default = 3
    "--chain-error-mode"
        arg_type = String
        default = "half-range"
    "--dr"
        arg_type = Float64
        default = 1.0
    "--r-max"
        arg_type = Float64
        default = 24.0
    "--fit-rmax"
        arg_type = Float64
        default = 20.0
    "--T-max"
        arg_type = Float64
        default = 16.0
    "--ntimes"
        arg_type = Int
        default = 8
    "--reference-points"
        arg_type = Int
        default = 5
    "--equilibrium-block-steps"
        arg_type = Int
        default = 512
    "--equilibrium-window-time"
        arg_type = Float64
        default = 16.0
    "--equilibrium-window-blocks"
        arg_type = Int
        default = 8
    "--maximum-equilibrium-blocks"
        arg_type = Int
        default = 256
    "--equilibrium-stable-checks"
        arg_type = Int
        default = 3
    "--energy-threshold"
        arg_type = Float64
        default = 0.02
    "--magnetization-threshold"
        arg_type = Float64
        default = 0.02
    "--grid-points"
        arg_type = Int
        default = 48
    "--report-seconds"
        arg_type = Float64
        default = 30.0
    "--seed"
        arg_type = Int
        default = 1_610_000
    "--output"
        arg_type = String
        default = "results/single_v_threaded_L96_v0p1.jld2"
end
args = ArgParse.parse_args(settings)

nchains = args["chains"]
Threads.nthreads() >= nchains ||
    error("start Julia with at least --threads=$(nchains)")
chain_error_mode = args["chain-error-mode"]
chain_error_mode in ("half-range", "standard-error") ||
    error("chain-error-mode must be half-range or standard-error")
temperature = args["temperature"]
J = inv(temperature)
L = args["L"]
Q = args["Q"]
v = args["v"]
dt = args["dt"]
params = ModelParams(; L, Q, J, v)
workspaces = [LatticeFlockingSDE.DriftWorkspace(params) for _ in 1:nchains]
solver = SRA1()
radii = collect(args["dr"]:args["dr"]:min(args["r-max"], L / 2))
lag_steps = round(Int, args["T-max"] / args["ntimes"] / dt)
schedule = lag_step_schedule(args["ntimes"], lag_steps; spacing=:geometric)
times = schedule.cum_steps .* dt
equilibrium_window = equilibrium_window_blocks(
    args["equilibrium-block-steps"] * dt,
    args["equilibrium-window-time"],
    args["equilibrium-window-blocks"],
)
config = (;
    temperature, J, L, Q, v, dt, nchains,
    block_windows=args["block-windows"],
    dr=args["dr"], r_max=args["r-max"], fit_rmax=args["fit-rmax"],
    T_max=args["T-max"], ntimes=args["ntimes"],
    reference_points=args["reference-points"],
    equilibrium_block_steps=args["equilibrium-block-steps"],
    equilibrium_window,
    equilibrium_stable_checks=args["equilibrium-stable-checks"],
    energy_threshold=args["energy-threshold"],
    magnetization_threshold=args["magnetization-threshold"],
    grid_points=args["grid-points"], seed=args["seed"],
    solver=string(typeof(solver)),
)

rngs = [MersenneTwister(args["seed"] + chain) for chain in 1:nchains]
theta = [initial_angles(rngs[chain], L, :ordered) for chain in 1:nchains]
equilibrium_energies = [Float64[] for _ in 1:nchains]
equilibrium_magnetizations = [Float64[] for _ in 1:nchains]
equilibrium_stable_counts = zeros(Int, nchains)
equilibrium_reached = falses(nchains)
chain_means = zeros(Float64, length(radii), length(times), nchains)
block_sums = zeros(Float64, size(chain_means))
block_counts = zeros(Int, nchains)
nrounds = 0
F_blocks = Array{Float64}(undef, length(radii), length(times), 0, nchains)
history = Any[]

if isfile(args["output"])
    saved = load(args["output"], "run")
    saved.config == config || error("the saved run has different simulation settings")
    rngs = saved.rngs
    theta = saved.theta
    equilibrium_energies = saved.equilibrium_energies
    equilibrium_magnetizations = saved.equilibrium_magnetizations
    equilibrium_stable_counts = saved.equilibrium_stable_counts
    equilibrium_reached = saved.equilibrium_reached
    chain_means = saved.chain_means
    block_sums = saved.block_sums
    block_counts = saved.block_counts
    nrounds = saved.nrounds
    F_blocks = saved.F_blocks
    history = Any[saved.history...]
    @info "resumed threaded run" nrounds blocks=size(F_blocks, 3)
end

mkpath(dirname(args["output"]))
last_report = time()

let rngs=rngs, theta=theta, chain_means=chain_means,
        block_sums=block_sums, block_counts=block_counts,
        nrounds=nrounds, F_blocks=F_blocks, last_report=last_report

while !all(equilibrium_reached) &&
        maximum(length.(equilibrium_energies)) < args["maximum-equilibrium-blocks"]
    Threads.@threads for chain in 1:nchains
        equilibrium_reached[chain] && continue
        theta[chain] = LatticeFlockingSDE.advance_fixed(
            theta[chain], args["equilibrium-block-steps"], dt,
            workspaces[chain], solver, rngs[chain])
        push!(equilibrium_energies[chain], xy_energy(theta[chain], params) / L^2)
        push!(equilibrium_magnetizations[chain], magnetization(theta[chain]))
    end

    energy_deltas = fill(Inf, nchains)
    magnetization_deltas = fill(Inf, nchains)
    for chain in 1:nchains
        equilibrium_reached[chain] && continue
        values = equilibrium_energies[chain]
        if length(values) >= 2equilibrium_window
            last_block = length(values)
            previous = (last_block - 2equilibrium_window + 1):(last_block - equilibrium_window)
            current = (last_block - equilibrium_window + 1):last_block
            energy_deltas[chain] = abs(mean(@view equilibrium_energies[chain][previous]) -
                mean(@view equilibrium_energies[chain][current]))
            magnetization_deltas[chain] = abs(
                mean(@view equilibrium_magnetizations[chain][previous]) -
                mean(@view equilibrium_magnetizations[chain][current]))
            plateau = energy_deltas[chain] <= args["energy-threshold"] &&
                magnetization_deltas[chain] <= args["magnetization-threshold"]
            equilibrium_stable_counts[chain] = plateau ?
                equilibrium_stable_counts[chain] + 1 : 0
            equilibrium_reached[chain] = equilibrium_stable_counts[chain] >=
                args["equilibrium-stable-checks"]
        end
    end

    run = (;
        config, rngs, theta, equilibrium_energies, equilibrium_magnetizations,
        equilibrium_stable_counts, equilibrium_reached, chain_means,
        block_sums, block_counts, nrounds, F_blocks, history,
    )
    temporary_output = args["output"] * ".tmp"
    jldsave(temporary_output; run)
    mv(temporary_output, args["output"]; force=true)

    if time() - last_report >= args["report-seconds"] || all(equilibrium_reached)
        finite_energy_deltas = filter(isfinite, energy_deltas)
        finite_magnetization_deltas = filter(isfinite, magnetization_deltas)
        @info(
            "threaded status",
            phase="equilibrating",
            reached=count(equilibrium_reached),
            chains=nchains,
            blocks=maximum(length.(equilibrium_energies)),
            max_energy_delta=isempty(finite_energy_deltas) ? 0.0 :
                maximum(finite_energy_deltas),
            max_magnetization_delta=isempty(finite_magnetization_deltas) ? 0.0 :
                maximum(finite_magnetization_deltas),
            zeta=NaN,
            estimated_error=Inf,
        )
        last_report = time()
    end
end

all(equilibrium_reached) || error("equilibration did not converge before the block limit")
converged = !isempty(history) && history[end].converged

while !converged && nrounds < args["maximum-rounds"]
    samples = Vector{Matrix{Float64}}(undef, nchains)
    Threads.@threads for chain in 1:nchains
        window = sample_fixed_window(theta[chain], schedule.advance_gaps, dt,
            workspaces[chain], solver, rngs[chain])
        theta[chain] = copy(window[end])
        samples[chain] = LatticeFlockingSDE.spin_aligned_correlators(window, L, radii)
    end

    nrounds += 1
    for chain in 1:nchains
        @views chain_means[:, :, chain] .+=
            (samples[chain] .- chain_means[:, :, chain]) ./ nrounds
        @views block_sums[:, :, chain] .+= samples[chain]
        block_counts[chain] += 1
    end

    if all(==(args["block-windows"]), block_counts)
        new_block = Array{Float64}(undef, length(radii), length(times), 1, nchains)
        for chain in 1:nchains
            @views new_block[:, :, 1, chain] .=
                block_sums[:, :, chain] ./ block_counts[chain]
        end
        F_blocks = cat(F_blocks, new_block; dims=3)
        fill!(block_sums, 0.0)
        fill!(block_counts, 0)
    end

    if size(F_blocks, 3) >= args["minimum-blocks"] && all(iszero, block_counts)
        mean_field = dropdims(mean(chain_means; dims=3), dims=3)
        positive = findall(>(0), times)
        reference_count = min(args["reference-points"], length(positive))
        reference_indices = positive[(end - reference_count + 1):end]
        reference = best_common_grid_collapse(
            radii, times, mean_field, reference_indices;
            rmax=args["fit-rmax"], grid_points=args["grid-points"])
        fit_at_boundary = reference.best.zeta <= 0.155 || reference.best.zeta >= 0.695

        time_counts = unique(clamp.((reference_count - 1):reference_count + 1,
            3, length(positive)))
        time_zetas = [
            best_common_grid_collapse(
                radii, times, mean_field, positive[(end - count + 1):end];
                rmax=args["fit-rmax"], grid_points=args["grid-points"]).best.zeta
            for count in time_counts
        ]
        radius_maxima = unique(min.((
            0.75args["fit-rmax"], args["fit-rmax"], 1.25args["fit-rmax"]),
            last(radii)))
        radius_zetas = [
            best_common_grid_collapse(
                radii, times, mean_field, reference_indices;
                rmax, grid_points=args["grid-points"]).best.zeta
            for rmax in radius_maxima
        ]

        nblocks = size(F_blocks, 3)
        split = nblocks ÷ 2
        first_field = dropdims(mean(
            @view(F_blocks[:, :, 1:split, :]); dims=(3, 4)), dims=(3, 4))
        second_field = dropdims(mean(
            @view(F_blocks[:, :, (split + 1):nblocks, :]); dims=(3, 4)), dims=(3, 4))
        first_fit = best_common_grid_collapse(
            radii, times, first_field, reference_indices;
            rmax=args["fit-rmax"], grid_points=args["grid-points"]).best
        second_fit = best_common_grid_collapse(
            radii, times, second_field, reference_indices;
            rmax=args["fit-rmax"], grid_points=args["grid-points"]).best
        half_run_halfwidth = abs(first_fit.zeta - second_fit.zeta) / 2

        chain_zetas = [
            best_common_grid_collapse(
                radii, times, @view(chain_means[:, :, chain]), reference_indices;
                rmax=args["fit-rmax"], grid_points=args["grid-points"]).best.zeta
            for chain in 1:nchains
        ]
        cross_chain_halfwidth = (maximum(chain_zetas) - minimum(chain_zetas)) / 2
        between_chain_error = nchains == 1 ? Inf : std(chain_zetas) / sqrt(nchains)
        chain_error = chain_error_mode == "half-range" ?
            cross_chain_halfwidth : between_chain_error

        groups_per_chain = min(2, nblocks)
        ngroups = nchains * groups_per_chain
        total_sum = dropdims(sum(F_blocks; dims=(3, 4)), dims=(3, 4))
        jackknife_zetas = Float64[]
        for chain in 1:nchains, group in 1:groups_per_chain
            first_omitted = fld((group - 1) * nblocks, groups_per_chain) + 1
            last_omitted = fld(group * nblocks, groups_per_chain)
            omitted = @view F_blocks[:, :, first_omitted:last_omitted, chain]
            omitted_sum = dropdims(sum(omitted; dims=3), dims=3)
            omitted_count = last_omitted - first_omitted + 1
            field = (total_sum .- omitted_sum) ./ (nblocks * nchains - omitted_count)
            eta_values = collect(
                (reference.best.eta - 0.06):0.01:(reference.best.eta + 0.06))
            zeta_values = collect(
                (reference.best.zeta - 0.06):0.005:(reference.best.zeta + 0.06))
            fit = LatticeFlockingSDE.scan_common_grid_collapse(
                radii, times, field, reference_indices, eta_values, zeta_values;
                rmax=args["fit-rmax"], grid_points=args["grid-points"]).best
            at_edge = abs(fit.eta - reference.best.eta) >= 0.059 ||
                abs(fit.zeta - reference.best.zeta) >= 0.059
            if at_edge
                fit = best_common_grid_collapse(
                    radii, times, field, reference_indices;
                    rmax=args["fit-rmax"], grid_points=args["grid-points"]).best
            end
            push!(jackknife_zetas, fit.zeta)
        end
        jackknife_mean = mean(jackknife_zetas)
        jackknife_error = sqrt((ngroups - 1) / ngroups *
            sum(abs2(zeta - jackknife_mean) for zeta in jackknife_zetas))
        time_halfspread = (maximum(time_zetas) - minimum(time_zetas)) / 2
        radius_halfspread = (maximum(radius_zetas) - minimum(radius_zetas)) / 2
        recent_zetas = [row.zeta for row in history[
            max(1, end - args["stable-checks"] + 2):end]]
        push!(recent_zetas, reference.best.zeta)
        stability_error = length(recent_zetas) < args["stable-checks"] ? Inf :
            (maximum(recent_zetas) - minimum(recent_zetas)) / 2
        estimated_error = fit_at_boundary ? Inf : maximum((
            jackknife_error, chain_error, half_run_halfwidth,
            time_halfspread, radius_halfspread, stability_error,
        ))
        converged = estimated_error <= args["tolerance"]

        row = (;
            checkpoint=length(history) + 1, nrounds,
            samples=nrounds * nchains, nblocks,
            eta=reference.best.eta, zeta=reference.best.zeta,
            estimated_error, jackknife_error, chain_error_mode,
            chain_error, between_chain_error, cross_chain_halfwidth,
            half_run_halfwidth, time_halfspread, radius_halfspread,
            stability_error, chain_zeta_min=minimum(chain_zetas),
            chain_zeta_max=maximum(chain_zetas),
            first_half_zeta=first_fit.zeta, second_half_zeta=second_fit.zeta,
            fit_at_boundary, converged,
        )
        push!(history, row)
        @info "threaded checkpoint" row
    end

    run = (;
        config, rngs, theta, equilibrium_energies, equilibrium_magnetizations,
        equilibrium_stable_counts, equilibrium_reached, chain_means,
        block_sums, block_counts, nrounds, F_blocks, history,
    )
    temporary_output = args["output"] * ".tmp"
    jldsave(temporary_output; run)
    mv(temporary_output, args["output"]; force=true)

    if !isempty(history)
        csv = replace(args["output"], r"\.jld2$" => ".csv")
        open(csv, "w") do io
            columns = keys(history[end])
            println(io, join(columns, ","))
            for row in history
                println(io, join((hasproperty(row, column) ?
                    getproperty(row, column) : "" for column in columns), ","))
            end
        end
    end

    if time() - last_report >= args["report-seconds"]
        zeta = isempty(history) ? NaN : history[end].zeta
        estimated_error = isempty(history) ? Inf : history[end].estimated_error
        @info(
            "threaded status",
            phase="measuring",
            nrounds,
            samples=nrounds * nchains,
            blocks=size(F_blocks, 3),
            zeta,
            estimated_error,
        )
        last_report = time()
    end
end

println()
println("threaded single-v convergence")
println("v = ", v, ", T = ", temperature, ", L = ", L,
    ", chains = ", nchains)
println("rounds = ", nrounds, ", samples = ", nrounds * nchains)
if isempty(history)
    println("zeta = not yet identified")
    println("estimated error = Inf")
    println("converged = false")
else
    println("zeta = ", @sprintf("%.4f", history[end].zeta))
    println("estimated error = ", @sprintf("%.4f", history[end].estimated_error))
    println("converged = ", history[end].converged)
end
println("saved = ", args["output"])
end
