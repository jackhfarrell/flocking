#!/usr/bin/env julia

# Follow one continuing trajectory and stop when temporal reblocking and fit-window
# checks agree that the collapse exponent has settled.

using ArgParse
using JLD2
using Printf
using Random
using Statistics
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(
    description="Single-chain convergence with adaptive reblocking.")
@add_arg_table! settings begin
    "--v"
        arg_type = Float64
        default = 10.0
    "--temperature"
        arg_type = Float64
        default = 0.8
    "--L"
        arg_type = Int
        default = 256
    "--Q"
        arg_type = Float64
        default = 1.0
    "--dt"
        arg_type = Float64
        default = 0.0
    "--tolerance"
        arg_type = Float64
        default = 0.01
    "--fit-step"
        arg_type = Float64
        default = 0.005
    "--zeta-min"
        arg_type = Float64
        default = 0.2
    "--zeta-max"
        arg_type = Float64
        default = 0.65
    "--maximum-blocks"
        arg_type = Int
        default = 512
    "--windows-per-block"
        arg_type = Int
        default = 2
    "--minimum-blocks"
        arg_type = Int
        default = 32
    "--minimum-reblock-groups"
        arg_type = Int
        default = 8
    "--reblock-plateau-fraction"
        arg_type = Float64
        default = 0.25
    "--stable-checks"
        arg_type = Int
        default = 3
    "--check-every"
        arg_type = Int
        default = 4
    "--dr"
        arg_type = Float64
        default = 1.0
    "--r-max"
        arg_type = Float64
        default = 96.0
    "--fit-rmax"
        arg_type = Float64
        default = 80.0
    "--maximum-edge-ratio"
        arg_type = Float64
        default = 0.25
    "--T-max"
        arg_type = Float64
        default = 8.0
    "--ntimes"
        arg_type = Int
        default = 8
    "--reference-points"
        arg_type = Int
        default = 5
    "--spatial-samples"
        arg_type = Int
        default = 8192
    "--equilibrium-block-steps"
        arg_type = Int
        default = 1024
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
    "--maximum-wall-seconds"
        arg_type = Float64
        default = 0.0
    "--seed"
        arg_type = Int
        default = 2_810_010
    "--output"
        arg_type = String
        default = ""
end
args = ArgParse.parse_args(settings)

temperature = args["temperature"]
J = inv(temperature)
L = args["L"]
Q = args["Q"]
v = args["v"]
target_dt = min(2.0^-9, 0.01 / max(abs(v), 1.0))
dt = args["dt"] > 0 ? args["dt"] : 2.0^floor(log2(target_dt))
if isempty(args["output"])
    temperature_label = replace(@sprintf("%.6g", temperature), "." => "p")
    velocity_label = replace(@sprintf("%.6g", v), "." => "p")
    args["output"] = joinpath(
        "results", "single_v_reblocked_L$(L)_T$(temperature_label)_v$(velocity_label).jld2")
end
params = ModelParams(; L, Q, J, v)
workspace = LatticeFlockingSDE.DriftWorkspace(params)
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
    temperature, J, L, Q, v, dt,
    windows_per_block=args["windows-per-block"],
    dr=args["dr"], r_max=args["r-max"], fit_rmax=args["fit-rmax"],
    T_max=args["T-max"], ntimes=args["ntimes"],
    reference_points=args["reference-points"],
    spatial_samples=min(args["spatial-samples"], L^2),
    equilibrium_block_steps=args["equilibrium-block-steps"],
    equilibrium_window,
    equilibrium_stable_checks=args["equilibrium-stable-checks"],
    energy_threshold=args["energy-threshold"],
    magnetization_threshold=args["magnetization-threshold"],
    grid_points=args["grid-points"], fit_step=args["fit-step"],
    zeta_min=args["zeta-min"], zeta_max=args["zeta-max"], seed=args["seed"],
    solver=string(typeof(solver)),
)

rng = MersenneTwister(args["seed"])
measurement_rng = MersenneTwister(args["seed"] + 1)
theta = initial_angles(rng, L, :ordered)
equilibrium_energies = Float64[]
equilibrium_magnetizations = Float64[]
equilibrium_stable_count = 0
equilibrium_reached = false
block_sum = zeros(Float64, length(radii), length(times))
block_count = 0
nwindows = 0
F_blocks = Array{Float64}(undef, length(radii), length(times), 0)
estimates = Any[]
history = Any[]
analysis_settings_changed = false

if isfile(args["output"])
    saved = load(args["output"], "run")
    shared_settings = intersect(keys(saved.config), keys(config))
    analysis_settings = (:fit_rmax, :reference_points, :grid_points,
        :fit_step, :zeta_min, :zeta_max)
    simulation_settings = filter(name -> name ∉ analysis_settings, shared_settings)
    all(getproperty(saved.config, name) == getproperty(config, name)
        for name in simulation_settings) ||
        error("the saved run has different simulation settings")
    analysis_settings_changed = any(!hasproperty(saved.config, name) ||
        getproperty(saved.config, name) != getproperty(config, name)
        for name in analysis_settings)
    rng = saved.rng
    measurement_rng = saved.measurement_rng
    theta = saved.theta
    equilibrium_energies = saved.equilibrium_energies
    equilibrium_magnetizations = saved.equilibrium_magnetizations
    equilibrium_stable_count = saved.equilibrium_stable_count
    equilibrium_reached = saved.equilibrium_reached
    block_sum = saved.block_sum
    block_count = saved.block_count
    nwindows = saved.nwindows
    F_blocks = saved.F_blocks
    estimates = hasproperty(saved, :estimates) ? Any[saved.estimates...] : Any[]
    history = Any[saved.history...]
    @info "resumed single-chain run" nwindows blocks=size(F_blocks, 3)
    analysis_settings_changed && @info "analysis settings changed; recomputing convergence"
    flush(stderr)
end

mkpath(dirname(args["output"]))
last_report = time()
wall_deadline = args["maximum-wall-seconds"] > 0 ?
    time() + args["maximum-wall-seconds"] : Inf

let rng=rng, theta=theta,
        measurement_rng=measurement_rng,
        equilibrium_energies=equilibrium_energies,
        equilibrium_magnetizations=equilibrium_magnetizations,
        equilibrium_stable_count=equilibrium_stable_count,
        equilibrium_reached=equilibrium_reached,
        block_sum=block_sum, block_count=block_count,
        nwindows=nwindows, F_blocks=F_blocks, estimates=estimates, history=history,
        last_report=last_report, wall_deadline=wall_deadline

if !isfile(args["output"])
    run = (;
        config, rng, measurement_rng, theta,
        equilibrium_energies, equilibrium_magnetizations,
        equilibrium_stable_count, equilibrium_reached, block_sum, block_count,
        nwindows, F_blocks, estimates, history,
    )
    temporary_output = args["output"] * ".tmp"
    jldsave(temporary_output; run)
    mv(temporary_output, args["output"]; force=true)
end

while !equilibrium_reached &&
        length(equilibrium_energies) < args["maximum-equilibrium-blocks"] &&
        time() < wall_deadline
    theta = LatticeFlockingSDE.advance_fixed(
        theta, args["equilibrium-block-steps"], dt, workspace, solver, rng)
    push!(equilibrium_energies, xy_energy(theta, params) / L^2)
    push!(equilibrium_magnetizations, magnetization(theta))

    energy_delta = Inf
    magnetization_delta = Inf
    if length(equilibrium_energies) >= 2equilibrium_window
        last_block = length(equilibrium_energies)
        previous = (last_block - 2equilibrium_window + 1):(last_block - equilibrium_window)
        current = (last_block - equilibrium_window + 1):last_block
        energy_delta = abs(mean(@view equilibrium_energies[previous]) -
            mean(@view equilibrium_energies[current]))
        magnetization_delta = abs(mean(@view equilibrium_magnetizations[previous]) -
            mean(@view equilibrium_magnetizations[current]))
        plateau = energy_delta <= args["energy-threshold"] &&
            magnetization_delta <= args["magnetization-threshold"]
        equilibrium_stable_count = plateau ? equilibrium_stable_count + 1 : 0
        equilibrium_reached = equilibrium_stable_count >= args["equilibrium-stable-checks"]
    end

    run = (;
        config, rng, measurement_rng, theta,
        equilibrium_energies, equilibrium_magnetizations,
        equilibrium_stable_count, equilibrium_reached, block_sum, block_count,
        nwindows, F_blocks, estimates, history,
    )
    temporary_output = args["output"] * ".tmp"
    jldsave(temporary_output; run)
    mv(temporary_output, args["output"]; force=true)

    if time() - last_report >= args["report-seconds"] || equilibrium_reached
        @info(
            "single-chain status",
            phase="equilibrating",
            blocks=length(equilibrium_energies),
            energy_delta,
            magnetization_delta,
        )
        flush(stderr)
        last_report = time()
    end
end

if !equilibrium_reached
    if time() >= wall_deadline
        @info "wall-time budget reached during equilibration" output=args["output"]
        flush(stderr)
        println("saved = ", args["output"])
        flush(stdout)
        exit()
    end
    error("equilibration did not converge before the block limit")
end

converged = false
if !isempty(history) && !analysis_settings_changed
    latest = history[end]
    recent_zetas = [row.zeta for row in history[
        max(1, end - args["stable-checks"] + 1):end]]
    stability_error = length(recent_zetas) < args["stable-checks"] ? Inf :
        (maximum(recent_zetas) - minimum(recent_zetas)) / 2
    estimated_error = maximum((
        latest.statistical_error, latest.quarter_halfspread,
        latest.time_halfspread, latest.radius_halfspread, stability_error,
    ))
    converged = latest.nblocks >= args["minimum-blocks"] &&
        length(latest.reblock_errors) >= 3 && latest.reblock_plateau &&
        !latest.profile_at_boundary && !latest.fit_at_boundary &&
        estimated_error <= args["tolerance"]
end

while !converged && size(F_blocks, 3) < args["maximum-blocks"] &&
        time() < wall_deadline
    window = sample_fixed_window(
        theta, schedule.advance_gaps, dt, workspace, solver, rng)
    theta = copy(window[end])
    spatial_samples = min(args["spatial-samples"], L^2)
    origins = spatial_samples <= 0 || spatial_samples == L^2 ? nothing :
        [(mod1(index, L), fld(index - 1, L) + 1) for
            index in @view(randperm(measurement_rng, L^2)[1:spatial_samples])]
    block_sum .+= LatticeFlockingSDE.spin_aligned_correlators(
        window, L, radii; origins)
    block_count += 1
    nwindows += 1

    if block_count == args["windows-per-block"]
        F_blocks = cat(F_blocks, block_sum ./ block_count; dims=3)
        fill!(block_sum, 0.0)
        block_count = 0
    end

    nblocks = size(F_blocks, 3)
    if block_count == 0
        mean_field = dropdims(mean(F_blocks; dims=3), dims=3)
        positive = findall(>(0), times)
        reference_count = min(args["reference-points"], length(positive))
        reference_indices = positive[(end - reference_count + 1):end]
        reference = best_common_grid_collapse(
            radii, times, mean_field, reference_indices;
            rmax=args["fit-rmax"], grid_points=args["grid-points"],
            zeta_min=args["zeta-min"], zeta_max=args["zeta-max"],
            fine_zeta_step=args["fit-step"])
        fit_at_boundary = reference.best.zeta <=
            args["zeta-min"] + args["fit-step"] / 2 ||
            reference.best.zeta >= args["zeta-max"] - args["fit-step"] / 2
        edge_points = min(5, length(radii))
        edge_ratios = [
            maximum(abs, view(mean_field, (length(radii) - edge_points + 1):length(radii),
                index)) /
            max(maximum(abs, view(mean_field, :, index)), eps(Float64))
            for index in reference_indices
        ]
        maximum_edge_ratio = maximum(edge_ratios)
        profile_at_boundary = maximum_edge_ratio > args["maximum-edge-ratio"]

        estimate = (;
            nwindows, nblocks,
            measurement_time=2nwindows * last(times),
            eta=reference.best.eta,
            zeta=reference.best.zeta,
            collapse_score=reference.best.score,
            maximum_edge_ratio,
            profile_at_boundary,
        )
        push!(estimates, estimate)
        @info "single-chain estimate" estimate
        flush(stderr)

        trace_csv = replace(args["output"], r"\.jld2$" => "_trace.csv")
        open(trace_csv, "w") do io
            columns = keys(estimates[end])
            println(io, join(columns, ","))
            for row in estimates
                println(io, join((getproperty(row, column) for column in columns), ","))
            end
        end
    end

    if block_count == 0 && nblocks >= 4 && nblocks % args["check-every"] == 0

        time_counts = unique(clamp.((reference_count - 1):reference_count + 1,
            3, length(positive)))
        time_zetas = [
            best_common_grid_collapse(
                radii, times, mean_field, positive[(end - count + 1):end];
                rmax=args["fit-rmax"], grid_points=args["grid-points"],
                zeta_min=args["zeta-min"], zeta_max=args["zeta-max"],
                fine_zeta_step=args["fit-step"]).best.zeta
            for count in time_counts
        ]
        radius_maxima = unique(min.((
            0.75args["fit-rmax"], args["fit-rmax"], 1.2args["fit-rmax"]),
            last(radii)))
        radius_zetas = [
            best_common_grid_collapse(
                radii, times, mean_field, reference_indices;
                rmax, grid_points=args["grid-points"],
                zeta_min=args["zeta-min"], zeta_max=args["zeta-max"],
                fine_zeta_step=args["fit-step"]).best.zeta
            for rmax in radius_maxima
        ]

        quarter_zetas = [
            best_common_grid_collapse(
                radii, times,
                dropdims(mean(@view(F_blocks[:, :,
                    (fld((quarter - 1) * nblocks, 4) + 1):fld(quarter * nblocks, 4)]);
                    dims=3), dims=3),
                reference_indices;
                rmax=args["fit-rmax"], grid_points=args["grid-points"],
                zeta_min=args["zeta-min"], zeta_max=args["zeta-max"],
                fine_zeta_step=args["fit-step"]).best.zeta
            for quarter in 1:4
        ]
        quarter_halfspread = (maximum(quarter_zetas) - minimum(quarter_zetas)) / 2

        reblock_sizes = Int[]
        reblock_errors = Float64[]
        superblock_size = 1
        while nblocks ÷ superblock_size >= args["minimum-reblock-groups"]
            ngroups = nblocks ÷ superblock_size
            used_blocks = ngroups * superblock_size
            total_sum = dropdims(sum(
                @view(F_blocks[:, :, 1:used_blocks]); dims=3), dims=3)
            jackknife_zetas = Float64[]
            for group in 1:ngroups
                first_omitted = (group - 1) * superblock_size + 1
                last_omitted = group * superblock_size
                omitted_sum = dropdims(sum(
                    @view(F_blocks[:, :, first_omitted:last_omitted]); dims=3), dims=3)
                field = (total_sum .- omitted_sum) ./ (used_blocks - superblock_size)
                eta_values = collect(
                    (reference.best.eta - 0.06):0.01:(reference.best.eta + 0.06))
                zeta_values = collect(
                    max(args["zeta-min"], reference.best.zeta - 0.06):
                    args["fit-step"]:
                    min(args["zeta-max"], reference.best.zeta + 0.06))
                fit = LatticeFlockingSDE.scan_common_grid_collapse(
                    radii, times, field, reference_indices, eta_values, zeta_values;
                    rmax=args["fit-rmax"], grid_points=args["grid-points"]).best
                at_edge = abs(fit.eta - reference.best.eta) >= 0.059 ||
                    abs(fit.zeta - reference.best.zeta) >= 0.059
                if at_edge
                    fit = best_common_grid_collapse(
                        radii, times, field, reference_indices;
                        rmax=args["fit-rmax"], grid_points=args["grid-points"],
                        zeta_min=args["zeta-min"], zeta_max=args["zeta-max"],
                        fine_zeta_step=args["fit-step"]).best
                end
                push!(jackknife_zetas, fit.zeta)
            end
            jackknife_mean = mean(jackknife_zetas)
            push!(reblock_sizes, superblock_size)
            push!(reblock_errors, sqrt((ngroups - 1) / ngroups *
                sum(abs2(zeta - jackknife_mean) for zeta in jackknife_zetas)))
            superblock_size *= 2
        end

        enough_reblock_levels = length(reblock_errors) >= 3
        reblock_plateau = enough_reblock_levels &&
            maximum(reblock_errors[(end - 2):end]) -
            minimum(reblock_errors[(end - 2):end]) <=
            args["reblock-plateau-fraction"] *
            maximum(reblock_errors[(end - 2):end])
        statistical_error = isempty(reblock_errors) ? Inf : maximum(reblock_errors)
        time_halfspread = (maximum(time_zetas) - minimum(time_zetas)) / 2
        radius_halfspread = (maximum(radius_zetas) - minimum(radius_zetas)) / 2
        recent_zetas = [row.zeta for row in history[
            max(1, end - args["stable-checks"] + 2):end]]
        push!(recent_zetas, reference.best.zeta)
        stability_error = length(recent_zetas) < args["stable-checks"] ? Inf :
            (maximum(recent_zetas) - minimum(recent_zetas)) / 2
        estimated_error = fit_at_boundary || profile_at_boundary ? Inf : maximum((
            statistical_error, quarter_halfspread, time_halfspread,
            radius_halfspread, stability_error,
        ))
        converged = nblocks >= args["minimum-blocks"] &&
            enough_reblock_levels && reblock_plateau &&
            estimated_error <= args["tolerance"]

        row = (;
            checkpoint=length(history) + 1, nwindows, nblocks,
            measurement_time=2nwindows * last(times),
            eta=reference.best.eta, zeta=reference.best.zeta,
            estimated_error, statistical_error,
            largest_reblock_size=isempty(reblock_sizes) ? 0 : reblock_sizes[end],
            largest_reblock_error=isempty(reblock_errors) ? Inf : reblock_errors[end],
            reblock_sizes, reblock_errors, reblock_plateau,
            quarter_halfspread, quarter_zeta_min=minimum(quarter_zetas),
            quarter_zeta_max=maximum(quarter_zetas),
            time_halfspread, radius_halfspread, stability_error,
            maximum_edge_ratio, profile_at_boundary, fit_at_boundary, converged,
        )
        push!(history, row)
        @info "single-chain checkpoint" row
        flush(stderr)
    end

    run = (;
        config, rng, measurement_rng, theta,
        equilibrium_energies, equilibrium_magnetizations,
        equilibrium_stable_count, equilibrium_reached, block_sum, block_count,
        nwindows, F_blocks, estimates, history,
    )
    temporary_output = args["output"] * ".tmp"
    jldsave(temporary_output; run)
    mv(temporary_output, args["output"]; force=true)

    if !isempty(history)
        csv = replace(args["output"], r"\.jld2$" => ".csv")
        open(csv, "w") do io
            columns = (
                :checkpoint, :nwindows, :nblocks, :measurement_time,
                :eta, :zeta, :estimated_error,
                :statistical_error, :largest_reblock_size, :largest_reblock_error,
                :reblock_plateau, :quarter_halfspread, :quarter_zeta_min,
                :quarter_zeta_max, :time_halfspread, :radius_halfspread,
                :stability_error, :maximum_edge_ratio, :profile_at_boundary,
                :fit_at_boundary, :converged,
            )
            println(io, join(columns, ","))
            for row in history
                println(io, join((getproperty(row, column) for column in columns), ","))
            end
        end
    end

    if time() - last_report >= args["report-seconds"]
        zeta = isempty(history) ? NaN : history[end].zeta
        estimated_error = isempty(history) ? Inf : history[end].estimated_error
        @info(
            "single-chain status",
            phase="measuring",
            nwindows,
            blocks=nblocks,
            zeta,
            estimated_error,
        )
        flush(stderr)
        last_report = time()
    end
end

println()
println("single-chain reblocked convergence")
println("v = ", v, ", T = ", temperature, ", L = ", L)
println("windows = ", nwindows, ", blocks = ", size(F_blocks, 3))
if isempty(history)
    println("zeta = not yet identified")
    println("estimated error = Inf")
    println("converged = false")
else
    println("zeta = ", @sprintf("%.4f", history[end].zeta))
    println("estimated error = ", @sprintf("%.4f", history[end].estimated_error))
    println("converged = ", converged)
end
println("wall-time budget reached = ", time() >= wall_deadline)
println("saved = ", args["output"])
flush(stdout)
end
