#!/usr/bin/env julia

# Local dt-convergence harness for the spin-aligned F correlator.
#
# For each anchor velocity it integrates the SAME Wiener path at dt and dt/2 using
# common-random-number (CRN) coupling: the fine (dt/2) solve records its Brownian
# path, and the coarse (dt) solve is driven by that same path via NoiseWrapper, which
# coarsens the fine increments onto the coarse grid. dt error is a local/intensive
# integration-accuracy question (see the PRD's "Solver and timestep" section), so
# convergence is judged directly on the correlator itself: the largest pointwise gap
# |F_coarse(r,t) - F_fine(r,t)| over the whole (r,t) grid. A ζ-based (least-squares
# collapse or trough-position) criterion was tried first and dropped — both require the
# correlator to have a well-resolved trough inside the box, which is a large-scale,
# L-sensitive question entirely separate from dt, and at small/calibration-scale L that
# requirement got contaminated by finite-size effects (the trough sitting near r_max)
# well before dt error was even in play. Comparing F(r,t) pointwise needs no trough, no
# fit, and no minimum box size to be meaningful. Convergence criterion: max |ΔF| < tol.

using ArgParse
using Dates
using JLD2
using Printf
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

function loginfo(message::String, fields::Pair...)
    pieces = ["$(first(field))=$(repr(last(field)))" for field in fields]
    println(stderr, Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"), " [info] ",
        message, isempty(pieces) ? "" : " " * join(pieces, " "))
    flush(stderr)
    flush(stdout)
    return nothing
end

function parse_args()
    settings = ArgParseSettings(
        description="Local CRN dt-convergence harness for the spin-aligned F correlator.",
    )
    @add_arg_table! settings begin
        "--L"
            arg_type = Int
            default = 40
        "--gamma"
            arg_type = Float64
            default = 1.0
        "--J"
            arg_type = Float64
            default = 2.0
        "--velocities"
            arg_type = String
            default = "0.1,1.0,10.0"
        "--dt"
            arg_type = Float64
            default = 0.001
        "--dr"
            arg_type = Float64
            default = 0.5
        "--r-max"
            arg_type = Float64
            default = 18.0
        "--burnin-time"
            arg_type = Float64
            default = 10.0
        "--T-max"
            arg_type = Float64
            default = 6.0
        "--ntimes"
            arg_type = Int
            default = 6
        "--nchunks"
            arg_type = Int
            default = 4
        "--solver"
            arg_type = String
            default = "SRA1"
        "--base-seed"
            arg_type = Int
            default = 1
        "--tol"
            arg_type = Float64
            default = 0.005
        "--output"
            arg_type = String
            default = ""
        "--csv"
            arg_type = String
            default = ""
    end
    return ArgParse.parse_args(settings)
end

function select_solver(name::String)
    name == "SRA1" && return SRA1()
    name == "SRA2" && return SRA2()
    name == "SRA3" && return SRA3()
    name == "SRIW1" && return SRIW1()
    name == "EM" && return EM()
    throw(ArgumentError("unknown solver: $name (choose SRA1, SRA2, SRA3, SRIW1, EM)"))
end

function steps_for_time(time::Real, dt::Real, name::String)
    steps = round(Int, time / dt)
    isapprox(steps * dt, time; atol=100eps(max(abs(time), abs(dt))), rtol=0) ||
        throw(ArgumentError("$name must be an integer multiple of dt"))
    return steps
end

function interpolated_spin_dot(cos_field, sin_field, L::Integer, px::Real, py::Real,
        ref_cos::Real, ref_sin::Real)
    x0 = floor(Int, px)
    y0 = floor(Int, py)
    fx = px - x0
    fy = py - y0
    ix0 = mod1(x0, L)
    ix1 = mod1(x0 + 1, L)
    iy0 = mod1(y0, L)
    iy1 = mod1(y0 + 1, L)

    c = 0.0
    s = 0.0
    @inbounds for (ix, wx) in ((ix0, 1 - fx), (ix1, fx))
        for (iy, wy) in ((iy0, 1 - fy), (iy1, fy))
            w = wx * wy
            idx = site_index(ix, iy, L)
            c += w * cos_field[idx]
            s += w * sin_field[idx]
        end
    end
    return ref_cos * c + ref_sin * s
end

function spin_aligned_f_correlator(window::AbstractVector, params::ModelParams,
        radii::AbstractVector)
    L = params.L
    nsites = L * L
    ntimes = length(window) ÷ 2
    mid = ntimes + 1
    cos_window = [cos.(state) for state in window]
    sin_window = [sin.(state) for state in window]
    cos_mid = cos_window[mid]
    sin_mid = sin_window[mid]
    F = zeros(Float64, length(radii), ntimes + 1)

    @inbounds for (ridx, r) in enumerate(radii)
        for lag in 0:ntimes
            cos_minus = cos_window[mid - lag]
            sin_minus = sin_window[mid - lag]
            cos_plus = cos_window[mid + lag]
            sin_plus = sin_window[mid + lag]
            accum = 0.0
            for y in 1:L, x in 1:L
                center = site_index(x, y, L)
                cx = cos_mid[center]
                sy = sin_mid[center]
                dx = r * cx
                dy = r * sy
                forward_plus = interpolated_spin_dot(cos_plus, sin_plus, L, x + dx, y + dy, cx, sy)
                backward_minus = interpolated_spin_dot(cos_minus, sin_minus, L, x - dx, y - dy, cx, sy)
                forward_minus = interpolated_spin_dot(cos_minus, sin_minus, L, x + dx, y + dy, cx, sy)
                backward_plus = interpolated_spin_dot(cos_plus, sin_plus, L, x - dx, y - dy, cx, sy)
                accum += 0.25 * (forward_plus + backward_minus - forward_minus - backward_plus)
            end
            F[ridx, lag + 1] = accum / nsites
        end
    end
    return F
end

# Chunk-averaged F(r, t): chunks are repeated windows within one anchor/dt run, purely to
# average down the correlator's own disorder noise. No stderr is needed here — CRN coupling
# already cancels the stochastic component between the coarse and fine solves directly, so
# the pointwise gap below reflects deterministic discretization bias, not sampling noise.
chunk_mean(chunks::AbstractVector{<:AbstractMatrix}) = sum(chunks) ./ length(chunks)

# Largest pointwise disagreement between the coarse and fine correlators, plus where it
# occurs, so a REFINE line points at which (r, t) drove it.
function max_pointwise_gap(F_coarse::AbstractMatrix, F_fine::AbstractMatrix,
        radii::AbstractVector, times::AbstractVector)
    diff = abs.(F_coarse .- F_fine)
    idx = argmax(diff)
    return diff[idx], radii[idx[1]], times[idx[2]]
end

function main()
    args = parse_args()
    L = args["L"]
    dt = args["dt"]
    dt_fine = dt / 2
    dr = args["dr"]
    ntimes = args["ntimes"]
    nchunks = args["nchunks"]
    tol = args["tol"]
    solver = select_solver(args["solver"])
    velocities = parse.(Float64, split(args["velocities"], ","))

    lag_steps = steps_for_time(args["T-max"] / ntimes, dt, "T-max / ntimes")
    schedule = lag_step_schedule(ntimes, lag_steps; spacing=:geometric)
    burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
    sample_times = [0; cumsum(schedule.advance_gaps)] .* dt
    radii = collect(dr:dr:min(args["r-max"], L / 2))
    times = schedule.cum_steps .* dt

    @printf("# CRN dt-convergence harness  L=%d  J=%.3g  gamma=%.3g  solver=%s\n",
        L, args["J"], args["gamma"], args["solver"])
    @printf("# dt=%.2e  dt/2=%.2e  ntimes=%d  nchunks=%d  tol=%.4f\n",
        dt, dt_fine, ntimes, nchunks, tol)
    @printf("# %-8s  %-10s  %-10s  %-10s  %s\n", "v", "max|dF|", "r@max", "t@max", "status")

    loginfo("starting dt-convergence harness", :L => L, :J => args["J"],
        :gamma => args["gamma"], :solver => args["solver"], :dt => dt, :dt_fine => dt_fine,
        :velocities => Tuple(velocities), :ntimes => ntimes, :nchunks => nchunks, :tol => tol)

    rows = NamedTuple{(:v, :dt, :dt_fine, :gap, :r_at_max, :t_at_max, :converged),
        Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Bool}}[]
    max_gap = 0.0
    for (vi, v) in enumerate(velocities)
        params = ModelParams(; L, Q=args["gamma"], J=args["J"], v)
        work = LatticeFlockingSDE.DriftWorkspace(params)
        rng = MersenneTwister(args["base-seed"] + vi - 1)
        theta = initial_angles(rng, L, :ordered)

        # Burn in at the fine resolution to reach a shared starting configuration.
        burn_prob = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!,
            theta, (0.0, burnin_steps * dt), work)
        theta = wrap_angles!(collect(solve(burn_prob, solver; dt=dt_fine, adaptive=false,
            save_everystep=false, save_start=false, rng).u[end]))

        chunks_fine = Matrix{Float64}[]
        chunks_coarse = Matrix{Float64}[]
        for chunk in 1:nchunks
            seed = args["base-seed"] + 1000 * (vi - 1) + chunk
            prob = SDEProblem(LatticeFlockingSDE.drift!, LatticeFlockingSDE.noise!,
                theta, (0.0, sample_times[end]), work)
            sol_fine = solve(prob, solver; dt=dt_fine, adaptive=false, saveat=sample_times,
                save_start=true, save_noise=true, seed=seed)
            # CRN: drive the coarse solve with the fine solve's Wiener path.
            sol_coarse = solve(remake(prob; noise=NoiseWrapper(sol_fine.W)), solver;
                dt=dt, adaptive=false, saveat=sample_times, save_start=true)

            window_fine = [wrap_angles!(collect(u)) for u in sol_fine.u]
            window_coarse = [wrap_angles!(collect(u)) for u in sol_coarse.u]
            push!(chunks_fine, spin_aligned_f_correlator(window_fine, params, radii))
            push!(chunks_coarse, spin_aligned_f_correlator(window_coarse, params, radii))
            theta = window_fine[end]

            # sol_fine.W holds the full per-step Wiener path (save_noise=true); at L=200,
            # dt=0.001 that's ~10 GB per chunk. Drop the references and force a collection
            # before the next chunk allocates, or peak RSS stacks across chunks and OOMs.
            sol_fine = nothing
            sol_coarse = nothing
            GC.gc()
        end

        F_mean_fine = chunk_mean(chunks_fine)
        F_mean_coarse = chunk_mean(chunks_coarse)
        gap, r_at_max, t_at_max = max_pointwise_gap(F_mean_coarse, F_mean_fine, radii, times)
        max_gap = max(max_gap, gap)
        converged = gap < tol
        push!(rows, (; v, dt, dt_fine, gap, r_at_max, t_at_max, converged))
        @printf("  %-8.3g  %-10.4f  %-10.4f  %-10.4f  %s\n",
            v, gap, r_at_max, t_at_max, converged ? "converged" : "REFINE")
        loginfo("dt-convergence anchor", :v => v, :dt => dt, :gap => gap,
            :r_at_max => r_at_max, :t_at_max => t_at_max, :converged => converged)
    end

    converged = max_gap < tol
    @printf("# max |dF| = %.4f  ->  %s (tol %.4f)\n",
        max_gap, converged ? "CONVERGED" : "NOT CONVERGED", tol)

    config = (; L, J=args["J"], gamma=args["gamma"], solver=args["solver"], dt, dt_fine,
        dr, ntimes, nchunks, tol, burnin_time=args["burnin-time"], T_max=args["T-max"],
        r_max=args["r-max"], base_seed=args["base-seed"])
    if !isempty(args["csv"])
        mkpath(dirname(abspath(args["csv"])))
        open(args["csv"], "w") do io
            println(io, "v,dt,dt_fine,gap,r_at_max,t_at_max,converged")
            for row in rows
                println(io, join((
                    @sprintf("%.12g", row.v), @sprintf("%.12g", row.dt),
                    @sprintf("%.12g", row.dt_fine), @sprintf("%.8f", row.gap),
                    @sprintf("%.6g", row.r_at_max), @sprintf("%.6g", row.t_at_max),
                    row.converged ? 1 : 0), ","))
            end
        end
    end
    if !isempty(args["output"])
        mkpath(dirname(abspath(args["output"])))
        jldsave(args["output"]; result=(; config, rows))
    end
    loginfo("finished dt-convergence harness", :max_gap => max_gap,
        :converged => converged, :csv => args["csv"], :output => args["output"])
    return converged
end

main()
