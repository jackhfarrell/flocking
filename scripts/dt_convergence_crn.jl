#!/usr/bin/env julia

# Local dt-convergence harness for the spin-aligned F correlator.
#
# For each anchor velocity it integrates the SAME Wiener path at dt and dt/2 using
# common-random-number (CRN) coupling: the fine (dt/2) solve records its Brownian
# path, and the coarse (dt) solve is driven by that same path via NoiseWrapper, which
# coarsens the fine increments onto the coarse grid. The dt and dt/2 correlator
# windows are collapsed independently and the harness reports |ζ(dt) − ζ(dt/2)|.
# Because the two solves share noise, the statistical fluctuation cancels and the
# reported gap isolates the time-discretization error. Convergence criterion:
# |Δζ| < ~0.005 (about a third of the statistical band).

using ArgParse
using Dates
using JLD2
using Printf
using Random
using StochasticDiffEq

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

# Reuses the production least-squares data-collapse fit (scan_grid/evaluate_collapse) as
# the zeta estimator. A simple trough-position fit was tried first and dropped: it isn't
# robust (see issue 01, fit-window-robustness), and at v=10 in particular it locks onto
# whatever bin is the argmin even when that's a boundary/noise artifact rather than the
# real scaling front. The collapse objective's reduced_chi2 gives an honest goodness-of-fit
# instead of silently trusting a single trough per lag.
include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const POLY_ORDER = 3

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

# Mean and standard error across chunks, the same sum/sumsq -> stderr construction
# check_v1_speedup.jl's load_streaming_stats uses across independent sample files.
function chunk_mean_stderr(chunks::AbstractVector{<:AbstractMatrix})
    n = length(chunks)
    sum_F = sum(chunks)
    sumsq_F = sum(F .^ 2 for F in chunks)
    mean_F = sum_F ./ n
    variance_F = max.((sumsq_F .- n .* mean_F .^ 2) ./ (n - 1), 0.0)
    return mean_F, sqrt.(variance_F ./ n)
end

# Raw trough radius at each lag, with no power-law fit through it — purely a boundary-
# contamination check, so it must survive exactly the noisy/near-zero traces that make a
# trough-amplitude fit (spin_aligned_f_analysis.jl's feature_estimate) throw on log of a
# slightly-negative amplitude.
function max_trough_radius(F_mean::AbstractMatrix, radii::AbstractVector,
        time_indices::AbstractVector{Int})
    return maximum(radii[argmin(@view F_mean[:, tidx])] for tidx in time_indices)
end

# Fit zeta by the production least-squares data-collapse (scan_grid/evaluate_collapse),
# coarse-then-fine as in check_v1_speedup.jl, rather than a trough-position power law: the
# latter isn't robust (issue 01) and at v=10 can lock onto a boundary/noise artifact instead
# of the real scaling front. Returns zeta, its reduced_chi2, and the trough radius at the
# largest lag (a boundary-contamination check: if it sits near r_max, the front ran out of
# room and even the collapse fit is being asked to fit a finite-size artifact).
function collapse_zeta(radii::AbstractVector, times::AbstractVector, F_mean, F_stderr,
        eta_values::AbstractVector, zeta_values::AbstractVector, nbins::Integer)
    radius_mask = trues(length(radii))
    time_indices = findall(>(0), times)
    _, coarse_best = scan_grid(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        eta_values, zeta_values, POLY_ORDER, nbins)
    eta_step = eta_values[2] - eta_values[1]
    zeta_step = zeta_values[2] - zeta_values[1]
    fine_eta = collect((coarse_best.eta - 3eta_step):(eta_step / 4):(coarse_best.eta + 3eta_step))
    fine_zeta = collect((coarse_best.zeta - 3zeta_step):(zeta_step / 4):(coarse_best.zeta + 3zeta_step))
    _, fine_best = scan_grid(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        fine_eta, fine_zeta, POLY_ORDER, nbins)
    return fine_best.zeta, fine_best.reduced_chi2, max_trough_radius(F_mean, radii, time_indices)
end

function main()
    args = parse_args()
    L = args["L"]
    dt = args["dt"]
    dt_fine = dt / 2
    dr = args["dr"]
    ntimes = args["ntimes"]
    nchunks = args["nchunks"]
    nchunks >= 2 || throw(ArgumentError("--nchunks must be at least 2 (chunks supply the collapse fit's per-point stderr)"))
    tol = args["tol"]
    solver = select_solver(args["solver"])
    velocities = parse.(Float64, split(args["velocities"], ","))

    lag_steps = steps_for_time(args["T-max"] / ntimes, dt, "T-max / ntimes")
    schedule = lag_step_schedule(ntimes, lag_steps; spacing=:geometric)
    burnin_steps = steps_for_time(args["burnin-time"], dt, "burnin-time")
    sample_times = [0; cumsum(schedule.advance_gaps)] .* dt
    radii = collect(dr:dr:min(args["r-max"], L / 2))
    times = schedule.cum_steps .* dt
    nbins = clamp(length(radii) ÷ 3, 5, 60)
    coarse_eta = collect(-0.2:0.02:1.6)
    coarse_zeta = collect(-0.2:0.02:1.2)

    @printf("# CRN dt-convergence harness  L=%d  J=%.3g  gamma=%.3g  solver=%s\n",
        L, args["J"], args["gamma"], args["solver"])
    @printf("# dt=%.2e  dt/2=%.2e  ntimes=%d  nchunks=%d  tol=%.4f\n",
        dt, dt_fine, ntimes, nchunks, tol)
    @printf("# %-8s  %-10s  %-10s  %-10s  %s\n", "v", "zeta(dt)", "zeta(dt/2)", "|dzeta|", "status")

    loginfo("starting dt-convergence harness", :L => L, :J => args["J"],
        :gamma => args["gamma"], :solver => args["solver"], :dt => dt, :dt_fine => dt_fine,
        :velocities => Tuple(velocities), :ntimes => ntimes, :nchunks => nchunks, :tol => tol)

    rows = NamedTuple{(:v, :dt, :dt_fine, :zeta_coarse, :zeta_fine, :dzeta, :converged,
        :chi2_coarse, :chi2_fine),
        Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Bool,Float64,Float64}}[]
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

        F_mean_fine, F_stderr_fine = chunk_mean_stderr(chunks_fine)
        F_mean_coarse, F_stderr_coarse = chunk_mean_stderr(chunks_coarse)
        zeta_fine, chi2_fine, r_last_fine = collapse_zeta(radii, times, F_mean_fine,
            F_stderr_fine, coarse_eta, coarse_zeta, nbins)
        zeta_coarse, chi2_coarse, r_last_coarse = collapse_zeta(radii, times, F_mean_coarse,
            F_stderr_coarse, coarse_eta, coarse_zeta, nbins)
        gap = abs(zeta_coarse - zeta_fine)
        max_gap = max(max_gap, gap)
        converged = gap < tol
        push!(rows, (; v, dt, dt_fine, zeta_coarse, zeta_fine, dzeta=gap, converged,
            chi2_coarse, chi2_fine))
        @printf("  %-8.3g  %-10.4f  %-10.4f  %-10.4f  %s\n",
            v, zeta_coarse, zeta_fine, gap, converged ? "converged" : "REFINE")
        loginfo("dt-convergence anchor", :v => v, :dt => dt, :zeta_coarse => zeta_coarse,
            :zeta_fine => zeta_fine, :dzeta => gap, :converged => converged,
            :chi2_coarse => chi2_coarse, :chi2_fine => chi2_fine)
        # Boundary-contamination check: if the trough at the largest lag sits near the
        # capped radius, the front ran out of room and the collapse fit is chasing a
        # finite-size artifact rather than the real scaling front.
        r_cap = radii[end]
        near_boundary = max(r_last_fine, r_last_coarse) > 0.8 * r_cap
        loginfo("trough boundary check", :v => v, :r_last_fine => r_last_fine,
            :r_last_coarse => r_last_coarse, :r_cap => r_cap, :L_half => L / 2,
            :near_boundary => near_boundary)
    end

    converged = max_gap < tol
    @printf("# max |dzeta| = %.4f  ->  %s (tol %.4f)\n",
        max_gap, converged ? "CONVERGED" : "NOT CONVERGED", tol)

    config = (; L, J=args["J"], gamma=args["gamma"], solver=args["solver"], dt, dt_fine,
        dr, ntimes, nchunks, tol, burnin_time=args["burnin-time"], T_max=args["T-max"],
        r_max=args["r-max"], base_seed=args["base-seed"])
    if !isempty(args["csv"])
        mkpath(dirname(abspath(args["csv"])))
        open(args["csv"], "w") do io
            println(io, "v,dt,dt_fine,zeta_coarse,zeta_fine,dzeta,converged,chi2_coarse,chi2_fine")
            for row in rows
                println(io, join((
                    @sprintf("%.12g", row.v), @sprintf("%.12g", row.dt),
                    @sprintf("%.12g", row.dt_fine), @sprintf("%.8f", row.zeta_coarse),
                    @sprintf("%.8f", row.zeta_fine), @sprintf("%.8f", row.dzeta),
                    row.converged ? 1 : 0, @sprintf("%.6g", row.chi2_coarse),
                    @sprintf("%.6g", row.chi2_fine)), ","))
            end
        end
    end
    if !isempty(args["output"])
        mkpath(dirname(abspath(args["output"])))
        jldsave(args["output"]; result=(; config, rows))
    end
    loginfo("finished dt-convergence harness", :max_dzeta => max_gap,
        :converged => converged, :csv => args["csv"], :output => args["output"])
    return converged
end

main()
