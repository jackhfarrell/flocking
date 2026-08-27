#!/usr/bin/env julia

# Choose the production T_max(v) window per anchor by the least-squares collapse
# objective (issue 08, output 2). Given a generous-T_max stage-2 measurement with
# geometric lags, for each velocity it scans contiguous prefixes of the positive lags
# [1:k], collapses each with the established coarse-then-fine (eta, zeta) scan, and picks
# the window where the objective is best AND zeta is stable under shrinking/extending it
# by one lag — no trough anchoring. Production T_max(v) is that trimmed window's end time.

using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))
include(joinpath(@__DIR__, "calibration_schedule.jl"))

const RADIUS_MAX = 40.0
const POLY_ORDER = 3
const COLLAPSE_BINS = 60
const MIN_WINDOW = 3          # fewest positive lags a power-law fit is meaningful over
const ZETA_STABILITY_TOL = 0.01

function collect_sample_files(input_dir::String)
    isdir(input_dir) || error("missing input directory: $input_dir")
    files = String[]
    for name in readdir(input_dir)
        startswith(name, "sample_") && endswith(name, ".jld2") || continue
        push!(files, joinpath(input_dir, name))
    end
    sort!(files)
    isempty(files) && error("no sample_*.jld2 files found in $input_dir")
    return files
end

function load_streaming_stats(files)
    first_result = load(files[1], "result")
    v_values = first_result.v_values
    radii = first_result.radii
    times = first_result.times
    sum_F = copy(first_result.F)
    sumsq_F = first_result.F .^ 2

    for file in files[2:end]
        result = load(file, "result")
        result.v_values == v_values || error("v_values mismatch in $file")
        result.radii == radii || error("radii mismatch in $file")
        result.times == times || error("times mismatch in $file")
        sum_F .+= result.F
        sumsq_F .+= result.F .^ 2
    end

    n = length(files)
    mean_F = sum_F ./ n
    variance_F = max.((sumsq_F .- n .* mean_F .^ 2) ./ (n - 1), 0.0)
    stderr_F = sqrt.(variance_F ./ n)
    return (; nsamples=n, config=first_result.config, v_values, radii, times,
        mean_F, stderr_F)
end

# Best-fit (zeta, objective) for one (v, time window) via the coarse-then-fine scan the
# rest of the pipeline uses. time_indices is the contiguous prefix of positive lags.
function best_collapse(data, vidx, radius_mask, time_indices)
    mean_field = data.mean_F[vidx, :, :]
    stderr_field = data.stderr_F[vidx, :, :]
    coarse_eta = collect(-0.2:0.02:1.6)
    coarse_zeta = collect(0.0:0.02:0.8)
    _, coarse_best = scan_grid(data.radii, data.times, mean_field, stderr_field,
        radius_mask, time_indices, coarse_eta, coarse_zeta, POLY_ORDER, COLLAPSE_BINS)
    fine_eta = collect((coarse_best.eta - 0.06):0.005:(coarse_best.eta + 0.06))
    fine_zeta = collect((coarse_best.zeta - 0.06):0.005:(coarse_best.zeta + 0.06))
    _, fine_best = scan_grid(data.radii, data.times, mean_field, stderr_field,
        radius_mask, time_indices, fine_eta, fine_zeta, POLY_ORDER, COLLAPSE_BINS)
    return (; zeta=fine_best.zeta, eta=fine_best.eta, objective=fine_best.reduced_chi2)
end

function window_scan_for_v(data, vidx, radius_mask, base_indices)
    candidates = NamedTuple{(:window_end, :ntimes, :T_max, :zeta, :eta, :objective),
        Tuple{Int,Int,Float64,Float64,Float64,Float64}}[]
    for k in MIN_WINDOW:length(base_indices)
        indices = base_indices[1:k]
        fit = best_collapse(data, vidx, radius_mask, indices)
        push!(candidates, (; window_end=k, ntimes=k, T_max=data.times[indices[end]],
            zeta=fit.zeta, eta=fit.eta, objective=fit.objective))
    end
    choice = select_stable_window(candidates; zeta_tol=ZETA_STABILITY_TOL)
    return (; v=data.v_values[vidx], candidates, selected=choice.selected,
        stable=choice.stable)
end

function write_csv(path, results)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "v,log10_v,T_max,ntimes,window_end,zeta,eta,objective,stable")
        for r in results
            s = r.selected
            println(io, join((
                @sprintf("%.12g", r.v), @sprintf("%.12g", log10(r.v)),
                @sprintf("%.8f", s.T_max), s.ntimes, s.window_end,
                @sprintf("%.8f", s.zeta), @sprintf("%.8f", s.eta),
                @sprintf("%.8f", s.objective), r.stable ? 1 : 0), ","))
        end
    end
end

function main()
    input_dir = length(ARGS) >= 1 ? ARGS[1] : error("usage: analyze_tmax_window.jl <input-dir> [csv] [archive]")
    csv_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(input_dir, "tmax_window.csv")
    archive_path = length(ARGS) >= 3 ? ARGS[3] : joinpath(input_dir, "tmax_window.jld2")

    files = collect_sample_files(input_dir)
    data = load_streaming_stats(files)
    radius_mask = data.radii .<= RADIUS_MAX
    base_indices = findall(t -> t > 0, data.times)
    length(base_indices) >= MIN_WINDOW ||
        error("need at least $MIN_WINDOW positive lags; measurement has $(length(base_indices))")

    results = [window_scan_for_v(data, vidx, radius_mask, base_indices)
        for vidx in eachindex(data.v_values)]

    @printf("# T_max window selection  samples=%d  radius<=%.0f\n", data.nsamples, RADIUS_MAX)
    @printf("# %-8s  %-8s  %-6s  %-8s  %-10s  %s\n",
        "v", "T_max", "ntimes", "zeta", "objective", "stable")
    for r in results
        s = r.selected
        @printf("  %-8.3g  %-8.3g  %-6d  %-8.4f  %-10.4f  %s\n",
            r.v, s.T_max, s.ntimes, s.zeta, s.objective, r.stable ? "yes" : "no")
    end

    write_csv(csv_path, results)
    mkpath(dirname(abspath(archive_path)))
    jldsave(archive_path; data, results)
    println("csv: ", csv_path)
    println("archive: ", archive_path)
    return results
end

main()
