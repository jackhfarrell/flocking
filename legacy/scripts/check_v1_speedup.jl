#!/usr/bin/env julia

# Speedup validation (issue 08): confirm the capped-radii + SRA solver path reproduces
# the established v = 1 collapse exponent (zeta ~ 0.38, band +/- 0.015). Isolates that the
# speedups changed cost, not results. Reads a v = 1 stage-2 measurement, runs the same
# coarse-then-fine collapse scan on the reference window (r <= 40, all positive lags), and
# reports PASS/FAIL against the band. Exits nonzero on FAIL so a driver can gate on it.

using JLD2
using Printf
using Statistics

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const RADIUS_MAX = 40.0
const POLY_ORDER = 3
const COLLAPSE_BINS = 60
const TARGET_ZETA = 0.38
const BAND = 0.015

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
    return (; nsamples=n, v_values, radii, times, mean_F, stderr_F)
end

function reference_zeta(data, vidx)
    mean_field = data.mean_F[vidx, :, :]
    stderr_field = data.stderr_F[vidx, :, :]
    radius_mask = data.radii .<= RADIUS_MAX
    time_indices = findall(t -> t > 0, data.times)
    coarse_eta = collect(-0.2:0.02:1.6)
    coarse_zeta = collect(0.0:0.02:0.8)
    _, coarse_best = scan_grid(data.radii, data.times, mean_field, stderr_field,
        radius_mask, time_indices, coarse_eta, coarse_zeta, POLY_ORDER, COLLAPSE_BINS)
    fine_eta = collect((coarse_best.eta - 0.06):0.005:(coarse_best.eta + 0.06))
    fine_zeta = collect((coarse_best.zeta - 0.06):0.005:(coarse_best.zeta + 0.06))
    _, fine_best = scan_grid(data.radii, data.times, mean_field, stderr_field,
        radius_mask, time_indices, fine_eta, fine_zeta, POLY_ORDER, COLLAPSE_BINS)
    return fine_best.zeta
end

function main()
    input_dir = length(ARGS) >= 1 ? ARGS[1] :
        error("usage: check_v1_speedup.jl <input-dir> [target-zeta] [band]")
    target = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : TARGET_ZETA
    band = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : BAND

    files = collect_sample_files(input_dir)
    data = load_streaming_stats(files)
    vidx = argmin(abs.(data.v_values .- 1.0))
    zeta = reference_zeta(data, vidx)
    delta = abs(zeta - target)
    pass = delta <= band

    @printf("# v=1 speedup validation  samples=%d  v=%.4g\n", data.nsamples, data.v_values[vidx])
    @printf("  zeta = %.4f   target = %.4f +/- %.4f   |delta| = %.4f   -> %s\n",
        zeta, target, band, delta, pass ? "PASS" : "FAIL")
    pass || error("v=1 zeta $(round(zeta; digits=4)) outside band $target +/- $band")
    return pass
end

main()
