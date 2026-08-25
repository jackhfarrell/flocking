using JLD2
using Statistics

function stage2_sample_files(input_dir::String)
    isdir(input_dir) || error("missing input directory: $input_dir")
    files = String[]
    for (dir, _, names) in walkdir(input_dir)
        for name in names
            startswith(name, "sample_") && endswith(name, ".jld2") || continue
            push!(files, joinpath(dir, name))
        end
    end
    sort!(files)
    isempty(files) && error("no sample_*.jld2 files found below $input_dir")
    return files
end

function load_legacy_sweep(files, runs)
    first_result = runs[1]
    v_values = first_result.v_values
    radii = first_result.radii
    times = first_result.times
    shape = size(first_result.F)
    all(run -> run.v_values == v_values && run.radii == radii &&
        run.times == times && size(run.F) == shape, runs) ||
        error("legacy sweep samples must share v_values, radii, times, and F shape")

    F_samples = cat((run.F for run in runs)...; dims=4)
    mean_F = dropdims(mean(F_samples; dims=4), dims=4)
    stderr_F = length(runs) == 1 ? zeros(size(mean_F)) :
        dropdims(std(F_samples; dims=4), dims=4) ./ sqrt(length(runs))
    return (; files, config=first_result.config, v_values, radii, times,
        mean_F, stderr_F, nsamples=length(runs), nfiles=length(files))
end

function load_stage2_sweep(input_dir::String)
    files = stage2_sample_files(input_dir)
    first_result = load(files[1], "result")
    if length(first_result.v_values) != 1
        runs = [first_result; [load(file, "result") for file in files[2:end]]]
        return load_legacy_sweep(files, runs)
    end

    radii = first_result.radii
    times = first_result.times
    entries = Dict{Tuple{Int,Int},Vector{Tuple{String,Int}}}()
    v_by_index = Dict{Int,Float64}()
    for file in files
        run = load(file, "result")
        run.radii == radii && run.times == times && size(run.F, 1) == 1 ||
            error("Stage-2 samples must share radii and times and contain exactly one velocity")
        hasproperty(run.config, :vi) && hasproperty(run.config, :traj) ||
            error("Stage-2 samples must record config.vi and config.traj")
        vi = run.config.vi
        traj = run.config.traj
        chunks = hasproperty(run.config, :chunks) ? run.config.chunks : 1
        push!(get!(entries, (vi, traj), Tuple{String,Int}[]), (file, chunks))
        v = only(run.v_values)
        haskey(v_by_index, vi) && v_by_index[vi] != v &&
            error("velocity mismatch for v index $vi")
        v_by_index[vi] = v
    end

    v_indices = sort!(collect(keys(v_by_index)))
    v_indices == collect(1:length(v_indices)) ||
        error("Stage-2 sweep is incomplete: found velocity indices $(v_indices)")
    trajectories = sort!(unique(last.(collect(keys(entries)))))
    for vi in v_indices
        available = sort!([traj for (entry_vi, traj) in keys(entries) if entry_vi == vi])
        available == trajectories || error(
            "Stage-2 sweep is incomplete at v index $vi: trajectories $available, expected $trajectories")
    end

    nv = length(v_indices)
    nr = length(radii)
    nt = length(times)
    mean_F = zeros(Float64, nv, nr, nt)
    stderr_F = zeros(Float64, nv, nr, nt)
    for vi in v_indices
        trajectory_means = zeros(Float64, nr, nt, length(trajectories))
        for (trajectory_index, traj) in enumerate(trajectories)
            total_chunks = 0
            for (file, chunks) in entries[(vi, traj)]
                run = load(file, "result")
                trajectory_means[:, :, trajectory_index] .+=
                    chunks .* dropdims(run.F; dims=1)
                total_chunks += chunks
            end
            trajectory_means[:, :, trajectory_index] ./= total_chunks
        end
        mean_F[vi, :, :] .= dropdims(mean(trajectory_means; dims=3), dims=3)
        if length(trajectories) > 1
            stderr_F[vi, :, :] .=
                dropdims(std(trajectory_means; dims=3), dims=3) ./ sqrt(length(trajectories))
        end
    end

    v_values = [v_by_index[vi] for vi in v_indices]
    return (; files, config=first_result.config, v_values, radii, times,
        mean_F, stderr_F, nsamples=length(trajectories), nfiles=length(files), trajectories)
end
