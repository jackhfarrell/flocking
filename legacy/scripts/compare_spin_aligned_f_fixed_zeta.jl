#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_J2_v1_gamma1")
const DEFAULT_RESULTS_DIR = joinpath(REPO_ROOT, "results", "spin_aligned_f_correlator_L200_J2_v1_gamma1_fixed_zeta_compare")
const DEFAULT_FIGURES_DIR = joinpath(REPO_ROOT, "figures", "spin_aligned_f_correlator_L200_J2_v1_gamma1_fixed_zeta_compare")

function parse_args()
    settings = ArgParseSettings(
        description="Compare the best F(r,t) collapse against a collapse with fixed zeta.",
    )
    @add_arg_table! settings begin
        "--input-dir"
            arg_type = String
            default = DEFAULT_INPUT_DIR
        "--results-dir"
            arg_type = String
            default = DEFAULT_RESULTS_DIR
        "--figures-dir"
            arg_type = String
            default = DEFAULT_FIGURES_DIR
        "--radius-max"
            arg_type = Float64
            default = 80.0
        "--time-min"
            arg_type = Float64
            default = 6.0
        "--poly-order"
            arg_type = Int
            default = 3
        "--collapse-bins"
            arg_type = Int
            default = 60
        "--eta-min"
            arg_type = Float64
            default = -0.2
        "--eta-max"
            arg_type = Float64
            default = 0.8
        "--eta-step"
            arg_type = Float64
            default = 0.02
        "--zeta-min"
            arg_type = Float64
            default = 0.0
        "--zeta-max"
            arg_type = Float64
            default = 0.8
        "--zeta-step"
            arg_type = Float64
            default = 0.02
        "--fine-window"
            arg_type = Float64
            default = 0.06
        "--fine-step"
            arg_type = Float64
            default = 0.005
        "--fixed-zeta"
            arg_type = Float64
            default = 0.375
    end
    return ArgParse.parse_args(settings)
end

function fixed_zeta_best(radii, times, F_mean, F_stderr, radius_mask, time_indices,
        fixed_zeta::Real, eta_values::AbstractVector, poly_order::Integer, nbins::Integer)
    objective = fill(Inf, length(eta_values))
    best = nothing
    for (i, eta) in enumerate(eta_values)
        result = evaluate_collapse(radii, times, F_mean, F_stderr, radius_mask,
            time_indices, eta, fixed_zeta, poly_order, nbins)
        result === nothing && continue
        isfinite(result.reduced_chi2) || continue
        objective[i] = result.reduced_chi2
        if best === nothing || result.reduced_chi2 < best.reduced_chi2
            best = result
        end
    end
    best === nothing && error("no valid fixed-zeta collapse found")
    return objective, best
end

function run_analysis(ensemble, args)
    radius_mask = ensemble.radii .<= args["radius-max"]
    any(radius_mask) || error("no radii satisfy r <= $(args["radius-max"])")
    time_indices = findall(t -> t > 0 && t >= args["time-min"], ensemble.times)
    isempty(time_indices) && error("no times satisfy t >= $(args["time-min"])")

    eta_values = collect(args["eta-min"]:args["eta-step"]:args["eta-max"])
    zeta_values = collect(args["zeta-min"]:args["zeta-step"]:args["zeta-max"])
    coarse_objective, coarse_best = scan_grid(ensemble.radii, ensemble.times,
        ensemble.F_mean, ensemble.F_stderr, radius_mask, time_indices, eta_values,
        zeta_values, args["poly-order"], args["collapse-bins"])

    fine_eta_values = collect((coarse_best.eta - args["fine-window"]):
        args["fine-step"]:(coarse_best.eta + args["fine-window"]))
    fine_zeta_values = collect((coarse_best.zeta - args["fine-window"]):
        args["fine-step"]:(coarse_best.zeta + args["fine-window"]))
    fine_objective, fine_best = scan_grid(ensemble.radii, ensemble.times,
        ensemble.F_mean, ensemble.F_stderr, radius_mask, time_indices, fine_eta_values,
        fine_zeta_values, args["poly-order"], args["collapse-bins"])

    fixed_eta_values = collect((fine_best.eta - args["fine-window"]):
        args["fine-step"]:(fine_best.eta + args["fine-window"]))
    fixed_objective, fixed_best = fixed_zeta_best(ensemble.radii, ensemble.times,
        ensemble.F_mean, ensemble.F_stderr, radius_mask, time_indices, args["fixed-zeta"],
        fixed_eta_values, args["poly-order"], args["collapse-bins"])

    feature = feature_estimate(ensemble.radii, ensemble.times, ensemble.F_mean,
        radius_mask, time_indices)

    return (; time_indices, coarse_objective, coarse_best, fine_eta_values, fine_zeta_values,
        fine_objective, fine_best, fixed_eta_values, fixed_objective, fixed_best, feature)
end

function add_collapse_panel(fig, slot, collapse, feature, title)
    ax = Axis(fig[slot...], xlabel="r / t^ζ", ylabel="t^η F(r,t)", title=title)
    palette = Makie.wong_colors()
    for (k, t) in enumerate(feature.times)
        inds = findall(==(t), collapse.time)
        local_order = sortperm(collapse.x[inds])
        ordered_inds = inds[local_order]
        band!(ax, collapse.x[ordered_inds],
            collapse.y[ordered_inds] .- collapse.sigma[ordered_inds],
            collapse.y[ordered_inds] .+ collapse.sigma[ordered_inds],
            color=(palette[mod1(k, length(palette))], 0.30))
        lines!(ax, collapse.x[ordered_inds], collapse.y[ordered_inds],
            color=palette[mod1(k, length(palette))], linewidth=3,
            label="t = $(round(t; digits=3))")
    end
    vlines!(ax, [collapse.overlap_min, collapse.overlap_max], color=:gray60,
        linestyle=:dash, linewidth=2)
    return ax
end

function add_collapse_panel_paper(fig, slot, collapse, feature, title)
    ax = Axis(fig[slot...],
        xlabel=L"r / t^{\zeta}",
        ylabel=L"t^{\eta_F} F(r,t)",
        title=title,
        titlesize=9,
        xlabelsize=9,
        ylabelsize=9,
        xticklabelsize=8,
        yticklabelsize=8)
    palette = Makie.wong_colors()
    for (k, t) in enumerate(feature.times)
        inds = findall(==(t), collapse.time)
        local_order = sortperm(collapse.x[inds])
        ordered_inds = inds[local_order]
        band!(ax, collapse.x[ordered_inds],
            collapse.y[ordered_inds] .- collapse.sigma[ordered_inds],
            collapse.y[ordered_inds] .+ collapse.sigma[ordered_inds],
            color=(palette[mod1(k, length(palette))], 0.22))
        lines!(ax, collapse.x[ordered_inds], collapse.y[ordered_inds],
            color=palette[mod1(k, length(palette))], linewidth=2.0,
            label=L"t = %$(round(t; digits=3))")
    end
    vlines!(ax, [collapse.overlap_min, collapse.overlap_max], color=:gray55,
        linestyle=:dash, linewidth=1.2)
    return ax
end

function plot_comparison(path::String, analysis, label::AbstractString, fixed_zeta::Real)
    mkpath(dirname(path))
    fig = Figure(size=(1400, 650))
    Label(fig[0, :], label, fontsize=24)

    best_title = @sprintf("Best free fit: η_F = %.3f, ζ = %.3f, χ² = %.2f",
        analysis.fine_best.eta, analysis.fine_best.zeta, analysis.fine_best.reduced_chi2)
    fixed_title = @sprintf("Fixed ζ = %.3f: η_F = %.3f, χ² = %.2f",
        fixed_zeta, analysis.fixed_best.eta, analysis.fixed_best.reduced_chi2)

    ax1 = add_collapse_panel(fig, (1, 1), analysis.fine_best, analysis.feature, best_title)
    ax2 = add_collapse_panel(fig, (1, 2), analysis.fixed_best, analysis.feature, fixed_title)
    axislegend(ax2, position=:rb, nbanks=2)
    linkxaxes!(ax1, ax2)
    linkyaxes!(ax1, ax2)
    save(path, fig)
end

function plot_comparison_paper(path::String, analysis, fixed_zeta::Real)
    mkpath(dirname(path))
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(1150, 480), fontsize=9)

        best_title = @sprintf("Free fit: η_F = %.3f, ζ = %.3f",
            analysis.fine_best.eta, analysis.fine_best.zeta)
        fixed_title = @sprintf("Fixed ζ = %.3f: η_F = %.3f",
            fixed_zeta, analysis.fixed_best.eta)

        ax1 = add_collapse_panel_paper(fig, (1, 1), analysis.fine_best,
            analysis.feature, best_title)
        ax2 = add_collapse_panel_paper(fig, (1, 2), analysis.fixed_best,
            analysis.feature, fixed_title)
        hidexdecorations!(ax1; grid=false, minorgrid=false)
        linkxaxes!(ax1, ax2)
        linkyaxes!(ax1, ax2)
        axislegend(ax2, position=:rb, nbanks=2, framevisible=false, labelsize=8,
            titlesize=8, patchsize=(14, 8))
        save(path, fig)
    end
end

function plot_fixed_eta_scan(path::String, analysis, fixed_zeta::Real, label::AbstractString)
    mkpath(dirname(path))
    fig = Figure(size=(900, 650))
    Label(fig[0, :], label, fontsize=24)
    ax = Axis(fig[1, 1], xlabel="η_F", ylabel="reduced χ²",
        title=@sprintf("Fixed-ζ scan at ζ = %.3f", fixed_zeta))
    lines!(ax, analysis.fixed_eta_values, analysis.fixed_objective, linewidth=3)
    scatter!(ax, [analysis.fixed_best.eta], [analysis.fixed_best.reduced_chi2], markersize=16)
    save(path, fig)
end

function write_summary(path::String, analysis, fixed_zeta::Real, time_min::Real)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Fixed-zeta comparison")
        println(io)
        println(io, "- Fixed zeta target: `", @sprintf("%.3f", fixed_zeta), "`")
        println(io, "- Time cutoff: `t >= ", @sprintf("%.1f", time_min), "`")
        println(io)
        times_str = join(string.(analysis.feature.times), ", ")
        println(io, "- Times: `", times_str, "`")
        println(io, "- Best free collapse: `eta_F = ",
            @sprintf("%.4f", analysis.fine_best.eta), "`, `zeta = ",
            @sprintf("%.4f", analysis.fine_best.zeta), "`, reduced `chi^2 = ",
            @sprintf("%.4f", analysis.fine_best.reduced_chi2), "`")
        println(io, "- Fixed-`zeta` collapse: `eta_F = ",
            @sprintf("%.4f", analysis.fixed_best.eta), "`, `zeta = ",
            @sprintf("%.4f", fixed_zeta), "`, reduced `chi^2 = ",
            @sprintf("%.4f", analysis.fixed_best.reduced_chi2), "`")
        println(io, "- Objective ratio: `chi2_fixed / chi2_best = ",
            @sprintf("%.4f", analysis.fixed_best.reduced_chi2 / analysis.fine_best.reduced_chi2), "`")
        println(io, "- Trough fit: `zeta = ",
            @sprintf("%.4f ± %.4f", analysis.feature.zeta, analysis.feature.zeta_stderr),
            "`, `eta_F = ",
            @sprintf("%.4f ± %.4f", analysis.feature.eta, analysis.feature.eta_stderr), "`")
    end
end

function main()
    args = parse_args()
    files = collect_job_files(args["input-dir"])
    ensemble = load_ensemble(files)
    analysis = run_analysis(ensemble, args)

    mkpath(args["results-dir"])
    mkpath(args["figures-dir"])

    label = @sprintf("Late-time comparison (t >= %.1f)", args["time-min"])
    compare_plot = joinpath(args["figures-dir"], "spin_aligned_f_fixed_zeta_compare.png")
    compare_plot_paper_png = joinpath(args["figures-dir"], "spin_aligned_f_fixed_zeta_compare_paper.png")
    compare_plot_paper_pdf = joinpath(args["figures-dir"], "spin_aligned_f_fixed_zeta_compare_paper.pdf")
    eta_scan_plot = joinpath(args["figures-dir"], "spin_aligned_f_fixed_zeta_eta_scan.png")
    plot_comparison(compare_plot, analysis, label, args["fixed-zeta"])
    plot_comparison_paper(compare_plot_paper_png, analysis, args["fixed-zeta"])
    plot_comparison_paper(compare_plot_paper_pdf, analysis, args["fixed-zeta"])
    plot_fixed_eta_scan(eta_scan_plot, analysis, args["fixed-zeta"], label)
    println("saved comparison plot: ", compare_plot)
    println("saved paper comparison: ", compare_plot_paper_png)
    println("saved paper comparison: ", compare_plot_paper_pdf)
    println("saved fixed-eta scan: ", eta_scan_plot)

    summary_path = joinpath(args["results-dir"], "spin_aligned_f_fixed_zeta_compare_summary.md")
    archive_path = joinpath(args["results-dir"], "spin_aligned_f_fixed_zeta_compare.jld2")
    write_summary(summary_path, analysis, args["fixed-zeta"], args["time-min"])
    jldsave(archive_path; analysis, fixed_zeta=args["fixed-zeta"], time_min=args["time-min"],
        input_dir=args["input-dir"])
    println("saved summary: ", summary_path)
    println("saved archive: ", archive_path)
end

main()
