#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2
using Printf

include(joinpath(@__DIR__, "spin_aligned_f_analysis.jl"))

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(REPO_ROOT, "results",
    "spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353")
const DEFAULT_PREFIX = "f_backbone_df_preference"

settings = ArgParseSettings(
    description="Infer the backbone fractal dimension preferred by the spin-aligned time-antisymmetric F correlator.",
)
@add_arg_table! settings begin
    "--input-dir"
        arg_type = String
        default = DEFAULT_INPUT_DIR
    "--radius-min"
        arg_type = Float64
        default = 0.0
    "--radius-max"
        arg_type = Float64
        default = 40.0
    "--time-min"
        arg_type = Float64
        default = 2.0
    "--eta-min"
        arg_type = Float64
        default = -0.2
    "--eta-max"
        arg_type = Float64
        default = 1.0
    "--eta-step"
        arg_type = Float64
        default = 0.01
    "--zeta-min"
        arg_type = Float64
        default = 0.25
    "--zeta-max"
        arg_type = Float64
        default = 0.6
    "--zeta-step"
        arg_type = Float64
        default = 0.005
    "--poly-order"
        arg_type = Int
        default = 3
    "--collapse-bins"
        arg_type = Int
        default = 60
    "--output-prefix"
        arg_type = String
        default = DEFAULT_PREFIX
end
args = parse_args(settings)

files = collect_job_files(args["input-dir"])
ensemble = load_ensemble(files)
radius_mask = (ensemble.radii .>= args["radius-min"]) .&
    (ensemble.radii .<= args["radius-max"])
time_indices = findall(t -> t > 0 && t >= args["time-min"], ensemble.times)
eta_values = collect(args["eta-min"]:args["eta-step"]:args["eta-max"])
zeta_values = collect(args["zeta-min"]:args["zeta-step"]:args["zeta-max"])

objective, best = scan_grid(ensemble.radii, ensemble.times, ensemble.F_mean,
    ensemble.F_stderr, radius_mask, time_indices, eta_values, zeta_values,
    args["poly-order"], args["collapse-bins"])
feature = feature_estimate(ensemble.radii, ensemble.times, ensemble.F_mean,
    radius_mask, time_indices)

targets = [
    (; label="SAW contour", zeta=3 / 8),
    (; label="lab diffusion", zeta=1 / 2),
    (; label="older 1/3 check", zeta=1 / 3),
]
target_rows = map(targets) do target
    j = argmin(abs.(zeta_values .- target.zeta))
    i = argmin(objective[:, j])
    chi2 = objective[i, j]
    (; target.label, target_zeta=target.zeta, grid_zeta=zeta_values[j],
        eta=eta_values[i], chi2, ratio=chi2 / best.reduced_chi2,
        df=1 / (2target.zeta), z=1 / target.zeta)
end

profile_chi2 = vec(minimum(objective; dims=1))
profile_eta_index = [argmin(objective[:, j]) for j in eachindex(zeta_values)]
profile_eta = eta_values[profile_eta_index]
profile_df = 1.0 ./ (2 .* zeta_values)
best_df = 1 / (2best.zeta)
feature_df = 1 / (2feature.zeta)

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
data_output = joinpath("results", "$(args["output-prefix"]).jld2")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1180, 520), backgroundcolor=:white)
ax1 = Axis(fig[1, 1], xlabel="ζ = 1/z", ylabel="minη reduced χ²",
    title="F-collapse exponent preference")
lines!(ax1, zeta_values, profile_chi2; color=:black, linewidth=3)
vlines!(ax1, [best.zeta]; color=:red, linewidth=2, label="best")
vlines!(ax1, [3 / 8]; color=:dodgerblue, linestyle=:dash, linewidth=2,
    label="SAW ζ=3/8")
vlines!(ax1, [1 / 2]; color=:gray40, linestyle=:dash, linewidth=2,
    label="diffusive ζ=1/2")
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2], xlabel="d_f = 1/(2ζ)", ylabel="minη reduced χ²",
    title="Same scan as inferred backbone d_f")
lines!(ax2, profile_df, profile_chi2; color=:black, linewidth=3)
vlines!(ax2, [best_df]; color=:red, linewidth=2, label="best")
vlines!(ax2, [4 / 3]; color=:dodgerblue, linestyle=:dash, linewidth=2,
    label="SAW d_f=4/3")
vlines!(ax2, [1.0]; color=:gray40, linestyle=:dash, linewidth=2,
    label="straight/lab d_f=1")
axislegend(ax2; position=:rt)

save(figure_output, fig)

open(summary_output, "w") do io
    println(io, "# F-correlator backbone df preference")
    println(io)
    println(io, "- Observable: spin-aligned time-antisymmetric `F(r,t)`")
    println(io, "- Input: `$(args["input-dir"])`")
    println(io, "- Runs: `$(ensemble.nruns)`")
    println(io, "- Window: `$(args["radius-min"]) <= r <= $(args["radius-max"])`, `t >= $(args["time-min"])`")
    println(io, "- Times: `$(join(round.(ensemble.times[time_indices]; digits=4), ", "))`")
    println(io)
    println(io, "## Best collapse")
    println(io)
    println(io, "- `eta_F = $(@sprintf("%.4f", best.eta))`")
    println(io, "- `zeta = 1/z = $(@sprintf("%.4f", best.zeta))`")
    println(io, "- `z = $(@sprintf("%.4f", 1 / best.zeta))`")
    println(io, "- Interpreting `z = 2 d_f`: `d_f = $(@sprintf("%.4f", best_df))`")
    println(io, "- Reduced `chi^2 = $(@sprintf("%.4f", best.reduced_chi2))`")
    println(io)
    println(io, "## Fixed-exponent checks")
    println(io)
    println(io, "| target | zeta | z | implied d_f | best eta_F | reduced chi^2 | ratio to best |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|")
    for row in target_rows
        println(io, "| $(row.label) | $(@sprintf("%.4f", row.target_zeta)) | $(@sprintf("%.4f", row.z)) | $(@sprintf("%.4f", row.df)) | $(@sprintf("%.4f", row.eta)) | $(@sprintf("%.4f", row.chi2)) | $(@sprintf("%.4f", row.ratio)) |")
    end
    println(io)
    println(io, "## Trough feature check")
    println(io)
    println(io, "- `r_min(t) ~ t^zeta`: `zeta = $(@sprintf("%.4f ± %.4f", feature.zeta, feature.zeta_stderr))`")
    println(io, "- Equivalent `z = $(@sprintf("%.4f", 1 / feature.zeta))`")
    println(io, "- Equivalent `d_f = 1/(2 zeta) = $(@sprintf("%.4f", feature_df))`")
    println(io, "- Reference SAW: `zeta = 0.3750`, `z = 2.6667`, `d_f = 1.3333`")
end

jldsave(data_output; args, input_dir=args["input-dir"], radii=ensemble.radii,
    times=ensemble.times, time_indices, eta_values, zeta_values, objective,
    profile_chi2, profile_eta, profile_df, best, best_df, feature, feature_df,
    target_rows)

println("saved figure: ", figure_output)
println("saved data: ", data_output)
println("saved summary: ", summary_output)
println("best zeta: ", best.zeta)
println("best df: ", best_df)
println("feature df: ", feature_df)
