#!/usr/bin/env julia

using ArgParse
using CairoMakie
using Random
using Statistics

function linear_fit(x, y)
    mx = mean(x)
    my = mean(y)
    slope = sum((x .- mx) .* (y .- my)) / sum((x .- mx).^2)
    intercept = my - slope * mx
    return slope, intercept
end

function d4_transform(dx, dy, code)
    code == 1 && return (dx, dy)
    code == 2 && return (-dy, dx)
    code == 3 && return (-dx, -dy)
    code == 4 && return (dy, -dx)
    code == 5 && return (-dx, dy)
    code == 6 && return (dx, -dy)
    code == 7 && return (dy, dx)
    return (-dy, -dx)
end

function pivot_saw_sample!(walk, rng, npivots)
    n = size(walk, 1)
    candidate = similar(walk)
    for _ in 1:npivots
        pivot = rand(rng, 1:n)
        code = rand(rng, 2:8)
        candidate .= walk
        px = walk[pivot, 1]
        py = walk[pivot, 2]
        occupied = Set{Tuple{Int, Int}}()
        accepted = true
        for i in 1:pivot
            push!(occupied, (walk[i, 1], walk[i, 2]))
        end
        for i in (pivot + 1):n
            dx = walk[i, 1] - px
            dy = walk[i, 2] - py
            tx, ty = d4_transform(dx, dy, code)
            x = px + tx
            y = py + ty
            if (x, y) in occupied
                accepted = false
                break
            end
            candidate[i, 1] = x
            candidate[i, 2] = y
            push!(occupied, (x, y))
        end
        accepted && (walk .= candidate)
    end
    return walk
end

settings = ArgParseSettings(
    description="Numerical SAW surrogate for contour-resolved F scaling versus lab-frame diffusion.",
)
@add_arg_table! settings begin
    "--length"
        arg_type = Int
        default = 768
    "--samples"
        arg_type = Int
        default = 800
    "--thermal-pivots"
        arg_type = Int
        default = 6000
    "--sample-pivots"
        arg_type = Int
        default = 1000
    "--seed"
        arg_type = Int
        default = 7
    "--fit-min"
        arg_type = Float64
        default = 16.0
    "--fit-max"
        arg_type = Float64
        default = 256.0
    "--output-prefix"
        arg_type = String
        default = "saw_f_correlator_sampling"
end
args = ArgParse.parse_args(settings)

rng = MersenneTwister(args["seed"])
nsteps = args["length"]
walk = zeros(Int, nsteps + 1, 2)
walk[:, 1] .= 0:nsteps
pivot_saw_sample!(walk, rng, args["thermal-pivots"])

lags = unique(round.(Int, exp10.(range(log10(4), log10(nsteps ÷ 2); length=42))))
r2_sum = zeros(Float64, length(lags))
r2_endpoint = zeros(Float64, length(lags))

for sample in 1:args["samples"]
    pivot_saw_sample!(walk, rng, args["sample-pivots"])
    for (j, lag) in enumerate(lags)
        acc = 0.0
        count = 0
        @inbounds for i in 1:(size(walk, 1) - lag)
            dx = walk[i + lag, 1] - walk[i, 1]
            dy = walk[i + lag, 2] - walk[i, 2]
            acc += dx^2 + dy^2
            count += 1
        end
        r2_sum[j] += acc / count
        dx = walk[lag + 1, 1] - walk[1, 1]
        dy = walk[lag + 1, 2] - walk[1, 2]
        r2_endpoint[j] += dx^2 + dy^2
    end
    sample % 100 == 0 && @info "sampled SAWs" sample total=args["samples"]
end

r2 = r2_sum ./ args["samples"]
r2_endpoint ./= args["samples"]
fit_mask = Float64.(lags) .>= args["fit-min"] .&& Float64.(lags) .<= args["fit-max"]
alpha, intercept = linear_fit(log.(Float64.(lags[fit_mask])), log.(r2[fit_mask]))
df = 2 / alpha
z_f = 2df
zeta_f = 1 / z_f

times = exp10.(range(log10(16.0), log10(4096.0); length=28))
contour_s = sqrt.(times)
contour_r = exp.(0.5 .* (intercept .+ alpha .* log.(contour_s)))
zeta_contour, contour_intercept = linear_fit(log.(times), log.(contour_r))
z_contour = 1 / zeta_contour

lab_r = sqrt.(times)
zeta_lab, lab_intercept = linear_fit(log.(times), log.(lab_r))
z_lab = 1 / zeta_lab

figure_output = joinpath("figures", "$(args["output-prefix"]).png")
summary_output = joinpath("results", "$(args["output-prefix"]).md")
mkpath(dirname(figure_output))
mkpath(dirname(summary_output))

fig = Figure(size=(1200, 520), backgroundcolor=:white)
ax1 = Axis(fig[1, 1], xscale=log10, yscale=log10, xlabel="contour lag s",
    ylabel="SAW mean R2(s)", title="SAW geometry")
scatterlines!(ax1, Float64.(lags), r2; color=:black, label="pivot-sampled SAWs")
lines!(ax1, Float64.(lags), exp.(intercept .+ alpha .* log.(Float64.(lags)));
    color=:red, linewidth=2, label="fit alpha=$(round(alpha; digits=3))")
lines!(ax1, Float64.(lags),
    exp.(mean(log.(r2[fit_mask]) .- 1.5 .* log.(Float64.(lags[fit_mask]))) .+
        1.5 .* log.(Float64.(lags)));
    color=:dodgerblue, linestyle=:dash, linewidth=2, label="SAW alpha=1.5")
axislegend(ax1; position=:lt)

ax2 = Axis(fig[1, 2], xscale=log10, yscale=log10, xlabel="time t",
    ylabel="length scale", title="Diffusive clock: contour versus lab")
scatterlines!(ax2, times, contour_r; color=:red,
    label="contour-resolved SAW, z=$(round(z_contour; digits=3))")
lines!(ax2, times, exp.(contour_intercept .+ zeta_contour .* log.(times));
    color=:red, linewidth=2)
scatterlines!(ax2, times, lab_r; color=:black,
    label="Euclidean diffusion control, z=$(round(z_lab; digits=3))")
lines!(ax2, times, exp.(lab_intercept .+ zeta_lab .* log.(times));
    color=:black, linewidth=2)
axislegend(ax2; position=:lt)
save(figure_output, fig)

open(summary_output, "w") do io
    println(io, "# SAW F-correlator sampling surrogate")
    println(io)
    println(io, "- SAW sampler: 2D square-lattice pivot sampler")
    println(io, "- Walk length: `$(nsteps)`, samples: `$(args["samples"])`")
    println(io, "- Geometry fit window: `$(args["fit-min"]) <= s <= $(args["fit-max"])`")
    println(io)
    println(io, "## Geometry")
    println(io)
    println(io, "- Fit `R2(s) ~ s^alpha`: `alpha = $(round(alpha; digits=5))`")
    println(io, "- Inferred `d_f = 2 / alpha`: `$(round(df; digits=5))`")
    println(io, "- Reference SAW: `alpha = 1.5`, `d_f = 1.33333`")
    println(io)
    println(io, "## Clock comparison")
    println(io)
    println(io, "- Contour diffusive clock plus SAW geometry: `r(t) ~ t^$(round(zeta_contour; digits=5))`, `z = $(round(z_contour; digits=5))`")
    println(io, "- Formula `z = 2 d_f`: `$(round(z_f; digits=5))`")
    println(io, "- Reference `8/3`: `$(round(8 / 3; digits=5))`")
    println(io, "- Euclidean lab-frame diffusion control: `r(t) ~ t^$(round(zeta_lab; digits=5))`, `z = $(round(z_lab; digits=5))`")
end

println("saved figure: ", figure_output)
println("saved summary: ", summary_output)
println("alpha: ", alpha)
println("df: ", df)
println("contour z: ", z_contour)
println("lab z: ", z_lab)
