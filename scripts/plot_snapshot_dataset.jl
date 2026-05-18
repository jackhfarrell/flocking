#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(
    description="Plot four angle-field snapshots with block-averaged arrows.",
)
@add_arg_table! settings begin
    "input"
        arg_type = String
    "--output"
        arg_type = String
        default = "figures/snapshot_dataset/snapshots.png"
    "--arrow-scale"
        arg_type = Float64
        default = 7.0
    "--arrow-size"
        arg_type = Float64
        default = 4.0
end
args = ArgParse.parse_args(settings)

dataset = load(args["input"], "dataset")
times = dataset.times
theta_snapshots = dataset.theta_snapshots
block_x = dataset.block_x
block_y = dataset.block_y
block_u = dataset.block_u
block_v = dataset.block_v
L = size(theta_snapshots, 1)

fig = Figure(size=(1500, 430), backgroundcolor=:white)
for k in 1:length(times)
    ax = Axis(
        fig[1, k],
        aspect=DataAspect(),
        title="t = $(round(times[k]; digits=2))",
        xlabel="x",
        ylabel=k == 1 ? "y" : "",
    )
    heatmap!(ax, 1:L, 1:L, theta_snapshots[:, :, k];
        colormap=:hsv, colorrange=(0, 2π))
    arrows2d!(
        ax,
        vec(block_x),
        vec(block_y),
        args["arrow-scale"] .* vec(block_u[:, :, k]),
        args["arrow-scale"] .* vec(block_v[:, :, k]);
        color=:black,
        tipwidth=args["arrow-size"],
        tiplength=1.5args["arrow-size"],
        shaftwidth=1.2,
        lengthscale=1,
    )
    xlims!(ax, 1, L)
    ylims!(ax, 1, L)
end

Colorbar(fig[2, 1:4], limits=(0, 2π), colormap=:hsv, vertical=false,
    label="spin angle θ")
mkpath(dirname(args["output"]))
save(args["output"], fig)
println("saved: ", args["output"])
