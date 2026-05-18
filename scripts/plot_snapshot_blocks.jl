#!/usr/bin/env julia

using ArgParse
using CairoMakie
using JLD2

include(joinpath(@__DIR__, "..", "src", "LatticeFlockingSDE.jl"))
using .LatticeFlockingSDE

settings = ArgParseSettings(
    description="Plot coarse rightward-alignment snapshots with block-averaged arrows.",
)
@add_arg_table! settings begin
    "input"
        arg_type = String
    "--output"
        arg_type = String
        default = "figures/snapshot_dataset/block_snapshots.png"
    "--arrow-scale"
        arg_type = Float64
        default = 0.7
    "--arrow-size"
        arg_type = Float64
        default = 4.0
end
args = ArgParse.parse_args(settings)

dataset = load(args["input"], "dataset")
times = dataset.times
block_x = dataset.block_x
block_y = dataset.block_y
block_u = dataset.block_u
block_v = dataset.block_v

fig = Figure(size=(1500, 430), backgroundcolor=:white)
for k in 1:length(times)
    ax = Axis(
        fig[1, k],
        aspect=DataAspect(),
        title="t = $(round(times[k]; digits=2))",
        xlabel="x",
        ylabel=k == 1 ? "y" : "",
    )
    heatmap!(ax, block_x[:, 1], block_y[1, :], block_u[:, :, k];
        colormap=:balance, colorrange=(-1, 1))
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
        lengthscale=10,
    )
end

Colorbar(fig[2, 1:4], limits=(-1, 1), colormap=:balance, vertical=false,
    label="block mean cos(θ)")
mkpath(dirname(args["output"]))
save(args["output"], fig)
println("saved: ", args["output"])
