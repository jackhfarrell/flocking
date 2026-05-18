#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import tempfile

import h5py

cache_dir = Path(tempfile.gettempdir()) / "flocking-matplotlib-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


REVTEX_TWO_COLUMN_WIDTH_IN = 510.0 / 72.27


def load_dataset(path):
    with h5py.File(path, "r") as handle:
        dataset = handle["dataset"][()]
        times = handle[dataset["times"]][()]
        theta = handle[dataset["theta_snapshots"]][()]
    return times, theta


def seaborn_cyclic_cmap(n=256, saturation=0.65, lightness=0.55):
    colors = sns.hls_palette(n, h=0.01, l=lightness, s=saturation)
    return ListedColormap(colors + [colors[0]], name="seaborn_husl_cyclic")


def block_mean(field, block_size):
    if block_size == 1:
        return field
    ny, nx = field.shape
    if ny % block_size != 0 or nx % block_size != 0:
        raise ValueError("coarsen must divide both snapshot dimensions")
    return field.reshape(ny // block_size, block_size, nx // block_size, block_size).mean(axis=(1, 3))


def main():
    parser = argparse.ArgumentParser(
        description="Plot four full-field theta snapshots at REVTeX two-column size.",
    )
    parser.add_argument("input")
    parser.add_argument(
        "--output",
        default="figures/snapshot_dataset/revtex_snapshots.pdf",
    )
    parser.add_argument(
        "--png-output",
        default="",
        help="Optional PNG output path.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--coarsen",
        type=int,
        default=1,
        help="Average phase vectors over coarsen x coarsen pixels before plotting.",
    )
    parser.add_argument("--saturation", type=float, default=0.65)
    parser.add_argument("--lightness", type=float, default=0.55)
    args = parser.parse_args()

    times, theta = load_dataset(args.input)
    if args.coarsen > 1:
        coarse_theta = []
        for snapshot in theta:
            c = block_mean(np.cos(snapshot), args.coarsen)
            s = block_mean(np.sin(snapshot), args.coarsen)
            coarse_theta.append(np.mod(np.arctan2(s, c), 2 * np.pi))
        theta = np.stack(coarse_theta)

    nsnapshots, ny, nx = theta.shape
    if nsnapshots != 4:
        raise ValueError(f"expected 4 snapshots, got {nsnapshots}")

    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        },
    )

    cmap = seaborn_cyclic_cmap(
        saturation=args.saturation,
        lightness=args.lightness,
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(REVTEX_TWO_COLUMN_WIDTH_IN, 1.95),
        sharey=True,
        constrained_layout=True,
    )

    image = None
    for ax, snapshot, time in zip(axes, theta, times):
        image = ax.imshow(
            snapshot.T,
            origin="lower",
            extent=(1, nx * args.coarsen, 1, ny * args.coarsen),
            cmap=cmap,
            vmin=0,
            vmax=2 * np.pi,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(rf"$t={time:g}$", pad=2)
        ax.set_xlabel(r"$x$")
        ax.set_aspect("equal")
        ax.set_xticks([50, 100, 150, 200])
        ax.set_yticks([50, 100, 150, 200])

    axes[0].set_ylabel(r"$y$")
    cbar = fig.colorbar(
        image,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.015,
        aspect=18,
    )
    cbar.set_label(r"$\theta$")
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"saved: {output}")

    if args.png_output:
        png_output = Path(args.png_output)
        png_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output, dpi=args.dpi)
        print(f"saved: {png_output}")


if __name__ == "__main__":
    main()
