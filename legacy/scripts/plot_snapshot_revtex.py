#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import tempfile

cache_dir = Path(tempfile.gettempdir()) / "flocking-matplotlib-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


REVTEX_TWO_COLUMN_WIDTH_IN = 510.0 / 72.27
FIGURE_TEXT_SIZE = 9.0
FIGURE_TICK_SIZE = 9.0

mpl.rcParams.update(
    {
        "figure.dpi": 720,
        "savefig.dpi": 720,
        "text.usetex": True,
        "axes.unicode_minus": False,
        "font.size": FIGURE_TEXT_SIZE,
        "axes.labelsize": FIGURE_TEXT_SIZE,
        "axes.titlesize": FIGURE_TEXT_SIZE,
        "xtick.labelsize": FIGURE_TICK_SIZE,
        "ytick.labelsize": FIGURE_TICK_SIZE,
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "mathtext.default": "rm",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }
)


def load_dataset(path):
    with h5py.File(path, "r") as handle:
        dataset = handle["dataset"][()]
        times = handle[dataset["times"]][()]
        theta = handle[dataset["theta_snapshots"]][()]
        block_x = handle[dataset["block_x"]][()]
        block_y = handle[dataset["block_y"]][()]
        block_u = handle[dataset["block_u"]][()]
        block_v = handle[dataset["block_v"]][()]
    return times, theta, block_x, block_y, block_u, block_v


def circular_cmap(n=256, saturation=0.70, lightness=0.56):
    colors = sns.hls_palette(n, h=0.01, l=lightness, s=saturation)
    return ListedColormap(colors + [colors[0]], name="red_green_cyclic")


def block_mean(field, block_size):
    if block_size == 1:
        return field
    ny, nx = field.shape
    if ny % block_size != 0 or nx % block_size != 0:
        raise ValueError("coarsen must divide both snapshot dimensions")
    return field.reshape(ny // block_size, block_size, nx // block_size, block_size).mean(axis=(1, 3))


def math_ticklabels(values):
    labels = []
    for value in values:
        if float(value).is_integer():
            labels.append(rf"${int(value)}$")
        else:
            labels.append(rf"${value:g}$")
    return labels


def add_axes_inches(fig, left, bottom, width, height, **kwargs):
    fig_width, fig_height = fig.get_size_inches()
    return fig.add_axes(
        [left / fig_width, bottom / fig_height, width / fig_width, height / fig_height],
        **kwargs,
    )


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
    parser.add_argument("--saturation", type=float, default=0.70)
    parser.add_argument("--lightness", type=float, default=0.56)
    parser.add_argument("--arrow-scale", type=float, default=7.0)
    parser.add_argument("--arrow-width", type=float, default=0.0022)
    parser.add_argument("--coarsen-arrows", type=int, default=1)
    args = parser.parse_args()

    times, theta, block_x, block_y, block_u, block_v = load_dataset(args.input)
    field = np.mod(theta, 2 * np.pi)
    if args.coarsen > 1:
        coarse_theta = []
        for snapshot in theta:
            c = block_mean(np.cos(snapshot), args.coarsen)
            s = block_mean(np.sin(snapshot), args.coarsen)
            coarse_theta.append(np.mod(np.arctan2(s, c), 2 * np.pi))
        field = np.stack(coarse_theta)

    arrow_x = block_x
    arrow_y = block_y
    arrow_u = block_u
    arrow_v = block_v

    nsnapshots, ny, nx = field.shape
    if nsnapshots != 4:
        raise ValueError(f"expected 4 snapshots, got {nsnapshots}")
    if args.coarsen_arrows < 1:
        raise ValueError("--coarsen-arrows must be at least 1")

    cmap = circular_cmap(saturation=args.saturation, lightness=args.lightness)
    fig_width = REVTEX_TWO_COLUMN_WIDTH_IN
    left = 0.48
    right = 0.43
    bottom = 0.42
    top = 0.26
    panel_gap = 0.085
    cbar_gap = 0.085
    cbar_width = 0.095
    panel_size = (fig_width - left - right - cbar_width - cbar_gap - 3 * panel_gap) / 4
    fig_height = bottom + panel_size + top
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = []
    for i in range(4):
        x0 = left + i * (panel_size + panel_gap)
        if i == 0:
            ax = add_axes_inches(fig, x0, bottom, panel_size, panel_size)
        else:
            ax = add_axes_inches(fig, x0, bottom, panel_size, panel_size, sharey=axes[0])
        axes.append(ax)
    cbar_x0 = left + 4 * panel_size + 3 * panel_gap + cbar_gap
    cax = add_axes_inches(fig, cbar_x0, bottom, cbar_width, panel_size)

    image = None
    extent_max = nx * args.coarsen
    tick_values = [50, 100, 150, 200] if extent_max >= 200 else [20, 40, 60, 80]
    for k, (ax, snapshot, time) in enumerate(zip(axes, field, times)):
        image = ax.imshow(
            snapshot,
            origin="lower",
            extent=(1, extent_max, 1, ny * args.coarsen),
            cmap=cmap,
            vmin=0,
            vmax=2 * np.pi,
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(rf"$t={time:g}$", pad=2)
        ax.set_xlabel(r"$x$")
        ax.set_aspect("equal")
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xticklabels(math_ticklabels(tick_values), fontsize=FIGURE_TICK_SIZE)
        ax.set_yticklabels(math_ticklabels(tick_values), fontsize=FIGURE_TICK_SIZE)
        step = args.coarsen_arrows
        ax.quiver(
            arrow_x[::step, ::step],
            arrow_y[::step, ::step],
            args.arrow_scale * arrow_u[k][::step, ::step],
            args.arrow_scale * arrow_v[k][::step, ::step],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=args.arrow_width,
            color="black",
            pivot="mid",
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
            minlength=0.0,
            zorder=3,
        )

    axes[0].set_ylabel(r"$y$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    cbar = fig.colorbar(image, cax=cax, orientation="vertical")
    cbar.set_label(r"$\theta$", labelpad=2.0)
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    cbar.ax.tick_params(labelsize=FIGURE_TICK_SIZE)

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
