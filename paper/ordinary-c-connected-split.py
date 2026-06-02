#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize, PowerNorm


# Match paper/F-correlator.py.
COLUMN_WIDTH = 3.375
FIGURE_HEIGHT = 2.30
FIGURE_TEXT_SIZE = 9.0
FIGURE_TICK_SIZE = 9.0
OUTPUT_STEM = "ordinary-c-connected-split"
SPLIT_COLOR = "#f6c19f"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PREFIX = (
    REPO_ROOT
    / "figures"
    / "ordinary_c_correlator_L200_J2_v0_v1_gamma1_20260525_142415"
    / "ordinary_c_connected_split_rmax30_tinterp8_sqrtcolor"
)


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
        "legend.fontsize": FIGURE_TEXT_SIZE,
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the paper-style connected ordinary C split heatmap.",
    )
    parser.add_argument("--data-prefix", type=Path, default=DATA_PREFIX)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_STEM,
    )
    parser.add_argument("--color-scale", choices=("linear", "mild", "sqrt", "quarter", "log"), default="linear")
    parser.add_argument("--cmap", default="mako")
    parser.add_argument("--r-max", type=float, default=20.0)
    parser.add_argument("--plot-transform", choices=("none", "log10"), default="none")
    parser.add_argument("--contours", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contour-levels", type=int, default=8)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    return parser.parse_args()


def load_split(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_path = prefix.with_name(prefix.name + "_x.csv")
    t_path = prefix.with_name(prefix.name + "_t.csv")
    c_path = prefix.with_name(prefix.name + "_C.csv")
    missing = [path for path in (x_path, t_path, c_path) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing exported split data: {joined}")

    x = np.loadtxt(x_path, delimiter=",")
    t = np.loadtxt(t_path, delimiter=",")
    c = np.loadtxt(c_path, delimiter=",")
    return np.atleast_1d(x), np.atleast_1d(t), np.atleast_2d(c)


def make_norm(values: np.ndarray, scale: str, vmin: float | None, vmax: float | None):
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    if scale == "linear":
        return Normalize(vmin=lo, vmax=hi)
    if scale == "mild":
        if lo < 0.0:
            raise ValueError("mild color scale requires nonnegative values")
        return PowerNorm(gamma=0.75, vmin=lo, vmax=hi)
    if scale == "sqrt":
        if lo < 0.0:
            raise ValueError("sqrt color scale requires nonnegative values")
        return PowerNorm(gamma=0.5, vmin=lo, vmax=hi)
    if scale == "quarter":
        if lo < 0.0:
            raise ValueError("quarter color scale requires nonnegative values")
        return PowerNorm(gamma=0.25, vmin=lo, vmax=hi)
    if lo <= 0.0:
        raise ValueError("log color scale requires positive values")
    return LogNorm(vmin=lo, vmax=hi)


def main() -> None:
    args = parse_args()
    x, t, c = load_split(args.data_prefix)
    x_mask = np.abs(x) <= args.r_max
    if not np.any(x_mask):
        raise ValueError(f"no x values satisfy --r-max {args.r_max}")
    x = x[x_mask]
    c = c[:, x_mask]
    if args.plot_transform == "log10":
        positive = c[c > 0.0]
        if positive.size == 0:
            raise ValueError("log10 plot transform requires positive values")
        c = np.log10(np.maximum(c, positive.min()))
    norm = make_norm(c, args.color_scale, args.vmin, args.vmax)
    cmap = sns.color_palette(args.cmap, as_cmap=True)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIGURE_HEIGHT), constrained_layout=False)
    fig.subplots_adjust(left=0.16, bottom=0.20, right=0.93, top=0.98)

    mesh = ax.pcolormesh(
        x,
        t,
        c,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    if args.contours:
        lo = np.nanmin(c) if args.vmin is None else args.vmin
        hi = np.nanmax(c) if args.vmax is None else args.vmax
        levels = np.linspace(lo, hi, args.contour_levels + 2)[1:-1]
        ax.contour(
            x,
            t,
            c,
            levels=levels,
            colors="white",
            linewidths=0.35,
            alpha=0.36,
        )
    ax.axvline(0.0, color=SPLIT_COLOR, linewidth=1.15)

    xmax = float(np.max(np.abs(x)))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(float(np.min(t)), float(np.max(t)))
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$t$")

    xticks = np.arange(-args.r_max, args.r_max + 0.1, 10)
    xticks = xticks[(xticks >= -xmax) & (xticks <= xmax)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([rf"${abs(int(tick))}$" for tick in xticks])
    ax.set_yticks([-10, 0, 10])

    ax.text(
        -0.50 * xmax,
        0.965,
        r"$v_0=0$",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color=SPLIT_COLOR,
        fontsize=FIGURE_TEXT_SIZE,
    )
    ax.text(
        0.50 * xmax,
        0.965,
        r"$v_0=1$",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color=SPLIT_COLOR,
        fontsize=FIGURE_TEXT_SIZE,
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar_label = (
        r"$\log_{10} C(r,t)$"
        if args.plot_transform == "log10"
        else r"$C(r,t)\equiv\langle\hat{\mathbf n}(r,t)\!\cdot\!\hat{\mathbf n}(0,0)\rangle_c$"
    )
    cbar.set_label(cbar_label, labelpad=3.0, fontsize=FIGURE_TEXT_SIZE)
    cbar.ax.yaxis.set_label_position("right")
    if args.plot_transform == "log10":
        log_ticks = np.arange(np.ceil(np.nanmin(c)), np.floor(np.nanmax(c)) + 1)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([rf"${tick:.0f}$" for tick in log_ticks])
    else:
        cbar.set_ticks([0.1, 0.2, 0.3])
        cbar.set_ticklabels([r"$0.1$", r"$0.2$", r"$0.3$"])
    cbar.ax.tick_params(labelsize=FIGURE_TICK_SIZE)

    output_stem = args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"))


if __name__ == "__main__":
    main()
