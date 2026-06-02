#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import ScalarFormatter


# RevTeX 4.2 single-column width in inches.
COLUMN_WIDTH = 3.375
FIGURE_HEIGHT = 2.45
OUTPUT_STEM = "F-correlator"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "results" / "spin_aligned_f_correlator_L200_J2_v1_gamma1_20260520_153353"
FIGURE_TEXT_SIZE = 10.0
FIGURE_TICK_SIZE = 9.0
COLLAPSE_ETA = 0.3100
COLLAPSE_ZETA = 0.3900


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


def load_reference_array(group: h5py.File, ref) -> np.ndarray:
    return np.array(group[ref][()])


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        result = handle["result"][()]
        radii = load_reference_array(handle, result["radii"])
        times = load_reference_array(handle, result["times"])
        # JLD2 arrays are column-major; h5py exposes them transposed.
        F_mean = load_reference_array(handle, result["F_mean"]).T
    return radii, times, F_mean


def load_ensemble(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(data_dir.glob("job_*.jld2"))
    if not files:
        raise FileNotFoundError(f"no job_*.jld2 files found in {data_dir}")

    radii, times, first = load_run(files[0])
    stack = [first]
    for path in files[1:]:
        run_radii, run_times, run_F = load_run(path)
        if not (np.array_equal(run_radii, radii) and np.array_equal(run_times, times)):
            raise ValueError(f"incompatible radii/times in {path}")
        stack.append(run_F)

    ensemble = np.stack(stack, axis=0)
    F_mean = ensemble.mean(axis=0)
    F_stderr = ensemble.std(axis=0, ddof=1) / np.sqrt(ensemble.shape[0])
    return radii, times, F_mean, F_stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the paper-style spin-aligned F-correlator figure.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-stem", type=Path, default=Path(__file__).resolve().parent / OUTPUT_STEM)
    parser.add_argument("--radius-max", type=float, default=30.0)
    parser.add_argument("--eta", type=float, default=COLLAPSE_ETA)
    parser.add_argument("--zeta", type=float, default=COLLAPSE_ZETA)
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="Draw point markers on the raw traces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    here.joinpath(".mplcache").mkdir(exist_ok=True)
    radii, times, F_mean, F_stderr = load_ensemble(args.data_dir)
    radius_mask = radii <= args.radius_max
    time_mask = times > 0.0

    plot_radii = radii[radius_mask]
    plot_times = times[time_mask]
    plot_mean = F_mean[radius_mask][:, time_mask].T
    plot_stderr = F_stderr[radius_mask][:, time_mask].T
    colors = sns.color_palette("flare", n_colors=len(plot_times))
    cmap = ListedColormap(colors)
    if len(plot_times) > 1:
        spacing = np.diff(plot_times)
        lower_edge = plot_times[0] - 0.5 * spacing[0]
        upper_edge = plot_times[-1] + 0.5 * spacing[-1]
        inner_edges = 0.5 * (plot_times[:-1] + plot_times[1:])
        boundaries = np.concatenate(([lower_edge], inner_edges, [upper_edge]))
    else:
        boundaries = np.array([plot_times[0] - 0.5, plot_times[0] + 0.5])
    norm = BoundaryNorm(boundaries, cmap.N)

    fig = plt.figure(figsize=(COLUMN_WIDTH, FIGURE_HEIGHT), constrained_layout=False)
    ax = fig.add_axes([0.13, 0.17, 0.70, 0.765])

    for color, time, trace, err in zip(colors, plot_times, plot_mean, plot_stderr):
        ax.fill_between(
            plot_radii,
            trace - err,
            trace + err,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        ax.plot(
            plot_radii,
            trace,
            color=color,
            linewidth=1.2,
            marker="o" if args.show_markers else None,
            markersize=3.2 if args.show_markers else 0.0,
            markeredgewidth=0.0,
        )

    inset = ax.inset_axes([0.52, 0.49, 0.43, 0.41])
    collapse_xmin = np.inf
    collapse_xmax = -np.inf
    collapse_ymin = np.inf
    collapse_ymax = -np.inf
    for color, time, trace in zip(colors, plot_times, plot_mean):
        x_scaled = plot_radii / (time ** args.zeta)
        y_scaled = trace * (time ** args.eta)
        inset.plot(x_scaled, y_scaled, color=color, linewidth=1.2)
        collapse_xmin = min(collapse_xmin, np.min(x_scaled))
        collapse_xmax = max(collapse_xmax, np.max(x_scaled))
        collapse_ymin = min(collapse_ymin, np.min(y_scaled))
        collapse_ymax = max(collapse_ymax, np.max(y_scaled))

    inset.set_xlim(collapse_xmin, collapse_xmax)
    inset.set_ylim(collapse_ymin - 0.05 * (collapse_ymax - collapse_ymin),
        collapse_ymax + 0.05 * (collapse_ymax - collapse_ymin))
    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(length=0)
    inset.set_xlabel(rf"$r/t^{{{args.zeta:.3f}}}$", labelpad=1.5)
    inset.set_ylabel(rf"$t^{{{args.eta:.3f}}} F$", labelpad=1.5)
    inset_formatter = ScalarFormatter(useMathText=True)
    inset_formatter.set_powerlimits((-3, -3))
    inset.yaxis.set_major_formatter(inset_formatter)
    inset.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    inset.yaxis.get_offset_text().set_visible(False)
    inset.text(0.0, 1.002, r"$\times 10^{-3}$", transform=inset.transAxes,
        ha="left", va="bottom", fontsize=FIGURE_TICK_SIZE)

    ax.axhline(0.0, color="0.75", linewidth=0.6, zorder=0)
    ax.set_xlim(1, args.radius_max)
    ax.set_xticks(np.arange(5, args.radius_max + 0.1, 5))
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$F(r,t)$")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(0.0, 1.002, r"$\times 10^{-3}$", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=FIGURE_TICK_SIZE)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.85, 0.17, 0.024, 0.765])
    cbar = fig.colorbar(sm, cax=cax, boundaries=boundaries, ticks=plot_times,
        spacing="proportional")
    cbar.set_label(r"$t$", labelpad=5.0)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=FIGURE_TICK_SIZE)

    output_stem = args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"))


if __name__ == "__main__":
    main()
