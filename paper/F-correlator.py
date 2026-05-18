#!/usr/bin/env python3

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
from matplotlib.ticker import ScalarFormatter


# RevTeX 4.2 single-column width in inches.
COLUMN_WIDTH = 3.375
FIGURE_HEIGHT = 2.45
OUTPUT_STEM = "F-correlator"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "results" / "spin_aligned_f_correlator_L200_J2_v1_gamma1"
FIGURE_TEXT_SIZE = 10.0
FIGURE_TICK_SIZE = 9.0
COLLAPSE_ETA = 0.31
COLLAPSE_ZETA = 0.375


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


def main() -> None:
    here = Path(__file__).resolve().parent
    here.joinpath(".mplcache").mkdir(exist_ok=True)
    radii, times, F_mean, F_stderr = load_ensemble(DATA_DIR)
    radius_mask = radii <= 30.0
    time_mask = times > 0.0

    plot_radii = radii[radius_mask]
    plot_times = times[time_mask]
    plot_mean = -F_mean[radius_mask][:, time_mask].T
    plot_stderr = F_stderr[radius_mask][:, time_mask].T
    cmap = sns.color_palette("flare", as_cmap=True)
    norm = mpl.colors.Normalize(vmin=plot_times.min(), vmax=plot_times.max())

    fig = plt.figure(figsize=(COLUMN_WIDTH, FIGURE_HEIGHT), constrained_layout=False)
    ax = fig.add_axes([0.13, 0.17, 0.70, 0.765])

    for time, trace, err in zip(plot_times, plot_mean, plot_stderr):
        color = cmap(norm(time))
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
            marker="o",
            markersize=3.2,
            markeredgewidth=0.0,
        )

    inset = ax.inset_axes([0.52, 0.49, 0.43, 0.41])
    collapse_xmin = np.inf
    collapse_xmax = -np.inf
    collapse_ymin = np.inf
    collapse_ymax = -np.inf
    for time, trace in zip(plot_times, plot_mean):
        x_scaled = plot_radii / (time ** COLLAPSE_ZETA)
        y_scaled = trace * (time ** COLLAPSE_ETA)
        color = cmap(norm(time))
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
    inset.set_xlabel(r"$r/t^{0.375}$", labelpad=1.5)
    inset.set_ylabel(r"$t^{0.31} F$", labelpad=1.5)
    inset_formatter = ScalarFormatter(useMathText=True)
    inset_formatter.set_powerlimits((-3, -3))
    inset.yaxis.set_major_formatter(inset_formatter)
    inset.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    inset.yaxis.get_offset_text().set_visible(False)
    inset.text(0.0, 1.002, r"$\times 10^{-3}$", transform=inset.transAxes,
        ha="left", va="bottom", fontsize=FIGURE_TICK_SIZE)

    ax.axhline(0.0, color="0.75", linewidth=0.6, zorder=0)
    ax.set_xlim(1, 30)
    ax.set_xticks([5, 10, 15, 20, 25, 30])
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
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$t$", labelpad=5.0)
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_ticks(plot_times)
    cbar.ax.tick_params(labelsize=FIGURE_TICK_SIZE)

    for suffix in ("pdf", "png"):
        fig.savefig(here / f"{OUTPUT_STEM}.{suffix}")


if __name__ == "__main__":
    main()
