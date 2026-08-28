#!/usr/bin/env python3

# Plot the late-time exponent together with the fit-window drift at one passive and
# one active velocity. The points and error bars remain unsmoothed. A three-point
# average only guides the connecting curve in the main panel.

import argparse
from pathlib import Path

import revtex_mpl as rt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
rt.apply_style(font_pt=10.0, tick_pt=9.0, cache_dir=HERE / ".mplcache")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import FixedFormatter, FixedLocator


parser = argparse.ArgumentParser(
    description="Render the late-time exponent and convergence panels.")
parser.add_argument(
    "--input-dir",
    type=Path,
    default=REPO_ROOT / "analysis" / "exponent_convergence",
)
parser.add_argument(
    "--output-stem",
    type=Path,
    default=HERE / "zeta-convergence",
)
parser.add_argument("--xmax", type=float, default=0.5)
parser.add_argument("--low-v", type=float, default=0.16)
parser.add_argument("--active-v", type=float, default=2.4)
args = parser.parse_args()

fits = np.genfromtxt(args.input_dir / "zeta_late_time.csv", delimiter=",", names=True)
convergence = np.genfromtxt(
    args.input_dir / "zeta_time_convergence.csv", delimiter=",", names=True)
temperatures = np.unique(fits["temperature"])
colors = ("#86AFCB", "#E3A878", "#89BEA5")

width = rt.TWO_COLUMN
height = 2.5
fig = plt.figure(figsize=(width, height))
ax_main = fig.add_axes([0.075, 0.19, 0.43, 0.73])
ax_low = fig.add_axes([0.59, 0.19, 0.18, 0.73])
ax_active = fig.add_axes([0.81, 0.19, 0.18, 0.73])

for temperature, color in zip(temperatures, colors):
    rows = fits[(fits["temperature"] == temperature) & (fits["log10_v"] <= args.xmax)]
    order = np.argsort(rows["log10_v"])
    rows = rows[order]
    x = rows["log10_v"]
    y = rows["zeta"]
    uncertainty = rows["uncertainty"]

    line_y = np.convolve(
        np.pad(y, 1, mode="edge"),
        np.array((0.25, 0.5, 0.25)),
        mode="valid",
    )
    dense_x = np.linspace(x.min(), x.max(), 500)
    ax_main.plot(
        dense_x,
        PchipInterpolator(x, line_y)(dense_x),
        color=color,
        linewidth=1.35,
        zorder=2,
    )
    ax_main.errorbar(
        x,
        y,
        yerr=uncertainty,
        fmt="o",
        linestyle="none",
        color=color,
        ecolor=color,
        elinewidth=0.7,
        capsize=1.6,
        capthick=0.7,
        markersize=3.3,
        markeredgecolor="white",
        markeredgewidth=0.35,
        label=rf"$T={temperature:g}$",
        zorder=3,
    )

velocity_values = np.unique(fits["v"])
low_v = velocity_values[np.argmin(np.abs(velocity_values - args.low_v))]
active_v = velocity_values[np.argmin(np.abs(velocity_values - args.active_v))]

for axis, velocity, target in (
    (ax_low, low_v, 0.5),
    (ax_active, active_v, 3 / 8),
):
    for temperature, color in zip(temperatures, colors):
        rows = convergence[
            (convergence["temperature"] == temperature)
            & np.isclose(convergence["v"], velocity)
        ]
        order = np.argsort(rows["t_min"])
        axis.plot(
            rows["t_min"][order],
            rows["zeta"][order],
            color=color,
            linewidth=1.1,
            marker="o",
            markersize=3.0,
            markeredgecolor="white",
            markeredgewidth=0.3,
        )
    axis.axhline(target, color="0.35", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_xscale("log")
    axis.set_xlim(1.8, 7.2)
    axis.xaxis.set_major_locator(FixedLocator((2.0, 4.0)))
    axis.xaxis.set_major_formatter(FixedFormatter((r"$2$", r"$4$")))
    axis.xaxis.set_minor_locator(FixedLocator(()))
    axis.set_ylim(0.30, 0.56)
    axis.set_xlabel(r"$t_{\min}$")
    axis.set_title(rf"$v={velocity:.3g}$", pad=3.0)
    axis.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)

ax_main.axhline(0.5, color="0.35", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
ax_main.axhline(3 / 8, color="0.35", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
ax_main.text(args.xmax - 0.02, 0.503, r"$1/2$", ha="right", va="bottom", fontsize=9.0)
ax_main.text(
    args.xmax - 0.02,
    3 / 8 + 0.003,
    r"$3/8$",
    ha="right",
    va="bottom",
    fontsize=9.0,
)
ax_main.set_xlim(-1.02, args.xmax)
ax_main.set_ylim(0.30, 0.56)
ax_main.set_xticks((-1.0, -0.5, 0.0, 0.5))
ax_main.set_xlabel(r"$\log_{10} v$")
ax_main.set_ylabel(r"$\zeta$")
ax_main.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
ax_main.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=False,
    handletextpad=0.3,
    columnspacing=0.8,
)

ax_low.set_ylabel(r"$\zeta$")
ax_active.tick_params(labelleft=False)
for label, axis in zip((r"$(a)$", r"$(b)$", r"$(c)$"),
        (ax_main, ax_low, ax_active)):
    axis.text(0.02, 0.97, label, transform=axis.transAxes,
        ha="left", va="top", fontsize=10.0)

residual = rt.fit_to_canvas(fig, pad=0.0)
rt.save(fig, args.output_stem)
print(
    f"wrote {args.output_stem}.pdf and {args.output_stem}.png, "
    f"residual={residual:.6g} in"
)
