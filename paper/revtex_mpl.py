"""RevTeX paper-figure helpers for matplotlib.

Goal: figures that are EXACTLY the RevTeX column/text width, with serif LaTeX
text at the chosen point size, and all ink (tick labels, axis labels, colorbar
labels, insets, titles) flush to the canvas edge -- zero outer padding, nothing
clipped -- without hand-tuning margins and without ``bbox_inches="tight"``
(which would silently change the physical width).

Two layout paths:

* Free-aspect content (line traces, heatmaps whose box may stretch): place axes
  roughly, then call :func:`fit_to_canvas`. It keeps the figure size fixed and
  grows/shrinks every axes so the ink is flush. This replaces all manual margin
  tuning.

* Aspect-locked content (square ``imshow`` panels): the height is not free, so
  use :func:`panel_row`, which derives the figure height from the panel aspect
  and auto-solves the margins from a measured draw.

Colormap convention: sequential seaborn palettes only. Traces -> ``flare`` or
``rocket`` (default ``flare``). Heatmaps -> ``mako`` or ``crest``.

Copy this file next to your figure script so the figure is reproducible on its
own (e.g. ``paper/revtex_mpl.py``), then ``import revtex_mpl as rt``.
"""
from __future__ import annotations

import os
from pathlib import Path

# RevTeX 4.2 widths in inches. \columnwidth = 246 pt (single column);
# \textwidth = 510 pt (two-column full width). RevTeX uses 72.27 pt/inch.
PT_PER_INCH = 72.27
SINGLE_COLUMN = 3.375          # = 246 / 72.27, rounded as used in the repo
TWO_COLUMN = 510.0 / PT_PER_INCH  # ~= 7.057 in, full text width

# Default aesthetic aspect for a single-column free-aspect figure (W:H).
DEFAULT_SINGLE_HEIGHT = 2.45


def pt_to_in(pt: float) -> float:
    """Convert a RevTeX length in points to inches (use \\the\\columnwidth)."""
    return pt / PT_PER_INCH


def apply_style(font_pt: float = 10.0, tick_pt: float = 9.0, dpi: int = 720,
                usetex: bool = True, cache_dir: str | os.PathLike | None = None) -> None:
    """Lock in the paper rcParams: serif Computer Modern via LaTeX, thin axes
    and ticks, high dpi. Call once before building a figure.

    ``font_pt`` sets text/axis-label size; ``tick_pt`` sets tick-label size.
    Set ``cache_dir`` (or rely on the default beside the cwd) to keep the
    matplotlib/usetex cache out of $HOME on clusters; must be set before the
    first matplotlib import in the process.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "text.usetex": usetex,
        "axes.unicode_minus": False,
        "font.size": font_pt,
        "axes.labelsize": font_pt,
        "axes.titlesize": font_pt,
        "xtick.labelsize": tick_pt,
        "ytick.labelsize": tick_pt,
        "legend.fontsize": font_pt,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.bbox": "standard",   # never 'tight': it would change the width
        "savefig.pad_inches": 0.0,
    })


# --- colormaps -------------------------------------------------------------

def trace_colors(n: int, name: str = "flare"):
    """``n`` discrete colors for line traces from a sequential seaborn palette.

    Default ``flare`` (warm); ``rocket`` is the other approved trace palette.
    Returns a list of RGB tuples ordered low->high, suitable for a per-line
    color and for building a ``ListedColormap`` for a matching colorbar.
    """
    import seaborn as sns
    if name not in ("flare", "rocket"):
        raise ValueError("trace palette must be 'flare' or 'rocket'")
    return sns.color_palette(name, n_colors=n)


def heatmap_cmap(name: str = "mako"):
    """Continuous colormap for heatmaps. ``mako`` (default) or ``crest``."""
    import seaborn as sns
    if name not in ("mako", "crest"):
        raise ValueError("heatmap palette must be 'mako' or 'crest'")
    return sns.color_palette(name, as_cmap=True)


# --- placement -------------------------------------------------------------

def add_axes_inches(fig, left, bottom, width, height, **kwargs):
    """``fig.add_axes`` with the rectangle given in INCHES rather than figure
    fractions. Lets you reason about layout in the same physical units as the
    RevTeX width."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh], **kwargs)


def _movable(ax) -> bool:
    """Top-level axes (main axes, manual colorbars) move; insets do not -- their
    _TransformedBoundsLocator already tracks the parent we are about to move."""
    loc = ax.get_axes_locator()
    return loc is None or type(loc).__name__ != "_TransformedBoundsLocator"


def fit_to_canvas(fig, pad: float = 0.0, max_iter: int = 10, tol: float = 5e-4) -> float:
    """Grow/shrink every (non-inset) axes uniformly so the figure's tight ink
    bounding box sits ``pad`` inches from each canvas edge, WITHOUT changing the
    figure's physical size. Returns the final residual in inches.

    Use for free-aspect figures (line traces, heatmaps, colorbars, insets). It
    applies one measured affine map to every axes position and iterates a few
    times to absorb the fixed-size text overhang; convergence is geometric.

    Do NOT use on aspect-locked panels (``ax.set_aspect('equal')``/``imshow``):
    a square box cannot stretch to fill an arbitrary canvas, leaving white
    bands. Use :func:`panel_row` for those.
    """
    fig.canvas.draw()
    fw, fh = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()
    movable = [ax for ax in fig.axes if _movable(ax)]
    residual = float("inf")
    for _ in range(max_iter):
        tb = fig.get_tightbbox(renderer)  # inches, figure coordinates
        residual = max(abs(tb.x0 - pad), abs(fw - tb.x1 - pad),
                       abs(tb.y0 - pad), abs(fh - tb.y1 - pad))
        if residual < tol:
            break
        sx = (fw - 2 * pad) / (tb.x1 - tb.x0)
        sy = (fh - 2 * pad) / (tb.y1 - tb.y0)
        for ax in movable:
            p = ax.get_position()
            nx0 = (pad + (p.x0 * fw - tb.x0) * sx) / fw
            nx1 = (pad + (p.x1 * fw - tb.x0) * sx) / fw
            ny0 = (pad + (p.y0 * fh - tb.y0) * sy) / fh
            ny1 = (pad + (p.y1 * fh - tb.y0) * sy) / fh
            ax.set_position([nx0, ny0, nx1 - nx0, ny1 - ny0])
        fig.canvas.draw()
    return residual


def panel_row(n: int, *, width: float = TWO_COLUMN, panel_aspect: float = 1.0,
              gap: float = 0.085, cbar: bool = False, cbar_width: float = 0.095,
              cbar_gap: float = 0.085, margins=(0.5, 0.1, 0.42, 0.26)):
    """Lay out a row of ``n`` equal panels at fixed total ``width`` (inches).

    ``panel_aspect`` is height/width of each panel; set it to the data aspect of
    your ``imshow`` (1.0 for a square field). The figure HEIGHT is derived so the
    panels hit that aspect exactly -- there is no free height to guess.

    ``margins`` = (left, right, bottom, top) in inches are only a starting
    guess; call :func:`fit_panels` after plotting to solve them from a measured
    draw. Returns ``(fig, axes, cax)`` where ``cax`` is ``None`` if ``cbar`` is
    False. Stores layout params on the figure for :func:`fit_panels`.
    """
    import matplotlib.pyplot as plt

    fig = _build_panel_row(width, n, panel_aspect, gap, cbar, cbar_width,
                           cbar_gap, margins, fig=None)
    return fig._panel_fig, fig._panel_axes, fig._panel_cax


def _build_panel_row(width, n, panel_aspect, gap, cbar, cbar_width, cbar_gap,
                     margins, fig):
    import matplotlib.pyplot as plt

    left, right, bottom, top = margins
    avail = width - left - right - (n - 1) * gap
    if cbar:
        avail -= cbar_width + cbar_gap
    pw = avail / n
    ph = pw * panel_aspect
    height = bottom + ph + top

    if fig is None:
        fig = plt.figure(figsize=(width, height))
        axes = [add_axes_inches(fig, left + i * (pw + gap), bottom, pw, ph)
                for i in range(n)]
        cax = None
        if cbar:
            cx = left + n * pw + (n - 1) * gap + cbar_gap
            cax = add_axes_inches(fig, cx, bottom, cbar_width, ph)
        fig._panel_axes = axes
        fig._panel_cax = cax
    else:
        fig.set_size_inches(width, height)
        for i, ax in enumerate(fig._panel_axes):
            _set_position_inches(ax, left + i * (pw + gap), bottom, pw, ph)
        if cbar and fig._panel_cax is not None:
            cx = left + n * pw + (n - 1) * gap + cbar_gap
            _set_position_inches(fig._panel_cax, cx, bottom, cbar_width, ph)

    fig._panel_fig = fig
    fig._panel_params = dict(width=width, n=n, panel_aspect=panel_aspect,
                             gap=gap, cbar=cbar, cbar_width=cbar_width,
                             cbar_gap=cbar_gap, margins=margins)
    return fig


def _set_position_inches(ax, left, bottom, width, height):
    fw, fh = ax.figure.get_size_inches()
    ax.set_position([left / fw, bottom / fh, width / fw, height / fh])


def fit_panels(fig, pad: float = 0.0, passes: int = 3) -> tuple:
    """Solve the :func:`panel_row` margins from a measured draw so ink is flush
    to ``pad`` inches on every side, then re-derive the height. Panels keep their
    aspect. Returns the final (left, right, bottom, top) margins in inches.
    """
    p = dict(fig._panel_params)
    for _ in range(passes):
        fig.canvas.draw()
        fw, fh = fig.get_size_inches()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        left, right, bottom, top = p["margins"]
        # ink overhang beyond the content block on each side (fixed-size text)
        new_left = (left - tb.x0) + pad
        new_right = (tb.x1 - (fw - right)) + pad
        new_bottom = (bottom - tb.y0) + pad
        new_top = (tb.y1 - (fh - top)) + pad
        new_margins = (new_left, new_right, new_bottom, new_top)
        if max(abs(a - b) for a, b in zip(new_margins, p["margins"])) < 5e-4:
            p["margins"] = new_margins
            break
        p["margins"] = new_margins
        _build_panel_row(fig=fig, **p)
    fig._panel_params = p
    return p["margins"]


# --- saving ----------------------------------------------------------------

def save(fig, stem, formats=("pdf", "png")) -> None:
    """Save to ``<stem>.<fmt>`` for each format. Never uses ``bbox_inches`` so
    the saved size equals the figure size exactly. ``stem`` is a path without
    extension."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"))
