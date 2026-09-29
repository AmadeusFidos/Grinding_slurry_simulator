"""Chart styling: one palette, two themes, applied everywhere.

Colour is assigned by the job it does rather than by series order:

*   **Grit size is ordered**, so grits get steps of a single blue ramp -- coarse
    pale, fine dark -- and every line is labelled directly, so the reader never
    has to decode a colour to know which grit they are looking at.
*   **Magnitude** (removal rate, damage depth) gets the same one-hue ramp as a
    continuous colour map.
*   **Feasibility has polarity** -- a recipe is inside the window or outside it --
    so it gets a diverging red/blue map with a neutral grey at exactly zero.  Grey
    means "on the boundary", which is what you want to see.
*   Genuinely unordered series (water / glycerine / soap) get the three
    categorical hues, which are validated for all-pairs colour-blind separation.

Both themes are stepped for their own surface, not flipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    categorical: tuple[str, ...]
    grit_ramp: tuple[str, ...]
    """Four steps, pale to dark, for coarse-to-fine grit."""

    sequential: tuple[str, ...]
    diverging_low: tuple[str, ...]
    """Dark to light red -- the "outside the window" arm."""

    diverging_high: tuple[str, ...]
    """Light to dark blue -- the "inside the window" arm."""

    neutral: str
    good: str
    warning: str
    critical: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb", page="#f9f9f7",
    ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    categorical=("#2a78d6", "#eb6834", "#1baf7a"),
    grit_ramp=("#86b6ef", "#3987e5", "#1c5cab", "#0d366b"),
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"),
    diverging_low=("#7a1d1c", "#b52d2c", "#e34948", "#ef8585", "#f5adad"),
    diverging_high=("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"),
    neutral="#f0efec",
    good="#0ca30c", warning="#fab219", critical="#d03b3b",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19", page="#0d0d0d",
    ink="#ffffff", ink_secondary="#c3c2b7", ink_muted="#898781",
    grid="#2c2c2a", axis="#383835",
    categorical=("#3987e5", "#d95926", "#199e70"),
    grit_ramp=("#6da7ec", "#3987e5", "#256abf", "#184f95"),
    sequential=("#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"),
    diverging_low=("#8f2322", "#c23b3a", "#e66767", "#ef8585", "#f5adad"),
    diverging_high=("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"),
    neutral="#383835",
    good="#0ca30c", warning="#fab219", critical="#d03b3b",
)

THEMES = {"light": LIGHT, "dark": DARK}


def apply(theme: Theme) -> None:
    """Push a theme into matplotlib's global style."""
    mpl.rcParams.update({
        "figure.facecolor": theme.page,
        "figure.edgecolor": theme.page,
        "savefig.facecolor": theme.page,
        "axes.facecolor": theme.surface,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.ink_secondary,
        "axes.titlecolor": theme.ink,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": theme.grid,
        "grid.linewidth": 0.6,
        "text.color": theme.ink,
        "xtick.color": theme.ink_muted,
        "ytick.color": theme.ink_muted,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.labelcolor": theme.ink_secondary,
        "ytick.labelcolor": theme.ink_secondary,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def sequential_cmap(theme: Theme, reverse: bool = False):
    """One-hue ramp for continuous magnitude (heat maps)."""
    colors = list(theme.sequential)
    if reverse:
        colors.reverse()
    return LinearSegmentedColormap.from_list(f"seq_{theme.name}", colors)


def diverging_cmap(theme: Theme):
    """Two hues with a neutral grey midpoint, for signed quantities.

    Used for the recipe window, where the sign of the margin is the whole point:
    red outside, grey exactly on the boundary, blue inside.
    """
    colors = list(theme.diverging_low) + [theme.neutral] + list(theme.diverging_high)
    return LinearSegmentedColormap.from_list(f"div_{theme.name}", colors)


def centred_norm(data: np.ndarray, centre: float = 0.0,
                 max_span: float | None = None) -> TwoSlopeNorm:
    """A norm that pins ``centre`` to the colour map's neutral midpoint.

    ``max_span`` caps how far out the ends reach.  Without it a few wildly
    infeasible points set the scale and wash out everything near the boundary,
    which is the only part anyone reads.
    """
    low = float(np.nanmin(data))
    high = float(np.nanmax(data))
    span = max(abs(low - centre), abs(high - centre), 1e-6)
    if max_span is not None:
        span = min(span, max_span)
    return TwoSlopeNorm(vmin=centre - span, vcenter=centre, vmax=centre + span)


MARGIN_STOPS = (-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0)
MARGIN_COLOURS = (
    "#7a1d1c", "#e34948", "#f5adad",
    "#f0efec",
    "#9ec5f4", "#3987e5", "#104281",
)
"""The diverging red - grey - blue ramp, as plain hex.

Kept here rather than in the plotting or GUI code because both draw the recipe
window -- matplotlib in the figures, and the desktop window on a canvas -- and
the two must agree. Pure arithmetic, so it imports nothing heavy.
"""


def margin_hex(margin: float) -> str:
    """A ``#rrggbb`` colour for a constraint margin, clipped to +-1.

    Grey sits exactly at zero, so the edge of the workable region is wherever the
    colour goes neutral. Clipping matters: a few hopeless mixes would otherwise
    set the scale and wash out the boundary, which is the only part worth
    reading.
    """
    value = max(-1.0, min(1.0, margin))
    for index in range(len(MARGIN_STOPS) - 1):
        low, high = MARGIN_STOPS[index], MARGIN_STOPS[index + 1]
        if value <= high:
            span = high - low
            t = 0.0 if span <= 0 else (value - low) / span
            start = MARGIN_COLOURS[index].lstrip("#")
            end = MARGIN_COLOURS[index + 1].lstrip("#")
            channels = (
                round(int(start[i:i + 2], 16)
                      + (int(end[i:i + 2], 16) - int(start[i:i + 2], 16)) * t)
                for i in (0, 2, 4)
            )
            return "#%02x%02x%02x" % tuple(channels)
    return MARGIN_COLOURS[-1]


def grit_colors(theme: Theme, count: int) -> list[str]:
    """``count`` steps of the grit ramp, coarse (pale) to fine (dark).

    Never cycled: if more grades than ramp steps are asked for, the ramp is
    resampled rather than repeated, so colour always tracks grain size.
    """
    ramp = LinearSegmentedColormap.from_list("grit", list(theme.grit_ramp))
    if count == 1:
        return [theme.grit_ramp[1]]
    return [mpl.colors.to_hex(ramp(i / (count - 1))) for i in range(count)]


def label_line(ax, x, y, text: str, color: str, theme: Theme,
               dx: float = 4.0, dy: float = 0.0, ha: str = "left",
               va: str = "center", size: float = 8.0) -> None:
    """Put a direct label at the end of a line.

    Direct labelling is what lets these charts stay readable when a colour is
    hard to tell apart -- and three of the light-mode hues sit below 3:1 contrast
    on the surface, so it is required rather than optional.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
        color=color, fontsize=size, fontweight="semibold", ha=ha, va=va,
        clip_on=False,
    )


def caption(fig, text: str, theme: Theme) -> None:
    """A single explanatory line under a figure."""
    fig.text(0.008, -0.012, text, color=theme.ink_muted, fontsize=7.7,
             va="top", ha="left", wrap=True)


def title_block(fig, title: str, subtitle: str, theme: Theme,
                lines: int = 1) -> float:
    """Left-aligned title and deck above the axes; returns the axes' top edge.

    Placed as figure text rather than a suptitle, so ``tight_layout(rect=...)``
    cannot slide the two on top of each other, and positioned in *inches* from
    the top rather than in figure fractions -- otherwise the gap between them
    grows with figure height and a tall figure ends up with its deck sitting in
    the middle of the first row of panels.

    Pass ``lines`` for a subtitle that wraps onto more than one line.
    """
    height = fig.get_size_inches()[1]
    fig.text(0.008, 1.0 - 0.30 / height, title, ha="left", va="top",
             fontsize=13.5, fontweight="bold", color=theme.ink)
    fig.text(0.008, 1.0 - 0.56 / height, subtitle, ha="left", va="top",
             fontsize=9, color=theme.ink_secondary)
    return 1.0 - (0.72 + 0.17 * (lines - 1)) / height
