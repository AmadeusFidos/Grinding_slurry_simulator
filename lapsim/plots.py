"""Figures.

Six charts, each answering one question:

1.  :func:`plot_vehicle` -- what the water/glycerine/soap mix does.
2.  :func:`plot_recipe_windows` -- where the workable recipes are, per grit.
3.  :func:`plot_drying` -- whether a recipe can dry to a crust.
4.  :func:`plot_contact` -- what the plate gap actually looks like.
5.  :func:`plot_grit_comparison` -- cut rate, finish and damage across grits.
6.  :func:`plot_sequence` -- the progression, and how long each stage takes.

Plus an optional animation, :func:`animate_gap`.

All of them honour :mod:`lapsim.style`, so ``theme="dark"`` restyles everything
against a dark surface with colours stepped for it.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from lapsim import style
from lapsim.contact import solve_contact
from lapsim.drying import dry_out
from lapsim.materials import ABRASIVES, stock_grades
from lapsim.recipes import ScanResult, SequencePlan, find_recipe, scan_window
from lapsim.simulate import Setup, simulate
from lapsim.slurry import LOW_SHEAR_RATE
from lapsim.vehicle import (
    DROP_VOLUME_ML,
    Vehicle,
    glycerine_fraction_at_activity,
    glycerol_water_viscosity,
    surface_tension,
    water_activity,
)
from lapsim.paths import FIGURE_DIR


def _save(fig, name: str, theme: style.Theme, directory: Path | None = None) -> Path:
    directory = Path(directory or FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme.name == "light" else f"_{theme.name}"
    path = directory / f"{name}{suffix}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1. The vehicle
# --------------------------------------------------------------------------- #


def plot_vehicle(setup: Setup | None = None, theme: str = "light",
                 directory: Path | None = None) -> Path:
    """What glycerine and soap actually do to the liquid."""
    setup = setup or Setup()
    th = style.THEMES[theme]
    style.apply(th)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    top = style.title_block(
        fig,
        "What the carrier liquid does",
        "Glycerine sets viscosity and how far the slurry can dry; soap sets "
        "wetting. Both saturate, in opposite directions.",
        th,
    )

    glycerine = np.linspace(0, 0.95, 200)

    # -- (a) viscosity, which is the reason glycerine is in the recipe at all --
    ax = axes[0]
    temperatures = [15.0, 21.0, 27.0]
    for color, temperature in zip(th.categorical, temperatures):
        viscosity = [glycerol_water_viscosity(g, temperature) * 1e3 for g in glycerine]
        ax.plot(glycerine * 100, viscosity, color=color,
                label=f"{temperature:.0f} °C")
        style.label_line(ax, 95, viscosity[-1], f"{temperature:.0f}°C", color, th)
    ax.set_yscale("log")
    ax.set_xlabel("glycerine in the liquid (wt %)")
    ax.set_ylabel("viscosity (mPa·s)")
    ax.set_title("a.  Viscosity is steeply non-linear")
    ax.set_xlim(0, 118)
    ax.legend(loc="upper left", title="liquid temperature",
              title_fontsize=8, labelcolor=th.ink_secondary)
    ax.annotate(
        "a cool room roughly\ndoubles the viscosity",
        xy=(84, glycerol_water_viscosity(0.84, 15.0) * 1e3),
        xytext=(30, 260), fontsize=7.8, color=th.ink_secondary,
        arrowprops=dict(arrowstyle="->", color=th.ink_muted, lw=0.8),
    )

    # -- (b) water activity: the ceiling on how far the slurry can dry --------
    ax = axes[1]
    activity = np.array([water_activity(g) for g in glycerine])
    ax.plot(glycerine * 100, activity * 100, color=th.grit_ramp[2])
    ax.set_xlabel("glycerine in the liquid (wt %)")
    ax.set_ylabel("water activity (%)")
    ax.set_title("b.  Evaporation stops at room humidity")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)

    for color, humidity in zip(th.categorical, (0.35, 0.50, 0.65)):
        endpoint = glycerine_fraction_at_activity(humidity) * 100
        ax.plot([0, endpoint], [humidity * 100] * 2, color=color, lw=1.2, ls=":")
        ax.plot([endpoint] * 2, [0, humidity * 100], color=color, lw=1.2, ls=":")
        ax.plot([endpoint], [humidity * 100], "o", color=color, ms=6,
                mec=th.surface, mew=1.5)
        style.label_line(ax, endpoint, humidity * 100 + 4,
                         f"{humidity * 100:.0f} % RH → {endpoint:.0f} wt%",
                         color, th, dx=-6, ha="right")
    ax.annotate(
        "the liquid ends here,\nnot dry",
        xy=(88, 46), xytext=(30, 22), fontsize=7.8, color=th.ink_secondary,
        arrowprops=dict(arrowstyle="->", color=th.ink_muted, lw=0.8),
    )

    # -- (c) soap: all of its effect happens in the first drop ---------------
    ax = axes[2]
    drops = np.logspace(-1.3, 1.8, 200)
    fraction = drops * DROP_VOLUME_ML / 100.0
    tension = np.array([surface_tension(0.4, c) for c in fraction])
    ax.plot(drops, tension * 1e3, color=th.grit_ramp[2])
    ax.set_xscale("log")
    ax.set_xlabel("dish soap (drops per 100 ml of liquid)")
    ax.set_ylabel("surface tension (mN/m)")
    ax.set_title("c.  Soap saturates after about one drop")
    ax.set_ylim(22, 76)
    ax.set_xlim(drops[0], drops[-1])

    cmc_drops = 2.0e-4 * 100.0 / DROP_VOLUME_ML
    ax.axvline(cmc_drops, color=th.ink_muted, lw=1.0, ls="--")
    ax.annotate("micelles form here", xy=(cmc_drops, 60),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7.8, color=th.ink_secondary, va="center")
    ax.axvspan(8, drops[-1], color=th.critical, alpha=0.10, lw=0)
    ax.annotate("past here it just foams,\nand the air unloads the grains",
                xy=(22, 25), fontsize=7.6, color=th.ink_secondary, ha="center")

    style.caption(
        fig,
        "Glycerol-water viscosity and density from Cheng (2008); water activity "
        "from Raoult's law with a Margules correction fitted to published "
        "glycerol equilibrium-humidity tables (within ~2 %). Surface tension is "
        "a Langmuir saturation through the critical micelle concentration.",
        th,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, "fig1_vehicle", th, directory)


# --------------------------------------------------------------------------- #
# 2. The recipe window
# --------------------------------------------------------------------------- #


def plot_recipe_windows(base: Setup | None = None,
                        grits: list[tuple[str, str]] | None = None,
                        theme: str = "light", resolution: int = 24,
                        directory: Path | None = None) -> Path:
    """Map the workable region for each grit, and mark the best recipe in it."""
    base = base or Setup()
    grits = grits or [(a.key, g.label) for a, g in stock_grades()]
    th = style.THEMES[theme]
    style.apply(th)

    columns = 2
    rows = math.ceil(len(grits) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.1 * columns, 4.5 * rows),
                             squeeze=False, layout="constrained")
    top = style.title_block(
        fig,
        "Where the workable recipes are",
        "Blue is inside every target band, red outside, grey exactly on the "
        "boundary. The marker is the most robust mix — the one furthest from "
        "every edge.",
        th,
    )

    cmap = style.diverging_cmap(th)
    scans: list[ScanResult] = []

    for index, (abrasive_key, grit) in enumerate(grits):
        ax = axes[index // columns][index % columns]
        scan = scan_window(
            replace(base, abrasive=abrasive_key, grit=grit),
            n_solids=resolution, n_glycerine=resolution,
        )
        scans.append(scan)

        x = scan.solids_mass_fractions * 100
        y = scan.glycerine_mass_fractions * 100
        # Clipped to +-1: beyond that a recipe is simply wrong, and letting
        # those values set the scale would wash out the boundary region.
        shown = np.clip(scan.robustness, -1.0, 1.0)
        mesh = ax.pcolormesh(
            x, y, shown, cmap=cmap,
            norm=style.centred_norm(shown, max_span=1.0), shading="gouraud",
        )
        # The window boundary: everything inside is usable.
        ax.contour(x, y, scan.robustness, levels=[0.0],
                   colors=[th.ink], linewidths=1.4)

        rate_levels = np.linspace(
            np.nanpercentile(scan.removal_rate_um_min, 20),
            np.nanpercentile(scan.removal_rate_um_min, 95), 5,
        )
        contours = ax.contour(x, y, scan.removal_rate_um_min,
                              levels=np.unique(rate_levels.round(1)),
                              colors=[th.ink], linewidths=0.6,
                              linestyles=":", alpha=0.45)
        ax.clabel(contours, fmt="%.0f \u00b5m/min", fontsize=6.4,
                  colors=th.ink_secondary)

        best = find_recipe(replace(base, abrasive=abrasive_key, grit=grit),
                           n_solids=18, n_glycerine=18)
        if best is not None:
            bx = best.setup.solids_mass_fraction * 100
            by = best.setup.glycerine_mass_fraction * 100
            ax.plot([bx], [by], marker="o", ms=10, color=th.surface,
                    mec=th.ink, mew=1.8, zorder=5)
            ax.plot([bx], [by], marker="o", ms=4, color=th.ink, zorder=6)
            recipe = best.recipe
            drops = max(round(recipe.soap_drops), 1)
            # Flip the callout to whichever side has room.
            on_left = bx < 0.5 * (x[0] + x[-1])
            ax.annotate(
                f"{recipe.grit_grams:.0f} g grit\n"
                f"{recipe.water_ml:.0f} ml water\n"
                f"{recipe.glycerine_ml:.0f} ml glycerine\n"
                f"{drops} drop{'s' if drops != 1 else ''} of soap",
                xy=(bx, by),
                xytext=(10 if on_left else -10, -8),
                textcoords="offset points",
                fontsize=7.4, color=th.ink, va="top",
                ha="left" if on_left else "right",
                bbox=dict(boxstyle="round,pad=0.35", fc=th.surface,
                          ec=th.axis, lw=0.7, alpha=0.94),
                zorder=7,
            )
        else:
            ax.annotate(
                "No mix works here.\nThe gap is small enough that the liquid\n"
                "needed to stop it running off also floats\nthe stone. "
                "Stroke slower, or press harder.",
                xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                va="center", fontsize=8.4, color=th.ink,
                bbox=dict(boxstyle="round,pad=0.5", fc=th.surface, ec=th.axis,
                          lw=0.8, alpha=0.94), zorder=7,
            )

        abrasive = ABRASIVES[abrasive_key]
        grade = abrasive.grade(grit)
        stock = "" if grade.in_stock else "  · not in stock"
        ax.set_title(
            f"{abrasive.name}  {grit}   (d50 {grade.d50_um:.0f} µm){stock}"
        )
        ax.set_xlabel("grit loading (wt % of the slurry)")
        ax.set_ylabel("glycerine in the liquid (wt %)")
        ax.grid(False)

        share = scan.feasible_fraction * 100
        ax.annotate(f"{share:.0f} % of this map works",
                    xy=(0.985, 0.03), xycoords="axes fraction",
                    ha="right", fontsize=7.6, color=th.ink_secondary,
                    bbox=dict(boxstyle="round,pad=0.28", fc=th.surface,
                              ec="none", alpha=0.88))

    for index in range(len(grits), rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.get_layout_engine().set(rect=(0, 0.02, 1, top))
    bar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="horizontal",
                       fraction=0.030, pad=0.035, shrink=0.55, aspect=45)
    bar.set_label("margin to the nearest limit  (negative = unusable)",
                  color=th.ink_secondary, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=th.ink_muted)
    bar.outline.set_visible(False)

    style.caption(
        fig,
        "Dotted contours are stone removal rate in µm/min. The limits are: "
        "not so thin it sheets off the plate, not so thick it drags, grit stays "
        "suspended, enough load on the grains rather than the film, and it must "
        "not be able to dry to a crust at the stated humidity.",
        th,
    )
    return _save(fig, "fig2_recipe_windows", th, directory)


# --------------------------------------------------------------------------- #
# 3. Drying
# --------------------------------------------------------------------------- #


def plot_drying(base: Setup | None = None, theme: str = "light",
                directory: Path | None = None) -> Path:
    """Why less grit stays wet longer -- the endpoint and the time course."""
    base = base or Setup()
    th = style.THEMES[theme]
    style.apply(th)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
    top = style.title_block(
        fig,
        "Why a lighter grit loading stays wet longer",
        "Only the water evaporates. Glycerine stays behind and holds the liquid "
        "at whatever strength matches the room, which fixes where drying stops.",
        th,
    )

    abrasive, grade, _, _ = base.resolve()
    reference = base.build_slurry()
    ceiling = base.calibration.workable_solids_headroom \
        * reference.max_packing_fraction

    # -- (a) the endpoint map: can this mix ever dry to a crust? -------------
    ax = axes[0]
    solids = np.linspace(0.10, 0.80, 90)
    glycerine = np.linspace(0.0, 0.85, 90)
    endpoint = np.zeros((glycerine.size, solids.size))
    for i, g in enumerate(glycerine):
        vehicle = Vehicle.from_fractions(float(g), base.soap_volume_fraction,
                                        temperature_C=base.temperature_C)
        for j, w in enumerate(solids):
            from lapsim.drying import equilibrium_state
            from lapsim.slurry import Slurry

            slurry = Slurry.from_mass_fraction(abrasive, grade, vehicle, float(w))
            endpoint[i, j] = equilibrium_state(slurry, base.relative_humidity)[0]

    mesh = ax.pcolormesh(solids * 100, glycerine * 100,
                         np.clip(endpoint * 100, 0, 100),
                         cmap=style.sequential_cmap(th), shading="gouraud",
                         vmin=0, vmax=100)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("solids after drying (vol %)", color=th.ink_secondary, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=th.ink_muted)
    bar.outline.set_visible(False)

    boundary = ax.contour(solids * 100, glycerine * 100, endpoint,
                          levels=[ceiling], colors=[th.critical], linewidths=2.0)
    ax.clabel(boundary, fmt=lambda v: "crusts over", fontsize=7.6,
              colors=th.critical)
    ax.set_title(f"a.  Endpoint at {base.relative_humidity * 100:.0f} % RH")
    ax.set_xlabel("grit loading (wt %)")
    ax.set_ylabel("glycerine in the liquid (wt %)")
    ax.grid(False)
    ax.annotate("safe: thickens and stops", xy=(14, 66), fontsize=8.4,
                color=th.ink, fontweight="semibold")
    ax.annotate("dries to a crust", xy=(58, 9), fontsize=8.4, color=th.surface,
                fontweight="semibold")

    # -- (b) the same boundary as a design rule, per humidity ----------------
    ax = axes[1]
    from lapsim.drying import equilibrium_state
    from lapsim.slurry import Slurry

    def loading_ceiling(glycerine_fraction: float, humidity: float) -> float:
        """Highest grit loading (wt %) that still cannot dry to a crust.

        Found by bisection rather than by scanning a grid, so the curve comes
        out smooth.
        """
        vehicle = Vehicle.from_fractions(glycerine_fraction,
                                         base.soap_volume_fraction,
                                         temperature_C=base.temperature_C)

        def crusts(mass_fraction: float) -> bool:
            slurry = Slurry.from_mass_fraction(abrasive, grade, vehicle,
                                               mass_fraction)
            return equilibrium_state(slurry, humidity)[0] >= ceiling

        low, high = 0.02, 0.92
        if crusts(low):
            return float("nan")
        for _ in range(40):
            middle = 0.5 * (low + high)
            low, high = (low, middle) if crusts(middle) else (middle, high)
        return low * 100

    glycerine_fine = np.linspace(0.0, 0.88, 120)
    label_offsets = {0.30: -9.0, 0.45: 0.0, 0.60: 9.0}
    for color, humidity in zip(th.categorical, (0.30, 0.45, 0.60)):
        limits = [loading_ceiling(float(g), humidity) for g in glycerine_fine]
        ax.plot(glycerine_fine * 100, limits, color=color,
                label=f"{humidity * 100:.0f} % RH")
        style.label_line(ax, glycerine_fine[-1] * 100, limits[-1],
                         f"{humidity * 100:.0f}% RH", color, th,
                         dy=label_offsets[humidity])
    ax.set_title("b.  The most grit you can safely use")
    ax.set_xlabel("glycerine in the liquid (wt %)")
    ax.set_ylabel("grit loading ceiling (wt %)")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 88)
    ax.legend(loc="lower right", title="room humidity", title_fontsize=8,
              labelcolor=th.ink_secondary)
    ax.annotate(
        "below a curve the slurry can never dry\nhard; above it, it eventually "
        "will.\nHumidity barely moves the line \u2014 the\nglycerine fraction "
        "is what decides it.",
        xy=(0.035, 0.955), xycoords="axes fraction", va="top", fontsize=7.8,
        color=th.ink_secondary,
    )

    # -- (c) the time course on the plate -----------------------------------
    ax = axes[2]
    loadings = (0.25, 0.40, 0.55, 0.70)
    colors = style.grit_colors(th, len(loadings))
    for color, loading in zip(colors, loadings):
        result = dry_out(
            replace(base, solids_mass_fraction=loading).build_slurry(),
            relative_humidity=base.relative_humidity,
            air_speed_m_s=base.air_speed_m_s,
            exposed_area=base.exposed_area,
            film_volume_ml=base.film_volume_ml,
            duration_s=2700.0,
        )
        minutes = result.times_s / 60.0
        ax.plot(minutes, result.solids_fraction * 100, color=color)
        style.label_line(ax, minutes[-1], result.solids_fraction[-1] * 100,
                         f"{loading * 100:.0f} wt%", color, th)
    ax.axhline(ceiling * 100, color=th.critical, lw=1.6, ls="--")
    ax.annotate("unworkable above here", xy=(0.5, ceiling * 100 + 1.6),
                fontsize=7.8, color=th.critical,
                bbox=dict(boxstyle="round,pad=0.18", fc=th.surface, ec="none"))
    ax.set_title("c.  The film on the plate, over time")
    ax.set_xlabel("minutes of exposure")
    ax.set_ylabel("solids in the film (vol %)")
    ax.set_xlim(0, 52)
    ax.set_ylim(0, min(ceiling * 100 * 1.45, 100))

    style.caption(
        fig,
        f"Panel c: {base.film_volume_ml:.0f} ml spread over "
        f"{base.exposed_area * 1e4:.0f} cm² at "
        f"{base.relative_humidity * 100:.0f} % RH and "
        f"{base.temperature_C:.0f} °C with light air movement, using "
        f"{base.glycerine_mass_fraction * 100:.0f} wt% glycerine. Curves flatten "
        "where the liquid reaches equilibrium with the room and stops losing "
        "water altogether.",
        th,
    )
    fig.tight_layout(rect=(0, 0, 1, top))
    return _save(fig, "fig3_drying", th, directory)


# --------------------------------------------------------------------------- #
# 4. The plate gap
# --------------------------------------------------------------------------- #


def _angular_grain(centre_x: float, centre_y: float, diameter: float,
                   rng: np.random.Generator, color: str,
                   edge: str, aspect: float = 1.0) -> Polygon:
    """A ragged polygon standing in for a real abrasive grain.

    Grit is angular, not spherical, and that is mechanically important -- the
    corners are what cut. Drawing circles would misrepresent the thing being
    modelled.
    """
    sides = rng.integers(5, 8)
    angles = np.sort(rng.uniform(0, 2 * math.pi, sides))
    radii = diameter / 2 * rng.uniform(0.72, 1.0, sides)
    points = np.column_stack([
        centre_x + radii * np.cos(angles) * aspect,
        centre_y + radii * np.sin(angles),
    ])
    return Polygon(points, closed=True, facecolor=color, edgecolor=edge,
                   linewidth=0.6, zorder=3)


def plot_contact(base: Setup | None = None, theme: str = "light",
                 directory: Path | None = None, seed: int = 7) -> Path:
    """Draw the solved gap to scale, and show which grains carry the load."""
    base = base or Setup()
    th = style.THEMES[theme]
    style.apply(th)
    rng = np.random.default_rng(seed)

    result = simulate(base)
    contact = result.contact
    abrasive, grade, stone, lap = base.resolve()

    fig = plt.figure(figsize=(12.6, 7.0))
    top = style.title_block(
        fig,
        "Inside the gap between the two plates",
        f"{abrasive.name} {grade.label} \u00b7 {stone.name} on a "
        f"{lap.name} \u00b7 {base.nominal_pressure / 1e3:.0f} kPa \u00b7 "
        f"{base.sliding_speed_m_s:.2f} m/s",
        th,
    )
    # Explicit gridspec geometry rather than tight_layout, which cannot handle
    # the fixed aspect ratio of the true-scale cross-section.
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], hspace=0.46,
                            wspace=0.24, top=top - 0.03, bottom=0.10,
                            left=0.065, right=0.985)
    ax_gap = fig.add_subplot(grid[0, :])
    ax_size = fig.add_subplot(grid[1, 0])
    ax_load = fig.add_subplot(grid[1, 1])

    # -- (a) cross-section, drawn to true scale in microns -------------------
    gap_um = contact.gap * 1e6
    window_um = 30.0 * grade.d50_um
    plate = 0.55 * gap_um

    ax_gap.add_patch(Rectangle((0, gap_um), window_um, plate, facecolor=th.grid,
                               edgecolor=th.axis, lw=0.9, zorder=4))
    ax_gap.add_patch(Rectangle((0, -plate), window_um, plate, facecolor=th.grid,
                               edgecolor=th.axis, lw=0.9, zorder=4))
    ax_gap.add_patch(Rectangle((0, 0), window_um, gap_um,
                               facecolor=th.categorical[0], alpha=0.08, lw=0,
                               zorder=1))
    ax_gap.annotate(f"the stone  \u2014  {stone.name}",
                    xy=(window_um * 0.5, gap_um + plate / 2), ha="center",
                    va="center", fontsize=8.8, color=th.ink,
                    fontweight="semibold", zorder=6)
    ax_gap.annotate(f"the lap  \u2014  {lap.name}",
                    xy=(window_um * 0.5, -plate / 2), ha="center", va="center",
                    fontsize=8.8, color=th.ink, fontweight="semibold", zorder=6)

    diameters, weights = grade.size_distribution()
    d_um = diameters * 1e6

    # Grains that merely circulate, at the density the slurry actually supplies.
    linear_density = math.sqrt(max(contact.available_grain_density, 1.0))
    n_floating = max(int(linear_density * window_um * 1e-6), 8)
    sizes = np.minimum(rng.choice(d_um, size=n_floating, p=weights), gap_um * 0.92)
    for x, d in zip(rng.uniform(0.02, 0.98, n_floating) * window_um, sizes):
        y = rng.uniform(d / 2, gap_um - d / 2)
        ax_gap.add_patch(_angular_grain(x, y, d, rng, th.grit_ramp[1],
                                        th.grit_ramp[3]))

    # The grain that matters: sampled from where the load actually sits.
    load_share = contact.grain_density_by_size * contact.grain_loads
    if load_share.sum() > 0:
        index = int(np.argmax(load_share))
        d_bridge = d_um[index]
        x_bridge = 0.42 * window_um
        ax_gap.add_patch(_angular_grain(x_bridge, gap_um / 2, d_bridge, rng,
                                        th.categorical[1], th.ink))
        ax_gap.annotate(
            f"a bridging grain: {d_bridge:.0f} \u00b5m across, carrying "
            f"{contact.grain_loads[index] * 1e3:.0f} mN\n"
            f"{contact.penetration_work[index] * 1e9:.0f} nm into the stone, "
            f"{contact.penetration_lap[index] * 1e9:.0f} nm into the lap",
            xy=(x_bridge, gap_um / 2),
            xytext=(0.52 * window_um, gap_um * 1.5 + plate),
            fontsize=7.8, color=th.ink,
            arrowprops=dict(arrowstyle="->", color=th.categorical[1], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.34", fc=th.surface, ec=th.axis, lw=0.7),
            zorder=7,
        )

    ax_gap.annotate("", xy=(0.03 * window_um, 0),
                    xytext=(0.03 * window_um, gap_um),
                    arrowprops=dict(arrowstyle="<->", color=th.ink, lw=1.1),
                    zorder=6)
    ax_gap.annotate(f"gap {gap_um:.0f} \u00b5m = {gap_um / grade.d50_um:.1f} \u00d7 d50",
                    xy=(0.01 * window_um, gap_um * 1.02 + plate), fontsize=8,
                    color=th.ink, va="bottom", zorder=6)
    ax_gap.annotate("stroke \u2192",
                    xy=(0.88 * window_um, gap_um + plate * 0.45), fontsize=8.4,
                    color=th.ink_secondary, zorder=7)
    ax_gap.set_xlim(0, window_um)
    ax_gap.set_ylim(-plate * 1.1, (gap_um + plate) * 1.9)
    ax_gap.set_xlabel("distance along the plate (\u00b5m)")
    ax_gap.set_title(
        f"a.  True scale \u2014 only {contact.active_fraction * 100:.2f} % of the "
        f"grains present touch both surfaces, about "
        f"{contact.active_grain_density / 1e6:.1f} per mm\u00b2, and they carry "
        f"{contact.load_fraction_on_grains * 100:.0f} % of the load"
    )
    ax_gap.set_aspect("equal")
    ax_gap.grid(False)
    ax_gap.spines["left"].set_visible(False)
    ax_gap.set_yticks([])

    # -- (b) the grading, and the sliver of it above the gap -----------------
    number_density = weights / np.gradient(d_um)
    number_density /= number_density.max()
    volume_density = weights * diameters**3 / np.gradient(d_um)
    volume_density /= volume_density.max()

    ax_size.fill_between(d_um, 0, number_density, color=th.grit_ramp[1],
                         alpha=0.28, lw=0)
    ax_size.plot(d_um, number_density, color=th.grit_ramp[2])
    ax_size.plot(d_um, volume_density, color=th.categorical[1], ls="--")
    style.label_line(ax_size, d_um[np.argmax(number_density)], 1.05,
                     "by grain count", th.grit_ramp[2], th, dx=-8, ha="right",
                     size=7.8)
    style.label_line(ax_size, d_um[np.argmax(volume_density)], 1.05,
                     "by volume \u2014 what the label means", th.categorical[1],
                     th, dx=6, size=7.8)

    ax_size.axvline(gap_um, color=th.ink, lw=1.4)
    tail = d_um >= gap_um
    ax_size.fill_between(d_um[tail], 0, number_density[tail],
                         color=th.categorical[1], alpha=0.65, lw=0)
    ax_size.annotate(f"the gap, {gap_um:.0f} \u00b5m:\nonly grains to the right\n"
                     "of this line can cut",
                     xy=(gap_um, 0.22), xytext=(-10, 30),
                     textcoords="offset points", ha="right", fontsize=7.8,
                     color=th.ink,
                     arrowprops=dict(arrowstyle="->", color=th.ink, lw=0.9))
    ax_size.axvline(grade.d50_um, color=th.ink_muted, lw=1.0, ls=":")
    ax_size.annotate(f"d50 {grade.d50_um:.0f} \u00b5m", xy=(grade.d50_um, 0.92),
                     xytext=(-5, 0), textcoords="offset points", ha="right",
                     fontsize=7.6, color=th.ink_secondary)
    ax_size.set_xscale("log")
    ax_size.set_xlabel("grain diameter (\u00b5m)")
    ax_size.set_ylabel("relative frequency")
    ax_size.set_title(f"b.  How {grade.label} grit is graded")
    ax_size.set_ylim(0, 1.18)
    ax_size.set_xlim(d_um[0] * 0.85, d_um[-1] * 1.25)

    # -- (c) where the load actually sits ------------------------------------
    pressure_share = contact.grain_density_by_size * contact.grain_loads
    total = pressure_share.sum()
    if total > 0:
        ax_load.fill_between(d_um, 0, pressure_share / total * 100,
                             color=th.categorical[1], alpha=0.35, lw=0)
        ax_load.plot(d_um, pressure_share / total * 100, color=th.categorical[1])
        cumulative = np.cumsum(pressure_share) / total * 100
        ax_load.plot(d_um, cumulative, color=th.grit_ramp[2], ls="--")
        style.label_line(ax_load, d_um[-1], 100.0,
                         "cumulative share", th.grit_ramp[2], th, dx=-6,
                         ha="right", va="bottom", size=7.8)
        style.label_line(ax_load, d_um[int(np.argmax(pressure_share))],
                         float(pressure_share.max() / total * 100),
                         "load by grain size", th.categorical[1], th, dx=6,
                         va="bottom", size=7.8)
    ax_load.axvline(gap_um, color=th.ink, lw=1.4)
    ax_load.set_xlabel("grain diameter (\u00b5m)")
    ax_load.set_ylabel("% of the grain-borne load")
    ax_load.set_title("c.  Only the coarse tail carries anything")
    # Zoomed to the loaded tail: everything below the gap is identically zero,
    # so showing the full range would compress the interesting part to a line.
    ax_load.set_xlim(gap_um * 0.93, d_um[-1] * 1.02)
    ax_load.set_ylim(0, 108)
    ax_load.annotate(
        f"nothing below {gap_um:.0f} \u00b5m\ncarries any load at all",
        xy=(gap_um, 8), xytext=(7, 26), textcoords="offset points",
        fontsize=7.8, color=th.ink_secondary,
        arrowprops=dict(arrowstyle="->", color=th.ink_muted, lw=0.8),
    )

    style.caption(
        fig,
        "The gap settles where bridging grains plus hydrodynamic film pressure "
        "together balance the applied load. Because a graded powder has a size "
        "distribution, the gap sits above d50 and only the coarse tail carries "
        "load \u2014 which is why the top cut of a grading matters more for "
        "scratching than the number on the tub.",
        th,
    )
    return _save(fig, "fig4_contact", th, directory)


# --------------------------------------------------------------------------- #
# 5. Comparing grits
# --------------------------------------------------------------------------- #


def plot_grit_comparison(base: Setup | None = None,
                         grits: list[tuple[str, str]] | None = None,
                         theme: str = "light", use_best_recipe: bool = True,
                         directory: Path | None = None) -> Path:
    """Cut rate, finish, damage and gap across the grit range.

    Four separate panels rather than one chart with two y-axes: these are
    different quantities on different scales, and overlaying them on twin axes
    would let the reader read a crossing point that does not exist.
    """
    base = base or Setup()
    grits = grits or [(a.key, g.label) for a, g in stock_grades()]
    th = style.THEMES[theme]
    style.apply(th)

    labels, results = [], []
    for abrasive_key, grit in grits:
        candidate_setup = replace(base, abrasive=abrasive_key, grit=grit)
        if use_best_recipe:
            best = find_recipe(candidate_setup, n_solids=16, n_glycerine=16)
            if best is not None:
                candidate_setup = best.setup
        results.append(simulate(candidate_setup))
        abrasive = ABRASIVES[abrasive_key]
        prefix = "" if abrasive_key == "sic_black" else abrasive.name.split()[0] + " "
        labels.append(f"{prefix}{grit}")

    colors = style.grit_colors(th, len(results))
    positions = np.arange(len(results))

    panels = [
        ("a.  Stone removal rate", "µm/min",
         [r.removal_rate_um_min for r in results], "{:.1f}"),
        ("b.  Surface roughness Ra", "µm",
         [r.finish.Ra_um for r in results], "{:.2f}"),
        ("c.  Subsurface damage left behind", "µm",
         [r.finish.subsurface_damage_um for r in results], "{:.1f}"),
        ("d.  Plate gap", "µm",
         [r.contact.gap * 1e6 for r in results], "{:.0f}"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0))
    top = style.title_block(
        fig,
        "What each grit delivers",
        
"Colour tracks grain size throughout: pale is coarse, dark is fine. "
        "Every bar is labelled, so nothing depends on reading the colour.",
        th,
    )

    for ax, (title, unit, values, fmt) in zip(axes, panels):
        bars = ax.bar(positions, values, color=colors, width=0.66,
                      edgecolor=th.surface, linewidth=2.0)
        # Not "top": that name holds the layout fraction from title_block,
        # which tight_layout needs below.
        peak = max(values) if values else 1.0
        for bar, value in zip(bars, values):
            ax.annotate(fmt.format(value),
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=th.ink_secondary)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, peak * 1.20)
        ax.grid(axis="x", visible=False)

    regimes = ", ".join(
        f"{label} {result.finish.regime}" for label, result in zip(labels, results)
    )
    style.caption(
        fig,
        f"Each grit is run at its own best recipe. Cutting regime by grade: "
        f"{regimes}. Panel c is the number that sets how long the next stage "
        "must run: until that much stone has gone, the previous grit's cracks "
        "are still under the surface waiting to appear during polishing.",
        th,
    )
    fig.tight_layout(rect=(0, 0.02, 1, top))
    return _save(fig, "fig5_grit_comparison", th, directory)


# --------------------------------------------------------------------------- #
# 6. The progression
# --------------------------------------------------------------------------- #


def plot_sequence(plan: SequencePlan, theme: str = "light",
                  directory: Path | None = None) -> Path:
    """Roughness and damage down the grit sequence, and time per stage."""
    th = style.THEMES[theme]
    style.apply(th)

    stages = plan.stages
    # Name the abrasive only where it changes, so a run of one abrasive reads as
    # a list of grades rather than the same word repeated.
    labels, previous = [], None
    for stage in stages:
        short = stage.abrasive_name.replace("silicon carbide", "SiC")
        labels.append(stage.grit if short == previous else f"{short}\n{stage.grit}")
        previous = short
    positions = np.arange(len(stages))
    colors = style.grit_colors(th, len(stages))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    top = style.title_block(
        fig,
        "The grit progression",
        "Each stage has to grind away the damage the previous one left before "
        "its own finish is real.",
        th,
    )

    # -- (a) two lengths, same units, so one shared axis is correct ----------
    ax = axes[0]
    Ra = [s.Ra_um for s in stages]
    damage = [s.subsurface_damage_um for s in stages]
    ax.plot(positions, damage, color=th.categorical[1], marker="o",
            mec=th.surface, mew=1.4, label="subsurface damage")
    ax.plot(positions, Ra, color=th.categorical[0], marker="o",
            mec=th.surface, mew=1.4, label="roughness Ra")
    style.label_line(ax, positions[-1], damage[-1], "damage below\nthe surface",
                     th.categorical[1], th)
    style.label_line(ax, positions[-1], Ra[-1], "roughness Ra",
                     th.categorical[0], th)
    for x, value in zip(positions, Ra):
        ax.annotate(f"{value:.2f}", xy=(x, value), xytext=(0, -13),
                    textcoords="offset points", ha="center", fontsize=7.4,
                    color=th.ink_secondary)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("depth (µm)")
    ax.set_title("a.  Finish improves; hidden damage shrinks with it")
    ax.set_xlim(-0.35, len(stages) - 0.35)
    ax.legend(loc="lower left", labelcolor=th.ink_secondary)

    # -- (b) time per stage --------------------------------------------------
    ax = axes[1]
    minutes = [s.minutes_required for s in stages]
    bars = ax.bar(positions, minutes, color=colors, width=0.64,
                  edgecolor=th.surface, linewidth=2.0)
    for bar, value, stage in zip(bars, minutes, stages):
        ax.annotate(
            f"{value:.0f} min\n{stage.damage_to_remove_um:.0f} µm off",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3), textcoords="offset points", ha="center",
            fontsize=7.6, color=th.ink_secondary,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"b.  Time needed — {plan.total_minutes:.0f} min in total")
    ax.set_ylim(0, max(minutes) * 1.30 if minutes else 1)
    ax.grid(axis="x", visible=False)

    style.caption(
        fig,
        "Times assume the damage depth from the preceding stage plus a 30 % "
        "margin, at each stage's own best recipe. The first stage is sized to "
        "clear an assumed rough starting surface.",
        th,
    )
    fig.tight_layout(rect=(0, 0.02, 1, top))
    return _save(fig, "fig6_sequence", th, directory)


# --------------------------------------------------------------------------- #
# Optional animation
# --------------------------------------------------------------------------- #


def animate_gap(base: Setup | None = None, theme: str = "light",
                directory: Path | None = None, frames: int = 90,
                seed: int = 3) -> Path | None:
    """Animate grains travelling through the gap as the stone strokes across.

    Bridging grains are pinned between the surfaces and drag along at half the
    stone's speed while rolling; the rest drift with the slurry. Returns ``None``
    if no animation writer is available.
    """
    from matplotlib import animation

    base = base or Setup()
    th = style.THEMES[theme]
    style.apply(th)
    rng = np.random.default_rng(seed)

    result = simulate(base)
    contact = result.contact
    abrasive, grade, stone, lap = base.resolve()

    gap_um = contact.gap * 1e6
    window_um = 30.0 * grade.d50_um
    plate = 0.4 * gap_um

    diameters, weights = grade.size_distribution()
    n_floating = max(int(math.sqrt(contact.available_grain_density)
                         * window_um * 1e-6), 10)
    floating = {
        "x": rng.uniform(0, window_um, n_floating),
        "d": np.minimum(rng.choice(diameters * 1e6, size=n_floating, p=weights),
                        gap_um * 0.9),
    }
    floating["y"] = rng.uniform(floating["d"] / 2, gap_um - floating["d"] / 2)

    bridging_mask = diameters * 1e6 > gap_um
    n_bridge = 3
    if bridging_mask.any():
        tail = weights[bridging_mask] / weights[bridging_mask].sum()
        bridge_d = rng.choice(diameters[bridging_mask] * 1e6, size=n_bridge, p=tail)
    else:
        bridge_d = np.full(n_bridge, gap_um)
    bridge_x = rng.uniform(0, window_um, n_bridge)

    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    speed_um_per_frame = base.sliding_speed_m_s * 1e6 / 900.0

    def draw(frame: int):
        ax.clear()
        ax.add_patch(Rectangle((0, gap_um), window_um, plate, facecolor=th.grid,
                               edgecolor=th.axis, lw=0.9, zorder=4))
        ax.add_patch(Rectangle((0, -plate), window_um, plate, facecolor=th.grid,
                               edgecolor=th.axis, lw=0.9, zorder=4))
        ax.add_patch(Rectangle((0, 0), window_um, gap_um,
                               facecolor=th.categorical[0], alpha=0.07, lw=0))

        for x, y, d in zip(floating["x"], floating["y"], floating["d"]):
            drift = (x + 0.35 * speed_um_per_frame * frame) % window_um
            ax.add_patch(_angular_grain(drift, y, d, np.random.default_rng(int(d * 1e3)),
                                        th.grit_ramp[0], th.grit_ramp[2]))
        for x, d in zip(bridge_x, bridge_d):
            travel = (x + 0.5 * speed_um_per_frame * frame) % window_um
            ax.add_patch(_angular_grain(travel, gap_um / 2, d,
                                        np.random.default_rng(int(d * 1e4)),
                                        th.categorical[1], th.ink))

        ax.annotate(f"{stone.name} →", xy=(window_um * 0.5, gap_um + plate / 2),
                    ha="center", va="center", fontsize=9, color=th.ink,
                    fontweight="semibold", zorder=6)
        ax.annotate(lap.name, xy=(window_um * 0.5, -plate / 2), ha="center",
                    va="center", fontsize=9, color=th.ink, fontweight="semibold",
                    zorder=6)
        ax.set_xlim(0, window_um)
        ax.set_ylim(-plate * 1.1, (gap_um + plate) * 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{abrasive.name} {grade.label} in a {gap_um:.0f} µm gap  ·  "
            f"orange grains bridge and cut; pale grains just circulate",
            fontsize=9.5, color=th.ink,
        )
        return []

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    directory = Path(directory or FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "anim_gap.gif"
    try:
        anim.save(path, writer=animation.PillowWriter(fps=18))
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return path


def plot_all(base: Setup | None = None, theme: str = "light",
             directory: Path | None = None, plan: SequencePlan | None = None,
             animate: bool = False) -> list[Path]:
    """Render every figure and return the paths written."""
    from lapsim.recipes import plan_sequence

    base = base or Setup()
    plan = plan or plan_sequence(base)
    paths = [
        plot_vehicle(base, theme, directory),
        plot_recipe_windows(base, theme=theme, directory=directory),
        plot_drying(base, theme, directory),
        plot_contact(base, theme, directory),
        plot_grit_comparison(base, theme=theme, directory=directory),
        plot_sequence(plan, theme, directory),
    ]
    if animate:
        animation_path = animate_gap(base, theme, directory)
        if animation_path is not None:
            paths.append(animation_path)
    return [p for p in paths if p is not None]
