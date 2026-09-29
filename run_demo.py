#!/usr/bin/env python3
"""Run the whole simulator end to end and write a report plus every figure.

    python run_demo.py                 light-theme figures, report to stdout
    python run_demo.py --theme dark    dark-theme figures
    python run_demo.py --animate       also render the gap animation
    python run_demo.py --quick         smaller search grids, ~4x faster

Everything it prints is also written to ``report.txt``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from lapsim import Setup, find_recipe, plan_sequence, simulate, stock_grades
from lapsim.recipes import Recipe
from lapsim.simulate import Targets

HERE = Path(__file__).resolve().parent


class Tee:
    """Write to stdout and to a file at once."""

    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        self.handle.write(text)

    def close(self) -> None:
        self.handle.close()


def rule(out: Tee, title: str) -> None:
    out.write("\n" + "=" * 78 + "\n")
    out.write(f"  {title}\n")
    out.write("=" * 78 + "\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="light", choices=("light", "dark"))
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="smaller search grids")
    parser.add_argument("--humidity", type=float, default=45.0)
    parser.add_argument("--stone", default="agate")
    args = parser.parse_args()

    resolution = 16 if args.quick else 22
    base = Setup(stone=args.stone, relative_humidity=args.humidity / 100.0)
    out = Tee(HERE / "report.txt")
    started = time.time()

    # ----------------------------------------------------------------- intro
    rule(out, "LAPIDARY SLURRY SIMULATOR")
    out.write(
        f"Stone:       {base.resolve()[2].name}\n"
        f"Lap:         {base.resolve()[3].name}\n"
        f"Pressure:    {base.nominal_pressure / 1e3:.0f} kPa "
        f"({base.normal_load_N:.0f} N over {base.contact_area_mm2:.0f} mm2)\n"
        f"Stroke:      {base.sliding_speed_m_s:.2f} m/s\n"
        f"Room:        {base.temperature_C:.0f} C, "
        f"{base.relative_humidity * 100:.0f} % RH\n"
        f"On the plate: {base.film_volume_ml:.0f} ml spread over "
        f"{base.exposed_area * 1e4:.0f} cm2\n"
    )

    # -------------------------------------------------- recommended recipes
    rule(out, "RECOMMENDED RECIPES  (one per grade you own)")
    for abrasive, grade in stock_grades():
        stage = replace(base, abrasive=abrasive.key, grit=grade.label)
        best = find_recipe(stage, n_solids=resolution, n_glycerine=resolution)
        out.write(f"--- {abrasive.name} {grade.label} "
                  f"(d50 {grade.d50_um:.1f} um) ---\n")
        if best is None:
            from lapsim.recipes import diagnose_infeasible, explain_infeasible

            remedies = diagnose_infeasible(stage, n_solids=14, n_glycerine=14)
            out.write(explain_infeasible(stage, remedies) + "\n\n")
            continue
        out.write(Recipe.from_setup(best.setup, 100.0).describe() + "\n")
        out.write("  " + best.describe_tolerances() + "\n")
        out.write(
            f"  Expect {best.removal_rate_um_min:.0f} um/min, Ra "
            f"{best.Ra_um:.2f} um, "
            f"{best.result.slurry.consistency_label}.\n"
            f"  Tightest limit: {best.binding_constraint} "
            f"(margin {best.robustness:+.2f}).\n\n"
        )

    # ------------------------------------------------------- one full report
    rule(out, "FULL DETAIL FOR ONE RECIPE  (600 grit at its best mix)")
    best_600 = find_recipe(replace(base, grit="600"),
                           n_solids=resolution, n_glycerine=resolution)
    if best_600 is not None:
        out.write(simulate(best_600.setup).report())

    # -------------------------------------------------------- the sequence
    rule(out, "GRIT SEQUENCE FROM A SAWN FACE")
    plan = plan_sequence(base, n_solids=resolution, n_glycerine=resolution)
    out.write(plan.table() + "\n")

    # ------------------------------------------- what the future grits add
    rule(out, "WHAT THE FINER ABRASIVES WOULD ADD")
    from lapsim import ABRASIVES

    # Finer than FEPA F1000, so named by micron size rather than grit number.
    future = [("sic_green", "3um"), ("sic_green", "2um"),
              ("alumina_white", "3um"), ("alumina_white", "2um"),
              ("alumina_white", "1um")]
    out.write(
        "Grades finer than FEPA F1000, by micron size. They run in a smaller\n"
        "gap, where the slurry film lifts harder.\n"
        "Where the default technique will not carry them, the change that does\n"
        "is given instead.\n\n"
    )
    out.write(f"  {'abrasive':<24} {'grade':>6} {'Ra um':>8} {'SSD um':>8} "
              f"{'um/min':>8}  how to run it\n")
    out.write("  " + "-" * 96 + "\n")
    for abrasive_key, grit in future:
        stage = replace(base, abrasive=abrasive_key, grit=grit)
        best = find_recipe(stage, n_solids=14, n_glycerine=14)
        note = "at the default stroke and pressure"
        if best is None:
            from lapsim.recipes import diagnose_infeasible

            remedies = diagnose_infeasible(stage, n_solids=12, n_glycerine=12)
            if not remedies:
                out.write(f"  {ABRASIVES[abrasive_key].name:<24} {grit:>6} "
                          f"{'':>8} {'':>8} {'':>8}  "
                          f"nothing in the list makes this work\n")
                continue
            note, best = remedies[0]
        finish = best.result.finish
        out.write(
            f"  {ABRASIVES[abrasive_key].name:<24} {grit:>6} {best.Ra_um:8.3f} "
            f"{finish.subsurface_damage_um:8.2f} "
            f"{best.removal_rate_um_min:8.1f}  {note}\n"
        )

    # ------------------------------------------ sensitivity to your choices
    rule(out, "HOW MUCH YOUR TARGETS MATTER")
    out.write(
        "The single most consequential setting is how thick 'not too runny' "
        "means.\nRelaxing it lets you use far less glycerine:\n\n"
    )
    out.write(f"  {'min low-shear viscosity':>26} {'glycerine':>10} "
              f"{'grit':>6} {'um/min':>8}\n")
    out.write("  " + "-" * 56 + "\n")
    for threshold in (0.005, 0.01, 0.02, 0.05):
        targets = Targets(min_low_shear_viscosity_Pas=threshold)
        stage = replace(base, grit="600", targets=targets)
        best = find_recipe(stage, n_solids=14, n_glycerine=14)
        if best is None:
            out.write(f"  {threshold * 1e3:23.0f} mPa.s {'none':>10}\n")
            continue
        out.write(
            f"  {threshold * 1e3:23.0f} mPa.s "
            f"{best.setup.glycerine_mass_fraction * 100:9.0f} % "
            f"{best.setup.solids_mass_fraction * 100:5.0f} % "
            f"{best.removal_rate_um_min:8.1f}\n"
        )
    out.write(
        "\n  The default of 20 mPa.s means 'clearly thicker than water, holds a\n"
        "  film on a tilted plate'. Lower it if you are happy to stir often.\n"
    )

    # ------------------------------------------------------------- figures
    rule(out, "FIGURES")
    from lapsim.plots import plot_all

    paths = plot_all(base, theme=args.theme, plan=plan, animate=args.animate)
    for path in paths:
        out.write(f"  {path}\n")

    out.write(f"\nDone in {time.time() - started:.0f} s. "
              f"This report is also in {HERE / 'report.txt'}\n")
    out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
