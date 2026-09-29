"""Command line interface.

    python -m lapsim list                     what materials are defined
    python -m lapsim report --grit 600        simulate one recipe in full
    python -m lapsim recipe --grit 600        search for the best recipe
    python -m lapsim plan                     the whole grit progression
    python -m lapsim figures                  render every chart
    python -m lapsim calibrate                check the fitted constants
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from lapsim.materials import ABRASIVES, LAPS, STONES, stock_grades
from lapsim.simulate import Setup, Targets, simulate


def _setup_from_args(args: argparse.Namespace) -> Setup:
    """Build a :class:`Setup` from whichever options the user supplied."""
    targets = Targets()
    if getattr(args, "min_wet_minutes", None) is not None:
        targets = replace(targets, min_workable_time_s=args.min_wet_minutes * 60.0)
    return Setup(
        abrasive=args.abrasive,
        grit=args.grit,
        stone=args.stone,
        lap=args.lap,
        solids_mass_fraction=getattr(args, "solids", 0.40),
        glycerine_mass_fraction=getattr(args, "glycerine", 0.35),
        soap_volume_fraction=getattr(args, "soap_drops", 1.0) * 0.05 / 100.0,
        normal_load_N=args.load,
        contact_area_mm2=args.area,
        sliding_speed_m_s=args.speed,
        contact_length_mm=args.face_length,
        temperature_C=args.temperature,
        relative_humidity=args.humidity / 100.0,
        air_speed_m_s=args.air_speed,
        film_volume_ml=args.film_ml,
        targets=targets,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--abrasive", default="sic_black",
                        choices=sorted(ABRASIVES), help="abrasive type")
    parser.add_argument("--grit", default="600",
                        help="FEPA F grit up to 1000 (e.g. 600), or a micron "
                             "size for anything finer (e.g. 2.5um)")
    parser.add_argument("--stone", default="agate", choices=sorted(STONES))
    parser.add_argument("--lap", default="glass", choices=sorted(LAPS))
    parser.add_argument("--load", type=float, default=40.0,
                        help="normal force in newtons (default 40)")
    parser.add_argument("--area", type=float, default=400.0,
                        help="contact area in mm^2 (default 400)")
    parser.add_argument("--speed", type=float, default=0.25,
                        help="sliding speed in m/s (default 0.25)")
    parser.add_argument("--face-length", type=float, default=20.0,
                        help="face length along the stroke, mm (default 20)")
    parser.add_argument("--temperature", type=float, default=21.0,
                        help="degrees C (default 21)")
    parser.add_argument("--humidity", type=float, default=45.0,
                        help="relative humidity in percent (default 45)")
    parser.add_argument("--air-speed", type=float, default=0.3,
                        help="air movement over the plate, m/s (default 0.3)")
    parser.add_argument("--film-ml", type=float, default=8.0,
                        help="slurry on the plate at once, ml (default 8)")
    parser.add_argument("--min-wet-minutes", type=float, default=None,
                        help="require the film to stay usable this long")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lapsim",
        description="Simulate lapidary grinding slurries between two flat plates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="show the materials that are defined")
    subparsers.add_parser("calibrate", help="check the fitted constants")

    report = subparsers.add_parser("report", help="simulate one recipe in full")
    _add_common(report)
    report.add_argument("--solids", type=float, default=0.40,
                        help="grit as a mass fraction of the slurry")
    report.add_argument("--glycerine", type=float, default=0.35,
                        help="glycerine as a mass fraction of the liquid")
    report.add_argument("--soap-drops", type=float, default=1.0,
                        help="drops of dish soap per 100 ml of liquid")

    recipe = subparsers.add_parser("recipe", help="search for the best recipe")
    _add_common(recipe)
    recipe.add_argument("--prefer", default="robust",
                        choices=("robust", "balanced", "fast"))
    recipe.add_argument("--batch-ml", type=float, default=100.0)

    plan = subparsers.add_parser("plan", help="plan the whole grit sequence")
    _add_common(plan)
    plan.add_argument("--prefer", default="robust",
                      choices=("robust", "balanced", "fast"))
    plan.add_argument("--start-damage", type=float, default=120.0,
                      help="damage depth on the stone as it arrives, um")

    figures = subparsers.add_parser("figures", help="render every chart")
    _add_common(figures)
    figures.add_argument("--theme", default="light", choices=("light", "dark"))
    figures.add_argument("--out", default=None, help="output directory")
    figures.add_argument("--animate", action="store_true",
                         help="also render the gap animation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        print("ABRASIVES")
        for abrasive in ABRASIVES.values():
            grades = ", ".join(
                f"{label}{'*' if grade.in_stock else ''}"
                for label, grade in abrasive.grades.items()
            )
            print(f"  {abrasive.key:<16} {abrasive.name}")
            print(f"  {'':16} grades: {grades}")
            print(f"  {'':16} {abrasive.notes}")
        print("  (* = marked as in stock)\n")
        print("STONES")
        for stone in STONES.values():
            print(f"  {stone.key:<12} {stone.name:<22} "
                  f"Mohs ~{stone.mohs_approx:.1f}, K_Ic "
                  f"{stone.fracture_toughness / 1e6:.2f} MPa.m^0.5")
        print("\nLAP PLATES")
        for lap in LAPS.values():
            print(f"  {lap.key:<12} {lap.name:<22} "
                  f"hardness {lap.hardness / 1e9:.1f} GPa")
        print("\nIN STOCK, COARSE TO FINE")
        for abrasive, grade in stock_grades():
            print(f"  {abrasive.name} {grade.label}  (d50 {grade.d50_um:.1f} um)")
        return 0

    if args.command == "calibrate":
        from lapsim.calibration import _report

        _report()
        return 0

    setup = _setup_from_args(args)
    try:
        setup.resolve()
    except (KeyError, ValueError) as error:
        # A grade that does not exist, or a grit number finer than F1000: the
        # message says what to type instead, so show it without a traceback.
        message = error.args[0] if error.args else str(error)
        print(f"lapsim: {message}")
        return 2

    if args.command == "report":
        print(simulate(setup).report())
        return 0

    if args.command == "recipe":
        from lapsim.recipes import (
            Recipe, diagnose_infeasible, explain_infeasible, find_recipe,
        )

        best = find_recipe(setup, prefer=args.prefer)
        if best is None:
            remedies = diagnose_infeasible(setup, n_solids=14, n_glycerine=14)
            print(explain_infeasible(setup, remedies))
            return 1
        print(Recipe.from_setup(best.setup, args.batch_ml).describe())
        print()
        print(best.describe_tolerances())
        print()
        print(simulate(best.setup).report())
        return 0

    if args.command == "plan":
        from lapsim.recipes import plan_sequence

        plan = plan_sequence(setup, prefer=args.prefer,
                             initial_damage_um=args.start_damage)
        print(plan.table())
        return 0

    if args.command == "figures":
        import matplotlib

        matplotlib.use("Agg")
        from lapsim.plots import plot_all

        for path in plot_all(setup, theme=args.theme, directory=args.out,
                             animate=args.animate):
            print(path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
