"""Lapidary slurry simulator.

A physics model of loose-abrasive lapping between two flat plates: abrasive
grains suspended in a water/glycerine/soap vehicle, ground between a stone and a
glass lap.  It answers the practical questions -- how much grit, how much
glycerine, how fast will it cut, how fine will it finish, how long before it
dries -- from mechanics rather than from trial and error.

Quick start::

    from lapsim import Setup, simulate, find_recipe

    print(simulate(grit="600").report())
    best = find_recipe(Setup(grit="600"))
    print(best.recipe.describe())

See ``README.md`` for the model description and its limits.
"""

from lapsim.calibration import DEFAULT as DEFAULT_CALIBRATION, Calibration
from lapsim.contact import solve_contact
from lapsim.drying import dry_out, equilibrium_state
from lapsim.materials import (
    ABRASIVES,
    LAPS,
    STONES,
    Abrasive,
    GritGrade,
    Solid,
    add_micron_grade,
    find_grade,
    register_abrasive,
    register_lap,
    register_stone,
    set_stock,
    stock_grades,
)
from lapsim.recipes import (
    Recipe,
    diagnose_infeasible,
    find_recipe,
    plan_sequence,
    scan_window,
)
from lapsim.removal import evaluate_removal
from lapsim.simulate import Result, Setup, Targets, simulate
from lapsim.slurry import Slurry
from lapsim.vehicle import Vehicle

__all__ = [
    "ABRASIVES", "LAPS", "STONES",
    "Abrasive", "Calibration", "GritGrade", "Recipe", "Result", "Setup",
    "Slurry", "Solid", "Targets", "Vehicle",
    "DEFAULT_CALIBRATION",
    "add_micron_grade", "diagnose_infeasible", "dry_out", "equilibrium_state", "evaluate_removal",
    "find_grade", "find_recipe", "plan_sequence", "register_abrasive",
    "register_lap", "register_stone", "scan_window", "set_stock", "simulate",
    "solve_contact", "stock_grades",
]

__version__ = "1.0.0"
