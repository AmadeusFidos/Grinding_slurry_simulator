"""Check that the desktop window and the engine still agree on names.

The window cannot be exercised on a machine with no display, and a widget
callback that reaches for an attribute which has been renamed fails only when a
person clicks that button. These tests walk the same accessors
:mod:`lapsim.gui` uses, so the mismatch surfaces here instead.

They deliberately do not import tkinter, so they run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lapsim.materials import ABRASIVES, LAPS, STONES
from lapsim.recipes import (
    constraint_margins, diagnose_infeasible, explain_infeasible, find_recipe,
    plan_sequence, scan_window,
)
from lapsim.simulate import Setup, simulate

SETUP = Setup(grit="600", solids_mass_fraction=0.45, glycerine_mass_fraction=0.62)


def test_registry_fields_the_combo_boxes_read():
    for registry in (ABRASIVES, STONES, LAPS):
        for entry in registry.values():
            assert entry.name and entry.key
    for abrasive in ABRASIVES.values():
        for grade in abrasive.grades.values():
            assert grade.label and grade.d50_um > 0
            assert isinstance(grade.in_stock, bool)


def test_glance_panel_accessors():
    """Every quantity the "At a glance" tab prints."""
    result = simulate(SETUP)
    for value in (
        result.removal_rate_um_min, result.lap_wear_um_min,
        result.removal.plate_to_stone_wear_ratio, result.removal.drag_force,
        result.removal.drag_ratio, result.removal.temperature_rise,
        result.finish.Ra_um, result.finish.Rz_um,
        result.finish.subsurface_damage_um, result.finish.deepest_scratch_um,
        result.slurry.plastic_viscosity, result.slurry.yield_stress,
        result.slurry.puddle_height, result.slurry.settling_time(),
        result.slurry.solids_mass_fraction, result.slurry.solids_volume_fraction,
        result.slurry.max_packing_fraction, result.contact.gap,
        result.contact.load_fraction_on_grains,
        result.contact.active_grain_density, result.contact.active_fraction,
        result.contact.fractured_fraction, result.drying.film_thickness_um,
        result.drying.evaporation_rate_g_per_min,
        result.drying.workable_time_s,
        result.drying.equilibrium_solids_fraction,
        result.drying.jamming_solids_fraction,
    ):
        assert isinstance(value, float)

    assert isinstance(result.slurry.consistency_label, str)
    assert result.contact.regime in {"abrasive", "hydroplaning", "plate contact"}
    assert result.finish.regime in {"ductile", "mixed", "brittle"}
    assert isinstance(result.drying.seizes, bool)
    assert result.setup.resolve()[1].d50_um > 0
    assert isinstance(result.report(), str)
    assert isinstance(result.warnings, list)


def test_recipe_tab_accessors():
    best = find_recipe(SETUP, n_solids=10, n_glycerine=10)
    assert best is not None
    assert isinstance(best.recipe.describe(), str)
    assert best.recipe.soap_drops >= 0
    spans = best.tolerances()
    assert set(spans) == {"solids", "glycerine"}
    assert all(len(pair) == 2 for pair in spans.values())
    assert isinstance(best.binding_constraint, str)
    assert isinstance(constraint_margins(best.result), dict)
    assert isinstance(explain_infeasible(SETUP, []), str)


def test_window_map_accessors():
    """The canvas indexes the grid as [row][column]; the axes must match."""
    scan = scan_window(SETUP, n_solids=6, n_glycerine=7)
    assert scan.robustness.shape == (7, 6)
    assert len(scan.glycerine_mass_fractions) == scan.robustness.shape[0]
    assert len(scan.solids_mass_fractions) == scan.robustness.shape[1]
    assert scan.removal_rate_um_min.shape == scan.robustness.shape
    assert isinstance(scan.binding_constraint[0][0], str)
    assert 0.0 <= scan.feasible_fraction <= 1.0
    assert scan.abrasive_name and scan.grit


def test_plan_tab_accessors():
    plan = plan_sequence(SETUP, n_solids=8, n_glycerine=8)
    assert plan.stages
    for stage in plan.stages:
        assert stage.abrasive_name and stage.grit
        assert stage.d50_um > 0 and stage.Ra_um > 0
        assert stage.subsurface_damage_um > 0
        assert stage.candidate.removal_rate_um_min > 0
        assert stage.candidate.recipe.grit_grams > 0
        assert isinstance(stage.in_stock, bool)
    for skip in plan.skipped:
        assert skip.abrasive_name and skip.grit and skip.explanation
    assert plan.total_minutes > 0


def test_colour_ramp_is_continuous_and_neutral_at_zero():
    """The map colour scheme, shared by the figures and the desktop window."""
    from lapsim.style import margin_hex as margin_colour

    assert margin_colour(0.0) == "#f0efec"        # neutral grey on the boundary
    assert margin_colour(-5.0) == margin_colour(-1.0)   # clipped, not wrapped
    assert margin_colour(5.0) == margin_colour(1.0)
    for value in (-1.0, -0.4, 0.0, 0.4, 1.0):
        colour = margin_colour(value)
        assert len(colour) == 7 and colour.startswith("#")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"  pass  {name}")
            except AssertionError as error:
                failures += 1
                print(f"  FAIL  {name}  {error}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
