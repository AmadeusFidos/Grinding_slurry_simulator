"""Sanity checks: physics that must hold, and regressions that must not return.

Run with ``python -m pytest tests/`` or ``python tests/test_lapsim.py``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lapsim import Setup, Vehicle, find_recipe, find_grade, simulate
from lapsim.drying import equilibrium_state
from lapsim.slurry import Slurry
from lapsim.vehicle import (
    glycerol_water_viscosity,
    water_activity,
    water_viscosity,
)


# --------------------------------------------------------------- the liquid

def test_water_viscosity_matches_literature():
    assert math.isclose(water_viscosity(20.0), 1.002e-3, rel_tol=0.01)
    assert math.isclose(water_viscosity(40.0), 0.653e-3, rel_tol=0.02)


def test_glycerol_viscosity_matches_literature():
    # Cheng (2008) table values at 20 C, in Pa.s.
    for fraction, expected in ((0.5, 6.0e-3), (0.8, 60.1e-3), (1.0, 1.412)):
        assert math.isclose(glycerol_water_viscosity(fraction, 20.0), expected,
                            rel_tol=0.05), fraction


def test_water_activity_matches_measured_values():
    """Measured at 25 C: 'Molecular Basis of Water Activity in Glycerol-Water
    Mixtures', PMC6839025, Table 1, as (glycerol mole fraction, a_w)."""
    from lapsim.vehicle import MOLAR_MASS_GLYCEROL, MOLAR_MASS_WATER

    measured = ((0.08, 0.91), (0.17, 0.80), (0.27, 0.68), (0.37, 0.56),
                (0.50, 0.42), (0.58, 0.34), (0.69, 0.25))
    for x_g, expected in measured:
        mass = x_g * MOLAR_MASS_GLYCEROL
        fraction = mass / (mass + (1.0 - x_g) * MOLAR_MASS_WATER)
        assert abs(water_activity(fraction) - expected) < 0.025, x_g


def test_glycerine_lowers_water_activity_below_raoult():
    from lapsim.vehicle import glycerine_mole_fraction

    for fraction in (0.3, 0.6, 0.85):
        assert water_activity(fraction) < 1.0 - glycerine_mole_fraction(fraction)


def test_soap_saturates():
    weak = Vehicle.from_fractions(0.4, soap_volume_fraction=1e-3)
    strong = Vehicle.from_fractions(0.4, soap_volume_fraction=1e-1)
    assert abs(weak.surface_tension - strong.surface_tension) < 0.003


# ------------------------------------------------------------- the suspension

def test_finer_grit_thickens_more_at_equal_loading():
    """The 1/d scaling of yield stress is a headline behaviour of the model."""
    vehicle = Vehicle.from_fractions(0.5)
    stresses = []
    for grit in ("320", "600", "1000"):
        abrasive, grade = find_grade("sic_black", grit)
        stresses.append(
            Slurry.from_mass_fraction(abrasive, grade, vehicle, 0.62).yield_stress
        )
    assert stresses[0] < stresses[1] < stresses[2]


def test_viscosity_diverges_towards_jamming():
    vehicle = Vehicle.from_fractions(0.3)
    abrasive, grade = find_grade("sic_black", "600")
    thin = Slurry(abrasive, grade, vehicle, 0.10)
    thick = Slurry(abrasive, grade, vehicle, 0.50)
    assert thick.plastic_viscosity > 8 * thin.plastic_viscosity


def test_mass_and_volume_fractions_are_consistent():
    abrasive, grade = find_grade("sic_black", "600")
    slurry = Slurry.from_mass_fraction(abrasive, grade, Vehicle.from_fractions(0.3),
                                       0.40)
    assert math.isclose(slurry.solids_mass_fraction, 0.40, rel_tol=1e-6)
    # SiC is 3.2x denser than the liquid, so volume fraction is much lower.
    assert slurry.solids_volume_fraction < 0.5 * slurry.solids_mass_fraction * 1.2


# ----------------------------------------------------------------- the drying

def test_pure_water_slurry_always_crusts():
    abrasive, grade = find_grade("sic_black", "600")
    vehicle = Vehicle.from_fractions(0.0)
    slurry = Slurry.from_mass_fraction(abrasive, grade, vehicle, 0.40)
    endpoint, _ = equilibrium_state(slurry, 0.45)
    assert endpoint > 0.9


def test_less_grit_ends_wetter():
    """The user-observed effect: a lighter loading dries to a lower solids
    fraction, so it stays workable."""
    abrasive, grade = find_grade("sic_black", "600")
    vehicle = Vehicle.from_fractions(0.4)
    light = Slurry.from_mass_fraction(abrasive, grade, vehicle, 0.25)
    heavy = Slurry.from_mass_fraction(abrasive, grade, vehicle, 0.60)
    assert (equilibrium_state(light, 0.45)[0]
            < equilibrium_state(heavy, 0.45)[0])


def test_more_glycerine_ends_wetter():
    abrasive, grade = find_grade("sic_black", "600")
    dry = Slurry.from_mass_fraction(abrasive, grade, Vehicle.from_fractions(0.2), 0.45)
    wet = Slurry.from_mass_fraction(abrasive, grade, Vehicle.from_fractions(0.6), 0.45)
    assert equilibrium_state(wet, 0.45)[0] < equilibrium_state(dry, 0.45)[0]


# ---------------------------------------------------------------- the contact

def test_gap_sits_just_below_the_top_cut():
    """A derived result: only the coarsest grains bridge, so the gap settles
    just under the top cut for every grade, whatever its median."""
    ratios = []
    for grit in ("320", "400", "600", "1000"):
        result = simulate(Setup(grit=grit, solids_mass_fraction=0.45))
        _, grade, _, _ = result.setup.resolve()
        assert grade.d50_um * 1e-6 < result.contact.gap < grade.d_max_um * 1e-6
        ratios.append(result.contact.gap / (grade.d_max_um * 1e-6))
    assert max(ratios) / min(ratios) < 1.1


def test_thick_vehicle_and_fast_stroke_causes_hydroplaning():
    result = simulate(Setup(grit="1000", glycerine_mass_fraction=0.92,
                            sliding_speed_m_s=1.0))
    assert result.contact.regime == "hydroplaning"
    assert not result.contact.is_cutting


def test_near_jammed_fine_paste_floats_instead_of_crashing():
    """Regression: a stiff paste lifts the stone at any gap, so the solver has
    no balance point; it must report hydroplaning rather than raise."""
    result = simulate(Setup(abrasive="sic_green", grit="2um",
                            solids_mass_fraction=0.75,
                            glycerine_mass_fraction=0.9))
    assert result.contact.regime == "hydroplaning"
    assert not result.contact.is_cutting


def test_load_is_carried_by_a_tiny_minority_of_grains():
    result = simulate(Setup(grit="600", solids_mass_fraction=0.45))
    assert 0.0 < result.contact.active_fraction < 0.02


def test_glass_lap_takes_more_penetration_than_quartz_stone():
    """Glass is softer than agate, so grains bed into the plate preferentially."""
    result = simulate(Setup(grit="600"))
    assert result.contact.penetration_lap.max() > result.contact.penetration_work.max()


# ---------------------------------------------------------------- the cutting

def test_finer_grit_cuts_slower_and_finishes_better():
    coarse = simulate(Setup(grit="320", solids_mass_fraction=0.45))
    fine = simulate(Setup(grit="1000", solids_mass_fraction=0.45))
    assert coarse.removal_rate_um_min > fine.removal_rate_um_min
    assert coarse.finish.Ra_um > fine.finish.Ra_um


def test_tough_stone_grinds_ductilely_and_slowly():
    """Nephrite is softer than agate but far tougher, so it refuses to chip."""
    agate = simulate(Setup(grit="600", stone="agate"))
    jade = simulate(Setup(grit="600", stone="nephrite"))
    assert jade.finish.regime == "ductile"
    assert jade.removal_rate_um_min < 0.4 * agate.removal_rate_um_min


def test_roughness_tracks_the_rule_of_thumb():
    """Ra should land near d50/30 when a grade is worked at a sensible recipe.

    Judged at each grade's own best mix rather than one fixed composition: a
    recipe that suits 320 grit is close to hydroplaning at 1000, where the stone
    floats and barely cuts at all.
    """
    for grit in ("320", "600"):
        best = find_recipe(Setup(grit=grit), n_solids=12, n_glycerine=12)
        assert best is not None, grit
        _, grade, _, _ = best.setup.resolve()
        ratio = best.Ra_um / (grade.d50_um / 30.0)
        assert 0.4 < ratio < 1.8, (grit, ratio)


def test_softer_abrasive_cuts_quartz_more_slowly():
    """Same FEPA grade, so only the grain material differs."""
    sic = simulate(Setup(abrasive="sic_black", grit="600", stone="agate"))
    alumina = simulate(Setup(abrasive="alumina_white", grit="600", stone="agate"))
    assert alumina.removal_rate_um_min < sic.removal_rate_um_min


# ------------------------------------------------------------ the grading

def test_numbered_grades_are_fepa_f():
    for grit, d50, ds3 in (("320", 29.2, 49.0), ("600", 9.3, 19.0),
                           ("1000", 4.5, 10.0)):
        _, grade = find_grade("sic_black", grit)
        assert (grade.d50_um, grade.d_max_um) == (d50, ds3), grit
    assert find_grade("sic_black", "F1000")[1].label == "1000"


def test_grit_numbers_finer_than_f1000_are_refused():
    for label in ("1200", "2000", "F3000"):
        for key in ("sic_black", "sic_green", "alumina_white"):
            try:
                find_grade(key, label)
            except KeyError as error:
                assert "um" in str(error)
            else:
                raise AssertionError(f"{key} accepted {label}")


def test_any_micron_size_works_without_registering():
    for text in ("2.5um", "2.5 um", "2.5 µm", "2.5 micron"):
        _, grade = find_grade("alumina_white", text)
        assert grade.label == "2.5um" and grade.d50_um == 2.5
        assert grade.d_max_um > grade.d50_um
    result = simulate(Setup(abrasive="sic_green", grit="2.5um"))
    assert result.removal_rate_um_min >= 0.0


def test_registered_micron_grade_keeps_its_spec_and_stock():
    from lapsim import ABRASIVES, add_micron_grade, stock_grades

    original = ABRASIVES["sic_green"]
    try:
        add_micron_grade("sic_green", 2.5, d_max_um=6.0, in_stock=True)
        _, grade = find_grade("sic_green", "2.5um")
        assert grade.d_max_um == 6.0 and grade.in_stock
        assert ("sic_green", "2.5um") in [(a.key, g.label) for a, g in stock_grades()]
        labels = list(ABRASIVES["sic_green"].grades)
        assert labels.index("3um") < labels.index("2.5um") < labels.index("2um")
    finally:
        ABRASIVES["sic_green"] = original


# ------------------------------------------------------------- the search

def test_search_finds_a_recipe_and_it_simulates_cleanly():
    best = find_recipe(Setup(grit="600"), n_solids=12, n_glycerine=12)
    assert best is not None
    assert best.feasible and best.robustness > 0
    assert simulate(best.setup).warnings == []


def test_recipe_round_trips_to_measurable_quantities():
    from lapsim.recipes import Recipe

    setup = Setup(grit="600", solids_mass_fraction=0.45,
                  glycerine_mass_fraction=0.55)
    recipe = Recipe.from_setup(setup, batch_ml=250.0)
    total = (recipe.water_ml + recipe.glycerine_ml
             + recipe.grit_grams / 3.21 + recipe.soap_drops * 0.05)
    assert math.isclose(total, 250.0, rel_tol=0.02)
    assert math.isclose(recipe.solids_mass_fraction, 0.45, rel_tol=1e-3)


def test_registry_is_extensible():
    """Adding an abrasive must need no change anywhere else."""
    from lapsim import ABRASIVES, Abrasive, GritGrade, register_abrasive

    register_abrasive(Abrasive(
        key="test_boron_carbide", name="Boron carbide", density=2520.0,
        hardness=30e9, youngs_modulus=460e9, grain_strength_ref=500e6,
        grades={"800": GritGrade("800", 11.5, 18.0, 1.28)},
    ))
    try:
        result = simulate(Setup(abrasive="test_boron_carbide", grit="800"))
        assert result.removal_rate_um_min > 0
    finally:
        del ABRASIVES["test_boron_carbide"]


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
