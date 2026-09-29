"""Material removal, surface finish and drag.

Given the solved gap state, each bridging grain is a tiny indenter dragged
across the stone.  What it does depends on how hard it is pressed:

*   **Below a critical load** the stone deforms plastically and the grain plows a
    groove.  Most of the displaced material is pushed into ridges either side
    rather than removed, so this regime is slow but leaves almost no damage
    below the surface.  This is the "ductile" regime, and it is what polishing
    and the very finest lapping stages operate in.
*   **Above it** the stone cracks.  Lateral cracks run out sideways from the
    plastic zone and then curve up to the surface, spalling out a chip much
    larger than the groove itself -- typically ten times larger.  Cutting is
    fast, but every pass leaves cracks *below* the finished surface.

That second point is the one that costs lapidaries the most time.  The cracked
layer is called subsurface damage, and it is invisible: the surface looks fine
until the next stage polishes down into it and scratches appear from nowhere.
The remedy is to remove at least the full damage depth at each subsequent grit,
and :attr:`Finish.subsurface_damage_um` is the number that tells you how much.

Scalings follow Lawn & Evans' indentation-fracture analysis, with the critical
load ``F_c = lambda K_c^4 / H^3`` and lateral cracks growing as ``F^(5/8)``.
Prefactors are collected in :mod:`lapsim.calibration`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lapsim.calibration import Calibration, DEFAULT
from lapsim.contact import ContactState
from lapsim.materials import Solid


def critical_crack_load(solid: Solid, calibration: Calibration = DEFAULT) -> float:
    """Load above which a sharp indenter cracks the solid rather than flowing it.

    ``F_c = lambda * K_Ic^4 / H^3``.  The fourth power of toughness makes this
    brutally selective: agate cracks above about 11 mN, while nephrite jade --
    only three times tougher -- needs over 100 times more load, which is exactly
    why jade will not cut and has to be ground down by plowing.
    """
    return (
        calibration.crack_initiation_coeff
        * solid.fracture_toughness**4
        / solid.hardness**3
    )


def lateral_crack(solid: Solid, load: np.ndarray, tip_semi_angle_deg: float,
                  calibration: Calibration = DEFAULT
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Lateral crack ``(half_length, depth)`` in metres for each applied load.

    Both follow Marshall--Lawn--Evans indentation-fracture scaling, written in
    dimensionally explicit form so the prefactors are pure numbers:

        ``c_L = alpha_L (cot t)^(5/12) (E/H)^(3/4) F^(5/8) / (K_c^(1/2) H^(1/8))``
        ``c_h = alpha_h (cot t)^(1/3)  (E/H)^(1/5) (F/H)^(1/2)``

    ``c_L`` sets how wide a chip comes out (and so the removal rate); ``c_h``
    sets how deep the damage goes (and so the finish and the subsurface damage).
    """
    cot_tip = 1.0 / math.tan(math.radians(tip_semi_angle_deg))
    modulus_ratio = solid.youngs_modulus / solid.hardness
    load = np.maximum(load, 0.0)

    half_length = (
        calibration.lateral_crack_length_coeff
        * cot_tip ** (5.0 / 12.0)
        * modulus_ratio**0.75
        * load**0.625
        / (solid.fracture_toughness**0.5 * solid.hardness**0.125)
    )
    depth = (
        calibration.lateral_crack_depth_coeff
        * cot_tip ** (1.0 / 3.0)
        * modulus_ratio**0.2
        * np.sqrt(load / solid.hardness)
    )
    return half_length, depth


@dataclass
class Finish:
    """Surface texture and damage left on the stone."""

    Ra_um: float
    Rz_um: float
    """Peak-to-valley height, microns -- the depth of the deepest pits, which is
    what you actually see as a matte."""

    subsurface_damage_um: float
    """Depth of cracked-but-still-attached stone below the finished surface.

    A rule of thumb (2.5 x Rz, see
    :attr:`lapsim.calibration.Calibration.subsurface_damage_ratio`), not a
    crack calculation, so it runs deeper than :attr:`deepest_scratch_um`."""

    deepest_scratch_um: float
    """Lateral-crack depth under the single coarsest grain the grading permits.

    This is your scratch risk, and it moves with the top cut: a grading with a
    sloppy top cut leaves the occasional scratch far deeper than the average
    finish.  It counts only the shallow lateral cracks that chip material out,
    not the deeper median cracks, so it is not a bound on the damage depth.
    """

    regime: str
    """``"ductile"``, ``"brittle"`` or ``"mixed"``."""

    brittle_fraction: float
    """Share of the load carried by grains pressed hard enough to crack the
    stone."""


@dataclass
class RemovalResult:
    """Everything that follows from the solved contact."""

    removal_rate: float
    """m/s of stone thickness."""

    lap_wear_rate: float
    """m/s of plate thickness.  A glass lap grinding quartz wears faster than the
    stone does, because it is the softer of the two."""

    volumetric_rate: float
    """m^3/s of stone."""

    finish: Finish
    preston_coefficient: float
    """``removal_rate / (pressure * speed)``, m^2/N.

    The classic lapping figure of merit.  Quoting it makes this model comparable
    with published lapping data and with your own stopwatch measurements.
    """

    drag_force: float
    """N, tangential force you have to push against."""

    drag_ratio: float
    """Drag divided by applied load -- the effective coefficient of friction.
    This is what your hand feels as "grabby" or "slippery"."""

    friction_power: float
    """W dissipated in the contact."""

    heat_flux: float
    """W/m^2 into the contact."""

    temperature_rise: float
    """K, steady-state rise of the contact zone over ambient."""

    plate_to_stone_wear_ratio: float
    """How fast the plate wears relative to the stone.  Above 1 you are
    consuming the glass faster than the stone, which means frequent replacement
    and a plate that dishes."""


def _load_weighted_quantile(values: np.ndarray, weights: np.ndarray,
                            quantile: float) -> float:
    """Quantile of ``values`` weighted by ``weights`` (both 1-D, same length)."""
    total = weights.sum()
    if total <= 0:
        return 0.0
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / total
    return float(np.interp(quantile, cumulative, values))


def evaluate_removal(contact: ContactState, work: Solid, lap: Solid,
                     nominal_pressure: float, contact_area: float,
                     sliding_speed: float, tip_semi_angle_deg: float,
                     ambient_temperature_C: float = 21.0,
                     calibration: Calibration = DEFAULT) -> RemovalResult:
    """Turn a solved contact into cutting rate, finish, drag and heat."""
    density = contact.grain_density_by_size   # grains per m^2, by size
    load = contact.grain_loads                # N, by size
    active = density.sum()

    tan_tip = math.tan(math.radians(tip_semi_angle_deg))

    def removal_per_metre(solid: Solid, penetration: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Swept area removed per metre of sliding, plus the crack geometry."""
        critical = critical_crack_load(solid, calibration)
        half_length, crack_depth = lateral_crack(
            solid, load, tip_semi_angle_deg, calibration
        )
        brittle = load > critical

        # Ductile: a triangular groove of depth `penetration`, most of which is
        # plowed aside rather than removed.
        groove_area = penetration**2 * tan_tip
        ductile = calibration.ductile_removal_efficiency * groove_area

        # Brittle: the lateral-crack envelope spalls out.
        chip_area = 2.0 * half_length * crack_depth
        fracture = calibration.brittle_removal_efficiency * chip_area

        swept = np.where(brittle, np.maximum(fracture, ductile), ductile)
        return swept, crack_depth, brittle

    swept_work, crack_depth_work, brittle_work = removal_per_metre(
        work, contact.penetration_work
    )
    swept_lap, _, _ = removal_per_metre(lap, contact.penetration_lap)

    # Depth removed per unit sliding distance, then per unit time.
    depth_per_metre_work = float((density * swept_work).sum())
    depth_per_metre_lap = float((density * swept_lap).sum())
    removal_rate = depth_per_metre_work * sliding_speed
    lap_wear_rate = depth_per_metre_lap * sliding_speed

    # ------------------------------------------------------------- finish
    if active > 0:
        brittle_fraction = float((density * brittle_work * load).sum()
                                 / max((density * load).sum(), 1e-30))
        crack_p90 = _load_weighted_quantile(crack_depth_work, density, 0.90)
        plow_p90 = _load_weighted_quantile(contact.penetration_work, density, 0.90)
    else:
        brittle_fraction, crack_p90, plow_p90 = 0.0, 0.0, 0.0

    if brittle_fraction > 0.8:
        regime = "brittle"
    elif brittle_fraction < 0.2:
        regime = "ductile"
    else:
        regime = "mixed"

    Ra = (
        brittle_fraction * calibration.roughness_from_crack_depth * crack_p90
        + (1.0 - brittle_fraction) * calibration.roughness_from_plow_depth * plow_p90
    )
    # The coarsest grain the grading allows sets the worst-case scratch.
    deepest = float(crack_depth_work[-1]) if crack_depth_work.size else 0.0

    finish = Finish(
        Ra_um=Ra * 1e6,
        Rz_um=Ra * calibration.peak_to_valley_ratio * 1e6,
        subsurface_damage_um=Ra * calibration.subsurface_damage_ratio
        * calibration.peak_to_valley_ratio * 1e6,
        deepest_scratch_um=deepest * 1e6,
        regime=regime,
        brittle_fraction=brittle_fraction,
    )

    # --------------------------------------------------------------- drag
    normal_load = nominal_pressure * contact_area
    grain_load = contact.grain_pressure * contact_area
    fluid_shear = (
        contact.slurry_viscosity * sliding_speed / max(contact.gap, 1e-9) * contact_area
    )
    drag = calibration.abrasive_friction_coeff * grain_load + fluid_shear
    power = drag * sliding_speed
    heat_flux = power / contact_area

    # Lumped convective balance for a wet, slurry-flooded contact.  Rough, but it
    # separates "warm" from "hot enough to shock a stone".
    wet_contact_conductance = 900.0     # W/(m^2 K)
    temperature_rise = heat_flux / wet_contact_conductance

    return RemovalResult(
        removal_rate=removal_rate,
        lap_wear_rate=lap_wear_rate,
        volumetric_rate=removal_rate * contact_area,
        finish=finish,
        preston_coefficient=removal_rate / max(nominal_pressure * sliding_speed, 1e-30),
        drag_force=drag,
        drag_ratio=drag / max(normal_load, 1e-30),
        friction_power=power,
        heat_flux=heat_flux,
        temperature_rise=temperature_rise,
        plate_to_stone_wear_ratio=lap_wear_rate / max(removal_rate, 1e-30),
    )
