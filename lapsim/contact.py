"""What happens in the gap between the two flat plates.

This is the mechanical heart of the simulator.  The stone is pressed against the
lap plate with a known force; slurry fills the gap; the question is how the load
divides up, and that single answer determines everything else -- cutting rate,
finish, drag, plate wear.

The gap width ``h`` is found by force balance.  Three things can hold the plates
apart, and they trade off against each other:

1.  **Grains bridging the gap.**  Only grains coarser than ``h`` touch both
    surfaces at once.  Because a graded powder has a size distribution, this is
    a small fraction of what is in there -- typically well under a percent.  Each
    bridging grain is squeezed by ``delta = d - h`` and indents both surfaces.
2.  **Hydrodynamic pressure.**  A converging film of viscous slurry generates
    lift.  Push this too far -- thick vehicle, fast strokes, fine grit -- and the
    stone floats clear of the grains and stops cutting altogether.  Lapidaries
    know this as the plate "going slick".
3.  **Nothing.**  If neither can carry the load the plates come into direct
    contact, which is how you get a glazed plate and burnished stone.

Both grain pressure and hydrodynamic pressure fall as the gap opens, so the
balance has exactly one solution and a bracketed root-find always converges.

Two further pieces of physics keep the answer honest:

*   **The load splits between the two surfaces by hardness.**  On a glass plate
    grinding agate, the glass is softer, so almost 60 % of each grain's
    penetration goes into the *plate*.  That is not a defect of the method -- it
    is how a glass lap works, and it is why the plate wears and slowly dishes.
*   **Grains fracture.**  A grain can only carry so much load before it breaks.
    That caps penetration once grains start to crush.  At ordinary hand
    pressures few do, so in this model the finish still coarsens as you press
    harder (Ra roughly doubles from 20 N to 80 N at 600 grit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from lapsim.calibration import Calibration, DEFAULT
from lapsim.materials import Solid
from lapsim.slurry import Slurry


@dataclass
class ContactState:
    """The solved state of the plate gap."""

    gap: float
    """m, separation of the two flat surfaces."""

    regime: str
    """``"abrasive"``, ``"hydroplaning"`` or ``"plate contact"``."""

    grain_pressure: float
    """Pa, share of the nominal pressure carried by bridging grains."""

    hydrodynamic_pressure: float
    """Pa, share carried by the squeezed slurry film."""

    load_fraction_on_grains: float
    """0-1.  Below about 0.5 the process is more float than cut."""

    active_grain_density: float
    """Grains per m^2 of contact actually bridging the gap."""

    available_grain_density: float
    """Grains per m^2 present in the gap, bridging or not."""

    active_fraction: float
    """Bridging grains divided by grains present."""

    diameters: np.ndarray
    """m, size grid of the grading."""

    grain_loads: np.ndarray
    """N, load carried by a bridging grain of each size (0 if not bridging)."""

    grain_density_by_size: np.ndarray
    """Grains per m^2 at each size, bridging only."""

    penetration_work: np.ndarray
    """m, depth each bridging grain reaches into the stone."""

    penetration_lap: np.ndarray
    """m, depth each bridging grain reaches into the lap plate."""

    fractured: np.ndarray
    """Boolean mask: grains whose load exceeds their crushing strength."""

    fractured_fraction: float
    """Fraction of bridging grains being crushed.  High values mean the slurry
    is breaking itself down fast and will need renewing."""

    shear_rate: float
    """1/s in the gap -- sets which part of the flow curve matters."""

    slurry_viscosity: float
    """Pa*s, apparent viscosity at that shear rate."""

    lap_anchoring: float
    """Mean penetration into the lap divided by grain diameter.

    Above roughly 0.1 the grains are pinned in the plate and cut like a fixed
    abrasive: faster, with directional scratches.  Below that they roll, giving a
    slower, more uniform matte.
    """

    @property
    def is_cutting(self) -> bool:
        return self.regime == "abrasive" and self.load_fraction_on_grains > 0.05


def composite_hardness(work: Solid, lap: Solid) -> tuple[float, float]:
    """Effective hardness of the pair, and the share of penetration taken by
    the stone.

    A grain pressed between two surfaces indents both.  The load is common, so
    for conical indentation ``F = pi tan^2(theta) H delta^2`` in each surface,
    giving ``delta_lap / delta_work = sqrt(H_work / H_lap)``.  Summing the two
    depths yields an effective hardness

        ``H_eff = H_work / (1 + sqrt(H_work / H_lap))^2``

    which is *softer than either material*: a glass lap against quartz gives
    about 1.9 GPa against the 10.5 GPa of the quartz alone.  That softness is
    the whole reason a glass plate grinds -- and the reason it wears.
    """
    hardness_ratio = math.sqrt(work.hardness / lap.hardness)
    effective = work.hardness / (1.0 + hardness_ratio) ** 2
    work_depth_share = 1.0 / (1.0 + hardness_ratio)
    return effective, work_depth_share


def hydrodynamic_pressure(viscosity: float, speed: float, length: float,
                          gap: float, convergence: float,
                          efficiency: float = 1.0) -> float:
    """Mean lift pressure of a converging slurry film, Pa.

    Reynolds' solution for a plane inclined slider of length ``length`` with
    inlet/outlet gap ratio ``convergence``.  A perfectly parallel gap generates
    no lift at all, which the formula respects: the bracket vanishes as the
    convergence ratio approaches one.

    The ``1/gap^2`` dependence is what makes this matter for fine grits.  Going
    from FEPA F320 to F1000 shrinks the gap about fivefold and so multiplies the
    lift about 25 times -- which is why a thick vehicle that behaves perfectly
    at 320 will make 1000 grit skate.

    ``efficiency`` scales the flooded, steady-state result down to what a
    starved, reciprocating contact actually develops; see
    :attr:`lapsim.calibration.Calibration.hydrodynamic_efficiency`.
    """
    K = max(convergence, 1.0)
    epsilon = K - 1.0
    if epsilon < 1e-3:
        bracket = epsilon**3 / 12.0  # series limit, avoids 0/0
    else:
        bracket = math.log(K) - 2.0 * epsilon / (K + 1.0)
    if epsilon <= 0:
        return 0.0
    flooded = 6.0 * viscosity * speed * length * bracket / (gap**2 * epsilon**2)
    return efficiency * flooded


def solve_contact(slurry: Slurry, work: Solid, lap: Solid,
                  nominal_pressure: float, sliding_speed: float,
                  contact_length: float, convergence: float = 1.5,
                  calibration: Calibration = DEFAULT) -> ContactState:
    """Find the gap that balances the applied pressure, and everything with it.

    Parameters
    ----------
    nominal_pressure:
        Pa -- applied load divided by the plan area of the stone face.
    contact_length:
        m -- length of the stone face along the stroke direction.
    convergence:
        Inlet gap divided by outlet gap.  Hand lapping is never perfectly
        parallel; 1.3-1.8 is realistic, and 1.0 switches hydrodynamic lift off.
    """
    abrasive = slurry.abrasive
    diameters, number_weights = slurry.grade.size_distribution()
    effective_hardness, work_share = composite_hardness(work, lap)

    tip_angle = math.radians(abrasive.tip_semi_angle_deg)
    tan_sq = math.tan(tip_angle) ** 2

    grains_per_volume = slurry.number_density * calibration.gap_enrichment
    mean_grain_area = float((number_weights * math.pi / 4.0 * diameters**2).sum())
    monolayer_cap = calibration.monolayer_packing / mean_grain_area

    crushing_loads = np.asarray(abrasive.crushing_load(diameters), dtype=float)

    def state_at(gap: float) -> dict:
        """Grain-borne pressure and its breakdown, at a trial gap."""
        available = min(grains_per_volume * gap, monolayer_cap)
        squeeze = np.maximum(diameters - gap, 0.0)
        bridging = squeeze > 0.0

        # Conical indentation of two surfaces in series, capped by grain fracture.
        elastic_load = math.pi * tan_sq * effective_hardness * squeeze**2
        load = np.minimum(elastic_load, crushing_loads)
        fractured = bridging & (elastic_load > crushing_loads)

        density_by_size = available * number_weights * bridging
        pressure = float((density_by_size * load).sum())
        return {
            "available": available,
            "load": load * bridging,
            "density_by_size": density_by_size,
            "pressure": pressure,
            "fractured": fractured,
            "squeeze": squeeze,
        }

    def hydro_at(gap: float) -> tuple[float, float, float]:
        shear_rate = sliding_speed / max(gap, 1e-9)
        viscosity = slurry.apparent_viscosity(shear_rate)
        pressure = hydrodynamic_pressure(
            viscosity, sliding_speed, contact_length, gap, convergence,
            calibration.hydrodynamic_efficiency,
        )
        return pressure, shear_rate, viscosity

    def excess(gap: float) -> float:
        """Pressure the gap can support, minus what is applied."""
        return state_at(gap)["pressure"] + hydro_at(gap)[0] - nominal_pressure

    d_max = float(diameters[-1])

    # Case 1: the film alone can float the stone off the grains entirely.
    if hydro_at(d_max)[0] >= nominal_pressure:
        widest = 200.0 * d_max
        if hydro_at(widest)[0] >= nominal_pressure:
            # A near-jammed paste: its yield stress holds the stone up at any
            # film thickness, so there is no balance point to find.
            gap = widest
        else:
            gap = brentq(lambda g: hydro_at(g)[0] - nominal_pressure,
                         d_max, widest, xtol=1e-12, rtol=1e-10)
        regime = "hydroplaning"
    else:
        lower = 1e-9
        if excess(lower) <= 0.0:
            # Nothing can carry the load: the surfaces touch directly.
            gap, regime = lower, "plate contact"
        else:
            gap = brentq(excess, lower, d_max, xtol=1e-14, rtol=1e-12)
            regime = "abrasive"

    solved = state_at(gap)
    hydro, shear_rate, viscosity = hydro_at(gap)

    density_by_size = solved["density_by_size"]
    active = float(density_by_size.sum())
    grain_pressure = solved["pressure"]
    total = grain_pressure + hydro

    # Split each grain's penetration between stone and plate by hardness.
    load = solved["load"]
    with np.errstate(invalid="ignore", divide="ignore"):
        total_penetration = np.sqrt(
            np.maximum(load, 0.0) / (math.pi * tan_sq * effective_hardness)
        )
    penetration_work = total_penetration * work_share
    penetration_lap = total_penetration * (1.0 - work_share)

    if active > 0:
        weights = density_by_size / active
        anchoring = float((weights * penetration_lap / diameters).sum())
        fractured_fraction = float((weights * solved["fractured"]).sum())
    else:
        anchoring = 0.0
        fractured_fraction = 0.0

    return ContactState(
        gap=gap,
        regime=regime,
        grain_pressure=grain_pressure,
        hydrodynamic_pressure=hydro,
        load_fraction_on_grains=grain_pressure / total if total > 0 else 0.0,
        active_grain_density=active,
        available_grain_density=solved["available"],
        active_fraction=active / solved["available"] if solved["available"] > 0 else 0.0,
        diameters=diameters,
        grain_loads=load,
        grain_density_by_size=density_by_size,
        penetration_work=penetration_work,
        penetration_lap=penetration_lap,
        fractured=solved["fractured"],
        fractured_fraction=fractured_fraction,
        shear_rate=shear_rate,
        slurry_viscosity=viscosity,
        lap_anchoring=anchoring,
    )
