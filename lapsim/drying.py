"""How long the slurry stays usable before it dries out.

This module explains -- and quantifies -- the effect you noticed at the bench:
**less grit in the mix stays wet longer.**  There are two reasons, and they
compound.

The simple one is inventory.  A given puddle of slurry is part grain and part
liquid.  Load it heavily and there is less liquid in the same puddle, so the same
evaporation rate empties it sooner.

The interesting one is the endpoint.  Only the *water* evaporates -- glycerine is
effectively non-volatile, and it is hygroscopic, so it holds water back.  A
glycerol solution stops losing water the moment its water activity falls to the
room's relative humidity.  That fixes the final liquid volume regardless of how
long you leave it, and therefore fixes the final solids fraction:

    phi_end = V_solids / (V_solids + V_liquid_at_equilibrium)

If ``phi_end`` stays below the jamming point, **the slurry cannot dry to a crust,
ever** -- it thickens to a glycerine paste and stops there, and a spritz of water
brings it straight back.  If ``phi_end`` is above jamming, the slurry will seize
no matter how much glycerine is in it, and the only fixes are less grit or more
glycerine.

That gives a genuinely useful design rule, and it is what
:func:`equilibrium_state` computes.  The full time course is in
:func:`dry_out`, which integrates

    dm_water/dt = -k * A * c_sat(T) * (a_w - RH)

with the mass transfer coefficient ``k`` rising with air movement over the plate,
and the exposed area ``A`` being the *spread* area, not the bottle's cross
section.  Spreading a few millilitres over a 200 mm plate is what makes slurry
dry in minutes rather than hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lapsim.calibration import Calibration, DEFAULT
from lapsim.slurry import Slurry
from lapsim.vehicle import (
    Vehicle,
    glycerine_fraction_at_activity,
    saturated_vapour_concentration,
)


@dataclass
class DryingResult:
    """Time course and endpoint of a drying slurry film."""

    times_s: np.ndarray
    solids_fraction: np.ndarray
    yield_stress_Pa: np.ndarray
    viscosity_Pas: np.ndarray
    glycerine_fraction: np.ndarray
    water_mass_g: np.ndarray

    equilibrium_solids_fraction: float
    """Solids fraction the slurry settles at once evaporation stops."""

    jamming_solids_fraction: float
    """Solids fraction at which it is no longer workable."""

    seizes: bool
    """True if the slurry will eventually dry to an unworkable crust."""

    time_to_seize_s: float
    """Seconds until unworkable; infinite if it never gets there."""

    workable_time_s: float
    """Seconds until the slurry leaves the target consistency band.

    This is the practical answer to "how often do I have to re-wet the plate?"
    """

    initial_water_g: float
    equilibrium_water_g: float
    evaporation_rate_g_per_min: float
    """Initial rate, before the liquid concentrates and slows down."""

    exposed_area_cm2: float
    film_volume_ml: float
    film_thickness_um: float
    notes: list[str] = field(default_factory=list)


def mass_transfer_coefficient(air_speed_m_s: float,
                              calibration: Calibration = DEFAULT) -> float:
    """Water-vapour mass transfer coefficient over the plate, m/s.

    Still air gives ~0.003 m/s; hand movement over the work is worth a few
    tenths of a m/s of effective air speed, and a fan or an open window can
    easily triple the drying rate.
    """
    return (
        calibration.mass_transfer_base_m_s
        + calibration.mass_transfer_per_air_speed * max(air_speed_m_s, 0.0)
    )


def equilibrium_state(slurry: Slurry, relative_humidity: float) -> tuple[float, float]:
    """``(equilibrium_solids_fraction, equilibrium_water_mass)`` for this slurry.

    The endpoint of drying, reached without integrating anything: the liquid
    ends at whatever glycerine strength has a water activity equal to the room's
    humidity, which fixes its volume and hence the final solids fraction.
    """
    vehicle = slurry.vehicle
    final_glycerine_fraction = glycerine_fraction_at_activity(relative_humidity)

    # Glycerine mass is conserved; solve for the water mass that dilutes it to
    # the equilibrium strength.
    if final_glycerine_fraction >= 1.0 - 1e-9:
        final_water_mass = 0.0
    else:
        final_water_mass = vehicle.glycerine_mass * (
            1.0 / final_glycerine_fraction - 1.0
        )
    final_water_mass = min(final_water_mass, vehicle.water_mass)

    final_vehicle = vehicle.with_water_mass(final_water_mass)
    solids_volume = slurry.solids_volume
    final_phi = solids_volume / (solids_volume + final_vehicle.volume)
    return final_phi, final_water_mass


def dry_out(slurry: Slurry, relative_humidity: float = 0.45,
            air_speed_m_s: float = 0.3, exposed_area: float = 0.020,
            film_volume_ml: float = 8.0,
            target_yield_stress_Pa: float = 25.0,
            target_viscosity_Pas: float = 2.0,
            duration_s: float = 3600.0, steps: int = 900,
            calibration: Calibration = DEFAULT) -> DryingResult:
    """Integrate water loss from a spread slurry film.

    Parameters
    ----------
    exposed_area:
        m^2 over which the slurry is spread.  Together with ``film_volume_ml``
        this dominates the answer: the same 5 ml lasts hours in a puddle and
        minutes smeared over a 200 mm plate.
    film_volume_ml:
        How much slurry is actually on the plate.  Only this is at risk of
        drying -- what is left in the bottle is safe.
    target_yield_stress_Pa, target_viscosity_Pas:
        Upper edge of the consistency you are willing to work with.  Crossing
        either ends the "workable time".
    """
    slurry = slurry.scaled_to_volume(film_volume_ml * 1e-6)
    vehicle = slurry.vehicle
    temperature = vehicle.temperature_C
    c_sat = saturated_vapour_concentration(temperature)
    k_mass = mass_transfer_coefficient(air_speed_m_s, calibration)

    phi_max = slurry.max_packing_fraction
    jamming = calibration.workable_solids_headroom * phi_max
    equilibrium_phi, equilibrium_water = equilibrium_state(slurry, relative_humidity)

    dt = duration_s / steps
    water_mass = vehicle.water_mass

    times, phis, yields, viscosities, glycerines, waters = [], [], [], [], [], []
    workable_time = float("inf")
    time_to_seize = float("inf")
    initial_rate = None

    for step in range(steps + 1):
        time = step * dt
        current = slurry.with_vehicle(vehicle.with_water_mass(water_mass))

        times.append(time)
        phis.append(current.solids_volume_fraction)
        yields.append(current.yield_stress)
        viscosities.append(current.plastic_viscosity)
        glycerines.append(current.vehicle.glycerine_mass_fraction)
        waters.append(water_mass * 1e3)

        too_stiff = (
            current.yield_stress > target_yield_stress_Pa
            or current.plastic_viscosity > target_viscosity_Pas
        )
        if too_stiff and workable_time == float("inf"):
            workable_time = time
        if current.solids_volume_fraction >= jamming and time_to_seize == float("inf"):
            time_to_seize = time

        driving_force = current.vehicle.water_activity - relative_humidity
        flux = k_mass * exposed_area * c_sat * max(driving_force, 0.0)
        if initial_rate is None:
            initial_rate = flux * 60.0 * 1e3   # g/min

        water_mass = max(water_mass - flux * dt, equilibrium_water)

    notes: list[str] = []
    seizes = equilibrium_phi >= jamming
    if seizes:
        notes.append(
            f"Will dry to an unworkable crust: at {relative_humidity * 100:.0f} % RH "
            f"the liquid concentrates until solids reach "
            f"{equilibrium_phi * 100:.0f} vol%, past the {jamming * 100:.0f} vol% "
            "workable ceiling. Use less grit or more glycerine."
        )
    else:
        notes.append(
            f"Cannot dry to a crust: glycerine holds the endpoint at "
            f"{equilibrium_phi * 100:.0f} vol% solids, below the "
            f"{jamming * 100:.0f} vol% ceiling. It thickens and stops; a spray of "
            "water restores it."
        )
    if workable_time < 300:
        notes.append(
            f"Leaves the target consistency after only {workable_time / 60:.1f} min "
            "of exposure -- expect to re-wet often."
        )

    return DryingResult(
        times_s=np.array(times),
        solids_fraction=np.array(phis),
        yield_stress_Pa=np.array(yields),
        viscosity_Pas=np.array(viscosities),
        glycerine_fraction=np.array(glycerines),
        water_mass_g=np.array(waters),
        equilibrium_solids_fraction=equilibrium_phi,
        jamming_solids_fraction=jamming,
        seizes=seizes,
        time_to_seize_s=time_to_seize,
        workable_time_s=workable_time,
        initial_water_g=vehicle.water_mass * 1e3,
        equilibrium_water_g=equilibrium_water * 1e3,
        evaporation_rate_g_per_min=initial_rate or 0.0,
        exposed_area_cm2=exposed_area * 1e4,
        film_volume_ml=film_volume_ml,
        film_thickness_um=film_volume_ml * 1e-6 / exposed_area * 1e6,
        notes=notes,
    )
