"""The slurry: abrasive grains suspended in the vehicle.

Three questions this module answers, all of which are really the same question
("how thick is it?") asked from different directions:

1.  **Will it stay on the plate, or run off?**  A slurry holds its position if
    it has a yield stress -- a stress it can support without flowing at all.
    Below that it is a viscous liquid and only drains slowly; above it, it sits
    in a ridge.  Both regimes are covered here.
2.  **Will the grains stay suspended, or settle out in a minute?**  Two
    mechanisms: viscous drag slows settling, and a yield stress can stop it
    completely.  Coarse grit in thin liquid settles in seconds; that is why the
    glycerine is there.
3.  **Will it drag, or hydroplane?**  Answered in :mod:`lapsim.contact`, but the
    effective viscosity it needs comes from here.

The central variable is the solids volume fraction ``phi``.  Note how different
that is from the mass fraction people actually weigh out: silicon carbide is
3.2x denser than water, so 30 wt% solids is only about 11 vol%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from lapsim.calibration import Calibration, DEFAULT
from lapsim.materials import Abrasive, GritGrade
from lapsim.vehicle import Vehicle

GRAVITY = 9.80665

STOKES_YIELD_NUMBER = 0.10
"""Critical yield number for a sphere held static in a yield-stress fluid.

A particle never settles at all when ``tau_y / (delta_rho * g * d)`` exceeds
this.  Measured values sit between 0.06 and 0.14 depending on shape; 0.10 is a
reasonable middle for angular grains.
"""

CONSISTENCY_LADDER = [
    # (upper bound of apparent viscosity at 10 1/s in Pa*s, everyday name)
    (0.005, "water-thin"),
    (0.03, "like milk"),
    (0.12, "like single cream"),
    (0.5, "like double cream"),
    (2.0, "like thin honey"),
    (8.0, "like thick honey"),
    (float("inf"), "a paste that holds its shape"),
]
"""Turns a viscosity into something you can check by eye at the bench."""

LOW_SHEAR_RATE = 10.0
"""1/s -- roughly what you impose tipping the bottle or lifting a spoon.  Used
for the consistency description, because that is the shear rate your intuition
about "runny" was built at."""


@dataclass
class Slurry:
    """An abrasive suspension, ready to have its properties interrogated.

    Treated as immutable once built: every "change" (:meth:`with_vehicle`,
    :meth:`scaled_to_volume`) returns a new instance.  That is what makes the
    :func:`cached_property` results below safe -- do not mutate the fields in
    place or the caches will go stale.
    """

    abrasive: Abrasive
    grade: GritGrade
    vehicle: Vehicle
    solids_volume_fraction: float
    calibration: Calibration = DEFAULT

    # ------------------------------------------------------------- builders
    @classmethod
    def from_mass_fraction(cls, abrasive: Abrasive, grade: GritGrade,
                           vehicle: Vehicle, solids_mass_fraction: float,
                           calibration: Calibration = DEFAULT) -> "Slurry":
        """Build from grams of grit per gram of slurry -- how you weigh it out."""
        w = min(max(solids_mass_fraction, 0.0), 0.95)
        vehicle_volume_per_kg = (1.0 - w) / vehicle.density
        solids_volume_per_kg = w / abrasive.density
        phi = solids_volume_per_kg / (solids_volume_per_kg + vehicle_volume_per_kg)
        return cls(abrasive, grade, vehicle, phi, calibration)

    @classmethod
    def from_recipe(cls, abrasive: Abrasive, grade: GritGrade,
                    grit_grams: float, water_ml: float, glycerine_ml: float,
                    soap_drops: float = 2.0, temperature_C: float = 21.0,
                    calibration: Calibration = DEFAULT) -> "Slurry":
        """Build from exactly what you would put in the bottle."""
        vehicle = Vehicle.from_volumes(water_ml, glycerine_ml, soap_drops,
                                       temperature_C)
        solids_volume = grit_grams * 1e-3 / abrasive.density
        phi = solids_volume / (solids_volume + vehicle.volume)
        return cls(abrasive, grade, vehicle, phi, calibration)

    def scaled_to_volume(self, volume: float) -> "Slurry":
        """The same mix, but only ``volume`` m^3 of it.

        Composition (solids fraction, glycerine ratio, soap strength) is
        intensive and unchanged; the *amount* is extensive and matters for
        drying, because what dries is the film on the plate, not the bottle.
        """
        factor = volume / self.volume
        vehicle = Vehicle(
            water_mass=self.vehicle.water_mass * factor,
            glycerine_mass=self.vehicle.glycerine_mass * factor,
            soap_mass=self.vehicle.soap_mass * factor,
            temperature_C=self.vehicle.temperature_C,
        )
        return Slurry(self.abrasive, self.grade, vehicle,
                      self.solids_volume_fraction, self.calibration)

    def with_vehicle(self, vehicle: Vehicle) -> "Slurry":
        """Same grains, different (e.g. partly evaporated) liquid.

        The grain volume is held fixed and phi recomputed, which is precisely
        what happens as the slurry dries.
        """
        solids_volume = self.solids_volume  # per unit reference volume
        phi = solids_volume / (solids_volume + vehicle.volume)
        return Slurry(self.abrasive, self.grade, vehicle, phi, self.calibration)

    # ---------------------------------------------------------- composition
    @cached_property
    def solids_volume(self) -> float:
        """m^3 of grain accompanying this slurry's vehicle.

        Derived from the vehicle's actual volume and phi, so the pair stays
        consistent as the vehicle evaporates.
        """
        phi = self.solids_volume_fraction
        return self.vehicle.volume * phi / max(1.0 - phi, 1e-9)

    @property
    def volume(self) -> float:
        return self.solids_volume + self.vehicle.volume

    @property
    def solids_mass_fraction(self) -> float:
        solids_mass = self.solids_volume * self.abrasive.density
        return solids_mass / (solids_mass + self.vehicle.total_mass)

    @property
    def density(self) -> float:
        phi = self.solids_volume_fraction
        return phi * self.abrasive.density + (1.0 - phi) * self.vehicle.density

    # ------------------------------------------------------------ grain size
    @cached_property
    def max_packing_fraction(self) -> float:
        """phi_max: where the grains jam and the slurry stops being a fluid.

        The grain shape sets the base value; a wide size distribution raises it
        (small grains fit in the gaps between large ones), and good dispersion
        by the soap raises it a little more (flocs pack worse than single
        grains).
        """
        polydispersity_bonus = 1.0 + 0.12 * math.log(self.grade.sigma_g) / math.log(1.4)
        dispersion_bonus = 1.0 + 0.05 * self.vehicle.dispersion_quality
        return min(
            self.abrasive.max_packing * polydispersity_bonus * dispersion_bonus,
            0.64,
        )

    @cached_property
    def surface_volume_diameter(self) -> float:
        """Sauter mean diameter d_32, m.

        The right average for anything driven by interparticle contact: it
        weights by surface area per unit volume, so it tracks the fines that
        dominate a slurry's stickiness.
        """
        d, weights = self.grade.size_distribution()
        return float((weights * d**3).sum() / (weights * d**2).sum())

    @cached_property
    def coarse_diameter(self) -> float:
        """d90 by volume, m: the grain size that the coarsest tenth of the grit
        (by weight) is larger than.

        Taken on a volume basis on purpose.  Counted by number the powder is
        dominated by fines, and its number-d90 is far finer than the grains
        that actually drop out first.
        """
        d, weights = self.grade.size_distribution()
        volume = weights * d**3
        return float(np.interp(0.90, np.cumsum(volume) / volume.sum(), d))

    @cached_property
    def mean_grain_volume(self) -> float:
        d, weights = self.grade.size_distribution()
        return float((weights * math.pi / 6.0 * d**3).sum())

    @property
    def number_density(self) -> float:
        """Grains per cubic metre of slurry."""
        return self.solids_volume_fraction / self.mean_grain_volume

    # ------------------------------------------------------------- rheology
    @cached_property
    def relative_viscosity(self) -> float:
        """Krieger-Dougherty: mu_slurry / mu_vehicle.

        Diverges as phi approaches phi_max, which is the mathematical statement
        of "add one more spoonful and it turns to mud".
        """
        phi = self.solids_volume_fraction
        phi_max = self.max_packing_fraction
        ratio = min(phi / phi_max, 0.995)
        exponent = self.abrasive.intrinsic_viscosity * phi_max
        return (1.0 - ratio) ** (-exponent)

    @cached_property
    def plastic_viscosity(self) -> float:
        """High-shear (Bingham plastic) viscosity, Pa*s."""
        return self.vehicle.viscosity * self.relative_viscosity

    @cached_property
    def gel_fraction(self) -> float:
        """Solids fraction at which a load-bearing grain network first forms.

        Strongly size-dependent, and that dependence is the reason coarse and
        fine slurries need completely different recipes.  Interparticle
        attraction grows like the grain diameter while grain weight grows like
        its cube, so fine powders cohere into a network at a modest loading
        while coarse ones stay a pourable liquid right up to jamming.

        Practically: **your 320 and 400 grit will never develop a yield stress at
        a workable loading, so their consistency has to come from glycerine.**
        Fine grades, and especially the micron alumina, gel on their own -- so
        there you control consistency with the amount of grit instead.
        """
        calibration = self.calibration
        d_um = self.surface_volume_diameter * 1e6
        coarseness = d_um / (d_um + calibration.cohesive_size_um)
        return (
            calibration.gel_fraction_cohesive
            + (calibration.gel_fraction_coarse - calibration.gel_fraction_cohesive)
            * coarseness
        )

    @cached_property
    def yield_stress(self) -> float:
        """Static yield stress, Pa.

        Modelled as an interparticle-contact stress: the stress scale is surface
        tension over grain size, and the geometric factor blows up as the grains
        jam.  Two consequences worth internalising:

        * **Fine grit thickens far more than coarse grit at equal loading.**  The
          1/d scaling means 1000 grit at 20 vol% is stiff where 320 grit at
          20 vol% is still pourable.  This is usually the thing that surprises
          people moving down the grit sequence.
        * **Soap thins it.**  Lowering surface tension and deflocculating both
          cut the yield stress, which is why a drop of detergent can rescue a
          slurry that has gone claggy.
        """
        phi = self.solids_volume_fraction
        phi_max = self.max_packing_fraction
        phi_gel = self.gel_fraction
        if phi <= phi_gel:
            return 0.0
        head = (phi - phi_gel) / max(phi_max - phi, 1e-4)
        stress_scale = self.vehicle.surface_tension / self.surface_volume_diameter
        flocculation = 1.0 - 0.45 * self.vehicle.dispersion_quality
        return (
            self.calibration.yield_stress_coeff * stress_scale * head**2 * flocculation
        )

    def apparent_viscosity(self, shear_rate: float) -> float:
        """Bingham apparent viscosity at a given shear rate, Pa*s.

        ``tau = tau_y + mu_p * gamma_dot``, so the apparent viscosity is
        ``tau_y / gamma_dot + mu_p``.  At the very low shear rates of sitting on
        a plate this is enormous (that is the slurry holding still); at the
        3000 1/s of the grinding gap it collapses to the plastic viscosity.
        """
        rate = max(shear_rate, 1e-6)
        return self.yield_stress / rate + self.plastic_viscosity

    @property
    def consistency_label(self) -> str:
        """An everyday name for how thick this slurry feels."""
        mu = self.apparent_viscosity(LOW_SHEAR_RATE)
        for bound, name in CONSISTENCY_LADDER:
            if mu < bound:
                return name
        return CONSISTENCY_LADDER[-1][1]

    # ------------------------------------------------------------- stability
    @property
    def yield_number(self) -> float:
        """``tau_y / (delta_rho * g * d)``.

        Above :data:`STOKES_YIELD_NUMBER` the grains are held in place
        permanently and the slurry never separates.
        """
        delta_rho = self.abrasive.density - self.vehicle.density
        denominator = delta_rho * GRAVITY * self.coarse_diameter
        return self.yield_stress / max(denominator, 1e-12)

    @property
    def is_permanently_stable(self) -> bool:
        """True if even the coarse tail of grains cannot settle."""
        return self.yield_number >= STOKES_YIELD_NUMBER

    @property
    def settling_speed(self) -> float:
        """Hindered Stokes settling speed of a coarse grain, m/s.

        Uses the volume-basis 90th-percentile grain (:attr:`coarse_diameter`)
        because it is among the first to drop out, and applies the
        Richardson-Zaki hindrance factor because neighbouring grains get in each
        other's way.
        """
        if self.is_permanently_stable:
            return 0.0
        d_coarse = self.coarse_diameter
        delta_rho = self.abrasive.density - self.vehicle.density
        stokes = delta_rho * GRAVITY * d_coarse**2 / (18.0 * self.vehicle.viscosity)
        hindrance = (1.0 - self.solids_volume_fraction) ** 4.65
        return stokes * hindrance

    def settling_time(self, depth: float = 0.03) -> float:
        """Seconds for coarse grains to fall ``depth`` (default: a 30 mm bottle).

        Returns infinity for a slurry with enough yield stress to hold them.
        Anything under a minute means you are shaking the bottle constantly and
        grinding on whatever happens to be in suspension at the time.
        """
        speed = self.settling_speed
        return float("inf") if speed <= 0 else depth / speed

    # ----------------------------------------------- behaviour on the plate
    @property
    def puddle_height(self) -> float:
        """Equilibrium thickness of a blob left on a level plate, m.

        A yield-stress fluid stops spreading when its own weight can no longer
        overcome tau_y, at ``h ~ 1.7 * tau_y / (rho * g)``.  Below the gel point
        there is no such limit and the slurry thins until surface tension stops
        it, so a capillary floor is applied.

        This is the most directly checkable number in the whole model: put a
        blob on the plate, let it settle, and see how tall it is.
        """
        yield_limited = 1.7 * self.yield_stress / (self.density * GRAVITY)
        capillary_floor = 0.4 * math.sqrt(
            self.vehicle.surface_tension / (self.density * GRAVITY)
        )
        return max(yield_limited, capillary_floor)

    def squeeze_out_halflife(self, load: float, contact_radius: float,
                             gap: float) -> float:
        """Seconds for the contact film to thin to half its gap, s.

        Stefan's squeeze-film result for two parallel discs:
        ``dh/dt = -2 W h^3 / (3 pi mu R^4)``.  This is how long the slurry
        survives under the stone before it is pushed out and you have to
        re-charge the plate.  A yield stress stops squeeze flow entirely once the
        gap is small enough, which is reported as infinite life.
        """
        mu = self.plastic_viscosity
        squeeze_stress = load / (math.pi * contact_radius**2)
        if self.yield_stress > 0.0:
            # Squeeze flow arrests when the driving stress can no longer exceed
            # the yield stress over the contact length scale.
            arrest_gap = 2.0 * self.yield_stress * contact_radius / squeeze_stress
            if arrest_gap >= gap:
                return float("inf")
        coefficient = 3.0 * math.pi * mu * contact_radius**4 / (2.0 * load)
        return coefficient * (1.0 / (gap / 2.0) ** 2 - 1.0 / gap**2) / 2.0

    def summary(self) -> dict[str, float | str]:
        return {
            "solids_vol_pct": self.solids_volume_fraction * 100,
            "solids_wt_pct": self.solids_mass_fraction * 100,
            "phi_max": self.max_packing_fraction,
            "phi_gel": self.gel_fraction,
            "d32_um": self.surface_volume_diameter * 1e6,
            "vehicle_mPas": self.vehicle.viscosity * 1e3,
            "plastic_mPas": self.plastic_viscosity * 1e3,
            "yield_stress_Pa": self.yield_stress,
            "puddle_height_mm": self.puddle_height * 1e3,
            "settling_time_s": self.settling_time(),
            "consistency": self.consistency_label,
        }
