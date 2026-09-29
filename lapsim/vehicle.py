"""The carrier liquid: water, glycerine and dish soap.

This module has no fitted constants worth arguing about -- it is almost all
published correlation:

*   **Viscosity** of glycerol-water follows Cheng (2008), *Formula for the
    viscosity of a glycerol-water mixture*, Ind. Eng. Chem. Res. 47, 3285.  It
    reproduces the measured table to about 1 % from 0 to 100 degC, which matters
    because glycerol-water viscosity is wildly non-linear: going from 50 % to
    80 % glycerine multiplies viscosity by roughly fifteen.
*   **Water activity** sets how far the slurry can dry.  Only the water
    evaporates; the glycerine stays behind and holds onto water hygroscopically.
    A glycerol solution stops losing water once its activity equals the room's
    relative humidity, which puts a hard floor on how dry the slurry can get.
    Glycerol holds water more tightly than an ideal solute, so activity falls
    *below* Raoult's law; a two-term correction fitted to measured values
    reproduces them to within 0.02 up to 92 wt% glycerol.
*   **Surface tension** drops from 72 mN/m to about 30 mN/m once dish soap is
    past its critical micelle concentration.  That does three things at once:
    wets the plate so the slurry spreads instead of beading, deflocculates the
    grains so they do not clump into scratch-makers, and lowers the slurry's
    yield stress.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MOLAR_MASS_WATER = 0.018015       # kg/mol
MOLAR_MASS_GLYCEROL = 0.092094    # kg/mol
GAS_CONSTANT = 8.31446            # J/(mol K)

DROP_VOLUME_ML = 0.05
"""A drop from a squeeze bottle. Used to turn soap volumes into drop counts."""

SOAP_CMC_VOLUME_FRACTION = 2.0e-4
"""Roughly where a household dish detergent saturates the air-water interface,
as a volume fraction of neat detergent in the liquid.  About 0.4 ml of soap in
2 litres, i.e. a few drops in a cupful -- which is why "a drop or two" is the
standard instruction and why more does nothing for wetting.
"""

SURFACE_TENSION_SATURATED = 0.030
"""N/m, surface tension of a detergent solution above the CMC."""


# --------------------------------------------------------------------------- #
# Pure-component properties
# --------------------------------------------------------------------------- #


def water_viscosity(temperature_C: float) -> float:
    """Dynamic viscosity of water, Pa*s (Cheng 2008)."""
    T = temperature_C
    return 1.790e-3 * math.exp(-(1230.0 + T) * T / (36100.0 + 360.0 * T))


def glycerol_viscosity(temperature_C: float) -> float:
    """Dynamic viscosity of pure glycerol, Pa*s (Cheng 2008)."""
    T = temperature_C
    return 12.100 * math.exp((-1233.0 + T) * T / (9900.0 + 70.0 * T))


def water_density(temperature_C: float) -> float:
    """kg/m^3."""
    return 1000.0 * (1.0 - (abs(temperature_C - 4.0) / 622.0) ** 1.7)


def glycerol_density(temperature_C: float) -> float:
    """kg/m^3."""
    return 1277.0 - 0.654 * temperature_C


def water_saturation_pressure(temperature_C: float) -> float:
    """Saturation vapour pressure of water, Pa (Magnus form)."""
    T = temperature_C
    return 610.94 * math.exp(17.625 * T / (T + 243.04))


def saturated_vapour_concentration(temperature_C: float) -> float:
    """Mass of water vapour per cubic metre of saturated air, kg/m^3."""
    p_sat = water_saturation_pressure(temperature_C)
    return p_sat * MOLAR_MASS_WATER / (GAS_CONSTANT * (temperature_C + 273.15))


# --------------------------------------------------------------------------- #
# Mixture properties
# --------------------------------------------------------------------------- #


def glycerol_water_viscosity(glycerine_mass_fraction: float,
                             temperature_C: float) -> float:
    """Viscosity of a glycerol-water mixture, Pa*s.

    Cheng's (2008) weighted-power-law blend.  ``glycerine_mass_fraction`` is on
    a water+glycerine basis, ignoring the trace of soap.
    """
    Cm = min(max(glycerine_mass_fraction, 0.0), 1.0)
    T = temperature_C
    a = 0.705 - 0.0017 * T
    b = (4.9 + 0.036 * T) * a**2.5
    alpha = (
        1.0 - Cm
        + (a * b * Cm * (1.0 - Cm)) / (a * Cm + b * (1.0 - Cm))
        if 0.0 < Cm < 1.0
        else 1.0 - Cm
    )
    mu_w = water_viscosity(T)
    mu_g = glycerol_viscosity(T)
    return mu_w**alpha * mu_g ** (1.0 - alpha)


def glycerol_water_density(glycerine_mass_fraction: float,
                           temperature_C: float) -> float:
    """Mixture density, kg/m^3, by ideal volume addition (~0.5 % low)."""
    w_g = min(max(glycerine_mass_fraction, 0.0), 1.0)
    specific_volume = (
        (1.0 - w_g) / water_density(temperature_C)
        + w_g / glycerol_density(temperature_C)
    )
    return 1.0 / specific_volume


def glycerine_mole_fraction(glycerine_mass_fraction: float) -> float:
    """Convert mass fraction to mole fraction of glycerol."""
    w_g = min(max(glycerine_mass_fraction, 0.0), 1.0)
    n_g = w_g / MOLAR_MASS_GLYCEROL
    n_w = (1.0 - w_g) / MOLAR_MASS_WATER
    total = n_g + n_w
    return n_g / total if total > 0 else 1.0


WATER_ACTIVITY_K = 1.39
WATER_ACTIVITY_C = 1.38
"""Coefficients of ``ln(gamma_w) = -K x_g^2 + C x_g^3``.

Least-squares fit to the measured water activities of glycerol-water mixtures
at 25 C in *Molecular Basis of Water Activity in Glycerol-Water Mixtures*
(PMC6839025, Table 1), over glycerol mole fractions up to 0.69 (92 wt%).  The
first term alone is Norrish's equation, whose usual glycerol constant of 1.16
fits well only up to about 60 wt%.
"""


def water_activity(glycerine_mass_fraction: float) -> float:
    """Activity of water over a glycerol solution, 0-1.

    ``a_w = x_w exp(-K x_g^2 + C x_g^3)``, which sits below Raoult's law because
    glycerol binds water.  This is the quantity that decides whether your
    slurry can ever dry out: evaporation stops dead once the activity falls to
    the room's relative humidity.
    """
    x_g = glycerine_mole_fraction(glycerine_mass_fraction)
    x_w = 1.0 - x_g
    log_gamma = -WATER_ACTIVITY_K * x_g**2 + WATER_ACTIVITY_C * x_g**3
    return min(x_w * math.exp(log_gamma), 1.0)


def glycerine_fraction_at_activity(target_activity: float) -> float:
    """Invert :func:`water_activity`: the glycerine mass fraction whose water
    activity equals ``target_activity``.

    In plain terms: given the humidity in your workshop, this is the glycerine
    strength your slurry's liquid will end up at once it stops evaporating.
    """
    target = min(max(target_activity, 1e-6), 1.0)
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if water_activity(mid) > target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def surface_tension(glycerine_mass_fraction: float,
                    soap_volume_fraction: float) -> float:
    """Liquid-air surface tension, N/m.

    Glycerine lowers it slightly (72 -> 63 mN/m pure).  Soap follows the
    Szyszkowski equation, ``gamma = gamma_0 - B ln(1 + c/a)``, which falls
    steeply and then stops dead at the critical micelle concentration: beyond
    that point extra detergent forms micelles in the bulk instead of adsorbing
    at the surface, so the interface cannot be made any more active.

    The practical consequence is that one drop per 100 ml already does the whole
    job, and the tenth drop does nothing but foam.
    """
    clean = 0.0720 - 0.0086 * min(max(glycerine_mass_fraction, 0.0), 1.0)
    concentration = max(soap_volume_fraction, 0.0)
    langmuir_constant = SOAP_CMC_VOLUME_FRACTION / 8.0
    reduction = (
        math.log1p(concentration / langmuir_constant)
        / math.log1p(SOAP_CMC_VOLUME_FRACTION / langmuir_constant)
    )
    return max(
        clean - (clean - SURFACE_TENSION_SATURATED) * reduction,
        SURFACE_TENSION_SATURATED,
    )


# --------------------------------------------------------------------------- #
# The vehicle itself
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vehicle:
    """A made-up carrier liquid at a given temperature.

    Construct it either from proportions (:meth:`from_fractions`) or from what
    you actually poured into the bottle (:meth:`from_volumes`).
    """

    water_mass: float
    """kg."""

    glycerine_mass: float
    """kg."""

    soap_mass: float
    """kg.  Tiny, and treated as water-like for bulk properties -- its whole
    effect is at interfaces."""

    temperature_C: float = 21.0

    # -------------------------------------------------------------- builders
    @classmethod
    def from_fractions(cls, glycerine_mass_fraction: float,
                       soap_volume_fraction: float = 5.0e-4,
                       total_mass: float = 0.100,
                       temperature_C: float = 21.0) -> "Vehicle":
        """Build from a glycerine mass fraction and a soap volume fraction."""
        soap_mass = total_mass * soap_volume_fraction * 1.03  # soap SG ~1.03
        remaining = total_mass - soap_mass
        return cls(
            water_mass=remaining * (1.0 - glycerine_mass_fraction),
            glycerine_mass=remaining * glycerine_mass_fraction,
            soap_mass=soap_mass,
            temperature_C=temperature_C,
        )

    @classmethod
    def from_volumes(cls, water_ml: float, glycerine_ml: float,
                     soap_drops: float = 2.0,
                     temperature_C: float = 21.0) -> "Vehicle":
        """Build from measured-out volumes -- the way you would at the bench."""
        return cls(
            water_mass=water_ml * 1e-6 * water_density(temperature_C),
            glycerine_mass=glycerine_ml * 1e-6 * glycerol_density(temperature_C),
            soap_mass=soap_drops * DROP_VOLUME_ML * 1e-6 * 1030.0,
            temperature_C=temperature_C,
        )

    def with_water_mass(self, water_mass: float) -> "Vehicle":
        """Same liquid, different water content -- used by the drying model."""
        return Vehicle(max(water_mass, 0.0), self.glycerine_mass,
                       self.soap_mass, self.temperature_C)

    # ------------------------------------------------------------ properties
    @property
    def total_mass(self) -> float:
        return self.water_mass + self.glycerine_mass + self.soap_mass

    @property
    def glycerine_mass_fraction(self) -> float:
        """Glycerine as a fraction of water+glycerine (the soap is ignored)."""
        base = self.water_mass + self.glycerine_mass
        return self.glycerine_mass / base if base > 0 else 1.0

    @property
    def density(self) -> float:
        """kg/m^3."""
        return glycerol_water_density(self.glycerine_mass_fraction, self.temperature_C)

    @property
    def volume(self) -> float:
        """m^3."""
        return self.total_mass / self.density

    @property
    def viscosity(self) -> float:
        """Pa*s."""
        return glycerol_water_viscosity(self.glycerine_mass_fraction,
                                        self.temperature_C)

    @property
    def soap_volume_fraction(self) -> float:
        soap_volume = self.soap_mass / 1030.0
        return soap_volume / max(self.volume, 1e-12)

    @property
    def surface_tension(self) -> float:
        """N/m."""
        return surface_tension(self.glycerine_mass_fraction,
                               self.soap_volume_fraction)

    @property
    def water_activity(self) -> float:
        return water_activity(self.glycerine_mass_fraction)

    @property
    def dispersion_quality(self) -> float:
        """0-1 measure of how well the soap is keeping grains apart.

        Unlike the air-water interface, which is saturated at the critical
        micelle concentration, the grain surfaces keep taking up a little more
        surfactant beyond it -- there is far more solid area than free surface.
        So this keeps creeping up past the CMC, but with sharply diminishing
        returns.
        """
        c = self.soap_volume_fraction
        return c / (c + SOAP_CMC_VOLUME_FRACTION)

    @property
    def foam_tendency(self) -> float:
        """0-1.  Soap far past the CMC entrains air, which unloads the grains
        and makes a mess without improving anything."""
        excess = self.soap_volume_fraction / (10.0 * SOAP_CMC_VOLUME_FRACTION)
        return min(excess**1.5, 1.0) if excess > 1.0 else 0.0

    def describe(self) -> str:
        return (
            f"{self.glycerine_mass_fraction * 100:.0f} wt% glycerine, "
            f"{self.soap_volume_fraction * 1e4:.1f} x 10^-4 soap, "
            f"mu = {self.viscosity * 1e3:.1f} mPa.s, "
            f"gamma = {self.surface_tension * 1e3:.0f} mN/m, "
            f"a_w = {self.water_activity:.3f}"
        )
