"""Abrasives, grit gradings, workpiece stones and lap plates.

Everything is a frozen dataclass in a registry, so adding the green silicon
carbide and white fused alumina you plan to buy is a matter of appending one
entry -- no other module needs to change.  See :func:`register_abrasive` and,
for powders finer than FEPA F1000, :func:`add_micron_grade`.

Units are SI throughout (metres, pascals, kilograms).  Grit sizes are the one
exception: they are quoted in microns because that is how they are sold, and
converted at the boundary.

How grades are named
--------------------
Grit numbers only mean something alongside the standard they come from, and the
standards disagree badly once the numbers get large: "1000 grit" is 4.5 um under
FEPA F, 11.5-15.5 um under JIS R6001 and about 18 um under FEPA P (sandpaper).
So this module uses two kinds of grade:

*   **Numbered grades, 240 to 1000**, are FEPA F microgrits (FEPA 42-2:2006, the
    grading used for bonded and loose fused alumina and silicon carbide), plus
    two sieved macrogrits for black SiC.  Grits sold by number with no standard
    stated are taken to be FEPA F.
*   **Anything finer than F1000 is named by its micron size**, e.g. ``"3um"``,
    taken from the supplier's specification.  Numbers above 1000 are refused
    rather than guessed.  Any micron size works without registering it first
    (``grit="2.5um"``); :func:`add_micron_grade` records one properly, with its
    top size and whether it is on the shelf.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

MICRON = 1.0e-6
GIGAPASCAL = 1.0e9


# --------------------------------------------------------------------------- #
# Grading standards
# --------------------------------------------------------------------------- #

FEPA_F_MICROGRITS: dict[str, tuple[float, float]] = {
    # grit: (ds50 nominal, ds3 maximum), microns.
    "240": (44.5, 70.0),
    "280": (36.5, 59.0),
    "320": (29.2, 49.0),
    "360": (22.8, 40.0),
    "400": (17.3, 32.0),
    "500": (12.8, 25.0),
    "600": (9.3, 19.0),
    "800": (6.5, 14.0),
    "1000": (4.5, 10.0),
}
"""FEPA 42-2:2006, Table 1: microgrits measured by photosedimentometer.

ds50 is the median (Stokes) diameter.  ds3 is the size that no more than 3 % of
a compliant powder may exceed, so it is the upper limit of what the standard
allows rather than a typical value -- a good supplier's powder is tighter.  The
same limits apply to fused alumina and silicon carbide.
"""

FEPA_F_MACROGRITS: dict[str, tuple[float, float]] = {
    # grit: (approximate median, approximate top size), microns.
    "120": (115.0, 160.0),
    "220": (63.0, 90.0),
}
"""Sieve-graded macrogrits (FEPA 42-1; ANSI B74.12 uses the same sieves).

The standard gives sieve fractions, not a median.  The median is placed between
the control sieves, and the top size between the control sieve and the sieve
100 % must pass: for 120 grit, 106-125 um and 125-177 um; for 220, 53-75 um and
75-106 um.
"""

FINEST_NUMBERED_GRIT = 1000
"""Grit numbers above this are refused: give the micron size instead."""

FINE_MICRON_PRESETS: dict[float, float | None] = {
    # d50 in microns: top size in microns, or None for the default ratio.
    3.0: 7.0,     # what FEPA calls F1200 (ds3 max 7 um)
    2.0: None,    # FEPA F1500
    1.2: None,    # FEPA F2000
}
"""Micron grades registered for every abrasive, as examples to start from.

They carry the FEPA F1200/F1500/F2000 medians so that nothing here depends on a
grit number.  Your own powders go in with :func:`add_micron_grade`.
"""

DEFAULT_MICRON_TOP_CUT = 2.2
"""Top size as a multiple of d50 when a micron grade's spec gives no top size.

FEPA allows ds3 of 2.2x d50 at F1000 and 2.3x at F1200, so this assumes a
powder only just inside the standard.  The coarsest grains are the ones that
carry load in the plate gap, so this matters: put the real figure in if the
specification lists a d3, ds3 or maximum size.
"""

_MICRON_LABEL = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:um|µm|μm|micron|microns)\s*$",
    re.IGNORECASE,
)


def parse_micron_label(label: str) -> float | None:
    """The d50 in microns if ``label`` is a micron size such as ``"2.5um"``.

    A unit is required, so a bare ``"600"`` stays a grit number and is never
    mistaken for 600 um.  ``"um"``, ``"µm"`` and ``"micron"`` are all accepted.
    """
    match = _MICRON_LABEL.match(str(label))
    if not match:
        return None
    value = float(match.group(1))
    return value if value > 0 else None


def micron_label(d50_um: float) -> str:
    """The canonical label for a micron grade: ``2.5`` -> ``"2.5um"``."""
    return f"{d50_um:g}um"


# --------------------------------------------------------------------------- #
# Grit gradings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GritGrade:
    """One grade of one abrasive, e.g. "black SiC, 320".

    Grading standards quote sizes on a *volume* (mass) basis, because they are
    measured by sieving or sedimentation.  That matters: the number-weighted
    mean of a powder is far finer than its quoted d50, and it is grain *counts*
    that carry load in the plate gap.  :meth:`size_distribution` does the
    conversion.
    """

    label: str
    d50_um: float
    """Median diameter by volume: FEPA ds50 for a numbered grade, the
    supplier's d50 for a micron grade."""

    d_max_um: float
    """Top cut: the coarsest grains the model lets into the powder.

    Numbered grades use FEPA's ds3 maximum; micron grades use the supplier's
    top size, or :data:`DEFAULT_MICRON_TOP_CUT` times d50 if none is given.
    The distribution is truncated here, and because the coarsest grains are the
    few that bridge the gap and carry the load, finish and damage predictions
    move a lot with this number.  A tub with a sloppy top cut leaves scratches
    far coarser than its median suggests.
    """

    sigma_g: float
    """Geometric standard deviation of the underlying lognormal, by volume.

    ~1.25 for micron-graded (sedimented) powders, ~1.35 for mesh-graded ones.
    """

    standard: str = ""
    in_stock: bool = False
    """Marks what is on your shelf today, so reports can flag the rest as
    "would need buying"."""

    @property
    def d_min_um(self) -> float:
        """Bottom cut, taken log-symmetric with the top cut about the median."""
        return self.d50_um**2 / self.d_max_um

    @property
    def is_micron_graded(self) -> bool:
        return parse_micron_label(self.label) is not None

    def size_distribution(self, n_bins: int = 240) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(diameters_m, number_fraction)`` for the graded powder.

        The volume distribution is a lognormal truncated to
        ``[d_min_um, d_max_um]``; the number distribution follows from
        ``n(d) ∝ v(d) / d³``.

        Memoised, because the recipe search calls this hundreds of thousands of
        times.  The returned arrays are marked read-only so a caller cannot
        corrupt the shared cache.
        """
        return _size_distribution(
            self.d50_um, self.d_max_um, self.d_min_um, self.sigma_g, n_bins
        )


@lru_cache(maxsize=256)
def _size_distribution(d50_um: float, d_max_um: float, d_min_um: float,
                       sigma_g: float, n_bins: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    d_um = np.logspace(math.log10(d_min_um), math.log10(d_max_um), n_bins)
    z = np.log(d_um / d50_um) / math.log(sigma_g)
    volume_pdf = np.exp(-0.5 * z**2) / d_um       # lognormal in d, unnormalised

    weights = volume_pdf / d_um**3 * np.gradient(d_um)   # -> number basis
    weights /= weights.sum()

    diameters = d_um * MICRON
    diameters.flags.writeable = False
    weights.flags.writeable = False
    return diameters, weights


def micron_grade(d50_um: float, d_max_um: float | None = None,
                 sigma_g: float = 1.25, in_stock: bool = False,
                 standard: str = "micron-graded (supplier d50)") -> GritGrade:
    """Build a grade from a supplier's micron specification.

    ``d50_um`` is the median size.  ``d_max_um`` is the top size if the spec
    gives one (often listed as d3, ds3 or "max"); without it the top is taken as
    :data:`DEFAULT_MICRON_TOP_CUT` times d50.
    """
    if d50_um <= 0:
        raise ValueError(f"d50 must be positive, got {d50_um}")
    top = d_max_um if d_max_um is not None else d50_um * DEFAULT_MICRON_TOP_CUT
    if top <= d50_um:
        raise ValueError(f"top size {top} um must be larger than d50 {d50_um} um")
    return GritGrade(
        label=micron_label(d50_um),
        d50_um=float(d50_um),
        d_max_um=float(top),
        sigma_g=sigma_g,
        standard=standard,
        in_stock=in_stock,
    )


# --------------------------------------------------------------------------- #
# Abrasives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Abrasive:
    """A loose abrasive grain type."""

    key: str
    name: str
    density: float
    """kg/m^3."""

    hardness: float
    """Knoop indentation hardness, Pa.  Must comfortably exceed the workpiece
    hardness or the grain blunts instead of cutting."""

    youngs_modulus: float
    """Pa."""

    grain_strength_ref: float
    """Characteristic single-grain crushing strength at ``strength_ref_size_um``,
    Pa.  Far below the bulk strength of the ceramic because abrasive grains are
    deliberately full of flaws."""

    strength_ref_size_um: float = 50.0

    weibull_modulus: float = 6.0
    """Governs how much stronger small grains are than large ones:
    ``sigma ∝ d^(-3/m)``.  Low m = strong size effect = fines survive while
    coarse grains shatter."""

    tip_semi_angle_deg: float = 70.0
    """Effective cone semi-angle of a load-bearing corner.  Smaller = sharper =
    cuts deeper for the same load, but dulls faster."""

    friability: float = 0.5
    """0 = grain survives, dulls and stops cutting; 1 = grain keeps splitting
    into fresh sharp fragments.  Drives breakdown of the slurry with use."""

    max_packing: float = 0.50
    """Maximum *flowable* solids fraction of a monodisperse batch of this grain.

    Lower than the dry random-close-packing fraction (0.64 for spheres, ~0.58
    for angular grains), because a suspension seizes up while lubricating films
    still separate the grains.  Measured values for irregular mineral slurries
    sit at 0.50-0.58; blockier grains sit at the top of that range.
    """

    intrinsic_viscosity: float = 3.4
    """[eta] in Krieger-Dougherty.  2.5 for spheres; angular grains disturb the
    flow more."""

    grades: dict[str, GritGrade] = field(default_factory=dict)

    micron_sigma_g: float = 1.25
    """Spread assumed for a micron grade given only by its size."""

    notes: str = ""

    def grade(self, label: str) -> GritGrade:
        """Look up a grade by label.

        Accepts a registered label (``"600"``, ``"3um"``), a FEPA-style
        ``"F600"``, or any micron size (``"2.5um"``, ``"2.5 µm"``), which is built
        on the fly if it is not registered.  Grit numbers finer than F1000 are
        refused with an explanation, because they have no single meaning.
        """
        text = str(label).strip()
        if text in self.grades:
            return self.grades[text]
        number = text[1:] if text[:1] in ("F", "f") and text[1:].isdigit() else text
        if number in self.grades:
            return self.grades[number]

        d50 = parse_micron_label(text)
        if d50 is not None:
            registered = self.grades.get(micron_label(d50))
            if registered is not None:
                return registered
            return micron_grade(d50, sigma_g=self.micron_sigma_g)

        if number.isdigit() and int(number) > FINEST_NUMBERED_GRIT:
            raise KeyError(
                f"Grit {text!r} is finer than FEPA F1000, where grit numbers mean "
                "different sizes under FEPA F, FEPA P, JIS and ANSI. Give the d50 "
                "from the supplier's specification instead, e.g. '3um'."
            )
        available = ", ".join(self.grades)
        raise KeyError(
            f"{self.name} has no grade {text!r}. Available: {available}, "
            "or any micron size such as '2.5um'."
        )

    def crushing_load(self, diameter_m: np.ndarray | float) -> np.ndarray | float:
        """Load at which a grain of this diameter fractures, N.

        Weibull size scaling on the strength, times the grain cross-section.
        This is what stops penetration running away at high pressure: press
        harder and the grains break rather than dig deeper.
        """
        d_um = np.asarray(diameter_m) / MICRON
        size_factor = (self.strength_ref_size_um / d_um) ** (3.0 / self.weibull_modulus)
        strength = self.grain_strength_ref * size_factor
        return 0.7 * strength * np.asarray(diameter_m) ** 2


# --------------------------------------------------------------------------- #
# Workpiece stones and lap plates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Solid:
    """A stone being ground, or the plate it is ground against."""

    key: str
    name: str
    density: float
    hardness: float
    """Knoop hardness, Pa."""

    youngs_modulus: float
    fracture_toughness: float
    """K_Ic in Pa*sqrt(m).  The single most important number for how a stone
    lapps: tough stones (nephrite, K_Ic ~ 3) refuse to chip and must be ground
    ductilely, which is slow.  Brittle ones (agate, obsidian) spall away fast
    but carry deep subsurface damage."""

    notes: str = ""

    @property
    def mohs_approx(self) -> float:
        """Rough Mohs number back-calculated from Knoop hardness, for orientation."""
        knoop_kgf_mm2 = self.hardness / 9.807e6
        return 0.675 * knoop_kgf_mm2 ** (1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

ABRASIVES: dict[str, Abrasive] = {}
STONES: dict[str, Solid] = {}
LAPS: dict[str, Solid] = {}


def register_abrasive(abrasive: Abrasive) -> Abrasive:
    """Add an abrasive to the registry.  This is the extension point.

    >>> register_abrasive(Abrasive(key="cbn", name="Cubic boron nitride", ...))
    """
    ABRASIVES[abrasive.key] = abrasive
    return abrasive


def register_stone(stone: Solid) -> Solid:
    STONES[stone.key] = stone
    return stone


def register_lap(lap: Solid) -> Solid:
    LAPS[lap.key] = lap
    return lap


def _coarse_to_fine(grades: dict[str, GritGrade]) -> dict[str, GritGrade]:
    return dict(sorted(grades.items(), key=lambda item: -item[1].d50_um))


def _standard_grades(sigma_g: float, micron_sigma_g: float,
                     macrogrits: bool = False,
                     in_stock: tuple[str, ...] = ()) -> dict[str, GritGrade]:
    """FEPA F microgrits 240-1000, optional sieved macrogrits, and the micron
    presets finer than F1000, coarse to fine."""
    grades: dict[str, GritGrade] = {}
    if macrogrits:
        for label, (d50, top) in FEPA_F_MACROGRITS.items():
            grades[label] = GritGrade(label, d50, top, sigma_g,
                                      standard="FEPA F / ANSI B74.12 (sieved)",
                                      in_stock=label in in_stock)
    for label, (d50, ds3) in FEPA_F_MICROGRITS.items():
        grades[label] = GritGrade(label, d50, ds3, sigma_g,
                                  standard="FEPA F (42-2:2006)",
                                  in_stock=label in in_stock)
    for d50, top in FINE_MICRON_PRESETS.items():
        grade = micron_grade(d50, top, sigma_g=micron_sigma_g)
        grades[grade.label] = replace(grade, in_stock=grade.label in in_stock)
    return _coarse_to_fine(grades)


# --------------------------------------------------------------------------- #
# Black silicon carbide -- what you have now
# --------------------------------------------------------------------------- #

register_abrasive(
    Abrasive(
        key="sic_black",
        name="Black silicon carbide",
        density=3210.0,
        hardness=24.5 * GIGAPASCAL,       # Knoop ~2500 kgf/mm^2
        youngs_modulus=410 * GIGAPASCAL,
        grain_strength_ref=420e6,
        weibull_modulus=6.0,
        tip_semi_angle_deg=68.0,
        friability=0.55,
        max_packing=0.50,
        intrinsic_viscosity=3.4,
        grades=_standard_grades(
            sigma_g=1.34, micron_sigma_g=1.25, macrogrits=True,
            # Bought by number with no standard stated; taken as FEPA F, which
            # matches how the 1000 behaves at the bench.
            in_stock=("320", "400", "600", "1000"),
        ),
        notes=(
            "Standard lapidary cutting grit.  Sharp, fast, breaks down at a "
            "moderate rate.  Slightly less pure than green SiC, which makes it "
            "a touch tougher and blockier."
        ),
    )
)

# --------------------------------------------------------------------------- #
# Green silicon carbide -- for the finer stages you are planning
# --------------------------------------------------------------------------- #

register_abrasive(
    Abrasive(
        key="sic_green",
        name="Green silicon carbide",
        density=3210.0,
        hardness=26.0 * GIGAPASCAL,       # purer, marginally harder
        youngs_modulus=420 * GIGAPASCAL,
        grain_strength_ref=320e6,         # more friable than black
        weibull_modulus=5.5,
        tip_semi_angle_deg=64.0,          # sharper: stays sharp by splitting
        friability=0.72,
        max_packing=0.49,
        intrinsic_viscosity=3.5,
        grades=_standard_grades(sigma_g=1.28, micron_sigma_g=1.25),
        notes=(
            "Purer and more friable than black SiC: fragments rather than dulls, "
            "so it keeps cutting and self-sharpens.  Cuts a little cleaner at "
            "fine sizes, which is why it is the usual choice for pre-polish."
        ),
    )
)

# --------------------------------------------------------------------------- #
# White fused alumina -- softer, blockier, gentler finish
# --------------------------------------------------------------------------- #

register_abrasive(
    Abrasive(
        key="alumina_white",
        name="White fused alumina",
        density=3950.0,
        hardness=20.0 * GIGAPASCAL,       # Knoop ~2050 kgf/mm^2
        youngs_modulus=380 * GIGAPASCAL,
        grain_strength_ref=520e6,
        weibull_modulus=7.0,
        tip_semi_angle_deg=76.0,          # blocky, not sharp
        friability=0.38,
        max_packing=0.55,                 # blockier grains pack better
        intrinsic_viscosity=3.1,
        grades=_standard_grades(sigma_g=1.28, micron_sigma_g=1.25),
        notes=(
            "Softer and much blockier than SiC, so it abrades by many shallow "
            "contacts instead of a few deep ones.  Slower, but the shallow "
            "damage layer it leaves is what makes it a good last step before "
            "polish.  Only marginally harder than quartz -- expect it to cut "
            "agate and jasper sluggishly compared with SiC."
        ),
    )
)


# --------------------------------------------------------------------------- #
# Stones
# --------------------------------------------------------------------------- #

register_stone(Solid("agate", "Agate / chalcedony", 2600, 10.5 * GIGAPASCAL,
                     95 * GIGAPASCAL, 0.90e6,
                     "Microcrystalline quartz. The lapidary default."))
register_stone(Solid("jasper", "Jasper", 2650, 9.8 * GIGAPASCAL,
                     88 * GIGAPASCAL, 1.20e6,
                     "Quartz with an impure, slightly tougher matrix."))
register_stone(Solid("quartz", "Crystalline quartz", 2650, 11.5 * GIGAPASCAL,
                     95 * GIGAPASCAL, 0.85e6,
                     "Single crystal; cleaves readily, so it bruises easily."))
register_stone(Solid("obsidian", "Obsidian", 2400, 6.0 * GIGAPASCAL,
                     70 * GIGAPASCAL, 0.75e6,
                     "A glass: soft and very brittle. Cuts fast, scratches "
                     "easily, unforgiving of stray coarse grains."))
register_stone(Solid("nephrite", "Nephrite jade", 3000, 6.5 * GIGAPASCAL,
                     90 * GIGAPASCAL, 3.00e6,
                     "Felted amphibole fibres. Soft but extremely tough: it "
                     "will not chip, so removal is slow and undercutting of "
                     "softer patches dominates."))
register_stone(Solid("lapis", "Lapis lazuli", 2800, 4.0 * GIGAPASCAL,
                     60 * GIGAPASCAL, 1.00e6,
                     "Soft and mixed-mineral; the calcite grinds faster than "
                     "the lazurite, so it dishes."))

# --------------------------------------------------------------------------- #
# Lap plates
# --------------------------------------------------------------------------- #

register_lap(Solid("glass", "Float glass plate", 2500, 5.5 * GIGAPASCAL,
                   72 * GIGAPASCAL, 0.75e6,
                   "Softer than quartz, so grains bed into the glass and cut "
                   "the stone in a partly two-body mode. The plate wears; "
                   "that wear is the price of the cutting action."))
register_lap(Solid("cast_iron", "Cast iron lap", 7200, 2.0 * GIGAPASCAL,
                   120 * GIGAPASCAL, 25.0e6,
                   "Charges heavily with abrasive. Fast and flat, but grains "
                   "embed so deeply that fine grades can stop cutting."))
register_lap(Solid("aluminium", "Aluminium plate", 2700, 0.9 * GIGAPASCAL,
                   70 * GIGAPASCAL, 30.0e6,
                   "Too soft: grains sink in and the plate does the wearing."))
register_lap(Solid("steel", "Hardened steel lap", 7800, 7.0 * GIGAPASCAL,
                   205 * GIGAPASCAL, 50.0e6,
                   "Hard enough that grains roll rather than bed in: a true "
                   "three-body action, slower but with a finer finish."))


def find_grade(abrasive_key: str, grade_label: str) -> tuple[Abrasive, GritGrade]:
    """Look up an ``(abrasive, grade)`` pair, with a helpful error."""
    try:
        abrasive = ABRASIVES[abrasive_key]
    except KeyError:
        raise KeyError(
            f"Unknown abrasive {abrasive_key!r}. Known: {', '.join(ABRASIVES)}"
        ) from None
    return abrasive, abrasive.grade(grade_label)


def stock_grades() -> list[tuple[Abrasive, GritGrade]]:
    """Every grade currently marked as on the shelf, coarse to fine."""
    pairs = [
        (abrasive, grade)
        for abrasive in ABRASIVES.values()
        for grade in abrasive.grades.values()
        if grade.in_stock
    ]
    return sorted(pairs, key=lambda pair: -pair[1].d50_um)


def set_stock(abrasive_key: str, grade_labels: tuple[str, ...]) -> None:
    """Redefine which grades of an abrasive you own.

    Labels are the registered ones (``"600"``, ``"3um"``).  To stock a micron
    size that is not registered yet, use :func:`add_micron_grade` with
    ``in_stock=True``.

    >>> set_stock("sic_green", ("600", "3um"))
    """
    abrasive = ABRASIVES[abrasive_key]
    wanted = set()
    for label in grade_labels:
        grade = abrasive.grade(label)
        if grade.label not in abrasive.grades:
            raise KeyError(
                f"{grade.label} is not registered for {abrasive.name}; use "
                f"add_micron_grade({abrasive_key!r}, {grade.d50_um:g}, "
                "in_stock=True) to add it."
            )
        wanted.add(grade.label)
    updated = {
        label: replace(grade, in_stock=label in wanted)
        for label, grade in abrasive.grades.items()
    }
    ABRASIVES[abrasive_key] = replace(abrasive, grades=updated)


def add_micron_grade(abrasive_key: str, d50_um: float,
                     d_max_um: float | None = None,
                     sigma_g: float | None = None,
                     in_stock: bool = False) -> GritGrade:
    """Register a micron-graded powder from the supplier's specification.

    This is how anything finer than FEPA F1000 gets into the model.  ``d50_um``
    is the median size on the spec sheet; ``d_max_um`` is its top size if listed
    (d3, ds3 or "max").  The grade is labelled by its size, e.g. ``"2.5um"``,
    and replaces any grade already registered under that label.

    >>> add_micron_grade("sic_green", 2.5, d_max_um=6.0, in_stock=True)
    """
    abrasive = ABRASIVES[abrasive_key]
    grade = micron_grade(
        d50_um, d_max_um,
        sigma_g=abrasive.micron_sigma_g if sigma_g is None else sigma_g,
        in_stock=in_stock,
    )
    grades = _coarse_to_fine({**abrasive.grades, grade.label: grade})
    ABRASIVES[abrasive_key] = replace(abrasive, grades=grades)
    return grade


# --------------------------------------------------------------------------- #
# Your own micron-graded powders
# --------------------------------------------------------------------------- #
# Anything finer than FEPA F1000 goes in by its micron size, straight from the
# supplier's specification.  Give the top size too if the spec lists one (often
# as d3, ds3 or "max"); without it the top is taken as 2.2 x d50.  For example:
#
#     add_micron_grade("sic_green", 2.5, d_max_um=6.0, in_stock=True)
#     add_micron_grade("alumina_white", 1.0)
#
# For a quick what-if nothing needs registering: anywhere a grade is asked for,
# a micron size works directly, e.g. grit="2.5um".
