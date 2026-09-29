"""Searching for slurry recipes that actually work.

There is no single "best" slurry -- there is a *window* of workable ones, and the
useful question is where that window sits and how wide it is.  A recipe balanced
in the middle of the window tolerates a sloppy measurement, a warm afternoon or a
humid day; one perched on the edge of it works today and fails tomorrow.  So this
module reports two things:

*   :func:`scan_window` maps the whole feasible region in the two variables you
    actually control -- how much grit, and how much of the liquid is glycerine.
*   :func:`find_recipe` picks a point in it.  By default it maximises
    *robustness*: the distance to the nearest constraint boundary, so the recipe
    sits as centrally as possible.  Ask for ``prefer="fast"`` instead and it takes
    the quickest-cutting feasible mix, edges and all.

:func:`plan_sequence` then does the thing that saves the most time at the bench:
it works out how long to spend at each grit.  Every stage leaves cracked stone
below the surface it just made, and the following stage has to grind that damage
away before its own finish means anything.  Stop early and the scratches you
thought you had removed reappear during polishing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from lapsim.materials import ABRASIVES, GritGrade, Abrasive
from lapsim.simulate import Result, Setup, simulate
from lapsim.slurry import LOW_SHEAR_RATE
from lapsim.vehicle import DROP_VOLUME_ML, glycerol_density, water_density


# --------------------------------------------------------------------------- #
# Turning fractions into something you can measure out
# --------------------------------------------------------------------------- #


@dataclass
class Recipe:
    """A batch of slurry, in units you can weigh and pour."""

    abrasive_name: str
    grit: str
    batch_ml: float
    grit_grams: float
    water_ml: float
    glycerine_ml: float
    soap_drops: float
    solids_mass_fraction: float
    glycerine_mass_fraction: float
    soap_volume_fraction: float

    @classmethod
    def from_setup(cls, setup: Setup, batch_ml: float = 100.0) -> "Recipe":
        abrasive, grade, _, _ = setup.resolve()
        slurry = setup.build_slurry()
        temperature = setup.temperature_C

        batch_volume = batch_ml * 1e-6
        phi = slurry.solids_volume_fraction
        vehicle_volume = (1.0 - phi) * batch_volume

        grit_mass = phi * batch_volume * abrasive.density
        soap_volume = setup.soap_volume_fraction * vehicle_volume
        liquid_volume = vehicle_volume - soap_volume

        # Split the liquid volume to hit the requested glycerine *mass* fraction.
        rho_water = water_density(temperature)
        rho_glycerine = glycerol_density(temperature)
        g = setup.glycerine_mass_fraction
        # v_g * rho_g / (v_g * rho_g + v_w * rho_w) = g, with v_g + v_w = liquid
        glycerine_volume = (
            liquid_volume * g * rho_water / (g * rho_water + (1.0 - g) * rho_glycerine)
        )

        return cls(
            abrasive_name=abrasive.name,
            grit=grade.label,
            batch_ml=batch_ml,
            grit_grams=grit_mass * 1e3,
            water_ml=(liquid_volume - glycerine_volume) * 1e6,
            glycerine_ml=glycerine_volume * 1e6,
            soap_drops=soap_volume * 1e6 / DROP_VOLUME_ML,
            solids_mass_fraction=slurry.solids_mass_fraction,
            glycerine_mass_fraction=setup.glycerine_mass_fraction,
            soap_volume_fraction=setup.soap_volume_fraction,
        )

    def describe(self) -> str:
        return (
            f"For {self.batch_ml:.0f} ml of {self.abrasive_name} {self.grit} slurry:\n"
            f"    {self.grit_grams:6.1f} g   grit\n"
            f"    {self.water_ml:6.1f} ml  water\n"
            f"    {self.glycerine_ml:6.1f} ml  glycerine\n"
            f"    {max(round(self.soap_drops), 1):6d}     "
            f"drop{'s' if max(round(self.soap_drops), 1) != 1 else ''} of dish soap\n"
            f"  ({self.solids_mass_fraction * 100:.0f} wt% solids, "
            f"{self.glycerine_mass_fraction * 100:.0f} wt% glycerine in the liquid)"
        )


# --------------------------------------------------------------------------- #
# Scoring a candidate
# --------------------------------------------------------------------------- #


def constraint_margins(result: Result) -> dict[str, float]:
    """How much room each constraint has left, normalised to roughly 0-1.

    Negative means the constraint is violated.  Zero means the recipe is sitting
    exactly on the boundary -- technically legal, practically fragile.
    """
    setup, slurry = result.setup, result.slurry
    targets = setup.targets
    low_shear = slurry.apparent_viscosity(LOW_SHEAR_RATE)
    settling = slurry.settling_time()
    workable = result.drying.workable_time_s

    def capped(value: float) -> float:
        return float(min(value, 1.0))

    margins = {
        "not too runny": capped(
            (low_shear - targets.min_low_shear_viscosity_Pas)
            / targets.min_low_shear_viscosity_Pas
        ),
        "not too stiff": (targets.max_yield_stress_Pa - slurry.yield_stress)
        / targets.max_yield_stress_Pa,
        "low drag": (
            targets.max_plastic_viscosity_Pas - slurry.plastic_viscosity
        ) / targets.max_plastic_viscosity_Pas,
        "grit stays up": capped(
            (settling - targets.min_settling_time_s) / targets.min_settling_time_s
        ),
        "no hydroplaning": (
            result.contact.load_fraction_on_grains - targets.min_grain_load_fraction
        ) / max(1.0 - targets.min_grain_load_fraction, 1e-9),
        "will not crust": (
            result.drying.jamming_solids_fraction
            - result.drying.equilibrium_solids_fraction
        ) / result.drying.jamming_solids_fraction,
        "stays wet": capped(
            (workable - targets.min_workable_time_s) / targets.min_workable_time_s
        ),
        "no foaming": (
            targets.max_foam_tendency - slurry.vehicle.foam_tendency
        ) / max(targets.max_foam_tendency, 1e-9),
        "grit not self-destructing": (
            targets.max_fractured_fraction - result.contact.fractured_fraction
        ) / targets.max_fractured_fraction,
    }
    if targets.enforce_finish:
        _, grade, _, _ = setup.resolve()
        ceiling = targets.roughness_ceiling(grade.d50_um, setup.calibration)
        margins["finish as fine as the grade allows"] = (
            (ceiling - result.finish.Ra_um) / ceiling
        )
    return margins


def robustness(result: Result) -> float:
    """The tightest constraint margin: how much slack the recipe has overall."""
    margins = constraint_margins(result)
    return min(margins.values()) if margins else 0.0


@dataclass
class Candidate:
    """One point in the recipe search."""

    setup: Setup
    result: Result
    robustness: float
    removal_rate_um_min: float
    Ra_um: float
    feasible: bool
    binding_constraint: str

    @property
    def recipe(self) -> Recipe:
        return Recipe.from_setup(self.setup)

    def tolerances(self, steps: int = 40) -> dict[str, tuple[float, float]]:
        """How far each ingredient can drift before something breaks.

        Returns ``{"solids": (low, high), "glycerine": (low, high)}`` as mass
        fractions, found by walking away from this recipe along each axis until
        a constraint fails. This is the number that decides whether a recipe
        survives being mixed by eye.
        """
        spans: dict[str, tuple[float, float]] = {}
        probe = replace(self.setup, drying_steps=150)
        axes = {
            "solids": ("solids_mass_fraction", 0.02, 0.90),
            "glycerine": ("glycerine_mass_fraction", 0.0, 0.95),
        }
        for name, (field_name, floor, cap) in axes.items():
            centre = getattr(self.setup, field_name)
            bounds = []
            for direction in (-1, +1):
                edge = centre
                limit = floor if direction < 0 else cap
                for step in range(1, steps + 1):
                    value = centre + direction * step * abs(limit - centre) / steps
                    if not (floor <= value <= cap):
                        break
                    if not evaluate(replace(probe, **{field_name: value})).feasible:
                        break
                    edge = value
                bounds.append(edge)
            spans[name] = (bounds[0], bounds[1])
        return spans

    def describe_tolerances(self) -> str:
        spans = self.tolerances()
        solids = spans["solids"]
        glycerine = spans["glycerine"]
        return (
            f"Still works between {solids[0] * 100:.0f} and "
            f"{solids[1] * 100:.0f} wt% grit, and between "
            f"{glycerine[0] * 100:.0f} and {glycerine[1] * 100:.0f} % glycerine "
            "(each varied on its own)."
        )


def evaluate(setup: Setup) -> Candidate:
    """Simulate one recipe and score it."""
    result = simulate(setup)
    margins = constraint_margins(result)
    binding = min(margins, key=margins.get) if margins else ""
    score = margins[binding] if margins else 0.0
    return Candidate(
        setup=setup,
        result=result,
        robustness=score,
        removal_rate_um_min=result.removal_rate_um_min,
        Ra_um=result.finish.Ra_um,
        feasible=score > 0.0 and result.contact.is_cutting,
        binding_constraint=binding,
    )


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #


@dataclass
class ScanResult:
    """A 2-D map of the recipe space for one grit."""

    grit: str
    abrasive_name: str
    solids_mass_fractions: np.ndarray
    glycerine_mass_fractions: np.ndarray
    robustness: np.ndarray
    """Shape (n_glycerine, n_solids).  Positive where the recipe is workable."""

    removal_rate_um_min: np.ndarray
    Ra_um: np.ndarray
    equilibrium_solids: np.ndarray
    workable_time_min: np.ndarray
    yield_stress_Pa: np.ndarray
    low_shear_viscosity_Pas: np.ndarray
    binding_constraint: np.ndarray
    """Name of the tightest constraint at each point -- tells you which way to
    move when a recipe fails."""

    @property
    def feasible(self) -> np.ndarray:
        return self.robustness > 0.0

    @property
    def feasible_fraction(self) -> float:
        return float(self.feasible.mean())


def scan_window(base: Setup, solids_range: tuple[float, float] = (0.10, 0.75),
                glycerine_range: tuple[float, float] = (0.0, 0.90),
                n_solids: int = 26, n_glycerine: int = 26) -> ScanResult:
    """Map the workable region in (grit loading, glycerine fraction) space."""
    solids = np.linspace(*solids_range, n_solids)
    glycerine = np.linspace(*glycerine_range, n_glycerine)
    shape = (n_glycerine, n_solids)

    arrays = {
        name: np.zeros(shape)
        for name in ("robustness", "rate", "Ra", "phi_end", "workable",
                     "yield_stress", "low_shear")
    }
    binding = np.empty(shape, dtype=object)

    fast = replace(base, drying_steps=150)
    for i, g in enumerate(glycerine):
        for j, w in enumerate(solids):
            candidate = evaluate(
                replace(fast, solids_mass_fraction=float(w),
                        glycerine_mass_fraction=float(g))
            )
            slurry = candidate.result.slurry
            arrays["robustness"][i, j] = candidate.robustness
            arrays["rate"][i, j] = candidate.removal_rate_um_min
            arrays["Ra"][i, j] = candidate.Ra_um
            arrays["phi_end"][i, j] = candidate.result.drying.equilibrium_solids_fraction
            workable = candidate.result.drying.workable_time_s / 60.0
            arrays["workable"][i, j] = min(workable, 120.0)
            arrays["yield_stress"][i, j] = slurry.yield_stress
            arrays["low_shear"][i, j] = slurry.apparent_viscosity(LOW_SHEAR_RATE)
            binding[i, j] = candidate.binding_constraint

    abrasive, grade, _, _ = base.resolve()
    return ScanResult(
        grit=grade.label,
        abrasive_name=abrasive.name,
        solids_mass_fractions=solids,
        glycerine_mass_fractions=glycerine,
        robustness=arrays["robustness"],
        removal_rate_um_min=arrays["rate"],
        Ra_um=arrays["Ra"],
        equilibrium_solids=arrays["phi_end"],
        workable_time_min=arrays["workable"],
        yield_stress_Pa=arrays["yield_stress"],
        low_shear_viscosity_Pas=arrays["low_shear"],
        binding_constraint=binding,
    )


# --------------------------------------------------------------------------- #
# Picking a recipe
# --------------------------------------------------------------------------- #

SOAP_CANDIDATES = (5.0e-4, 1.0e-3, 2.0e-3)
"""Soap levels to try, as volume fractions of the liquid: one, two and four
drops per 100 ml.

Deliberately few, and deliberately whole drops. Soap saturates the interface
just past the critical micelle concentration, which for a household detergent
lands below a single drop per 100 ml, so every candidate here is already past
the point of diminishing returns -- the search is really only choosing how much
margin to leave, and you cannot measure a quarter of a drop anyway.
"""


def find_recipe(base: Setup, prefer: str = "robust",
                solids_range: tuple[float, float] = (0.10, 0.75),
                glycerine_range: tuple[float, float] = (0.0, 0.90),
                n_solids: int = 22, n_glycerine: int = 22) -> Candidate | None:
    """Search for the best recipe for one grit.

    ``prefer`` selects the objective:

    ``"robust"``
        Sit as far from every constraint as possible.  Use this when you want a
        recipe that keeps working on a different day.  This is the default.
    ``"fast"``
        Highest removal rate among feasible mixes, accepting a tighter margin.
    ``"balanced"``
        Compromise: normalise both and weight them equally.

    Returns ``None`` if nothing in the search range is feasible -- which is itself
    informative, and usually means the targets need relaxing or the grit is wrong
    for the job.
    """
    fast = replace(base, drying_steps=150)
    candidates: list[Candidate] = []
    for soap in SOAP_CANDIDATES:
        for g in np.linspace(*glycerine_range, n_glycerine):
            for w in np.linspace(*solids_range, n_solids):
                candidates.append(evaluate(replace(
                    fast,
                    solids_mass_fraction=float(w),
                    glycerine_mass_fraction=float(g),
                    soap_volume_fraction=float(soap),
                )))

    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        return None

    if prefer == "fast":
        best = max(feasible, key=lambda c: c.removal_rate_um_min)
    elif prefer == "balanced":
        max_robust = max(c.robustness for c in feasible)
        max_rate = max(c.removal_rate_um_min for c in feasible)
        best = max(
            feasible,
            key=lambda c: 0.5 * c.robustness / max_robust
            + 0.5 * c.removal_rate_um_min / max_rate,
        )
    else:
        best = max(feasible, key=lambda c: (c.robustness, c.removal_rate_um_min))

    # Re-run the winner at full drying resolution.
    return evaluate(replace(best.setup, drying_steps=base.drying_steps))


# --------------------------------------------------------------------------- #
# When no recipe works
# --------------------------------------------------------------------------- #

PROCESS_REMEDIES: tuple[tuple[str, dict], ...] = (
    ("stroke at half speed (0.12 m/s)", dict(sliding_speed_m_s=0.12)),
    ("stroke slowly (0.06 m/s)", dict(sliding_speed_m_s=0.06)),
    ("press about twice as hard", dict(normal_load_N=None)),
    ("work a smaller face at a time", dict(contact_length_mm=None)),
    ("stroke at half speed and press harder",
     dict(sliding_speed_m_s=0.12, normal_load_N=None)),
    ("lap on flatter, better mated plates", dict(gap_convergence=1.15)),
)
"""Changes of technique to try when no slurry composition works.

Every one of these attacks the same thing: hydrodynamic lift, which scales as
viscosity times speed divided by the square of the gap. Fine grits run in a
small gap, so lift is strong, and no amount of tuning the liquid will fix a
stroke that is simply too fast for the grit.
"""


def diagnose_infeasible(base: Setup, **search_kwargs) -> list[tuple[str, Candidate]]:
    """Find changes of technique that would open a closed recipe window.

    Returns ``(description, best_candidate)`` pairs, most robust first, for the
    remedies that work. An empty list means the grit is wrong for the job.
    """
    working: list[tuple[str, Candidate]] = []
    for description, changes in PROCESS_REMEDIES:
        resolved = dict(changes)
        # A couple of remedies are relative to whatever the user already set.
        if "normal_load_N" in resolved and resolved["normal_load_N"] is None:
            resolved["normal_load_N"] = base.normal_load_N * 2.0
        if "contact_length_mm" in resolved and resolved["contact_length_mm"] is None:
            resolved["contact_length_mm"] = base.contact_length_mm * 0.5
            resolved["contact_area_mm2"] = base.contact_area_mm2 * 0.25
            resolved["normal_load_N"] = base.normal_load_N * 0.25
        candidate = find_recipe(replace(base, **resolved), **search_kwargs)
        if candidate is not None:
            working.append((description, candidate))
    working.sort(key=lambda pair: -pair[1].robustness)
    return working


def explain_infeasible(base: Setup, remedies: list[tuple[str, Candidate]]) -> str:
    """A sentence or two on why nothing works, and what to do about it."""
    abrasive, grade, _, _ = base.resolve()
    head = (
        f"No slurry composition satisfies every target for {abrasive.name} "
        f"{grade.label} at {base.sliding_speed_m_s:.2f} m/s and "
        f"{base.nominal_pressure / 1e3:.0f} kPa. The liquid has to be thick "
        "enough not to run off the plate, but at this grit the gap is small "
        "enough that the same thickness starts floating the stone."
    )
    if not remedies:
        return head + " No change of technique in the list fixes it either."
    options = "; ".join(
        f"{description} (then {candidate.setup.solids_mass_fraction * 100:.0f} wt% "
        f"grit with {candidate.setup.glycerine_mass_fraction * 100:.0f} % "
        f"glycerine works, cutting "
        f"{candidate.removal_rate_um_min:.0f} \u00b5m/min)"
        for description, candidate in remedies[:3]
    )
    return head + " What does work: " + options + "."


# --------------------------------------------------------------------------- #
# The whole grit sequence
# --------------------------------------------------------------------------- #


@dataclass
class Stage:
    """One grit in a planned progression."""

    abrasive_key: str
    abrasive_name: str
    grit: str
    d50_um: float
    candidate: Candidate
    damage_to_remove_um: float
    """Subsurface damage left by the *previous* stage, which this one must grind
    away before its own finish is real."""

    minutes_required: float
    in_stock: bool

    @property
    def Ra_um(self) -> float:
        return self.candidate.Ra_um

    @property
    def subsurface_damage_um(self) -> float:
        return self.candidate.result.finish.subsurface_damage_um


@dataclass
class SkippedStage:
    """A grit with no workable recipe under the given process settings."""

    abrasive_name: str
    grit: str
    explanation: str
    remedies: list[tuple[str, Candidate]] = field(default_factory=list)


@dataclass
class SequencePlan:
    """A full coarse-to-fine progression with times and recipes."""

    stages: list[Stage] = field(default_factory=list)
    skipped: list[SkippedStage] = field(default_factory=list)

    @property
    def total_minutes(self) -> float:
        return sum(stage.minutes_required for stage in self.stages)

    def table(self) -> str:
        header = (
            f"  {'abrasive':<22} {'grit':>6} {'d50':>6} {'Ra':>7} {'SSD':>7}"
            f" {'cut':>9} {'clear':>7} {'time':>7}  recipe (per 100 ml)"
        )
        units = (
            f"  {'':<22} {'':>6} {'um':>6} {'um':>7} {'um':>7}"
            f" {'um/min':>9} {'um':>7} {'min':>7}"
        )
        lines = [header, units, "  " + "-" * 118]
        for stage in self.stages:
            recipe = stage.candidate.recipe
            drops = max(round(recipe.soap_drops), 1)
            summary = (
                f"{recipe.grit_grams:.0f} g + {recipe.water_ml:.0f} ml water"
                f" + {recipe.glycerine_ml:.0f} ml glyc"
                f" + {drops} drop{'s' if drops != 1 else ''}"
            )
            flag = "" if stage.in_stock else "  (to buy)"
            lines.append(
                f"  {stage.abrasive_name:<22} {stage.grit:>6} {stage.d50_um:6.1f}"
                f" {stage.Ra_um:7.3f} {stage.subsurface_damage_um:7.2f}"
                f" {stage.candidate.removal_rate_um_min:9.1f}"
                f" {stage.damage_to_remove_um:7.2f} {stage.minutes_required:7.1f}"
                f"  {summary}{flag}"
            )
        lines.append("  " + "-" * 118)
        lines.append(
            f"  Total {self.total_minutes:.0f} min of continuous grinding at the "
            "stated pressure and speed"
        )
        lines.append(
            "  (real bench time is longer: strokes lift, pressure varies, and "
            "the plate needs re-charging)."
        )
        for skip in self.skipped:
            lines.append("")
            lines.append(f"  SKIPPED  {skip.abrasive_name} {skip.grit}")
            for chunk in _wrap(skip.explanation, 112):
                lines.append(f"    {chunk}")
        return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width) or [""]


def plan_sequence(base: Setup, stages: list[tuple[str, str]] | None = None,
                  prefer: str = "robust",
                  initial_damage_um: float = 120.0,
                  safety_factor: float = 1.3,
                  diagnose: bool = True,
                  n_solids: int = 22, n_glycerine: int = 22) -> SequencePlan:
    """Build a grit progression, with a recipe and a time for each stage.

    Each stage must remove the subsurface damage its predecessor left, times a
    safety factor.  ``initial_damage_um`` stands for whatever the stone arrived
    with -- saw marks, a coarse-wheel finish -- and defaults to a fairly rough
    starting surface.

    Parameters
    ----------
    stages:
        ``[(abrasive_key, grit_label), ...]`` coarse to fine.  Defaults to
        everything currently marked as in stock.
    initial_damage_um:
        Damage depth on the stone as it arrives.  120 um suits a sawn face; drop
        it to 30-40 um if you are starting from an already-ground surface.
    diagnose:
        When a grit has no workable recipe, search for changes of technique that
        would give it one, instead of silently leaving it out.
    """
    if stages is None:
        from lapsim.materials import stock_grades

        stages = [
            (abrasive.key, grade.label) for abrasive, grade in stock_grades()
        ]

    plan = SequencePlan()
    damage_to_clear = initial_damage_um

    for abrasive_key, grit in stages:
        abrasive = ABRASIVES[abrasive_key]
        grade = abrasive.grade(grit)
        stage_setup = replace(base, abrasive=abrasive_key, grit=grit)
        candidate = find_recipe(stage_setup, prefer=prefer,
                                n_solids=n_solids, n_glycerine=n_glycerine)
        if candidate is None:
            remedies = (
                diagnose_infeasible(stage_setup, n_solids=14, n_glycerine=14)
                if diagnose else []
            )
            plan.skipped.append(SkippedStage(
                abrasive_name=abrasive.name,
                grit=grit,
                explanation=explain_infeasible(stage_setup, remedies),
                remedies=remedies,
            ))
            continue

        required = damage_to_clear * safety_factor
        rate = candidate.removal_rate_um_min
        minutes = required / rate if rate > 0 else math.inf

        plan.stages.append(Stage(
            abrasive_key=abrasive_key,
            abrasive_name=abrasive.name,
            grit=grit,
            d50_um=grade.d50_um,
            candidate=candidate,
            damage_to_remove_um=required,
            minutes_required=minutes,
            in_stock=grade.in_stock,
        ))
        damage_to_clear = candidate.result.finish.subsurface_damage_um

    return plan
