"""Top-level simulation: one recipe, one process, one complete answer.

:class:`Setup` describes everything you would have to decide before starting --
which grit, how much of it, how much glycerine, how hard you press, how humid the
room is.  :func:`simulate` runs the whole chain and returns a :class:`Result`
that knows how to print itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lapsim.calibration import Calibration, DEFAULT
from lapsim.contact import ContactState, solve_contact
from lapsim.drying import DryingResult, dry_out
from lapsim.materials import ABRASIVES, LAPS, STONES, Abrasive, GritGrade, Solid
from lapsim.removal import RemovalResult, evaluate_removal
from lapsim.slurry import Slurry
from lapsim.vehicle import Vehicle

SECONDS_PER_MINUTE = 60.0


@dataclass
class Targets:
    """The consistency band you want to stay inside.

    Defaults encode "thick enough not to run off the plate or let the grit
    settle, thin enough to still flow into the contact and not drag".  Adjust
    them to your own taste -- they are the constraints the recipe search honours.
    """

    max_yield_stress_Pa: float = 25.0
    """Above this the slurry sits in a lump instead of spreading under the stone."""

    max_plastic_viscosity_Pas: float = 2.0
    """Above this it drags and traps heat."""

    min_low_shear_viscosity_Pas: float = 0.02
    """Below this it is runny: it sheets off the plate and needs constant
    re-application."""

    min_settling_time_s: float = 120.0
    """Grit should stay suspended for at least this long, or you are grinding on
    whatever happens to still be in the liquid."""

    min_grain_load_fraction: float = 0.70
    """Below this the stone is floating on the slurry film more than riding on
    the grains, and the cutting rate collapses."""

    min_workable_time_s: float = 600.0
    """How long the film on the plate should last before needing re-wetting."""

    enforce_finish: bool = False
    """Hold recipes to a finish ceiling (see :attr:`roughness_tolerance`).

    Off by default because the model gets the direction wrong at the bench.  It
    has fewer grains each carrying more load, so a lean slurry digs deeper and
    finishes *rougher*; in practice a lean charge of friable SiC breaks down in
    use and finishes *brighter*, and the model does not follow grain breakdown.
    Switched on, this constraint pushes every recipe towards heavy loadings.
    Ra is still reported either way.
    """

    max_Ra_um: float | None = None
    """Hard cap on the finish in microns, or ``None`` to derive one from the
    grade (see :attr:`roughness_tolerance`).  Only used when
    :attr:`enforce_finish` is on."""

    roughness_tolerance: float = 1.25
    """How much worse than its nominal finish a grade may be allowed to come out,
    when :attr:`enforce_finish` is on: the ceiling is ``tolerance * d50 / 30``.
    """

    def roughness_ceiling(self, d50_um: float, calibration: Calibration) -> float:
        """The Ra ceiling in microns for a grade of this median size."""
        if self.max_Ra_um is not None:
            return self.max_Ra_um
        return self.roughness_tolerance * d50_um / calibration.nominal_roughness_divisor

    max_foam_tendency: float = 0.2
    """How much air entrainment is tolerable, 0-1.

    Detergent past a few times the critical micelle concentration buys no extra
    wetting and starts whipping air into the slurry, which unloads the grains
    and makes a mess.
    """

    max_fractured_fraction: float = 0.15
    """Most of the cutting grains that may be crushing at any moment.

    Beyond this the slurry is destroying its own grit: it drifts finer as you
    work, the cut rate falls off and the grade stops meaning anything. It is the
    signature of too little abrasive, or too much pressure, or both.
    """

    forbid_seizing: bool = True
    """Reject recipes that would eventually dry to an unworkable crust."""


@dataclass
class Setup:
    """A complete specification of one grinding run."""

    # ---- what you are using -------------------------------------------------
    abrasive: str = "sic_black"
    grit: str = "600"
    """A FEPA F grit number up to 1000 (``"600"`` or ``"F600"``), or a micron
    size for anything finer, e.g. ``"3um"`` -- see :mod:`lapsim.materials`."""
    stone: str = "agate"
    lap: str = "glass"

    # ---- the recipe ---------------------------------------------------------
    solids_mass_fraction: float = 0.40
    """Grams of grit per gram of finished slurry."""

    glycerine_mass_fraction: float = 0.35
    """Glycerine as a fraction of the water+glycerine liquid."""

    soap_volume_fraction: float = 5.0e-4
    """Neat dish soap as a volume fraction of the liquid.  5e-4 is about one drop
    per 100 ml -- already past the point where more does any good."""

    # ---- how you work -------------------------------------------------------
    normal_load_N: float = 40.0
    """Downward force on the stone.  Firm hand pressure on a cabochon is 20-60 N."""

    contact_area_mm2: float = 400.0
    """Plan area of the face being ground -- 4 cm^2 is a large cabochon."""

    sliding_speed_m_s: float = 0.25
    """Average speed of the stone over the plate."""

    contact_length_mm: float = 20.0
    """Length of the ground face along the stroke, which sets hydrodynamic lift."""

    gap_convergence: float = 1.5
    """Inlet/outlet gap ratio.  Hand lapping is never parallel; 1.0 disables
    hydrodynamic lift entirely."""

    # ---- the plate and the film on it --------------------------------------
    plate_area_cm2: float = 400.0
    """Area of the glass plate, 200 x 200 mm by default."""

    wetted_fraction: float = 0.5
    """Fraction of the plate actually carrying slurry."""

    film_volume_ml: float = 8.0
    """How much slurry is spread on the plate at a time."""

    # ---- the room -----------------------------------------------------------
    temperature_C: float = 21.0
    relative_humidity: float = 0.45
    air_speed_m_s: float = 0.3

    drying_steps: int = 900
    """Time steps in the drying integration.  Lower it for bulk recipe scans."""

    targets: Targets = field(default_factory=Targets)
    calibration: Calibration = DEFAULT

    # ---- resolution ---------------------------------------------------------
    def resolve(self) -> tuple[Abrasive, GritGrade, Solid, Solid]:
        abrasive = ABRASIVES[self.abrasive]
        return abrasive, abrasive.grade(self.grit), STONES[self.stone], LAPS[self.lap]

    @property
    def nominal_pressure(self) -> float:
        """Pa."""
        return self.normal_load_N / (self.contact_area_mm2 * 1e-6)

    @property
    def exposed_area(self) -> float:
        """m^2 of slurry open to the air."""
        return self.plate_area_cm2 * 1e-4 * self.wetted_fraction

    def build_slurry(self) -> Slurry:
        vehicle = Vehicle.from_fractions(
            self.glycerine_mass_fraction,
            self.soap_volume_fraction,
            temperature_C=self.temperature_C,
        )
        abrasive, grade, _, _ = self.resolve()
        return Slurry.from_mass_fraction(
            abrasive, grade, vehicle, self.solids_mass_fraction, self.calibration
        )


@dataclass
class Result:
    """Everything the simulator knows about one recipe."""

    setup: Setup
    slurry: Slurry
    contact: ContactState
    removal: RemovalResult
    drying: DryingResult
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------- convenience accessors
    @property
    def finish(self):
        return self.removal.finish

    @property
    def removal_rate_um_min(self) -> float:
        return self.removal.removal_rate * 1e6 * SECONDS_PER_MINUTE

    @property
    def lap_wear_um_min(self) -> float:
        return self.removal.lap_wear_rate * 1e6 * SECONDS_PER_MINUTE

    def minutes_to_remove(self, depth_um: float) -> float:
        rate = self.removal_rate_um_min
        return float("inf") if rate <= 0 else depth_um / rate

    @property
    def is_viable(self) -> bool:
        return self.contact.is_cutting and not self.warnings

    # -------------------------------------------------------------- reporting
    def report(self) -> str:
        setup = self.setup
        abrasive, grade, stone, lap = setup.resolve()
        slurry, contact, removal, drying = (
            self.slurry, self.contact, self.removal, self.drying
        )
        finish = removal.finish

        def minutes(seconds: float) -> str:
            if seconds == float("inf"):
                return "indefinitely"
            return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"

        lines = [
            "=" * 74,
            f"  {abrasive.name} {grade.label}  on  {stone.name}  over a {lap.name}",
            "=" * 74,
            "",
            "SLURRY",
            f"  grit loading           {slurry.solids_mass_fraction * 100:5.1f} wt%"
            f"  = {slurry.solids_volume_fraction * 100:.1f} vol%"
            f"   (jams at {slurry.max_packing_fraction * 100:.0f} vol%)",
            f"  grain size             d50 {grade.d50_um:.1f} um,"
            f" top cut {grade.d_max_um:.1f} um, Sauter mean"
            f" {slurry.surface_volume_diameter * 1e6:.1f} um",
            f"  liquid                 {slurry.vehicle.describe()}",
            f"  slurry viscosity       {slurry.plastic_viscosity * 1e3:.0f} mPa.s"
            f" at high shear;  yield stress {slurry.yield_stress:.1f} Pa",
            f"  consistency            {slurry.consistency_label}"
            f"   (gels above {slurry.gel_fraction * 100:.0f} vol% solids)",
            f"  blob height on plate   {slurry.puddle_height * 1e3:.2f} mm"
            "   <- measurable: leave a blob and see",
            f"  grit stays suspended   {minutes(slurry.settling_time())}",
            "",
            "CONTACT  (regime: " + contact.regime + ")",
            f"  pressure               {setup.nominal_pressure / 1e3:.0f} kPa"
            f"  ({setup.normal_load_N:.0f} N over {setup.contact_area_mm2:.0f} mm2)",
            f"  plate gap              {contact.gap * 1e6:.1f} um"
            f"  = {contact.gap / (grade.d50_um * 1e-6):.2f} x d50",
            f"  load on grains         {contact.load_fraction_on_grains * 100:.0f} %"
            f"   (rest is carried hydrodynamically by the film)",
            f"  grains actually cutting {contact.active_grain_density / 1e6:.2f} per mm2"
            f"  = {contact.active_fraction * 100:.3f} % of those present",
            f"  penetration into stone {contact.penetration_work.max() * 1e9:.0f} nm"
            f" at the deepest grain;  into the plate"
            f" {contact.penetration_lap.max() * 1e9:.0f} nm",
            f"  grains being crushed   {contact.fractured_fraction * 100:.1f} %",
            f"  grain anchoring        {contact.lap_anchoring:.3f}"
            f"  ({'grains bedded in the plate, cutting two-body' if contact.lap_anchoring > 0.1 else 'grains rolling freely, three-body'})",
            "",
            "CUTTING",
            f"  stone removal          {self.removal_rate_um_min:.1f} um/min"
            f"   ({finish.regime} regime,"
            f" {finish.brittle_fraction * 100:.0f} % of load cracking)",
            f"  plate wear             {self.lap_wear_um_min:.1f} um/min"
            f"   ({removal.plate_to_stone_wear_ratio:.2f} x the stone)",
            f"  Preston coefficient    {removal.preston_coefficient:.2e} m2/N",
            f"  drag                   {removal.drag_force:.1f} N"
            f"  ({removal.drag_ratio:.2f} x the load);"
            f" {removal.friction_power:.1f} W, contact +{removal.temperature_rise:.0f} K",
            "",
            "FINISH",
            f"  roughness              Ra {finish.Ra_um:.2f} um,"
            f" Rz {finish.Rz_um:.2f} um",
            f"  subsurface damage      {finish.subsurface_damage_um:.1f} um"
            "   <- the next grit must remove at least this much",
            f"  worst-case scratch     {finish.deepest_scratch_um:.2f} um deep,"
            f" from the {grade.d_max_um:.0f} um top-cut grains",
            "",
            "WORKING LIFE OF THE FILM ON THE PLATE",
            f"  {drying.film_volume_ml:.0f} ml over"
            f" {drying.exposed_area_cm2:.0f} cm2 = a"
            f" {drying.film_thickness_um:.0f} um film,"
            f" losing {drying.evaporation_rate_g_per_min:.2f} g water/min",
            f"  stays in the target band for {minutes(drying.workable_time_s)}",
            f"  dries down to          {drying.equilibrium_solids_fraction * 100:.0f} vol% solids"
            f"  (workable ceiling {drying.jamming_solids_fraction * 100:.0f} vol%)",
        ]
        for note in drying.notes:
            lines.append(f"  - {note}")

        if self.warnings:
            lines += ["", "WARNINGS"]
            lines += [f"  ! {warning}" for warning in self.warnings]
        else:
            lines += ["", "No warnings: this recipe sits inside every target band."]
        lines.append("")
        return "\n".join(lines)


def _collect_warnings(setup: Setup, slurry: Slurry, contact: ContactState,
                      removal: RemovalResult, drying: DryingResult) -> list[str]:
    """Check the solved state against the targets and against physical sense."""
    targets = setup.targets
    abrasive, grade, stone, _ = setup.resolve()
    warnings: list[str] = []

    hardness_ratio = abrasive.hardness / stone.hardness
    if hardness_ratio < 1.5:
        warnings.append(
            f"{abrasive.name} is only {hardness_ratio:.2f} x as hard as "
            f"{stone.name}: it will blunt rather than cut. Use a harder abrasive."
        )

    if contact.regime == "hydroplaning":
        warnings.append(
            "Hydroplaning: the slurry film is thick enough to float the stone "
            "clear of the grains, so nothing is cutting. Thin the vehicle, slow "
            "down, or press harder."
        )
    elif contact.load_fraction_on_grains < targets.min_grain_load_fraction:
        warnings.append(
            f"Only {contact.load_fraction_on_grains * 100:.0f} % of the load is on "
            f"the grains (target {targets.min_grain_load_fraction * 100:.0f} %); "
            "the film is taking the rest and the cut is slow for the effort."
        )
    elif contact.regime == "plate contact":
        warnings.append(
            "Not enough abrasive in the gap to carry the load: the stone is "
            "rubbing on the plate. Add grit or reduce pressure."
        )

    if slurry.yield_stress > targets.max_yield_stress_Pa:
        warnings.append(
            f"Yield stress {slurry.yield_stress:.0f} Pa exceeds the "
            f"{targets.max_yield_stress_Pa:.0f} Pa target: too stiff to spread "
            "under the stone."
        )
    if slurry.plastic_viscosity > targets.max_plastic_viscosity_Pas:
        warnings.append(
            f"Slurry viscosity {slurry.plastic_viscosity:.2f} Pa.s exceeds the "
            f"{targets.max_plastic_viscosity_Pas:.2f} Pa.s target: heavy drag."
        )

    from lapsim.slurry import LOW_SHEAR_RATE
    low_shear = slurry.apparent_viscosity(LOW_SHEAR_RATE)
    if low_shear < targets.min_low_shear_viscosity_Pas:
        warnings.append(
            f"Too runny: {low_shear * 1e3:.0f} mPa.s at low shear "
            f"({slurry.consistency_label}). It will sheet off the plate."
        )

    settling = slurry.settling_time()
    if settling < targets.min_settling_time_s:
        warnings.append(
            f"Grit settles out in {settling:.0f} s: add glycerine, or use a "
            "finer grade, or expect to keep stirring."
        )

    if slurry.vehicle.foam_tendency > targets.max_foam_tendency:
        warnings.append(
            f"Too much soap ({slurry.vehicle.soap_volume_fraction * 1e4:.1f}e-4 "
            "by volume): it will foam and the air bubbles unload the grains. "
            "One or two drops per 100 ml is plenty."
        )

    if targets.forbid_seizing and drying.seizes:
        warnings.append(
            f"Will dry to a crust: the endpoint is "
            f"{drying.equilibrium_solids_fraction * 100:.0f} vol% solids against a "
            f"{drying.jamming_solids_fraction * 100:.0f} vol% ceiling."
        )
    if drying.workable_time_s < targets.min_workable_time_s:
        warnings.append(
            f"Film leaves the target band after only "
            f"{drying.workable_time_s / 60:.0f} min "
            f"(want {targets.min_workable_time_s / 60:.0f} min)."
        )

    if contact.fractured_fraction > targets.max_fractured_fraction:
        warnings.append(
            f"{contact.fractured_fraction * 100:.0f} % of the cutting grains are "
            "being crushed: the slurry is destroying its own grit and will drift "
            "finer as you work. Add abrasive or reduce pressure."
        )

    ceiling = targets.roughness_ceiling(grade.d50_um, setup.calibration)
    if targets.enforce_finish and removal.finish.Ra_um > ceiling:
        warnings.append(
            f"Finish is Ra {removal.finish.Ra_um:.2f} um where this grade should "
            f"manage {ceiling:.2f} um: too little abrasive in the gap, so the few "
            "grains carrying load dig in and scratch instead of cutting evenly."
        )

    if removal.plate_to_stone_wear_ratio > 3.0:
        warnings.append(
            f"The plate wears {removal.plate_to_stone_wear_ratio:.1f} x faster "
            "than the stone; expect it to dish. A harder lap would last longer."
        )

    return warnings


def simulate(setup: Setup | None = None, **overrides) -> Result:
    """Run the full chain for one recipe.

    >>> result = simulate(grit="600", solids_mass_fraction=0.35)
    >>> print(result.report())
    """
    if setup is None:
        setup = Setup(**overrides)
    elif overrides:
        from dataclasses import replace

        setup = replace(setup, **overrides)

    abrasive, grade, stone, lap = setup.resolve()
    slurry = setup.build_slurry()

    contact = solve_contact(
        slurry, stone, lap,
        nominal_pressure=setup.nominal_pressure,
        sliding_speed=setup.sliding_speed_m_s,
        contact_length=setup.contact_length_mm * 1e-3,
        convergence=setup.gap_convergence,
        calibration=setup.calibration,
    )
    removal = evaluate_removal(
        contact, stone, lap,
        nominal_pressure=setup.nominal_pressure,
        contact_area=setup.contact_area_mm2 * 1e-6,
        sliding_speed=setup.sliding_speed_m_s,
        tip_semi_angle_deg=abrasive.tip_semi_angle_deg,
        ambient_temperature_C=setup.temperature_C,
        calibration=setup.calibration,
    )
    drying = dry_out(
        slurry,
        relative_humidity=setup.relative_humidity,
        air_speed_m_s=setup.air_speed_m_s,
        exposed_area=setup.exposed_area,
        film_volume_ml=setup.film_volume_ml,
        target_yield_stress_Pa=setup.targets.max_yield_stress_Pa,
        target_viscosity_Pas=setup.targets.max_plastic_viscosity_Pas,
        steps=setup.drying_steps,
        calibration=setup.calibration,
    )

    warnings = _collect_warnings(setup, slurry, contact, removal, drying)
    return Result(setup, slurry, contact, removal, drying, warnings)
