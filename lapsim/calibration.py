"""Tunable constants, collected in one place so they can be re-fitted.

Everything in this simulator splits into two kinds of number:

*   **Physical constants and measured material properties.**  Viscosity of
    glycerol-water, hardness of quartz, Stokes settling, vapour pressure of
    water.  These come from the literature and live in :mod:`lapsim.materials`
    and :mod:`lapsim.vehicle`.
*   **Coupling coefficients.**  How much of the groove a grain plows actually
    leaves as swarf; how a distribution of lateral cracks adds up to an Ra
    number; how many of the grains in the gap really carry load.  Nobody
    measures these directly -- they are fitted, and they live *here*.

The consequence, stated plainly: **relative** answers from this model (is 600
grit at 18 wt% better than at 30 wt%? does more glycerine help?) rest on the
physics and are trustworthy.  **Absolute** answers (35 um/min, Ra 0.9 um) rest
on the constants below.  If you ever measure your own removal rate or finish,
retune these and everything downstream sharpens up.  Run
``python -m lapsim.calibration`` to see how the current values score against
the anchor cases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    """Fitted coefficients of the contact / removal / drying models."""

    # ---------------------------------------------------------------- contact
    gap_enrichment: float = 1.0
    """Solids fraction in the plate gap divided by bulk slurry solids fraction.

    Slurry entering a loaded contact can be filtered (liquid squeezes out
    faster than grains, enriching the gap) or starved (grains are swept aside).
    1.0 means "the gap sees the bulk mix"; raise it if you believe your contact
    packs grains in.
    """

    monolayer_packing: float = 0.55
    """Maximum areal fraction a single layer of angular grains can occupy.

    Caps the number of grains that can physically fit in the gap, independent
    of how much abrasive the slurry carries.
    """

    hydrodynamic_efficiency: float = 0.35
    """Fraction of the fully-flooded Reynolds lift a real lapped contact develops.

    Reynolds' inclined-slider solution assumes a steady wedge with its inlet
    completely full of liquid. A hand-lapped contact is neither: the slurry film
    is thin enough that the inlet is *starved*, which typically costs half the
    load capacity or more, and a reciprocating stroke passes through zero speed
    at every reversal, where entrainment stops and the film collapses. Averaged
    over a stroke, measured lift comes out well below the flooded steady value.

    The flooded number is therefore an upper bound, and taking it literally
    predicts that fine grits cannot be lapped at ordinary speeds at all, which is
    plainly untrue. Raise this towards 1 if your plate genuinely feels slick and
    stops cutting; lower it if fine grades cut more readily than the model says.
    """

    indenter_semi_angle_deg: float = 70.0
    """Effective semi-apex angle of the load-bearing grain tip, degrees.

    A real grain is not a sphere; the corner that happens to be pointing at the
    work behaves like a blunt cone.  90 deg would be a flat punch, 45 deg a
    needle.  Individual abrasives override this via
    :attr:`lapsim.materials.Abrasive.tip_semi_angle_deg`.
    """

    # ---------------------------------------------------------------- removal
    ductile_removal_efficiency: float = 0.15
    """Fraction of the plowed groove that leaves as swarf below the brittle
    threshold.  The rest is pushed into the ridges either side of the scratch.
    """

    brittle_removal_efficiency: float = 0.30
    """Fraction of the lateral-crack envelope (2 * c_L * c_h) that actually
    spalls out per pass.  Less than one because cracks partly arrest and
    re-close.
    """

    lateral_crack_length_coeff: float = 0.10
    """alpha_L in the lateral-crack length scaling (see
    :func:`lapsim.removal.lateral_crack`)."""

    lateral_crack_depth_coeff: float = 0.226
    """alpha_h in the lateral-crack depth scaling.  Marshall, Lawn & Evans
    (1982) give ~0.226 for Vickers indentation of brittle solids."""

    crack_initiation_coeff: float = 2.0e4
    """lambda in F_crit = lambda * K_c^4 / H^3, the load below which a sharp
    indenter deforms a brittle solid plastically instead of cracking it."""

    # -------------------------------------------------------------- roughness
    roughness_from_crack_depth: float = 0.72
    """Ra divided by the 90th-percentile lateral-crack depth, brittle regime."""

    roughness_from_plow_depth: float = 0.30
    """Ra divided by the 90th-percentile plastic penetration, ductile regime."""

    peak_to_valley_ratio: float = 5.5
    """Rz / Ra for a lapped surface.  5-7 is typical of a randomly pitted,
    non-directional finish."""

    nominal_roughness_divisor: float = 30.0
    """Ra a grade should deliver, as ``d50 / this``.

    The standard free-abrasive lapping rule of thumb.  The calibration check
    scores the model against it, and the recipe search holds recipes to it only
    when :attr:`lapsim.simulate.Targets.enforce_finish` is switched on.
    """

    subsurface_damage_ratio: float = 2.5
    """Depth of cracked-but-attached material divided by Rz (peak-to-valley).

    So the damage depth is ``2.5 * Rz``, about 14 x Ra -- within the 1.5-4 x Rz
    usually quoted for loose-abrasive work.  It is a rule of thumb applied to the
    finish, not a crack calculation, and it runs several times deeper than the
    lateral cracks the model computes, because the deepest damage comes from
    median cracks that the model does not follow.

    This is the number that tells you how long to spend on the *next* grit:
    until you have removed this much, you are only polishing the tops of
    scratches the previous grit left below the surface.
    """

    # ----------------------------------------------------------------- stress
    yield_stress_coeff: float = 8.0e-3
    """C in tau_y = C * (gamma / d_sv) * f(phi).

    Sets the absolute scale of the slurry's yield stress.  The surface-tension
    over particle-size grouping is what makes fine grit thicken a slurry so
    much more than coarse grit at the same solids loading.
    """

    gel_fraction_cohesive: float = 0.05
    """Gel point for grains fine enough that surface forces dominate gravity.

    Below the gel point a suspension has no measurable yield stress at all -- it
    is just a viscous liquid with grains in it.
    """

    gel_fraction_coarse: float = 0.40
    """Gel point for grains so coarse that only near-jamming makes a network."""

    cohesive_size_um: float = 5.0
    """Grain size at which the gel point sits halfway between the two limits.

    Interparticle attraction scales with grain size while grain weight scales
    with its cube, so somewhere around a few microns the two cross over.  Above
    that size a suspension is held up by viscosity alone; below it, a genuine
    yield stress appears at modest loading.  This is the single biggest
    behavioural difference between your 320 grit and the 3 um alumina you are
    planning to buy.
    """

    # ---------------------------------------------------------------- friction
    abrasive_friction_coeff: float = 0.28
    """Coefficient of friction of grain-on-stone sliding contact."""

    # ----------------------------------------------------------------- drying
    mass_transfer_base_m_s: float = 0.0030
    """Water-vapour mass transfer coefficient in still air, m/s."""

    mass_transfer_per_air_speed: float = 0.020
    """Extra mass transfer coefficient per m/s of air movement over the plate.
    Hand motion alone is worth a few tenths of a m/s.
    """

    workable_solids_headroom: float = 0.90
    """Slurry is declared seized once phi reaches this fraction of phi_max."""


DEFAULT = Calibration()


# --------------------------------------------------------------------------- #
# Anchors: what the defaults above were tuned to reproduce.
# --------------------------------------------------------------------------- #

ANCHORS = [
    # (description, quantity, expected value or None for d50/30, acceptable factor)
    ("320 grit SiC on agate, 40 N over 4 cm^2", "removal_rate_um_min", 35.0, 2.5),
    ("320 grit SiC on agate", "Ra_um", None, 2.0),
    ("600 grit SiC on agate", "Ra_um", None, 2.0),
    ("1000 grit SiC on agate", "Ra_um", None, 2.0),
]
"""Published / workshop-experience values the fitted constants aim at.

Sources are rules of thumb rather than a single paper: free-abrasive lapping of
silicate rock leaves Ra of roughly d50/30 (``None`` above means "d50/30 for this
grade"), and hand lapping a cabochon on a glass plate with coarse SiC takes tens
of microns per minute.  All are checked at the default setup (40 N over 4 cm^2).
Treat the tolerance column as "same ballpark", not "same number".
"""


def _report() -> None:
    """Print predicted vs. anchor values for the current constants."""
    from lapsim.simulate import Setup, simulate

    print("Calibration anchor check")
    print("=" * 72)
    rows = []
    for grit in ("320", "600", "1000"):
        result = simulate(Setup(grit=grit))
        rows.append((grit, result))

    print(f"{'grit':>6} {'removal (um/min)':>18} {'Ra (um)':>10} {'regime':>12}")
    for grit, result in rows:
        print(
            f"{grit:>6} {result.removal_rate_um_min:18.1f}"
            f" {result.finish.Ra_um:10.3f} {result.finish.regime:>12}"
        )

    print()
    for description, quantity, expected, tolerance in ANCHORS:
        grit = description.split()[0]
        result = dict(rows)[grit]
        if expected is None:
            grade = result.setup.resolve()[1]
            expected = grade.d50_um / result.setup.calibration.nominal_roughness_divisor
        got = getattr(result, quantity, None)
        if got is None:
            got = getattr(result.finish, quantity)
        ratio = got / expected
        verdict = "ok" if 1 / tolerance <= ratio <= tolerance else "OFF"
        print(
            f"[{verdict:>3}] {description:<42} {quantity:<20}"
            f" got {got:8.3f}  want ~{expected:.3f}  (x{ratio:.2f})"
        )


if __name__ == "__main__":
    _report()
