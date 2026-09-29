"""JSON bridge, so a front end in another language can drive the simulator.

Run it as a long-lived server::

    python -m lapsim.api

It then reads one JSON request per line from standard input and writes one JSON
response per line to standard output.  Keeping the process alive matters: most
of the cost of a request is importing numpy and scipy, so a persistent server
answers in milliseconds where a fresh process would take most of a second --
which is the difference between a slider that tracks your finger and one that
stutters.

For a single shot, pass ``--once`` and the request as an argument::

    python -m lapsim.api --once '{"command": "capabilities"}'

Every response carries ``{"ok": true, ...}`` or
``{"ok": false, "error": "..."}``; nothing raises across the boundary.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace
from typing import Any

from lapsim.materials import ABRASIVES, LAPS, STONES
from lapsim.simulate import Setup, Targets, simulate

PROTOCOL_VERSION = 1


# --------------------------------------------------------------------------- #
# Request -> Setup
# --------------------------------------------------------------------------- #

_SETUP_FIELDS = {
    "abrasive", "grit", "stone", "lap",
    "solids_mass_fraction", "glycerine_mass_fraction", "soap_volume_fraction",
    "normal_load_N", "contact_area_mm2", "sliding_speed_m_s",
    "contact_length_mm", "gap_convergence",
    "plate_area_cm2", "wetted_fraction", "film_volume_ml",
    "temperature_C", "relative_humidity", "air_speed_m_s",
}

_TARGET_FIELDS = {
    "max_yield_stress_Pa", "max_plastic_viscosity_Pas",
    "min_low_shear_viscosity_Pas", "min_settling_time_s",
    "min_grain_load_fraction", "min_workable_time_s", "forbid_seizing",
    "enforce_finish", "max_Ra_um", "roughness_tolerance", "max_foam_tendency",
    "max_fractured_fraction",
}


def build_setup(payload: dict[str, Any]) -> Setup:
    """Turn a request body into a :class:`Setup`, ignoring unknown keys."""
    target_values = {k: v for k, v in payload.items() if k in _TARGET_FIELDS}
    setup_values = {k: v for k, v in payload.items() if k in _SETUP_FIELDS}
    targets = replace(Targets(), **target_values) if target_values else Targets()
    return Setup(targets=targets, **setup_values)


# --------------------------------------------------------------------------- #
# Serialisers
# --------------------------------------------------------------------------- #


def _finite(value: float) -> float | None:
    """JSON has no infinity; a null reads as 'never' on the far side."""
    return None if value in (float("inf"), float("-inf")) else round(value, 6)


def describe_result(result) -> dict[str, Any]:
    """Flatten a :class:`lapsim.simulate.Result` into plain JSON types."""
    setup, slurry, contact, removal = (
        result.setup, result.slurry, result.contact, result.removal
    )
    drying, finish = result.drying, result.finish
    abrasive, grade, stone, lap = setup.resolve()
    from lapsim.slurry import LOW_SHEAR_RATE

    return {
        "abrasive_name": abrasive.name,
        "grit": grade.label,
        "d50_um": grade.d50_um,
        "d_max_um": round(grade.d_max_um, 2),
        "stone_name": stone.name,
        "lap_name": lap.name,
        "slurry": {
            "solids_wt_pct": round(slurry.solids_mass_fraction * 100, 2),
            "solids_vol_pct": round(slurry.solids_volume_fraction * 100, 2),
            "jamming_vol_pct": round(slurry.max_packing_fraction * 100, 1),
            "gel_vol_pct": round(slurry.gel_fraction * 100, 1),
            "vehicle_viscosity_mPas": round(slurry.vehicle.viscosity * 1e3, 2),
            "slurry_viscosity_mPas": round(slurry.plastic_viscosity * 1e3, 2),
            "low_shear_viscosity_mPas": round(
                slurry.apparent_viscosity(LOW_SHEAR_RATE) * 1e3, 2),
            "yield_stress_Pa": round(slurry.yield_stress, 3),
            "surface_tension_mNm": round(slurry.vehicle.surface_tension * 1e3, 1),
            "water_activity": round(slurry.vehicle.water_activity, 4),
            "puddle_height_mm": round(slurry.puddle_height * 1e3, 3),
            "settling_time_s": _finite(slurry.settling_time()),
            "consistency": slurry.consistency_label,
        },
        "contact": {
            "regime": contact.regime,
            "pressure_kPa": round(setup.nominal_pressure / 1e3, 1),
            "gap_um": round(contact.gap * 1e6, 3),
            "gap_over_d50": round(contact.gap / (grade.d50_um * 1e-6), 3),
            "load_fraction_on_grains": round(contact.load_fraction_on_grains, 4),
            "active_grains_per_mm2": round(contact.active_grain_density / 1e6, 3),
            "active_fraction_pct": round(contact.active_fraction * 100, 4),
            "max_penetration_stone_nm": round(
                float(contact.penetration_work.max()) * 1e9, 1),
            "max_penetration_lap_nm": round(
                float(contact.penetration_lap.max()) * 1e9, 1),
            "fractured_pct": round(contact.fractured_fraction * 100, 2),
            "lap_anchoring": round(contact.lap_anchoring, 4),
        },
        "cutting": {
            "removal_um_min": round(result.removal_rate_um_min, 3),
            "lap_wear_um_min": round(result.lap_wear_um_min, 3),
            "plate_to_stone_wear": round(removal.plate_to_stone_wear_ratio, 3),
            "preston_coefficient": removal.preston_coefficient,
            "drag_N": round(removal.drag_force, 2),
            "drag_ratio": round(removal.drag_ratio, 3),
            "power_W": round(removal.friction_power, 2),
            "temperature_rise_K": round(removal.temperature_rise, 1),
        },
        "finish": {
            "Ra_um": round(finish.Ra_um, 4),
            "Rz_um": round(finish.Rz_um, 4),
            "subsurface_damage_um": round(finish.subsurface_damage_um, 3),
            "deepest_scratch_um": round(finish.deepest_scratch_um, 3),
            "regime": finish.regime,
            "brittle_fraction": round(finish.brittle_fraction, 3),
        },
        "drying": {
            "film_volume_ml": drying.film_volume_ml,
            "exposed_area_cm2": round(drying.exposed_area_cm2, 1),
            "film_thickness_um": round(drying.film_thickness_um, 1),
            "evaporation_g_min": round(drying.evaporation_rate_g_per_min, 4),
            "workable_time_s": _finite(drying.workable_time_s),
            "time_to_seize_s": _finite(drying.time_to_seize_s),
            "equilibrium_solids_pct": round(
                drying.equilibrium_solids_fraction * 100, 1),
            "jamming_solids_pct": round(drying.jamming_solids_fraction * 100, 1),
            "seizes": drying.seizes,
            "notes": drying.notes,
        },
        "warnings": result.warnings,
        "report": result.report(),
    }


def describe_recipe(recipe) -> dict[str, Any]:
    drops = max(round(recipe.soap_drops), 1)
    return {
        "abrasive_name": recipe.abrasive_name,
        "grit": recipe.grit,
        "batch_ml": recipe.batch_ml,
        "grit_grams": round(recipe.grit_grams, 1),
        "water_ml": round(recipe.water_ml, 1),
        "glycerine_ml": round(recipe.glycerine_ml, 1),
        "soap_drops": drops,
        "solids_wt_pct": round(recipe.solids_mass_fraction * 100, 1),
        "glycerine_wt_pct": round(recipe.glycerine_mass_fraction * 100, 1),
        "text": recipe.describe(),
    }


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    """What this installation can do, and what it knows about."""
    try:
        import matplotlib  # noqa: F401

        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    return {
        "protocol": PROTOCOL_VERSION,
        "python": sys.version.split()[0],
        "can_plot": has_matplotlib,
        "abrasives": [
            {
                "key": abrasive.key,
                "name": abrasive.name,
                "notes": abrasive.notes,
                "grades": [
                    {"label": grade.label, "d50_um": grade.d50_um,
                     "d_max_um": round(grade.d_max_um, 1),
                     "in_stock": grade.in_stock}
                    for grade in abrasive.grades.values()
                ],
            }
            for abrasive in ABRASIVES.values()
        ],
        "stones": [{"key": s.key, "name": s.name,
                    "mohs": round(s.mohs_approx, 1)} for s in STONES.values()],
        "laps": [{"key": l.key, "name": l.name} for l in LAPS.values()],
        "defaults": {
            field: getattr(Setup(), field) for field in sorted(_SETUP_FIELDS)
        },
        "targets": {
            field: getattr(Targets(), field) for field in sorted(_TARGET_FIELDS)
        },
    }


def cmd_simulate(payload: dict[str, Any]) -> dict[str, Any]:
    return describe_result(simulate(build_setup(payload)))


def cmd_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    from lapsim.recipes import (
        Recipe, constraint_margins, diagnose_infeasible, explain_infeasible,
        find_recipe,
    )

    setup = build_setup(payload)
    resolution = int(payload.get("resolution", 18))
    best = find_recipe(setup, prefer=payload.get("prefer", "robust"),
                       n_solids=resolution, n_glycerine=resolution)
    if best is None:
        remedies = diagnose_infeasible(setup, n_solids=12, n_glycerine=12)
        return {
            "found": False,
            "explanation": explain_infeasible(setup, remedies),
            "remedies": [
                {"change": change,
                 "recipe": describe_recipe(candidate.recipe),
                 "removal_um_min": round(candidate.removal_rate_um_min, 2),
                 "robustness": round(candidate.robustness, 4)}
                for change, candidate in remedies
            ],
        }

    spans = best.tolerances() if payload.get("tolerances", True) else {}
    return {
        "found": True,
        "recipe": describe_recipe(
            Recipe.from_setup(best.setup, float(payload.get("batch_ml", 100.0)))),
        "robustness": round(best.robustness, 4),
        "binding_constraint": best.binding_constraint,
        "margins": {k: round(v, 4) for k, v in constraint_margins(best.result).items()},
        "tolerances": {
            name: [round(low * 100, 1), round(high * 100, 1)]
            for name, (low, high) in spans.items()
        },
        "tolerance_text": best.describe_tolerances() if spans else "",
        "solids_mass_fraction": best.setup.solids_mass_fraction,
        "glycerine_mass_fraction": best.setup.glycerine_mass_fraction,
        "soap_volume_fraction": best.setup.soap_volume_fraction,
        "result": describe_result(simulate(best.setup)),
    }


def cmd_window(payload: dict[str, Any]) -> dict[str, Any]:
    """The feasibility map, as a grid the front end can draw itself."""
    from lapsim.recipes import scan_window

    setup = build_setup(payload)
    resolution = int(payload.get("resolution", 26))
    scan = scan_window(setup, n_solids=resolution, n_glycerine=resolution)
    return {
        "grit": scan.grit,
        "abrasive_name": scan.abrasive_name,
        "solids_wt_pct": [round(v * 100, 3) for v in scan.solids_mass_fractions],
        "glycerine_wt_pct": [round(v * 100, 3)
                             for v in scan.glycerine_mass_fractions],
        "robustness": [[round(v, 4) for v in row] for row in scan.robustness],
        "removal_um_min": [[round(v, 3) for v in row]
                           for row in scan.removal_rate_um_min],
        "Ra_um": [[round(v, 4) for v in row] for row in scan.Ra_um],
        "binding": [[str(v) for v in row] for row in scan.binding_constraint],
        "feasible_fraction": round(scan.feasible_fraction, 4),
    }


def cmd_plan(payload: dict[str, Any]) -> dict[str, Any]:
    from lapsim.recipes import plan_sequence

    setup = build_setup(payload)
    stages = payload.get("stages")
    if stages:
        stages = [(entry["abrasive"], entry["grit"]) for entry in stages]
    resolution = int(payload.get("resolution", 16))
    plan = plan_sequence(
        setup, stages=stages, prefer=payload.get("prefer", "robust"),
        initial_damage_um=float(payload.get("initial_damage_um", 120.0)),
        n_solids=resolution, n_glycerine=resolution,
    )
    return {
        "stages": [
            {
                "abrasive": stage.abrasive_key,
                "abrasive_name": stage.abrasive_name,
                "grit": stage.grit,
                "d50_um": stage.d50_um,
                "Ra_um": round(stage.Ra_um, 4),
                "subsurface_damage_um": round(stage.subsurface_damage_um, 3),
                "removal_um_min": round(stage.candidate.removal_rate_um_min, 2),
                "clear_um": round(stage.damage_to_remove_um, 2),
                "minutes": round(stage.minutes_required, 2),
                "in_stock": stage.in_stock,
                "recipe": describe_recipe(stage.candidate.recipe),
            }
            for stage in plan.stages
        ],
        "skipped": [
            {"abrasive_name": skip.abrasive_name, "grit": skip.grit,
             "explanation": skip.explanation}
            for skip in plan.skipped
        ],
        "total_minutes": round(plan.total_minutes, 2),
        "table": plan.table(),
    }


def cmd_figures(payload: dict[str, Any]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    from lapsim.plots import plot_all

    setup = build_setup(payload)
    paths = plot_all(setup, theme=payload.get("theme", "light"),
                     directory=payload.get("directory"),
                     animate=bool(payload.get("animate", False)))
    return {"paths": [str(path) for path in paths]}


COMMANDS = {
    "capabilities": cmd_capabilities,
    "simulate": cmd_simulate,
    "recipe": cmd_recipe,
    "window": cmd_window,
    "plan": cmd_plan,
    "figures": cmd_figures,
}


def handle(request: dict[str, Any]) -> dict[str, Any]:
    """Run one request, turning any failure into an error response."""
    command = request.get("command")
    handler = COMMANDS.get(command)
    if handler is None:
        return {"ok": False, "id": request.get("id"),
                "error": f"unknown command {command!r}; "
                         f"expected one of {', '.join(sorted(COMMANDS))}"}
    try:
        body = handler(request.get("payload") or {})
        return {"ok": True, "id": request.get("id"), "result": body}
    except Exception as error:  # noqa: BLE001 - the boundary must never raise
        return {
            "ok": False,
            "id": request.get("id"),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def serve(stream_in=None, stream_out=None) -> int:
    """Answer line-delimited JSON requests until standard input closes."""
    stream_in = stream_in or sys.stdin
    stream_out = stream_out or sys.stdout
    for line in stream_in:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            response = {"ok": False, "id": None, "error": f"bad JSON: {error}"}
        else:
            response = handle(request)
        stream_out.write(json.dumps(response) + "\n")
        stream_out.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--once":
        request = json.loads(argv[1]) if len(argv) > 1 else json.load(sys.stdin)
        print(json.dumps(handle(request), indent=2))
        return 0
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
