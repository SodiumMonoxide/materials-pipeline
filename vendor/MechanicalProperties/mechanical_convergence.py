"""Test k mesh and cutoff convergence of every raw elastic tensor component."""

from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np

from mech import VOIGT_ORDER


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _mesh_tuple(mesh):
    if not isinstance(mesh, (list, tuple)) or len(mesh) != 3:
        raise ValueError("mesh must contain three positive integers")
    return tuple(_positive_integer(value, "mesh component") for value in mesh)


def _run_ref(settings, result):
    mesh, encut = settings
    path = result.get("result_path")
    return {"mesh": list(mesh), "encut_eV": encut, "result_path": str(path) if path is not None else None}


def _validated_tensor(result, settings):
    mesh, encut = settings
    if not isinstance(result, Mapping) or result.get("status", "ok") != "ok":
        raise ValueError(f"evaluation did not return a successful result for mesh {mesh}, ENCUT {encut:g} eV")
    try:
        stiffness = np.asarray(result["elastic"]["C_raw_GPa"], dtype=float)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"missing or invalid elastic.C_raw_GPa for mesh {mesh}, ENCUT {encut:g} eV") from error
    if stiffness.shape != (6, 6) or not np.isfinite(stiffness).all():
        raise ValueError(f"elastic.C_raw_GPa must be finite and 6×6 for mesh {mesh}, ENCUT {encut:g} eV")
    inputs = result.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("evaluation inputs metadata must be a mapping")
    if "k_mesh" in inputs and _mesh_tuple(inputs["k_mesh"]) != mesh:
        raise ValueError(f"evaluation returned different k mesh than requested {mesh}")
    if "encut_eV" in inputs and inputs["encut_eV"] != encut:
        raise ValueError(f"evaluation returned different ENCUT than requested {encut:g} eV")
    if "voigt_order" in result and tuple(result["voigt_order"]) != VOIGT_ORDER:
        raise ValueError("evaluation returned a different Voigt order")
    return stiffness.copy()


def _comparison(base, probe, tolerance):
    reference, base_tensor, base_result = base
    probe_ref, probe_tensor, probe_result = probe
    if base_result.get("method") != probe_result.get("method"):
        raise ValueError(f"cannot compare different elastic calculation methods at {reference} and {probe_ref}")
    try:
        with np.errstate(over="raise"):
            delta = probe_tensor - base_tensor
    except FloatingPointError as error:
        raise ValueError(f"non-finite stiffness difference between {reference} and {probe_ref}") from error
    row, column = np.unravel_index(np.abs(delta).argmax(), delta.shape)
    max_delta = abs(float(delta[row, column]))
    return {
        "run": probe_ref,
        "passed": max_delta <= tolerance,
        "max_abs_delta_GPa": max_delta,
        "delta_GPa": delta.tolist(),
        "max_component": {"row": VOIGT_ORDER[row], "column": VOIGT_ORDER[column]},
    }


def converge_elastic(evaluate, mesh, encut, *, k_step=2, encut_step=80.0, tolerance_gpa=1.0, max_rounds=3):
    """Find a tested base meeting separate k mesh and ENCUT refinement criteria.

    `evaluate(k_mesh_tuple, encut_eV)` returns a JSON-ready run result containing
    elastic.C_raw_GPa in canonical Voigt order and optionally result_path. Each
    round tests both axes separately from the same base. Only failing axes advance.
    Exact setting pairs are evaluated once. This measures numerical sensitivity
    along the tested steps independently of physical elastic stability.
    """
    if not callable(evaluate):
        raise ValueError("evaluate must be callable")
    initial = (_mesh_tuple(mesh), _positive_number(encut, "encut"))
    k_step = _positive_integer(k_step, "k_step")
    encut_step = _positive_number(encut_step, "encut_step")
    tolerance = _positive_number(tolerance_gpa, "tolerance_gpa")
    rounds = _positive_integer(max_rounds, "max_rounds")
    cache = {}

    def run(settings):
        if settings not in cache:
            result = evaluate(*settings)
            tensor = _validated_tensor(result, settings)
            cache[settings] = (_run_ref(settings, result), tensor, result)
        return cache[settings]

    settings = initial
    history = []
    selected = None
    selected_result = None
    for number in range(1, rounds + 1):
        base_mesh, base_encut = settings
        cutoff_probe = base_encut + encut_step
        if not math.isfinite(cutoff_probe) or cutoff_probe <= base_encut:
            raise ValueError("encut_step must create a larger finite cutoff at each round")
        k_settings = (tuple(value + k_step for value in base_mesh), base_encut)
        encut_settings = (base_mesh, cutoff_probe)
        base = run(settings)
        if history:
            history[-1]["next_base"] = base[0]
        k_probe = _comparison(base, run(k_settings), tolerance)
        encut_probe = _comparison(base, run(encut_settings), tolerance)
        step = {"round": number, "base": base[0], "k_probe": k_probe,
                "encut_probe": encut_probe, "next_base": None}
        history.append(step)
        if k_probe["passed"] and encut_probe["passed"]:
            selected, selected_result = base[0], base[2]
            break
        if number < rounds:
            settings = (base_mesh if k_probe["passed"] else k_settings[0],
                        base_encut if encut_probe["passed"] else encut_settings[1])

    return {
        "status": "converged" if selected is not None else "max_rounds_reached",
        "criterion": {"metric": "maximum absolute difference of all 36 raw stiffness components (GPa)",
                      "matrix": "C_raw_GPa", "voigt_order": VOIGT_ORDER,
                      "tolerance_GPa": tolerance, "k_step": k_step, "encut_step_eV": encut_step,
                      "max_rounds": rounds},
        "initial": cache[initial][0],
        "selected": selected,
        "selected_result": selected_result,
        "last_base": history[-1]["base"],
        "history": history,
        "evaluations": [item[0] for item in cache.values()],
    }
