"""Adapt a shared material record to the existing VASP Berry-phase workflow."""

import json
import math


_RESOURCE_KEYS = {"vasp", "paw_dir", "launcher", "ranks", "jobs", "threads"}
_GAP_BLOCKS = {"nonpositive_gap", "unsupported_occupations", "gap_below_threshold"}


def _result_metrics(row, summary):
    metrics = {
        "minimum_gap_ev": row["minimum_gap_ev"],
        "sampled_peak_above_first_ev_cell": row["maximum_energy_above_first_ev_cell"],
    }
    for field, label in (
        ("reference_to_polar_c_m2", "delta_p_reference_to_polar"),
        ("inversion_switch_delta_c_m2", "delta_p_inversion_switch"),
    ):
        vector = row[field]
        if vector is not None:
            for axis, value in zip("xyz", vector):
                metrics[f"{label}_{axis}_c_m2"] = value
            metrics[f"{label}_norm_c_m2"] = math.hypot(*vector)
    if row["inversion_switch_delta_c_m2"] is not None:
        metrics["sampled_peak_above_endpoint_mean_ev_cell"] = summary["energy"][
            "above_endpoint_mean_ev_cell"
        ]
    return metrics


def run(ctx):
    """Run the native stage and normalize its status without changing its checks."""
    options = dict(ctx.settings)
    misplaced = options.keys() & _RESOURCE_KEYS
    if misplaced:
        return {
            "status": "failed",
            "message": f"Set these resource options in the pipeline: {', '.join(sorted(misplaced))}",
        }
    if not options.get("reference"):
        return {
            "status": "blocked",
            "message": "Ferroelectric calculation requires a fixed-cell reference POSCAR.",
        }
    for key in ("reference", "polar", "opposite", "potcar"):
        if key in options:
            path = ctx.resolve(options[key])
            if not path.is_file():
                return {"status": "failed", "message": f"{key} file does not exist: {path}"}
            options[key] = str(path)
    options.setdefault("polar", str(ctx.structure.resolve()))

    directory = ctx.output.resolve()
    native = directory / "native"
    manifest = directory / "ferroelectric-manifest.json"
    action = "prepare" if ctx.prepare else "run"
    if not ctx.mpi and ctx.ranks > 1 and not ctx.prepare:
        return {"status": "blocked", "message": "MPI is required for polarization jobs with multiple ranks."}
    launcher = [str(ctx.mpi or "mpirun"), "-np", "{ranks}", "--bind-to", "none"]
    if not ctx.mpi and ctx.ranks == 1:
        launcher = []
    defaults = {"launcher": launcher, "ranks": ctx.ranks, "jobs": 1}
    if ctx.vasp:
        defaults["vasp"] = str(ctx.vasp)
    if ctx.paw_dir:
        defaults["paw_dir"] = str(ctx.paw_dir)
    config = {
        "output": str(native),
        "defaults": defaults,
        "cases": [{**options, "id": ctx.material_id}],
    }
    directory.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    exit_code = ctx.command(
        [
            ctx.python, "-B", str(ctx.vendor / "scripts" / "fe_workflow.py"), action,
            str(manifest),
        ]
    )
    batch = native / "batch-summary.json"
    if exit_code not in (0, 1) or not batch.is_file():
        return {"status": "failed", "message": f"VASP polarization driver exited with code {exit_code}."}
    reports = json.loads(batch.read_text(encoding="utf-8"))
    if reports["action"] != action or [row["case"] for row in reports["cases"]] != [ctx.material_id]:
        return {"status": "failed", "message": "Native polarization report does not match this calculation."}
    row = reports["cases"][0]
    state = row["status"]
    case_dir = native / ctx.material_id
    if exit_code == 0 and state == "prepared" and ctx.prepare:
        plan = case_dir / "plan.json"
        if plan.is_file():
            return {"status": "prepared", "result": str(plan)}
    if exit_code == 0 and state in ("passed", "review") and not ctx.prepare:
        result = case_dir / "summary.json"
        if result.is_file():
            summary = json.loads(result.read_text(encoding="utf-8"))
            response = {
                "status": "completed" if state == "passed" else "review",
                "result": str(result),
                "metrics": _result_metrics(row, summary),
            }
            if state == "review":
                response["message"] = " ".join(summary["warnings"])
            return response
    return {
        "status": "blocked" if row.get("code") in _GAP_BLOCKS else "failed",
        "result": str(batch),
        "message": row.get("message") or f"VASP polarization driver returned {state!r} (exit {exit_code}).",
    }
