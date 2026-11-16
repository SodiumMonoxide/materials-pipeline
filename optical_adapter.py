"""Run the existing VASP optical workflow as one pipeline stage."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


METRICS = (
    "n_energy",
    "energy_min_eV",
    "energy_max_eV",
    "sampled_ks_gap_eV",
    "min_empty_transition_ceiling_eV",
    "negative_eps2_diagonal_rows",
    "elapsed_seconds",
)

CONVERGENCE_FLOATS = {"rtol": "--conv-rtol", "atol": "--conv-atol",
                      "k_growth": "--conv-k-growth", "band_growth": "--conv-band-growth"}
CONVERGENCE_INTS = {"k_steps": "--conv-k-steps", "band_steps": "--conv-band-steps"}


def merge_settings(defaults: dict, overrides: dict) -> dict:
    """Apply the original optical manifest's defaults and override rules."""
    result = {}
    for layer in (defaults, overrides):
        if "kmesh" in layer and "kspacing" in layer:
            raise ValueError("Use either kmesh or kspacing in an optical settings object")
        if "kmesh" in layer:
            result.pop("kspacing", None)
        if "kspacing" in layer:
            result.pop("kmesh", None)
        if layer.get("metal") is False and "drude_gamma" not in layer:
            result.pop("drude_gamma", None)
        for key, value in layer.items():
            if key in {"incar", "potentials"}:
                if not isinstance(value, dict):
                    raise ValueError(f"optical.{key} must be a JSON object")
                if key == "incar":
                    value = {tag.upper(): setting for tag, setting in value.items()}
                result[key] = {**result.get(key, {}), **value}
            else:
                result[key] = value
    if result.get("component") == "tensor":
        result["component"] = None
    return result


def _numeric_args(key, values, count, types):
    if not isinstance(values, list) or len(values) != count or any(type(value) not in types for value in values):
        raise ValueError(f"optical.convergence.{key} expects {count} numeric value(s) of the required type")
    return [str(value) for value in values]


def convergence_arguments(config, material_id):
    """Map the opt-in manifest setting to the native convergence CLI."""
    if config is False:
        return []
    arguments = ["--converge", material_id]
    if config is True:
        return arguments
    if not isinstance(config, dict):
        raise ValueError("optical.convergence must be a boolean or an options object")
    unexpected = config.keys() - (CONVERGENCE_FLOATS.keys() | CONVERGENCE_INTS.keys() | {"window_eV", "kmeshes"})
    if unexpected:
        raise ValueError(f"Unknown optical.convergence settings: {', '.join(sorted(unexpected))}")
    for key, flag in CONVERGENCE_FLOATS.items():
        if key in config:
            arguments.extend((flag, *_numeric_args(key, [config[key]], 1, (int, float))))
    for key, flag in CONVERGENCE_INTS.items():
        if key in config:
            arguments.extend((flag, *_numeric_args(key, [config[key]], 1, (int,))))
    if "window_eV" in config:
        arguments.extend(("--conv-window", *_numeric_args("window_eV", config["window_eV"], 2, (int, float))))
    if "kmeshes" in config:
        if not isinstance(config["kmeshes"], list):
            raise ValueError("optical.convergence.kmeshes must be a list of k meshes")
        for mesh in config["kmeshes"]:
            arguments.extend(("--conv-kmesh", *_numeric_args("kmeshes", mesh, 3, (int,))))
    return arguments


def run(ctx):
    settings = dict(ctx.settings)
    potcar = settings.pop("potcar", None)
    scratch = settings.pop("scratch", None)
    rerun = settings.pop("rerun", False)
    convergence = convergence_arguments(settings.pop("convergence", False), ctx.material_id)
    if not isinstance(rerun, bool):
        raise ValueError("optical.rerun must be a boolean")
    job = {
        "id": ctx.material_id,
        "poscar": str(ctx.structure.resolve()),
        "settings": settings,
    }
    if potcar is not None:
        job["potcar"] = str(ctx.resolve(potcar))

    directory = ctx.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "optical-manifest.json"
    manifest.write_text(
        json.dumps({"materials": [job]}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    native_output = directory / "vasp"
    command = [
        ctx.python, "-B", "-m", "optical_workflow", "--manifest", str(manifest),
        "--out", str(native_output), "--ranks", str(ctx.ranks), "--jobs", "1",
    ]
    for flag, value in (("--potcar-root", ctx.paw_dir), ("--vasp", ctx.vasp), ("--mpi", ctx.mpi)):
        if value:
            command.extend([flag, str(value)])
    if scratch is not None:
        command.extend(["--scratch", str(ctx.resolve(scratch))])
    if rerun:
        command.append("--rerun")
    if ctx.prepare:
        command.append("--dry-run")
    command.extend(convergence)

    exit_code = ctx.command(command, cwd=ctx.vendor)
    if ctx.prepare and exit_code == 0:
        return {"status": "prepared", "result": str(manifest)}
    message = f"Optical workflow exited with code {exit_code}; see the stage log."
    summary = native_output / "summary.csv"
    if not ctx.prepare and exit_code in {0, 1} and summary.is_file():
        row = _summary_row(summary, ctx.material_id)
        if row and (exit_code == 0 or row.get("status") == "failed" or (
            convergence and row.get("convergence_status") in {"failed", "not_converged"}
        )):
            return _read_result(native_output, ctx.material_id, row, bool(convergence))
    return {"status": "blocked" if exit_code == 2 else "failed", "message": message}


def _summary_row(summary: Path, material_id: str) -> dict | None:
    with summary.open(encoding="utf-8", newline="") as source:
        return next((row for row in csv.DictReader(source) if row.get("material_id") == material_id), None)


def _read_result(native_output: Path, material_id: str, row: dict, convergence: bool) -> dict:
    directory = native_output / material_id
    convergence_status = row.get("convergence_status") if convergence else None
    convergence_message = row.get("convergence_message", "") if convergence else ""
    result = {}
    if convergence:
        if convergence_status not in {"converged", "not_converged", "failed"}:
            raise ValueError(f"Missing convergence outcome for {material_id}")
        result["convergence_status"] = convergence_status
        if report_path := row.get("convergence_report"):
            report = (native_output / report_path).resolve()
            if report.parent != directory or not report.is_file():
                raise ValueError(f"Missing convergence report for {material_id}")
            result["convergence_report"] = str(report)
        if not convergence_message:
            convergence_message = f"Optical convergence: {convergence_status}"
    if row["status"] == "failed":
        return {**result, "status": "failed", "result": str(native_output / "summary.csv"),
                "message": row.get("message") or convergence_message or "Optical screening failed"}
    if row["status"] not in {"completed", "reused"}:
        raise ValueError(f"Optical workflow did not record a completed result for {material_id}")
    final_attempt = row.get("convergence_attempt") if convergence else None
    if convergence_status == "converged" and not final_attempt:
        raise ValueError(f"Missing final convergence attempt for {material_id}")
    attempt = (native_output / (final_attempt or row["attempt"])).resolve()
    if attempt.parent != directory:
        raise ValueError(f"Optical attempt is outside its material directory for {material_id}")
    state = json.loads((attempt / "run.json").read_text(encoding="utf-8"))
    results = state["results"]
    spectrum = (attempt / results["spectrum"]).resolve()
    if spectrum.parent != attempt or not spectrum.is_file():
        raise ValueError(f"Missing optical spectrum for {material_id}")

    metrics = {}
    for name in METRICS:
        value = results.get(name)
        if type(value) in (int, float) and math.isfinite(value):
            metrics[name] = value
    warnings = results.get("warnings", [])
    status = "review" if warnings else "completed"
    if convergence_status in {"failed", "not_converged"}:
        status = convergence_status
    message = " | ".join(filter(None, (convergence_message, *warnings)))
    return {**result, "status": status, "result": str(spectrum), "metrics": metrics, "message": message}
