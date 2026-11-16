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


def run(ctx):
    settings = dict(ctx.settings)
    potcar = settings.pop("potcar", None)
    scratch = settings.pop("scratch", None)
    rerun = settings.pop("rerun", False)
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

    exit_code = ctx.command(command, cwd=ctx.vendor)
    if ctx.prepare and exit_code == 0:
        return {"status": "prepared", "result": str(manifest)}
    if exit_code != 0:
        message = f"Optical workflow exited with code {exit_code}; see the stage log."
        summary = native_output / "summary.csv"
        if not ctx.prepare and exit_code == 1 and summary.is_file():
            row = _summary_row(summary, ctx.material_id)
            if row and row.get("status") == "failed":
                return {"status": "failed", "result": str(summary),
                        "message": row.get("message") or message}
        return {"status": "blocked" if exit_code == 2 else "failed", "message": message}
    return _read_result(native_output, ctx.material_id)


def _summary_row(summary: Path, material_id: str) -> dict | None:
    with summary.open(encoding="utf-8", newline="") as source:
        return next((row for row in csv.DictReader(source) if row.get("material_id") == material_id), None)


def _read_result(native_output: Path, material_id: str) -> dict:
    row = _summary_row(native_output / "summary.csv", material_id)
    if row is None or row["status"] not in {"completed", "reused"}:
        raise ValueError(f"Optical workflow did not record a completed result for {material_id}")

    attempt = (native_output / row["attempt"]).resolve()
    spectrum = (native_output / row["spectrum"]).resolve()
    if not attempt.is_relative_to(native_output) or spectrum.parent != attempt or not spectrum.is_file():
        raise ValueError(f"Missing optical spectrum for {material_id}")

    state = json.loads((attempt / "run.json").read_text(encoding="utf-8"))
    results = state["results"]
    metrics = {}
    for name in METRICS:
        value = results.get(name)
        if type(value) in (int, float) and math.isfinite(value):
            metrics[name] = value
    warnings = results.get("warnings", [])
    message = " | ".join(warnings)
    return {"status": "review" if warnings else "completed", "result": str(spectrum),
            "metrics": metrics, "message": message}
