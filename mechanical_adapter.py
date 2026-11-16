"""Adapt the relaxed-ion elasticity workflow and its optional convergence check."""

import json
import os
import shlex
import shutil
from pathlib import Path


def _launcher(ctx, settings):
    explicit = settings.get("command")
    if explicit is not None:
        if not isinstance(explicit, str) or not shlex.split(explicit):
            raise ValueError("mechanical.command must be a nonempty shell-free launcher string")
        return explicit
    if not ctx.vasp:
        return None  # Use the original workflow's VASP detection.

    executable = Path(ctx.vasp).expanduser()
    incar = settings.get("incar", {})
    noncollinear = isinstance(incar, dict) and any(
        str(value).strip().strip(".").lower() in {"true", "t", "1"}
        for tag, value in incar.items()
        if tag.upper() in {"LNONCOLLINEAR", "LSORBIT"}
    )
    if executable.name in {"vasp-cpu", "vasp-gpu"}:
        return shlex.join([str(executable), "{ranks}", "ncl" if noncollinear else "std"])
    if executable.name in {"vasp_std", "vasp_gam"} and noncollinear:
        raise ValueError("SOC/noncollinear settings require vasp_ncl or an explicit mechanical.command")
    if not ctx.mpi:
        raise ValueError("configure an MPI launcher or specify mechanical.command")
    return shlex.join([str(ctx.mpi), "-np", "{ranks}", str(executable)])


def _report(path, material_id):
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload["results"]
    if len(reports) != 1 or reports[0]["id"] != material_id:
        raise ValueError(f"{path} does not describe {material_id}")
    return reports[0]


def run(ctx):
    runner = ctx.vendor / "elastic_workflow.py"
    if not runner.is_file():
        return {"status": "blocked", "message": f"Mechanical workflow missing: {runner}"}
    settings = dict(ctx.settings)
    # Rank and thread limits apply at pipeline level, with a single strain active per stage.
    settings.update(ranks=ctx.ranks, threads=ctx.threads, strain_workers=1)
    potcar_root = settings.get("potcar_root")
    if potcar_root is not None:
        potcar_root = ctx.resolve(potcar_root)
    elif not (potcar_root := ctx.paw_dir or os.environ.get("VASP_POTCAR_ROOT")):
        return {"status": "blocked", "message": "Configure runtime.paw_dir or VASP_POTCAR_ROOT."}
    potcar_root = Path(potcar_root).expanduser().resolve()
    if not potcar_root.is_dir():
        return {"status": "blocked", "message": f"VASP POTCAR directory missing: {potcar_root}"}
    settings["potcar_root"] = str(potcar_root)
    command = _launcher(ctx, settings)
    if command is not None:
        settings["command"] = command
    if not ctx.prepare and ctx.vasp and not ctx.settings.get("command"):
        if not shutil.which(str(Path(ctx.vasp).expanduser())):
            return {"status": "blocked", "message": f"VASP executable or launcher unavailable: {ctx.vasp}"}

    native = ctx.output / "native"
    manifest = ctx.output / "mechanical-manifest.json"
    manifest.write_text(
        json.dumps(
            {"defaults": settings, "materials": [{"id": ctx.material_id, "structure": str(ctx.structure.resolve())}]},
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    args = [ctx.python, str(runner), "--manifest", str(manifest), "--out", str(native)]
    if ctx.prepare:
        args.append("--prepare-only")
    rc = ctx.command(args, cwd=ctx.output)
    native_report = native / ("prepared.json" if ctx.prepare else "batch_summary.json")
    if not native_report.is_file():
        return {"status": "failed", "message": f"Mechanical workflow exited with {rc}; see stage log"}
    entry = _report(native_report, ctx.material_id)
    native_status = entry["status"]
    valid_statuses = {"prepared"} if ctx.prepare else {"ok", "not_converged"}
    not_converged = native_status == "not_converged"
    if native_status not in valid_statuses or rc != int(not_converged):
        return {"status": "failed", "result": str(native_report),
                "message": entry.get("error") or f"Mechanical workflow exited with {rc}; see stage log"}
    if ctx.prepare:
        return {"status": "prepared", "result": str(native_report)}

    result_path = native / entry["result"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    elastic = result["elastic"]
    references = [data for name, data in result["stages"].items() if name.startswith("reference")]
    structure = Path(references[-1]["directory"]) / "POSCAR"
    if not structure.is_file():
        raise ValueError(f"validated elastic reference structure missing: {structure}")
    stability = elastic["stability"]
    diagnostics = []
    if not_converged:
        diagnostics.append("Elastic k mesh and cutoff convergence was not reached")
    if stability != "positive_definite":
        diagnostics.append(f"Elastic stiffness requires review ({stability})")
    status = "review" if diagnostics else "completed"
    if not_converged:
        status = "not_converged"
    outcome = {
        "status": status,
        "result": str(result_path),
        "structure": str(structure),
        "metrics": {
            "reference_max_abs_stress_GPa": result["reference_max_abs_stress_GPa"],
            "elastic_min_eigenvalue_GPa": elastic["eigenvalues_GPa"][0],
            "elastic_max_asymmetry_GPa": elastic["max_asymmetry_GPa"],
            **(elastic["vrh"] or {}),
        },
    }
    if convergence := entry.get("convergence"):
        outcome["convergence_report"] = str(native / convergence["report"])
        outcome["convergence_status"] = convergence["status"]
        outcome["metrics"]["elastic_convergence_rounds"] = convergence["rounds"]
    if diagnostics:
        outcome["message"] = ". ".join(diagnostics)
    return outcome
