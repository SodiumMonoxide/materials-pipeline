#!/usr/bin/env python3
"""Run bulk relaxed-ion elasticity with VASP on Linux or WSL."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from ase.io import write

from mech import VOIGT_ORDER, analyze_outcar, write_report
from mechanical_convergence import converge_elastic
from mechanical_fit import deformed_cell, fit_stress_strain
from mechanical_inputs import normalize_incar, prepare_vasp_inputs, read_structure, write_incar
from mechanical_vasp_output import read_run


PRESETS = {
    "quick": {"encut_factor": 1.30, "k_spacing": 0.30, "ediff": 1e-6,
              "force_tol": 0.015, "stress_tol": 0.20},
    "standard": {"encut_factor": 1.30, "k_spacing": 0.22, "ediff": 1e-7,
                 "force_tol": 0.005, "stress_tol": 0.10},
    "tight": {"encut_factor": 1.50, "k_spacing": 0.16, "ediff": 1e-8,
              "force_tol": 0.001, "stress_tol": 0.05},
    "metal": {"encut_factor": 1.40, "k_spacing": 0.15, "ediff": 1e-7,
              "force_tol": 0.005, "stress_tol": 0.10, "ismear": 1, "sigma": 0.20},
}
DEFAULTS = {
    "preset": "standard", "method": "auto", "potcar_root": None, "potcars": {},
    "encut": None, "mesh": None, "format": None, "assume_relaxed": False,
    "ranks": 4, "kpar": 1, "ncore": 1, "threads": 1, "strain_workers": 1,
    "command": None, "strain": [0.005], "potim": 0.015, "nfree": 2,
    "ismear": 0, "sigma": 0.05, "relax_steps": 100, "relax_retries": 1, "incar": {},
}
CONVERGENCE_DEFAULTS = {
    "converge": False, "convergence_tol_gpa": 1.0, "convergence_max_rounds": 3,
    "convergence_k_step": 2, "convergence_encut_step": 80.0,
}
CONTROLLED_TAGS = {
    "ENCUT", "EDIFF", "EDIFFG", "IBRION", "ISIF", "NSW", "NFREE", "POTIM", "KSPACING",
    "KGAMMA", "KPAR", "NCORE", "NPAR", "PSTRESS", "ISMEAR", "SIGMA", "ISTART", "ICHARG",
    "LATTICE_CONSTRAINTS", "CHECKPOINT_FD", "IMAGES", "ICHAIN",
}


def normalize_settings(overrides):
    preset = overrides.get("preset", DEFAULTS["preset"])
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}")
    settings = {**DEFAULTS, **PRESETS[preset], **overrides}
    unknown = set(overrides) - DEFAULTS.keys() - PRESETS[preset].keys()
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
    if settings["method"] not in {"auto", "vasp", "stress-strain"}:
        raise ValueError("method must be auto, vasp, or stress-strain")
    if not isinstance(settings["assume_relaxed"], bool):
        raise ValueError("assume_relaxed must be a boolean")
    for key in ("ranks", "threads", "strain_workers", "ncore", "kpar", "relax_steps"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["ranks"] % (settings["kpar"] * settings["ncore"]):
        raise ValueError("ranks must be divisible by kpar × ncore")
    if type(settings["relax_retries"]) is not int or settings["relax_retries"] < 0:
        raise ValueError("relax_retries must be a nonnegative integer")
    if settings["nfree"] not in {2, 4} or type(settings["ismear"]) is not int:
        raise ValueError("nfree must be 2 or 4, and ismear must be an integer")
    for key in ("encut_factor", "k_spacing", "ediff", "force_tol", "stress_tol", "potim", "sigma", "encut"):
        value = settings[key]
        if key == "encut" and value is None:
            continue
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    mesh = settings["mesh"]
    if mesh is not None and (not isinstance(mesh, (list, tuple)) or len(mesh) != 3
                             or any(type(n) is not int or n < 1 for n in mesh)):
        raise ValueError("mesh must contain three positive integers")
    strain = settings["strain"]
    if not isinstance(strain, (list, tuple)) or not strain or any(
        type(amp) not in {int, float} or not 0 < amp <= 0.05 for amp in strain
    ):
        raise ValueError("strain magnitudes must be in (0, 0.05]")
    settings["strain"] = sorted(set(strain))
    if not isinstance(settings["incar"], dict) or not isinstance(settings["potcars"], dict):
        raise ValueError("incar and potcars must be mappings")
    settings["incar"] = normalize_incar(settings["incar"])
    reserved = CONTROLLED_TAGS.intersection(settings["incar"])
    if reserved:
        raise ValueError(f"use workflow settings for controlled INCAR tags: {', '.join(sorted(reserved))}")
    return settings


def needs_noncollinear_binary(settings):
    return any(str(settings["incar"].get(tag, "")).strip().strip(".").lower() in {"t", "true", "1"}
               for tag in ("LNONCOLLINEAR", "LSORBIT"))


def vasp_command(settings):
    command = settings["command"] or os.environ.get("VASP_COMMAND")
    if command:
        argv = shlex.split(command)
    else:
        variant = "ncl" if needs_noncollinear_binary(settings) else "std"
        launcher = os.environ.get("VASP_LAUNCHER")
        argv = [launcher, "{ranks}", variant] if launcher else ["mpirun", "-np", "{ranks}", f"vasp_{variant}"]
    executable = shutil.which(os.path.expanduser(argv[0])) if argv else None
    if not executable:
        raise ValueError("VASP launcher unavailable; set --command or VASP_COMMAND")
    argv[0] = os.path.abspath(executable)
    return [arg.replace("{ranks}", str(settings["ranks"])) for arg in argv]


@contextmanager
def material_lock(directory):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another workflow is using {directory}") from error
        yield


def immutable_manifest(path, data):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError(f"settings differ from {path}; choose a new output directory")
    else:
        write_report(path, data)


def stage_parameters(settings, inputs, kind):
    tags = {
        "ENCUT": inputs["encut_eV"], "PREC": "Accurate", "EDIFF": settings["ediff"],
        "LREAL": False, "LASPH": True, "ALGO": "Normal", "LWAVE": False, "LCHARG": False,
        "NELM": 100, "ISMEAR": settings["ismear"], "SIGMA": settings["sigma"],
        **settings["incar"], "KPAR": settings["kpar"], "NCORE": settings["ncore"],
    }
    if kind in {"relax", "strain"}:
        tags.update(IBRION=2, ISIF=3 if kind == "relax" else 2, NSW=settings["relax_steps"],
                    EDIFFG=-settings["force_tol"])
    elif kind == "elastic":
        tags.update(IBRION=6, ISIF=3, NSW=1, NCORE=1, POTIM=settings["potim"], NFREE=settings["nfree"])
    else:
        tags.update(IBRION=-1, ISIF=2, NSW=0)
    return tags


def prepare_stage(attempt, atoms, common, tags):
    attempt.mkdir(exist_ok=True)
    path = attempt / "POSCAR"
    buffer = io.StringIO()
    write(buffer, atoms, format="vasp", direct=True)
    content = buffer.getvalue()
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"prepared structure differs from {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    write_incar(attempt / "INCAR", tags)
    for name in ("POTCAR", "KPOINTS"):
        target = common / name
        link = attempt / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise ValueError(f"unexpected input at {link}")
        link.symlink_to(os.path.relpath(target, attempt))


def check_stage(attempt, kind, force_tol):
    outcar = attempt / "OUTCAR"
    state = read_run(outcar, ionic=kind in {"relax", "strain"})
    expected = {"relax": (2, 3), "reference": (-1, 2), "elastic": (6, 3), "strain": (2, 2)}[kind]
    if (state["ibrion"], state["isif"]) != expected:
        raise ValueError(f"{outcar}: unexpected IBRION or ISIF for {kind}")
    if not state["scf_all_converged"]:
        raise ValueError(f"{outcar}: electronic convergence {state['scf_converged']}/{state['scf_started']}")
    if kind in {"relax", "strain"}:
        max_force = state["max_force_eV_A"]
        if max_force is None or max_force > force_tol * 1.01:
            raise ValueError(f"{outcar}: final maximum force {max_force} exceeds {force_tol:g} eV/Å")
        if not (attempt / "CONTCAR").is_file() or not (attempt / "CONTCAR").stat().st_size:
            raise ValueError(f"{attempt}: missing relaxed CONTCAR")
    if kind == "elastic":
        state["elastic"] = analyze_outcar(outcar)
    return {**state, "directory": str(attempt)}


def run_stage(directory, atoms, common, tags, command, kind, settings):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "completed.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        attempt = directory / previous["attempt"]
        return {"directory": str(attempt)} if command is None else check_stage(attempt, kind, settings["force_tol"])

    attempts = sorted(directory.glob("attempt-*"))
    number = int(attempts[-1].name.split("-")[-1]) if attempts else 1
    if attempts and (attempts[-1] / "vasp.log").exists():
        number += 1
    attempt = directory / f"attempt-{number:04d}"
    prepare_stage(attempt, atoms, common, tags)
    if command is None:
        return {"directory": str(attempt)}

    env = {**os.environ, "OMP_NUM_THREADS": str(settings["threads"]), "OPENBLAS_NUM_THREADS": "1"}
    for layout in (tags, {**tags, "KPAR": 1}):
        log = attempt / "vasp.log"
        print(f"VASP {attempt}", file=sys.stderr, flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, cwd=attempt, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode == 0:
            result = check_stage(attempt, kind, settings["force_tol"])
            write_report(marker, {"attempt": attempt.name})
            return result
        if not (kind == "elastic" and layout["KPAR"] > 1 and
                "requested a change of the k-point set" in log.read_text(encoding="utf-8", errors="replace")):
            raise ValueError(f"VASP exited with {process.returncode}; see {log}")
        number += 1
        attempt = directory / f"attempt-{number:04d}"
        prepare_stage(attempt, atoms, common, {**tags, "KPAR": 1})
    raise ValueError(f"VASP did not complete at {attempt}")


def selected_method(settings, atoms):
    if settings["method"] != "auto":
        return settings["method"]
    symmetry_off = str(settings["incar"].get("ISYM", "")).strip() in {"0", "-1"}
    return "stress-strain" if len(atoms) > 12 or symmetry_off or needs_noncollinear_binary(settings) else "vasp"


def run_single_material(source, directory, overrides, *, prepare_only=False):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    settings = normalize_settings(overrides)
    potcar_root = settings["potcar_root"] or os.environ.get("VASP_POTCAR_ROOT")
    if not potcar_root:
        raise ValueError("Set VASP_POTCAR_ROOT or pass a PAW-PBE library path")
    command = None if prepare_only else vasp_command(settings)
    with material_lock(directory):
        if (directory / "convergence" / "baseline" / "workflow.json").exists():
            raise ValueError(f"{directory} contains a convergence run; choose a new output directory for a single calculation")
        request = {**settings, "source": str(source), "potcar_root": str(Path(potcar_root).expanduser().resolve())}
        immutable_manifest(directory / "workflow.json", json.loads(json.dumps(request)))
        common = directory / "inputs"
        inputs = prepare_vasp_inputs(
            source, common, potcar_root=request["potcar_root"], potcar_overrides=settings["potcars"],
            k_spacing=settings["k_spacing"], k_mesh=settings["mesh"], encut=settings["encut"],
            encut_factor=settings["encut_factor"], incar_overrides=settings["incar"],
            structure_format=settings["format"],
        )
        atoms = read_structure(common / "POSCAR", "vasp")
        if atoms.constraints:
            raise ValueError("relaxed-ion elasticity requires an input structure without selective dynamics constraints")
        if command is not None:
            immutable_manifest(directory / "execution.json", {"argv": command})
        method = selected_method(settings, atoms)
        stages = {}

        def calculate(name, geometry, kind):
            tags = stage_parameters(settings, inputs, kind)
            return run_stage(directory / name, geometry, common, tags, command, kind, settings)

        if prepare_only:
            first = "reference" if settings["assume_relaxed"] else "relax"
            stage = calculate(first, atoms, first)
            return {"status": "prepared", "method": method, "inputs": inputs, "first_stage": stage}

        retries = 0 if settings["assume_relaxed"] else settings["relax_retries"]
        for cycle in range(retries + 1):
            suffix = f"-{cycle}" if cycle else ""
            if not settings["assume_relaxed"]:
                name = "relax" + suffix
                stages[name] = calculate(name, atoms, "relax")
                atoms = read_structure(Path(stages[name]["directory"]) / "CONTCAR", "vasp")
            name = "reference" + suffix
            reference = stages[name] = calculate(name, atoms, "reference")
            residual = max(abs(value) for value in reference["stress_GPa"])
            if (residual <= settings["stress_tol"] and
                    reference["max_force_eV_A"] <= settings["force_tol"]):
                break
        else:
            raise ValueError(
                f"reference has {residual:.4g} GPa maximum residual stress and "
                f"{reference['max_force_eV_A']:.4g} eV/Å maximum force; see {reference['directory']}"
            )

        if method == "vasp":
            result = stages["elastic"] = calculate("elastic", atoms, "elastic")
            tensor = result["elastic"]
        else:
            deformations = [(component, sign * amp) for amp in settings["strain"]
                            for component in VOIGT_ORDER for sign in (-1, 1)]

            def strain_case(case):
                component, amplitude = case
                geometry = atoms.copy()
                geometry.set_cell(deformed_cell(atoms.cell.array, component, amplitude), scale_atoms=True)
                name = f"strain/{component}_{amplitude:+}"
                (directory / "strain").mkdir(exist_ok=True)
                state = calculate(name, geometry, "strain")
                return {**state, "component": component, "amplitude": amplitude}

            with ThreadPoolExecutor(max_workers=settings["strain_workers"]) as pool:
                deformed = list(pool.map(strain_case, deformations))
            stages["strain"] = deformed
            tensor = fit_stress_strain(deformed, reference["stress_GPa"])

        result = {
            "schema_version": 1, "status": "ok", "scope": "3D bulk, relaxed ion, near zero stress", "method": method,
            "structure": str(source), "formula": atoms.get_chemical_formula(), "atoms": len(atoms),
            "reference_cell_A": atoms.cell.array.tolist(),
            "stress_convention": "tensile positive", "voigt_order": VOIGT_ORDER,
            "inputs": inputs, "vasp_command": command, "stages": stages,
            "reference_max_abs_stress_GPa": residual, "elastic": tensor,
        }
        write_report(directory / "result.json", result)
        return result


def convergence_controls(overrides):
    controls = {key: overrides.pop(key, value) for key, value in CONVERGENCE_DEFAULTS.items()}
    if not isinstance(controls["converge"], bool):
        raise ValueError("converge must be a boolean")
    for key in ("convergence_k_step", "convergence_max_rounds"):
        if type(controls[key]) is not int or controls[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("convergence_tol_gpa", "convergence_encut_step"):
        value = controls[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    return controls


def run_convergence(source, directory, overrides, controls, *, prepare_only):
    directory = Path(directory).resolve()
    sweep = directory / "convergence"
    with material_lock(directory):
        if (directory / "workflow.json").exists():
            raise ValueError(f"{directory} contains a single calculation; choose a new output directory for convergence")

        baseline_dir = sweep / "baseline"
        baseline = run_single_material(source, baseline_dir, overrides, prepare_only=prepare_only)
        if prepare_only:
            return {**baseline, "convergence": {"status": "prepared", "controls": controls}}
        baseline_mesh = tuple(baseline["inputs"]["k_mesh"])
        baseline_encut = baseline["inputs"]["encut_eV"]
        baseline["result_path"] = (baseline_dir / "result.json").relative_to(directory).as_posix()
        results = {(baseline_mesh, baseline_encut): baseline}

        def evaluate(mesh, encut):
            key = (tuple(mesh), encut)
            if key not in results:
                mesh_label = "x".join(map(str, mesh))
                target = sweep / f"k-{mesh_label}_encut-{encut!r}"
                print(f"Elastic convergence: mesh {mesh_label}, ENCUT {encut:g} eV", file=sys.stderr, flush=True)
                result = run_single_material(source, target, {**overrides, "mesh": list(mesh), "encut": encut})
                for label in ("structure_sha256", "potcar_sha256"):
                    if result["inputs"][label] != baseline["inputs"][label]:
                        raise ValueError(f"{label} changed during convergence at {target}; use a new output directory")
                result["result_path"] = (target / "result.json").relative_to(directory).as_posix()
                results[key] = result
            return results[key]

        report = converge_elastic(
            evaluate, baseline_mesh, baseline_encut,
            k_step=controls["convergence_k_step"], encut_step=controls["convergence_encut_step"],
            tolerance_gpa=controls["convergence_tol_gpa"], max_rounds=controls["convergence_max_rounds"],
        )
        selected = report.pop("selected_result")
        if selected is None:
            last_base = report["last_base"]
            selected = evaluate(last_base["mesh"], last_base["encut_eV"])
        write_report(directory / "convergence.json", report)
        result = {
            **selected, "status": "ok" if report["status"] == "converged" else "not_converged",
            "convergence": {"status": report["status"], "rounds": len(report["history"]),
                            "report": "convergence.json"},
        }
        write_report(directory / "result.json", result)
        return result


def run_material(source, directory, overrides=None, *, prepare_only=False):
    """Prepare or run one bulk structure, optionally checking tensor convergence."""
    settings = dict(overrides or {})
    controls = convergence_controls(settings)
    if controls["converge"]:
        return run_convergence(source, directory, settings, controls, prepare_only=prepare_only)
    return run_single_material(source, directory, settings, prepare_only=prepare_only)


def key_values(values, token):
    label, separator, value = token.partition("=")
    if not separator or not label or not value:
        raise ValueError(f"expected NAME=VALUE: {token}")
    values[label] = value


def inputs_from_args(args):
    cli = {key: getattr(args, key) for key in {*DEFAULTS, *PRESETS["standard"], *CONVERGENCE_DEFAULTS}
           if getattr(args, key, None) is not None}
    for key in ("incar", "potcars"):
        values = {}
        for pair in cli.pop(key, []):
            key_values(values, pair)
        if values:
            cli[key] = values
    if args.manifest:
        if args.structures:
            raise ValueError("provide positional structures or --manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object with a materials list")
        defaults = manifest.get("defaults", {})
        tasks = manifest.get("materials", [])
        base = args.manifest.resolve().parent
    else:
        tasks = [{"structure": str(path)} for path in args.structures]
        defaults, base = {}, Path.cwd()
    if not isinstance(tasks, list) or not tasks or not isinstance(defaults, dict):
        raise ValueError("provide at least one material and a valid defaults mapping")
    jobs = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("structure"), str):
            raise ValueError("each material needs a structure path")
        layers = (defaults, cli, {k: v for k, v in task.items() if k not in {"id", "structure"}})
        settings = {}
        for layer in layers:
            for key, value in layer.items():
                if key in {"incar", "potcars"} and isinstance(value, dict):
                    settings[key] = {**settings.get(key, {}), **value}
                else:
                    settings[key] = value
        source = base / task["structure"]
        if "potcar_root" in settings and settings["potcar_root"]:
            potcar_base = Path.cwd() if "potcar_root" in cli and "potcar_root" not in task else base
            settings["potcar_root"] = str(potcar_base / Path(settings["potcar_root"]).expanduser())
        identity = task.get("id", re.sub(r"[^A-Za-z0-9_.-]+", "-", source.stem).strip(".-"))
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity):
            raise ValueError(f"invalid material id: {identity!r}")
        if identity in jobs:
            raise ValueError(f"duplicate material id: {identity}; assign unique ids in a manifest")
        jobs[identity] = (source, settings)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structures", nargs="*", type=Path, help="bulk POSCAR, CIF or other ASE structure")
    parser.add_argument("--manifest", type=Path, help="JSON with defaults and materials")
    parser.add_argument("--out", type=Path, required=True, help="output directory for this batch")
    parser.add_argument("--prepare-only", action="store_true", help="prepare shared inputs and first stage")
    parser.add_argument("--assume-relaxed", action="store_true", default=None, help="skip relaxation; verify stress and forces")
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--method", choices=("auto", "vasp", "stress-strain"))
    parser.add_argument("--potcar-root", type=Path)
    parser.add_argument("--potcar", dest="potcars", action="append", metavar="ELEMENT=DATASET")
    parser.add_argument("--incar", action="append", metavar="TAG=VALUE")
    parser.add_argument("--command", help="shell-free launcher string, accepts {ranks}")
    parser.add_argument("--format", help="ASE format for an ambiguous input filename")
    parser.add_argument("--mesh", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--strain", action="append", type=float, metavar="AMPLITUDE")
    parser.add_argument("--converge", action=argparse.BooleanOptionalAction, default=None,
                        help="probe k mesh and ENCUT convergence of the elastic tensor")
    parser.add_argument("--convergence-tol-gpa", type=float, help="maximum absolute change of any Cij; default 1 GPa")
    parser.add_argument("--convergence-max-rounds", type=int, help="maximum probe pairs; default 3")
    parser.add_argument("--convergence-k-step", type=int, help="add this many subdivisions on each axis; default 2")
    parser.add_argument("--convergence-encut-step", type=float, help="increase ENCUT by this many eV; default 80")
    parser.add_argument("--materials-workers", type=int, default=1)
    for key in ("encut", "encut_factor", "k_spacing", "ediff", "force_tol", "stress_tol", "potim", "sigma"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("ranks", "kpar", "ncore", "threads", "strain_workers", "relax_steps", "relax_retries", "ismear", "nfree"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args()
    try:
        jobs = inputs_from_args(args)
        if args.materials_workers < 1:
            raise ValueError("materials-workers must be positive")
        args.out.mkdir(parents=True, exist_ok=True)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    def material(job):
        identity, (source, settings) = job
        directory = args.out / identity
        try:
            result = run_material(source, directory, settings, prepare_only=args.prepare_only)
            tensor = result.get("elastic", {})
            item = {"id": identity, "structure": str(source), "status": result["status"], "method": result["method"],
                    "stability": tensor.get("stability"), "vrh": tensor.get("vrh"),
                    "max_asymmetry_GPa": tensor.get("max_asymmetry_GPa"),
                    "reference_max_abs_stress_GPa": result.get("reference_max_abs_stress_GPa"),
                    "result": f"{identity}/result.json" if not args.prepare_only else None}
            if "convergence" in result:
                item["convergence"] = result["convergence"].copy()
                if not args.prepare_only:
                    item["convergence"]["report"] = f"{identity}/convergence.json"
            return item
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            print(f"{identity}: {error}", file=sys.stderr)
            return {"id": identity, "structure": str(source), "status": "error", "error": str(error)}

    name = "prepared.json" if args.prepare_only else "batch_summary.json"
    try:
        with material_lock(args.out):
            with ThreadPoolExecutor(max_workers=args.materials_workers) as pool:
                results = list(pool.map(material, jobs.items()))
            errors = sum(item["status"] == "error" for item in results)
            batch = {"schema_version": 1, "errors": errors, "results": results}
            not_converged = sum(item["status"] == "not_converged" for item in results)
            if any("convergence" in item for item in results):
                batch["not_converged"] = not_converged
            write_report(args.out / name, batch)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(args.out.resolve() / name)
    return int(errors > 0 or not_converged > 0)


if __name__ == "__main__":
    sys.exit(main())
