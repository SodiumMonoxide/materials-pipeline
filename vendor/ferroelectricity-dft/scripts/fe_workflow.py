#!/usr/bin/env python3
"""Run fixed-cell ferroelectric polarization batches in VASP on WSL/Linux."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

from fe_analysis import analyze, read_outputs
from fe_structures import build_path
from fe_vasp_jobs import load_potentials, run_jobs, verify_results
from polarization_path import dual_basis
from vasp_collect_batio3 import read_manifest


PRESETS = {
    "fast": {"images": 5, "kspacing": 0.40, "encut_factor": 1.20, "ediff": 1e-7},
    "standard": {"images": 9, "kspacing": 0.28, "encut_factor": 1.30, "ediff": 1e-8},
}
DEFAULTS = {
    "preset": "fast", "sigma": 0.03, "nelm": 140, "dipol": [0.25, 0.25, 0.25],
    "ranks": 4, "jobs": 2, "min_gap": 0.05, "occ_tol": 0.001,
    "max_displacement": 1.5, "symprec": 0.01, "quantum_multiple": 1,
    "mp_symprec": 0.03, "mp_max_strain": 0.08,
    "mirror": False, "ordered": False, "probe": True,
    "launcher": ["mpirun", "-np", "{ranks}", "--bind-to", "none"],
}
OPTIONS = DEFAULTS.keys() | PRESETS["fast"].keys() | {
    "vasp", "paw_dir", "potcar", "variants", "encut", "kmesh", "kpar", "mp_cache", "optics",
}
CASE_FIELDS = OPTIONS | {
    "id", "reference", "polar", "opposite", "formula", "material_id", "polar_id",
    "reference_id", "selection",
}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def atomic_json(path, value):
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as handle:
            pending = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def fields(value, expected, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"{label}: unknown keys: {', '.join(sorted(unknown))}")
    return value


def path_from(value, base, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    expanded = os.path.expandvars(os.path.expanduser(value))
    return (base / expanded).resolve()


def scalar(value, key, minimum=0, *, integer=False, allow_minimum=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (integer and not isinstance(value, int))
            or value < minimum or (value == minimum and not allow_minimum)):
        kind = "integer" if integer else "number"
        operator = "at least" if allow_minimum else "above"
        raise ValueError(f"{key} must be a finite {kind} {operator} {minimum}")
    return value


def case_options(defaults, case, overrides, base):
    configured = {**defaults, **case, **overrides}
    preset = configured.get("preset", "fast")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("preset must be fast or standard")
    values = {**DEFAULTS, **PRESETS[preset], **configured}
    values["infer_mirror"] = "mirror" not in configured
    if ("kspacing" in overrides
            or ("kspacing" in case and "kmesh" not in case and "kmesh" not in overrides)):
        values.pop("kmesh", None)
    for key in ("mirror", "ordered", "probe"):
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("images", "nelm", "ranks", "jobs", "quantum_multiple", "kpar"):
        if key in values:
            scalar(values[key], key, integer=True)
    for key in ("kspacing", "ediff", "sigma", "encut_factor", "encut", "symprec",
                "max_displacement", "mp_symprec", "mp_max_strain"):
        if key in values:
            scalar(values[key], key)
    for key in ("min_gap", "occ_tol"):
        scalar(values[key], key, allow_minimum=True)
    if values["occ_tol"] >= 0.5:
        raise ValueError("occ_tol must be below 0.5")
    if "kmesh" in values:
        mesh = values["kmesh"]
        if not isinstance(mesh, list) or len(mesh) != 3:
            raise ValueError("kmesh must be three positive integers")
        for k in mesh:
            scalar(k, "kmesh entries", integer=True)
    dipol = values["dipol"]
    if not isinstance(dipol, list) or len(dipol) != 3:
        raise ValueError("dipol must be three fractional coordinates")
    for x in dipol:
        scalar(x, "dipol coordinates", allow_minimum=True)
        if x >= 1:
            raise ValueError("dipol coordinates must be below 1")
    launcher = values["launcher"]
    if not isinstance(launcher, list) or any(not isinstance(token, str) or not token for token in launcher):
        raise ValueError("launcher must be an array of argument strings")
    if not launcher and values["ranks"] != 1:
        raise ValueError("direct launch requires ranks=1")
    if values["mp_max_strain"] >= 1:
        raise ValueError("mp_max_strain is a fractional strain and must be below 1")
    for key in ("reference", "polar", "opposite", "potcar", "mp_cache"):
        if key in values:
            values[key] = path_from(values[key], base, key)
    vasp = values.get("vasp", os.environ.get("VASP_EXECUTABLE") or os.environ.get("VASP_EXE") or "vasp_std")
    if not isinstance(vasp, str) or not vasp:
        raise ValueError("vasp must be a command name or path")
    values["vasp"] = str(path_from(vasp, base, "vasp")) if "/" in vasp or "$" in vasp else vasp
    paw = values.get("paw_dir", os.environ.get("VASP_POTCAR_ROOT") or os.environ.get("VASP_PSP_DIR"))
    values["paw_dir"] = path_from(paw, base, "paw_dir") if paw else None
    for source in (defaults.get("variants", {}), case.get("variants", {})):
        if not isinstance(source, dict) or any(not isinstance(v, str) or not v for v in source.values()):
            raise ValueError("variants must map element symbols to PAW subdirectory names")
    values["variants"] = {**defaults.get("variants", {}), **case.get("variants", {})}
    structure_query = set(values) & {"formula", "material_id"}
    mp_fields = {"polar_id", "reference_id", "selection"}
    for key in structure_query | (mp_fields & values.keys()):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
        values[key] = values[key].strip()
    if len(structure_query) > 1 or (structure_query and {"reference", "polar", "opposite"} & values.keys()):
        raise ValueError("use one formula or MP ID, or supply the local structure files")
    if not structure_query and mp_fields & values.keys():
        raise ValueError("MP pair selection requires a formula or material_id")
    return values


def kpoint_mesh(lattice, values):
    if "kmesh" in values:
        return values["kmesh"]
    spacing = values["kspacing"]
    return [max(2, math.ceil(2 * math.pi * math.hypot(*axis) / spacing))
            for axis in dual_basis(lattice)]


def prepare(case_root, values):
    for key in ("reference", "polar"):
        if key not in values:
            raise ValueError(f"missing {key} POSCAR")
    lattice, species, frames, geometry = build_path(
        values["reference"], values["polar"], values.get("opposite"), values["images"],
        values["mirror"], values["ordered"], values["max_displacement"], values["symprec"],
    )
    potcar = values.get("potcar")
    variants = {element: variant for element, variant in values["variants"].items() if element in species}
    potentials, enmax, potential_info = load_potentials(
        species, None if potcar is not None else values["paw_dir"],
        {} if potcar is not None else variants, potcar,
    )
    encut = values.get("encut", math.ceil(enmax * values["encut_factor"] / 10) * 10)
    if encut < enmax:
        raise ValueError(f"ENCUT={encut:g} is below the largest PAW ENMAX={enmax:g} eV")
    mesh = kpoint_mesh(lattice, values)
    ranks = values["ranks"]
    kpar = values.get("kpar", next(k for k in (4, 3, 2, 1) if ranks % k == 0 and math.prod(mesh) % k == 0))
    if ranks % kpar or kpar > math.prod(mesh):
        raise ValueError("KPAR must divide ranks and cannot exceed the k-point count")
    incar = (
        f"SYSTEM = {case_root.name} fixed-cell polarization\n"
        f"ENCUT = {encut:g}\nGGA = PE\nPREC = Accurate\n"
        f"EDIFF = {values['ediff']:g}\nNELM = {values['nelm']}\n"
        f"KPAR = {kpar}\nISPIN = 1\nISMEAR = 0\nSIGMA = {values['sigma']:g}\n"
        "ALGO = Normal\nLREAL = .FALSE.\nLASPH = .TRUE.\n"
        "ISYM = -1\nIBRION = -1\nNSW = 0\n"
        "LWAVE = .FALSE.\nLCHARG = .FALSE.\nLCALCPOL = .TRUE.\n"
        f"DIPOL = {' '.join(map(str, values['dipol']))}\n"
    )
    kpoints = f"Gamma-centered mesh\n0\nGamma\n{' '.join(map(str, mesh))}\n0 0 0\n"
    return install_plan(case_root, lattice, geometry, potential_info, encut, mesh, kpar,
                        frames, incar, kpoints, values.get("structure_source")), potentials


def install_plan(root, lattice, geometry, potential_info, encut, mesh, kpar, frames, incar, kpoints,
                 structure_source):
    inputs = {}
    manifest = io.StringIO(newline="")
    writer = csv.writer(manifest, lineterminator="\n")
    writer.writerow(("image", "lambda"))
    names = []
    for index, (coordinate, poscar) in enumerate(frames):
        name = f"{index:03d}"
        names.append(name)
        writer.writerow((name, f"{coordinate:.15g}"))
        for filename, body in (("POSCAR", poscar), ("INCAR", incar), ("KPOINTS", kpoints)):
            inputs[f"{name}/{filename}"] = body.encode("utf-8")
    inputs["images.csv"] = manifest.getvalue().encode("ascii")
    plan = {
        "schema": 1, "lattice_angstrom": lattice, "geometry": geometry, "potentials": potential_info,
        "encut_ev": encut, "kmesh": mesh, "kpar": kpar, "images": names,
        "input_sha256": {name: digest(blob) for name, blob in inputs.items()},
    }
    if structure_source is not None:
        plan["structure_source"] = structure_source
    plan = json.loads(json.dumps(plan, allow_nan=False))
    if root.exists():
        saved = read_plan(root)
        if saved != plan:
            raise ValueError(f"{root}: inputs or settings changed. Use a new case ID or output directory")
        return plan
    with tempfile.TemporaryDirectory(prefix=".fe-prepare-", dir=root.parent) as directory:
        staged = Path(directory) / "case"
        staged.mkdir()
        for name, blob in inputs.items():
            destination = staged / name
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(blob)
        atomic_json(staged / "plan.json", plan)
        staged.rename(root)
    return plan


def read_plan(root):
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("schema") != 1:
        raise ValueError("unsupported plan schema")
    for name, expected in plan["input_sha256"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"{root / name}: prepared input was modified")
    return plan


def set_status(root, state, **detail):
    status = {"case": root.name, "status": state, **detail}
    atomic_json(root / "case_status.json", status)
    return status


def query_case(query, kind=None, identifier=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("material query must be a nonempty string")
    query = query.strip()
    if kind is None:
        kind = "material_id" if re.fullmatch(r"(?:mp|mvc)-[A-Za-z0-9]+", query, re.I) else "formula"
    label = query.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-.") or "material"
    return {"id": identifier or name, kind: query}


def run_case(action, root, values, *, choose=False, refresh=False):
    if action != "analyze" and ("formula" in values or "material_id" in values):
        from fe_mp_resolve import resolve

        by_id = "material_id" in values
        database = root.parent / "_mp"
        try:
            resolution = resolve(
                database / root.name, values.get("mp_cache", database / "cache"),
                query=values["material_id" if by_id else "formula"], by_id=by_id,
                symprec=values["mp_symprec"], max_strain=values["mp_max_strain"],
                max_displacement=values["max_displacement"], polar_id=values.get("polar_id"),
                reference_id=values.get("reference_id"), selector=values.get("selection"),
                choose=choose, refresh=refresh,
            )
        except (OSError, ValueError, RuntimeError, ArithmeticError) as error:
            return {"case": root.name, "status": "structure_error", "message": str(error)}
        result = {key: val for key, val in resolution.items()
                  if key not in {"reference", "polar", "mirror"}}
        if action == "discover" or resolution["status"] != "resolved":
            return {"case": root.name, **result}
        for key in ("reference", "polar", "structure_source"):
            values[key] = resolution[key]
        if values["infer_mirror"]:
            values["mirror"] = resolution["mirror"]
    elif action == "discover":
        return {"case": root.name, "status": "manual_structures"}

    try:
        if action == "analyze":
            plan = read_plan(root)
        else:
            plan, potentials = prepare(root, values)
    except (OSError, ValueError, KeyError, ArithmeticError) as error:
        return {"case": root.name, "status": "input_error", "message": str(error)}
    source = {"structure_source": plan["structure_source"]} if "structure_source" in plan else {}
    if action == "prepare":
        return {"case": root.name, "status": "prepared", "images": len(plan["images"]),
                "encut_ev": plan["encut_ev"], "kmesh": plan["kmesh"], **source}

    (root / "summary.json").unlink(missing_ok=True)
    set_status(root, "running" if action == "run" else "analyzing")
    stage = "solver"
    try:
        if action == "run":
            names = plan["images"]
            def launch(subset):
                run_jobs(root, subset, potentials, values["vasp"], values["launcher"],
                         values["ranks"], values["jobs"])

            if values["probe"]:
                indices = [0, len(names) - 1]
                if plan["geometry"]["path_kind"] != "half":
                    indices.insert(1, len(names) // 2)
                probes = [names[i] for i in indices]
                launch(probes)
                stage = "probe"
                probe_manifest = [(name, coordinate) for name, coordinate in read_manifest(root / "images.csv")
                                  if name in probes]
                read_outputs(root, probe_manifest, plan["lattice_angstrom"], values["min_gap"],
                             values["occ_tol"])
                stage = "solver"
                pending = [name for name in names if name not in probes]
                if pending:
                    launch(pending)
            else:
                launch(names)
        stage = "analysis"
        verify_results(root, plan["images"])
        geometry = plan["geometry"]
        result = analyze(root, plan["lattice_angstrom"], min_gap=values["min_gap"],
                         occ_tol=values["occ_tol"], quantum_multiple=values["quantum_multiple"],
                         centrosymmetric=geometry["reference_centrosymmetric"],
                         inversion_pair=bool(geometry["endpoints_share_reference_inversion"]))
        polarization = result["polarization"]
        reference = polarization["reference"]
        return set_status(
            root, result["status"], summary="summary.json",
            minimum_gap_ev=result["gap"]["minimum_ev"],
            reference_to_polar_c_m2=reference["last_minus_ref_c_m2"] if reference else None,
            inversion_switch_delta_c_m2=polarization["inversion_switch_delta_c_m2"],
            maximum_energy_above_first_ev_cell=result["energy"]["above_first_ev_cell"],
            warnings=len(result["warnings"]),
            **source,
        )
    except (OSError, ValueError, RuntimeError, ArithmeticError) as error:
        return set_status(root, f"{stage}_error", message=str(error),
                          code=getattr(error, "code", None))


def main():
    if sys.argv[1:2] == ["optics"]:
        from fe_optics import main as optical_main

        return optical_main(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__, epilog="For optical convergence: %(prog)s optics --help")
    parser.add_argument("action", choices=("discover", "prepare", "run", "analyze"))
    parser.add_argument("config", type=Path, nargs="?")
    query = parser.add_mutually_exclusive_group()
    query.add_argument("--formula", help="resolve an MP chemical formula directly")
    query.add_argument("--material-id", help="resolve an MP material ID directly")
    parser.add_argument("--id", help="case directory name for a direct query")
    parser.add_argument("--select", dest="selection", help="select a candidate key in a single case")
    parser.add_argument("--polar-id", help="pin the polar MP ID in a single case")
    parser.add_argument("--reference-id", help="pin the reference MP ID in a single case")
    parser.add_argument("--choose", action="store_true", help="prompt in a terminal when several pairs match")
    parser.add_argument("--refresh-mp", action="store_true", help="replace the cached MP query response")
    parser.add_argument("--mp-cache", type=Path, help="shared MP cache directory or local response JSON")
    parser.add_argument("--case", action="append", dest="selected", help="run selected IDs, repeatable")
    parser.add_argument("--output", type=Path, help="override the batch output directory")
    parser.add_argument("--preset", choices=PRESETS)
    mesh_group = parser.add_mutually_exclusive_group()
    mesh_group.add_argument("--kmesh", type=int, nargs=3)
    mesh_group.add_argument("--kspacing", type=float)
    for key in ("images", "ranks", "jobs", "kpar"):
        parser.add_argument(f"--{key}", type=int)
    for key in ("encut", "min_gap"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=float)
    args = parser.parse_args()
    try:
        config = {}
        if args.config:
            config = fields(json.loads(args.config.read_text(encoding="utf-8-sig")),
                            {"output", "defaults", "cases"}, "config")
        defaults = fields(config.get("defaults", {}), OPTIONS, "defaults")
        cases = config.get("cases")
        if args.formula is not None or args.material_id is not None:
            direct = args.material_id if args.material_id is not None else args.formula
            kind = "material_id" if args.material_id is not None else "formula"
            cases = [query_case(direct, kind, args.id)]
        elif args.id:
            raise ValueError("--id is only used with a direct formula or MP ID query")
        if not isinstance(cases, list) or not cases:
            raise ValueError("supply a nonempty case configuration or a direct formula/MP ID query")
        ids, parsed_cases = set(), []
        for entry in cases:
            case = query_case(entry) if isinstance(entry, str) else fields(entry, CASE_FIELDS, "case")
            if "id" not in case:
                query_fields = set(case) & {"material_id", "formula"}
                if len(query_fields) == 1:
                    kind = query_fields.pop()
                    case = {**query_case(case[kind], kind), **case}
            ident = case.get("id")
            if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ident)
                    or ident.casefold() in ids):
                raise ValueError(f"invalid or repeated case ID: {ident!r}")
            ids.add(ident.casefold())
            parsed_cases.append(case)
        selected = {identifier.casefold() for identifier in args.selected or ()}
        if selected - ids:
            raise ValueError(f"unknown case IDs: {', '.join(sorted(selected - ids))}")
        overrides = {key: value for key, value in vars(args).items() if key in OPTIONS and value is not None}
        if args.mp_cache is not None:
            overrides["mp_cache"] = str(args.mp_cache.resolve())
        pair_overrides = {key: getattr(args, key) for key in ("polar_id", "reference_id", "selection")
                          if getattr(args, key) is not None}
        if pair_overrides and len(selected or ids) != 1:
            raise ValueError("candidate overrides require a single selected case")
        overrides.update(pair_overrides)
        if args.action == "analyze" and overrides.keys() - {"min_gap"}:
            raise ValueError("analyze uses the calculation settings saved in plan.json")
        base = args.config.resolve().parent if args.config else Path.cwd()
        output = args.output.resolve() if args.output else path_from(config.get("output", "fe-runs"), base, "output")
        output.mkdir(parents=True, exist_ok=True)
        good = {"prepared", "passed", "review", "resolved", "manual_structures"}
        with (output / ".fe.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"{output}: another workflow is using this batch directory") from None
            results = []
            for entry in parsed_cases:
                ident = entry["id"]
                if selected and ident.casefold() not in selected:
                    continue
                try:
                    values = case_options(defaults, entry, overrides, base)
                    result = run_case(args.action, output / ident, values, choose=args.choose,
                                      refresh=args.refresh_mp)
                except (ValueError, OSError, ArithmeticError) as error:
                    result = {"case": ident, "status": "input_error", "message": str(error)}
                results.append(result)
                print(f"{ident}: {result['status']}", flush=True)
                if "message" in result:
                    print(f"  {result['message']}", file=sys.stderr, flush=True)
            atomic_json(output / "batch-summary.json", {"action": args.action, "cases": results})
        statuses = {result["status"] for result in results}
        if statuses - good - {"needs_selection"}:
            return 1
        return 3 if "needs_selection" in statuses else 0
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
