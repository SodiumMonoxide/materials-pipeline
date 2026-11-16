"""Run the bundled optical workflow for selected ferroelectric configurations."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def from_fe_config(config, selected, fe_output):
    from fe_workflow import path_from, query_case, read_plan

    data = json.loads(config.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("FE configuration needs a cases array")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("FE defaults must be a JSON object")
    base = config.resolve().parent
    directory = fe_output or path_from(data.get("output", "fe-runs"), base, "FE output")
    selected = {ident.casefold() for ident in selected or ()}
    names = set()
    materials = []
    for case in data["cases"]:
        if isinstance(case, str):
            case = query_case(case)
        if not isinstance(case, dict):
            raise ValueError("FE cases must be objects, formulas, or Materials Project IDs")
        if "id" not in case:
            queries = case.keys() & {"formula", "material_id"}
            if len(queries) == 1:
                kind = next(iter(queries))
                case = {**query_case(case[kind], kind), **case}
        ident = case.get("id")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ident)
                or ident.casefold() in names):
            raise ValueError(f"invalid or duplicate FE case ID: {ident!r}")
        names.add(ident.casefold())
        if selected and ident.casefold() not in selected:
            continue

        case_dir = directory / ident
        plan_path = case_dir / "plan.json"
        if "polar" in case:
            poscar = path_from(case["polar"], base, "polar POSCAR")
            planned_potentials = {}
        elif plan_path.is_file():
            plan = read_plan(case_dir)
            source = plan.get("structure_source", {})
            for key in ("material_id", "formula"):
                if key in case and (source.get("query_type") != key or source.get("query") != str(case[key]).strip()):
                    raise ValueError(f"{ident}: this FE plan belongs to another structure query")
            if "selection" in case and source.get("selection") != case["selection"]:
                raise ValueError(f"{ident}: this FE plan uses another confirmed structure pair")
            if "polar_id" in case:
                from fe_mp_api import material_id_key

                if material_id_key(source.get("polar_id", "")) != material_id_key(case["polar_id"]):
                    raise ValueError(f"{ident}: this FE plan uses another polar material ID")
            poscar = case_dir / plan["images"][-1] / "POSCAR"
            planned_potentials = {entry["element"]: entry["variant"] for entry in plan["potentials"]}
        else:
            raise ValueError(f"{ident}: prepare the FE case to resolve its polar POSCAR first, "
                             "or supply a POSCAR to the optics command")

        settings = {}
        # Keep optical defaults for the measurement. Reuse PAW variants from the
        # FE case unless the optical settings explicitly select different ones.
        potentials = {}
        for layer in (defaults, case):
            variants = layer.get("variants", {})
            extras = layer.get("optics", {})
            if not isinstance(variants, dict) or not isinstance(extras, dict):
                raise ValueError(f"{ident}: variants and optics settings must be JSON objects")
            potentials.update(variants)
            if "kmesh" in extras and "kspacing" in extras:
                raise ValueError(f"{ident}: set kmesh or kspacing in each optics settings object")
            if "kmesh" in extras:
                settings.pop("kspacing", None)
            if "kspacing" in extras:
                settings.pop("kmesh", None)
            for key, value in extras.items():
                if key in ("incar", "potentials"):
                    if not isinstance(value, dict):
                        raise ValueError(f"{ident}: optics.{key} must be a JSON object")
                    settings[key] = {**settings.get(key, {}), **value}
                else:
                    settings[key] = value
        potentials.update(planned_potentials)
        potentials.update(settings.get("potentials", {}))
        material = {"id": ident, "poscar": str(poscar), "settings": settings}
        potcar = case.get("potcar", defaults.get("potcar"))
        if potcar is not None:
            material["potcar"] = str(path_from(potcar, base, "POTCAR"))
            settings.pop("potentials", None)
        elif potentials:
            from fe_structures import read_poscar

            elements = set(read_poscar(poscar).elements)
            settings["potentials"] = {element: value for element, value in potentials.items()
                                      if element in elements}
        materials.append(material)
    if unknown := selected - names:
        raise ValueError(f"unknown FE case IDs: {', '.join(sorted(unknown))}")
    if not materials:
        raise ValueError("no FE case selected for optical screening")
    runtime = {}
    if "paw_dir" in defaults:
        runtime["VASP_POTCAR_ROOT"] = str(path_from(defaults["paw_dir"], base, "PAW directory"))
    if "vasp" in defaults:
        vasp = defaults["vasp"]
        if not isinstance(vasp, str) or not vasp:
            raise ValueError("FE defaults.vasp must be a command name or path")
        runtime["VASP_EXECUTABLE"] = str(path_from(vasp, base, "VASP")) if "/" in vasp or "$" in vasp else vasp
    return {"materials": materials}, runtime


def main(argv):
    parser = argparse.ArgumentParser(prog="fe_workflow.py optics", add_help=False, allow_abbrev=False)
    parser.add_argument("--optics-root", type=Path, help="path to the OpticalProperties project")
    parser.add_argument("--fe-config", type=Path, help="load structures from a ferroelectric component JSON")
    parser.add_argument("--fe-output", type=Path, help="FE output directory with the prepared POSCARs")
    parser.add_argument("--case", action="append", help="filter --fe-config to those case IDs")
    args, backend_args = parser.parse_known_args(argv)
    try:
        default_root = Path(__file__).resolve().parents[2] / "OpticalProperties"
        root = args.optics_root or Path(os.environ.get("FE_OPTICS_ROOT", str(default_root)))
        root = root.expanduser().resolve()
        if not (root / "optical_workflow" / "__main__.py").is_file() or not (root / "optics.py").is_file():
            raise ValueError(f"OpticalProperties engine not found at {root}; set --optics-root or FE_OPTICS_ROOT")
        if "--help" in backend_args or "-h" in backend_args:
            parser.print_help()
            sys.stdout.flush()
        if args.fe_config is None and (args.case or args.fe_output):
            raise ValueError("--case and --fe-output require --fe-config")
        if args.fe_config is not None and any(arg.split("=", 1)[0] == "--manifest" for arg in backend_args):
            raise ValueError("use --fe-config or the optical --manifest")
        env = os.environ.copy()
        paths = [str(root), env.get("PYTHONPATH")]
        env["PYTHONPATH"] = os.pathsep.join(path for path in paths if path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for optical, ferroelectric in (("VASP_EXECUTABLE", "VASP_EXE"), ("VASP_PAW_PBE", "VASP_PSP_DIR")):
            if ferroelectric in env:
                env.setdefault(optical, env[ferroelectric])
        command = [sys.executable, "-m", "optical_workflow", *backend_args]
        if args.fe_config is None:
            return subprocess.run(command, env=env).returncode
        manifest, runtime = from_fe_config(args.fe_config, args.case,
                                           args.fe_output.resolve() if args.fe_output else None)
        env.update(runtime)
        with tempfile.TemporaryDirectory(prefix="fe-optics-") as work:
            path = Path(work) / "manifest.json"
            path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
            return subprocess.run([*command, "--manifest", str(path)], env=env).returncode
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    except KeyboardInterrupt:
        return 130
