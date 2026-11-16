"""Prepare and run a batch of fixed-structure VASP optical calculations."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from xml.etree.ElementTree import ParseError

from .convergence import ConvergenceOptions, mesh_levels, run_convergence
from .inputs import build_inputs
from .results import collect_result


SCHEMA_VERSION = 1
OUTPUTS = ("OUTCAR", "vasprun.xml", "OSZICAR", "EIGENVAL", "IBZKPT", "CONTCAR")
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


@dataclass(frozen=True)
class Material:
    name: str
    poscar: Path
    settings: dict
    potcar: Path | None


@dataclass(frozen=True)
class Runner:
    output: Path
    vasp: Path
    mpi: Path | None
    ranks: int
    taskset: Path | None
    scratch: Path | None
    rerun: bool

    def command(self):
        if self.mpi is None:
            return [str(self.vasp)]
        return [str(self.mpi), "-np", str(self.ranks), str(self.vasp)]


def parser_arguments(cpu_count):
    parser = argparse.ArgumentParser(prog="python3 -m optical_workflow", description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="POSCAR files or folders containing POSCAR files")
    parser.add_argument("--manifest", type=Path, help="JSON batch manifest")
    parser.add_argument("--out", type=Path, help="persistent results directory")
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate inputs without launching VASP")
    parser.add_argument("--rerun", action="store_true", help="create new attempts even if matching results exist")
    parser.add_argument("--vasp", help="path or name of the VASP standard executable")
    parser.add_argument("--mpi", help="MPI launcher accepting -np N (default: mpirun)")
    parser.add_argument("--ranks", type=int, default=min(8, cpu_count), help="MPI ranks per material")
    parser.add_argument("--jobs", type=int, default=1, help="number of simultaneous materials")
    parser.add_argument("--scratch", type=Path, help="existing WSL directory for temporary VASP files")
    parser.add_argument("--potcar-root", type=Path, help="PAW-PBE root with Element/POTCAR files")
    parser.add_argument("--potcar", type=Path, help="ordered PAW-PBE POTCAR for a single material")
    parser.add_argument("--profile", choices=("screen", "standard", "dense"))
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--kspacing", type=float)
    grid.add_argument("--kmesh", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    for flag in ("encut", "cshift", "sigma", "omegamax", "ediff", "drude-gamma"):
        parser.add_argument(f"--{flag}", type=float)
    for flag in ("nbands", "nedos", "nelm", "kpar", "ncore"):
        parser.add_argument(f"--{flag}", type=int)
    parser.add_argument("--dimension", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--vacuum-axes", type=int, choices=(1, 2, 3), nargs="+",
                        help="1-based nonperiodic lattice directions")
    parser.add_argument("--metal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--component", choices=("tensor", "isotropic", "xx", "yy", "zz"))
    parser.add_argument("--incar", action="append", metavar="KEY=VALUE", help="allowed extra INCAR tag, repeatable")
    conv = parser.add_argument_group("automatic convergence")
    conv.add_argument("--converge", nargs="*", metavar="MATERIAL",
                      help="refine these IDs after screening; omit IDs to refine all")
    conv.add_argument("--conv-window", type=float, nargs=2, default=(0.0, 8.0),
                      metavar=("MIN", "MAX"), help="comparison energy window in eV (default: 0 8)")
    conv.add_argument("--conv-rtol", type=float, default=0.05, help="relative RMS tolerance (default: 0.05)")
    conv.add_argument("--conv-atol", type=float, default=0.02, help="absolute RMS tolerance (default: 0.02)")
    conv.add_argument("--conv-k-growth", type=float, default=1.4, help="automatic k-mesh multiplier (default: 1.4)")
    conv.add_argument("--conv-k-steps", type=int, default=5, help="maximum automatic mesh increments (default: 5)")
    conv.add_argument("--conv-kmesh", type=int, nargs=3, action="append", metavar=("NX", "NY", "NZ"),
                      help="explicit refinement mesh; repeat to define the ladder after screening")
    conv.add_argument("--conv-band-growth", type=float, default=1.4,
                      help="multiplier for actual NBANDS (default: 1.4)")
    conv.add_argument("--conv-band-steps", type=int, default=3,
                      help="maximum NBANDS increments (default: 3)")
    return parser


def selection_settings(args, materials):
    if args.converge is None:
        return set(), None
    names = {material.name for material in materials}
    selected = set(args.converge) if args.converge else names
    if missing := selected - names:
        raise ValueError(f"Unknown convergence material IDs: {', '.join(sorted(missing))}")
    options = ConvergenceOptions(
        tuple(args.conv_window), args.conv_rtol, args.conv_atol, args.conv_k_growth, args.conv_k_steps,
        tuple(tuple(mesh) for mesh in args.conv_kmesh or ()), args.conv_band_growth, args.conv_band_steps,
    )
    options.validate()
    return selected, options


def merge_settings(*layers):
    merged = {}
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("settings and defaults must be JSON objects")
        if "kmesh" in layer and "kspacing" in layer:
            raise ValueError("Use either kmesh or kspacing in the same settings object")
        if "kmesh" in layer:
            merged.pop("kspacing", None)
        if "kspacing" in layer:
            merged.pop("kmesh", None)
        if layer.get("metal") is False and "drude_gamma" not in layer:
            merged.pop("drude_gamma", None)
        for key, value in layer.items():
            if key in ("incar", "potentials"):
                if not isinstance(value, dict):
                    raise ValueError(f"{key} must be a JSON object")
                previous = merged.get(key, {})
                if key == "incar":
                    value = {tag.upper(): setting for tag, setting in value.items()}
                merged[key] = {**previous, **value}
            else:
                merged[key] = value
    if merged.get("component") == "tensor":
        merged["component"] = None
    return merged


def command_settings(args):
    keys = ("profile", "kspacing", "kmesh", "encut", "nbands", "cshift", "sigma", "omegamax", "nedos",
            "ediff", "nelm", "dimension", "vacuum_axes", "metal", "drude_gamma", "kpar", "ncore", "component")
    settings = {key: getattr(args, key) for key in keys if getattr(args, key) is not None}
    if args.incar:
        settings["incar"] = {}
        for declaration in args.incar:
            key, sep, value = declaration.partition("=")
            if not sep or not key.strip() or not value.strip():
                raise ValueError(f"Expected --incar KEY=VALUE, got {declaration!r}")
            key = key.strip().upper()
            value = value.strip()
            if key not in {"MAGMOM", "LDAUL", "LDAUU", "LDAUJ"}:
                boolean = value.strip(".").lower()
                if boolean in {"true", "false", "t", "f"}:
                    value = boolean in {"true", "t"}
                elif re.fullmatch(r"[+-]?\d+", value):
                    value = int(value)
                else:
                    try:
                        value = float(value.replace("D", "E").replace("d", "e"))
                    except ValueError:
                        pass
            settings["incar"][key] = value
    return settings


def path_from(base, name, label):
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} must be a nonempty path")
    return (base / Path(name).expanduser()).resolve()


def load_materials(args):
    global_settings = command_settings(args)
    materials = []
    if args.manifest:
        manifest = args.manifest.expanduser().resolve()
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if (not isinstance(data, dict) or set(data) - {"defaults", "materials"}
                or not isinstance(data.get("materials"), list)):
            raise ValueError("Manifest must contain a materials array and optional defaults")
        for entry in data["materials"]:
            if (not isinstance(entry, dict) or set(entry) - {"id", "poscar", "potcar", "settings"}
                    or not isinstance(entry.get("id"), str)):
                raise ValueError("Each material needs id and poscar, with optional potcar and settings")
            poscar = path_from(manifest.parent, entry.get("poscar"), "poscar")
            potcar = path_from(manifest.parent, entry["potcar"], "potcar") if "potcar" in entry else None
            settings = merge_settings(data.get("defaults", {}), entry.get("settings", {}), global_settings)
            materials.append(Material(entry["id"], poscar, settings, potcar))
    else:
        files = set()
        output = args.out.expanduser().resolve() if args.out else None
        for requested in args.inputs:
            path = requested.expanduser().resolve()
            if path.is_dir():
                candidates = [path / "POSCAR"] if (path / "POSCAR").is_file() else path.rglob("POSCAR")
                paths = {item.resolve() for item in candidates if item.is_file()}
                if output is not None:
                    paths = {item for item in paths if not item.is_relative_to(output)}
                if not paths:
                    raise ValueError(f"No POSCAR found in {path}")
                files.update(paths)
            else:
                files.add(path)
        for path in sorted(files):
            name = path.parent.name if path.name.upper() in ("POSCAR", "CONTCAR") else path.stem
            name = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip(".-_")[:80] or "material"
            materials.append(Material(name, path, merge_settings(global_settings), None))
    if not materials:
        raise ValueError("No materials to calculate")
    if args.potcar:
        if len(materials) != 1:
            raise ValueError("--potcar requires exactly one material; use per-material potcar in a batch")
        job = materials[0]
        materials[0] = Material(job.name, job.poscar, job.settings, args.potcar.expanduser().resolve())
    used = set()
    for job in materials:
        if not ID_PATTERN.fullmatch(job.name) or job.name.endswith(".") or job.name.casefold() in used:
            raise ValueError(f"Invalid or duplicate material ID {job.name!r}; set a unique ID in a manifest")
        used.add(job.name.casefold())
    return materials


def executable(value):
    if value is None or not (resolved := shutil.which(os.path.expanduser(str(value)))):
        raise ValueError(f"Executable not found: {value}")
    return Path(resolved).resolve()


def resolve_runner(args, parallel_jobs):
    vasp = executable(args.vasp or os.environ.get("VASP_EXECUTABLE") or "vasp_std")
    mpi = executable(args.mpi or os.environ.get("VASP_MPIRUN") or "mpirun") if args.ranks > 1 else None
    taskset = executable("taskset") if parallel_jobs > 1 else None
    scratch = args.scratch.expanduser().resolve() if args.scratch else None
    if scratch is not None and not scratch.is_dir():
        raise ValueError(f"Scratch must be an existing directory: {scratch}")
    return Runner(args.out.expanduser().resolve(), vasp, mpi, args.ranks, taskset, scratch, args.rerun)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(inputs, runner):
    command = runner.command()
    signatures = []
    for program in (runner.vasp, runner.mpi):
        if program is not None:
            stat = program.stat()
            signatures.append([str(program), stat.st_size, stat.st_mtime_ns])
    data = [SCHEMA_VERSION, command, signatures, hashlib.sha256(inputs.poscar).hexdigest(),
            inputs.metadata["potcar_sha256"], inputs.incar, inputs.kpoints, inputs.metadata]
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_state(directory, value):
    temporary = directory / "run.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(directory / "run.json")


def prior_attempts(directory):
    attempts = []
    for path in directory.iterdir():
        match = re.fullmatch(r"attempt-(\d{4,})", path.name)
        if match:
            attempts.append((int(match[1]), path))
    return sorted(attempts)


def cached_state(attempt, key):
    if not attempt.is_dir() or attempt.is_symlink():
        return None
    try:
        state = json.loads((attempt / "run.json").read_text(encoding="utf-8"))
        if (state.get("status") == "completed" and state.get("fingerprint") == key
                and all((attempt / name).is_file() for name in ("OUTCAR", "vasprun.xml", "spectrum.csv"))
                and file_hash(attempt / "spectrum.csv") == state.get("spectrum_sha256")):
            return state
    except (OSError, ValueError, AttributeError):
        pass
    return None


def stop_solver(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return process.wait()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    return process.wait()


def launch(inputs, attempt, runner, cpu_group, stop):
    command = runner.command()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if cpu_group:
        command = [str(runner.taskset), "-c", ",".join(map(str, cpu_group)), *command]
        env["OMPI_MCA_hwloc_base_binding_policy"] = "none"

    for name, content in (("POSCAR", inputs.poscar), ("INCAR", inputs.incar.encode()),
                          ("KPOINTS", inputs.kpoints.encode())):
        (attempt / name).write_bytes(content)
    with tempfile.TemporaryDirectory(prefix="vasp-optics-", dir=runner.scratch) as temp:
        scratch = Path(temp)
        for name in ("POSCAR", "INCAR", "KPOINTS"):
            shutil.copyfile(attempt / name, scratch / name)
        (scratch / "POTCAR").write_bytes(inputs.potcar)
        try:
            with (attempt / "vasp.log").open("wb") as log:
                with subprocess.Popen(command, cwd=scratch, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True) as process:
                    while not stop.is_set():
                        try:
                            return process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            continue
                    stop_solver(process)
                    raise ValueError("Interrupted during VASP; partial outputs were retained")
        finally:
            for name in OUTPUTS:
                if (scratch / name).is_file():
                    shutil.copyfile(scratch / name, attempt / name)


def process_material(job, root_potcar, runner, slots, stop, expected_structure=None):
    cpu_group = slots.get()
    state = {"material_id": job.name, "source": str(job.poscar)}
    attempt = None
    try:
        if stop.is_set():
            raise ValueError("Interrupted before starting VASP")
        inputs = build_inputs(job.poscar, job.settings, job.potcar, None if job.potcar else root_potcar,
                              runner.ranks)
        poscar_sha256 = hashlib.sha256(inputs.poscar).hexdigest()
        if expected_structure is not None and expected_structure != (
            poscar_sha256, inputs.metadata["potcar_sha256"]
        ):
            raise ValueError("POSCAR or POTCAR changed after screening; rerun the batch with fixed inputs")
        key = fingerprint(inputs, runner)
        directory = runner.output / job.name
        if directory.is_symlink():
            raise ValueError(f"Material output directory is a symlink: {directory}")
        directory.mkdir(exist_ok=True)
        attempts = prior_attempts(directory)
        if not runner.rerun:
            for _, previous in reversed(attempts):
                if state_cached := cached_state(previous, key):
                    return {**state_cached, **state, "status": "reused", "poscar_sha256": poscar_sha256,
                            "attempt": str(previous.relative_to(runner.output))}
        number = attempts[-1][0] + 1 if attempts else 1
        attempt = directory / f"attempt-{number:04d}"
        attempt.mkdir()
        state.update({"status": "running", "fingerprint": key, "schema_version": SCHEMA_VERSION,
                      "attempt": str(attempt.relative_to(runner.output)), "inputs": inputs.metadata,
                      "command": runner.command(), "cpu_group": cpu_group, "poscar_sha256": poscar_sha256})
        write_state(attempt, state)
        state["exit_code"] = launch(inputs, attempt, runner, cpu_group, stop)
        if state["exit_code"] != 0:
            raise ValueError(f"VASP exited with status {state['exit_code']}; see vasp.log")
        state["results"] = collect_result(attempt, inputs.metadata)
        state["spectrum_sha256"] = file_hash(attempt / "spectrum.csv")
        state["status"] = "completed"
        write_state(attempt, state)
    except (OSError, ValueError, EOFError, ParseError) as exc:
        state.update({"status": "failed", "message": str(exc)})
        if attempt is not None and attempt.is_dir():
            try:
                write_state(attempt, state)
            except OSError as state_error:
                state["message"] += f" | Could not save run.json: {state_error}"
    finally:
        slots.put(cpu_group)
    return state


def write_summary(output, states):
    states = list(states)
    results_fields = ("vasp_version", "kmesh", "nk_written", "ENCUT", "NBANDS", "elapsed_seconds",
                      "energy_min_eV", "energy_max_eV", "sampled_ks_gap_eV", "partial_occupation_count",
                      "drude_included")
    fields = ("material_id", "status", "source", "attempt", "spectrum", "dimension",
              *results_fields, "warnings", "message")
    if any("convergence_status" in state for state in states):
        fields += ("convergence_status", "convergence_attempt", "convergence_report", "convergence_message")
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        for state in sorted(states, key=lambda row: row["material_id"]):
            row = {key: state.get(key, "") for key in fields}
            result = state.get("results", {})
            row["dimension"] = state.get("inputs", {}).get("dimension", "")
            for key in results_fields:
                row[key] = result.get(key, "")
            if "spectrum" in result:
                row["spectrum"] = f"{row['attempt']}/{result['spectrum']}"
            row["warnings"] = " | ".join(result.get("warnings", []))
            writer.writerow(row)
    temporary.replace(output / "summary.csv")


def dry_run(materials, root_potcar, ranks, selected, options):
    failed = False
    for job in materials:
        record = {"material_id": job.name, "source": str(job.poscar)}
        try:
            inputs = build_inputs(job.poscar, job.settings, job.potcar, None if job.potcar else root_potcar, ranks)
            record.update({"status": "ready", "inputs": inputs.metadata})
            if job.name in selected:
                record["convergence"] = {
                    "window_eV": options.window, "rtol": options.rtol, "atol": options.atol,
                    "mesh_ladder": mesh_levels(inputs.metadata, options), "band_steps_limit": options.band_steps,
                }
        except (OSError, ValueError, EOFError) as exc:
            record.update({"status": "failed", "message": str(exc)})
            failed = True
        print(json.dumps(record, ensure_ascii=False, allow_nan=False))
    return int(failed)


def process_convergence(job, baseline, root_potcar, runner, slots, stop, options):
    def solve(settings):
        material = Material(job.name, job.poscar, merge_settings(job.settings, settings), job.potcar)
        signature = baseline["poscar_sha256"], baseline["inputs"]["potcar_sha256"]
        return process_material(material, root_potcar, runner, slots, stop, signature)

    try:
        report = run_convergence(runner.output, baseline, options, solve,
                                 lambda message: print(f"{job.name}: {message}", flush=True))
    except (OSError, ValueError) as exc:
        return {"convergence_status": "failed", "convergence_message": str(exc)}
    return {"convergence_status": report["status"], "convergence_message": report["message"],
            "convergence_attempt": report["final_attempt"],
            "convergence_report": f"{job.name}/convergence.md"}


def run_batch(materials, root_potcar, runner, parallel_jobs, cpus, selected, options):
    import fcntl

    runner.output.mkdir(parents=True, exist_ok=True)
    with (runner.output / ".optics.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another batch is running in this output directory") from exc
        slots = SimpleQueue()
        for index in range(parallel_jobs):
            start = index * runner.ranks
            slots.put(cpus[start:start + runner.ranks] if runner.taskset else [])
        states = [{"material_id": job.name, "source": str(job.poscar), "status": "queued"} for job in materials]
        state_by_name = {state["material_id"]: state for state in states}
        for name in selected:
            state_by_name[name]["convergence_status"] = "queued"
        write_summary(runner.output, state_by_name.values())
        stop = Event()
        with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
            pending = {}
            try:
                for job in materials:
                    future = pool.submit(process_material, job, root_potcar, runner, slots, stop)
                    pending[future] = job.name
                for future in as_completed(pending):
                    result = future.result()
                    state_by_name[result["material_id"]].update(result)
                    write_summary(runner.output, state_by_name.values())
                    detail = result.get("message") or result.get("attempt", "")
                    print(f"{result['material_id']}: {result['status']} {detail}", flush=True)
                pending.clear()
                for job in materials:
                    if job.name in selected:
                        future = pool.submit(process_convergence, job, state_by_name[job.name],
                                             root_potcar, runner, slots, stop, options)
                        pending[future] = job.name
                for future in as_completed(pending):
                    name = pending[future]
                    result = future.result()
                    state_by_name[name].update(result)
                    write_summary(runner.output, state_by_name.values())
                    print(f"{name}: {result['convergence_status']} {result['convergence_message']}", flush=True)
            except KeyboardInterrupt:
                stop.set()
                pool.shutdown(wait=True, cancel_futures=True)
                for future, name in pending.items():
                    if not future.cancelled():
                        state_by_name[name].update(future.result())
                for state in state_by_name.values():
                    if state["status"] == "queued":
                        state.update(status="failed", message="Interrupted before starting VASP")
                    if state.get("convergence_status") == "queued":
                        state.update(convergence_status="failed", convergence_message="Batch interrupted")
                write_summary(runner.output, state_by_name.values())
                print("Batch interrupted. Finished attempts are available for reuse.", file=sys.stderr)
                return 130
    return int(any(state["status"] == "failed" or state.get("convergence_status") in {"failed", "not_converged"}
                   for state in state_by_name.values()))


def main():
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    parser = parser_arguments(len(cpus))
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.manifest):
        parser.error("Provide POSCAR paths or --manifest")
    if args.ranks < 1 or args.jobs < 1:
        parser.error("--ranks and --jobs must be positive integers")
    if not args.dry_run and not args.out:
        parser.error("--out is required for VASP runs")
    if not args.dry_run and sys.platform != "linux":
        parser.error("Launch this component inside WSL or Linux")
    try:
        materials = load_materials(args)
        selected, options = selection_settings(args, materials)
        parallel_jobs = min(args.jobs, len(materials))
        if args.ranks * parallel_jobs > len(cpus):
            raise ValueError(f"Requested {args.ranks * parallel_jobs} CPUs, but {len(cpus)} are available")
        root_potcar = args.potcar_root or os.environ.get("VASP_POTCAR_ROOT") or os.environ.get("VASP_PAW_PBE")
        if root_potcar is not None:
            root_potcar = Path(root_potcar).expanduser().resolve()
        if args.dry_run:
            return dry_run(materials, root_potcar, args.ranks, selected, options)
        runner = resolve_runner(args, parallel_jobs)
        return run_batch(materials, root_potcar, runner, parallel_jobs, cpus, selected, options)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    except KeyboardInterrupt:
        return 130


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
