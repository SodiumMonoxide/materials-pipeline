#!/usr/bin/env python3
"""Run VASP workflows for ferroelectric, optical and elastic properties on Linux."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
from queue import SimpleQueue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parent
STAGES = {
    "mechanical": ("mechanical_adapter", "MechanicalProperties"),
    "ferroelectric": ("ferro_adapter", "ferroelectricity-dft"),
    "optical": ("optical_adapter", "OpticalProperties"),
}
READY = {"completed", "review"}
RUNTIME_DEFAULTS = {
    "python": sys.executable,
    "vasp": os.environ.get("VASP_EXECUTABLE") or "vasp_std",
    "paw_dir": os.environ.get("VASP_POTCAR_ROOT") or None,
    "mpi": "mpirun",
    "workers": 1,
    "ranks": 1,
    "threads": 1,
}
PATH_KEYS = {"reference", "polar", "opposite", "potcar", "potcar_root", "paw_dir"}


def resolve_path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a nonempty path, got {value!r}")
    path = Path(os.path.expandvars(value)).expanduser()
    return (base / path).resolve()


def executable(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid executable: {value!r}")
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "/" in expanded or (base / expanded).is_file():
        # Following the executable symlink would turn a venv Python into system Python.
        return os.path.abspath(base / expanded)
    return shutil.which(expanded) or expanded


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def object_value(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_manifest(path):
    cfg = object_value(json.loads(path.read_text(encoding="utf-8-sig")), "Manifest")
    unknown = cfg.keys() - {"runtime", "defaults", "materials"}
    if unknown:
        raise ValueError(f"Unknown manifest keys: {sorted(unknown)}")
    runtime = dict(RUNTIME_DEFAULTS)
    requested = object_value(cfg.get("runtime", {}), "runtime")
    unknown = requested.keys() - runtime.keys()
    if unknown:
        raise ValueError(f"Unknown runtime keys: {sorted(unknown)}")
    runtime.update(requested)
    for key in ("python", "vasp", "mpi"):
        runtime[key] = executable(path.parent, runtime[key])
    if runtime["paw_dir"] is not None:
        runtime["paw_dir"] = str(resolve_path(path.parent, runtime["paw_dir"]))
    for key in ("workers", "ranks", "threads"):
        positive_integer(runtime[key], f"runtime.{key}")
    defaults = object_value(cfg.get("defaults", {}), "defaults")
    if defaults.keys() - STAGES.keys():
        raise ValueError(f"Unknown property defaults: {sorted(defaults.keys() - STAGES.keys())}")
    for name, value in defaults.items():
        object_value(value, f"defaults.{name}")
    materials = cfg.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("materials must be a nonempty array")
    items = {}
    for material in materials:
        material = object_value(material, "material")
        name = material.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
            raise ValueError(f"Unsafe material id: {name!r}")
        if name in items:
            raise ValueError(f"Duplicate material id: {name}")
        if material.keys() - {"id", "structure", "properties"}:
            raise ValueError(f"{name}: expected id, structure and properties")
        structure = resolve_path(path.parent, material.get("structure"))
        props = object_value(material.get("properties"), f"{name}.properties")
        settings = {}
        for stage, value in props.items():
            if stage not in STAGES:
                raise ValueError(f"{name}: unknown property {stage!r}")
            if value is False:
                continue
            overrides = {} if value is True else object_value(value, f"{name}.{stage}")
            stage_defaults = defaults.get(stage, {})
            if stage == "optical":
                from optical_adapter import merge_settings
                settings[stage] = merge_settings(stage_defaults, overrides)
            else:
                settings[stage] = {**stage_defaults, **overrides}
        if not settings:
            raise ValueError(f"{name}: enable at least one property")
        for stage, values in settings.items():
            source = values.get("structure_from")
            if source is not None and (stage != "optical" or source != "mechanical" or source not in settings):
                raise ValueError(f"{name}.{stage}: structure_from requires an enabled mechanical stage and is only available for optical")
        items[name] = structure, settings
    return runtime, items


@dataclass
class Context:
    material_id: str
    structure: Path
    settings: dict
    output: Path
    vendor: Path
    base: Path
    ranks: int
    threads: int
    python: str
    vasp: str
    paw_dir: str | None
    mpi: str
    prepare: bool
    cpus: tuple
    cancel: threading.Event

    def resolve(self, value):
        return resolve_path(self.base, value)

    def command(self, args, *, cwd=None):
        if self.cancel.is_set():
            return 130
        env = dict(os.environ, OMP_NUM_THREADS=str(self.threads),
                   OPENBLAS_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        # Disable Open MPI's default core rebinding inside the assigned CPU set.
        env["OMPI_MCA_hwloc_base_binding_policy"] = "none"
        argv = ["taskset", "-c", ",".join(map(str, self.cpus)), *map(str, args)]
        with (self.output / "driver.log").open("a", encoding="utf-8") as log:
            log.write(f"\n$ {shlex.join(argv)}\n")
            log.flush()
            proc = subprocess.Popen(argv, cwd=cwd or self.output, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while not self.cancel.is_set():
                try:
                    return proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=timeout)
                    return 130
                except subprocess.TimeoutExpired:
                    pass
            return 130


def run_stage(name, stage, structure, settings, base, out, runtime, cpus, prepare, cancel):
    record = {"material": name, "property": stage}
    try:
        files = {"structure": {"path": str(structure), "sha256": sha256(structure)}}
        for key in sorted(PATH_KEYS & settings.keys()):
            if settings[key] is not None:
                source = resolve_path(base, settings[key])
                files[key] = {"path": str(source), "sha256": sha256(source) if source.is_file() else None}
        solver = {key: value for key, value in runtime.items() if key != "workers"}
        request = {"property": stage, "settings": settings, "inputs": files, "runtime": solver}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
        output = out / name / stage / key
        output.mkdir(parents=True, exist_ok=True)
        log = output / "driver.log"
        record.update(directory=str(output), log=str(log))
        write_json(output / "request.json", request)
        adapter, vendor = STAGES[stage]
        context = Context(name, structure, settings, output, ROOT / "vendor" / vendor,
                          base, prepare=prepare, cpus=cpus, cancel=cancel, **solver)
        print(f"{name} / {stage} → {output}", flush=True)
        result = importlib.import_module(adapter).run(context)
        if result.get("status") not in READY | {"prepared", "blocked", "failed"}:
            raise ValueError(f"Invalid adapter status: {result!r}")
        record.update(result)
        if cancel.is_set():
            record.update(status="interrupted", message="Batch interrupted")
        write_json(output / ("prepared.json" if prepare else "result.json"), record)
    except Exception as exc:
        record.update(status="failed", message=f"{type(exc).__name__}: {exc}")
    print(f"{name} / {stage}: {record['status']}", flush=True)
    return record


def run_material(name, structure, properties, base, out, runtime, cpu_slots, prepare, cancel, publish):
    cpus = cpu_slots.get()
    records = {}
    try:
        for stage in STAGES:
            if stage not in properties:
                continue
            settings = dict(properties[stage])
            source = settings.pop("structure_from", None)
            stage_structure = structure
            if cancel.is_set():
                result = {"status": "interrupted", "message": "Batch interrupted"}
            elif source:
                upstream = records[source]
                # A mechanical review can mean residual stress or failed stability checks.
                if prepare:
                    result = {"status": "deferred", "message": f"Requires the structure from {source} during run"}
                elif upstream["status"] != "completed" or not upstream.get("structure"):
                    result = {"status": "blocked", "message": f"Requires a completed {source} stage with an equilibrium structure"}
                else:
                    stage_structure = Path(upstream["structure"])
                    result = run_stage(name, stage, stage_structure, settings, base, out, runtime, cpus, prepare, cancel)
                    result["structure_from"] = source
            else:
                result = run_stage(name, stage, stage_structure, settings, base, out, runtime, cpus, prepare, cancel)
            records[stage] = {"material": name, "property": stage, **result}
            publish(records[stage])
    finally:
        cpu_slots.put(cpus)


def write_summary(out, manifest, action, records):
    rows = sorted(records, key=lambda row: (row["material"], list(STAGES).index(row["property"])))
    write_json(out / f"{action}-summary.json", {"manifest": str(manifest), "stages": rows})
    columns = ["material", "property", "status", "message", "directory", "result", "log", "structure", "structure_from"]
    metric_names = sorted({f"metric.{key}" for row in rows for key in row.get("metrics", {})})
    temporary = out / f"{action}-summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + metric_names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{f"metric.{key}": val for key, val in row.get("metrics", {}).items()}})
    temporary.replace(out / f"{action}-summary.csv")


def check_runtime(runtime, properties):
    problems = []
    for tool in ("python", "vasp", "mpi"):
        value = runtime[tool]
        if not shutil.which(value):
            problems.append(f"{tool} is missing or not executable: {value}")
    if not runtime["paw_dir"]:
        problems.append("Set VASP_POTCAR_ROOT or runtime.paw_dir to your licensed PAW-PBE library")
    elif not Path(runtime["paw_dir"]).is_dir():
        problems.append(f"PBE POTCAR library is missing: {runtime['paw_dir']}")
    if not shutil.which("taskset"):
        problems.append("taskset is missing")
    if "mechanical" in properties and shutil.which(runtime["python"]):
        check = subprocess.run([runtime["python"], "-B", "-c", "import ase, numpy, spglib"],
                               capture_output=True, text=True)
        if check.returncode:
            problems.append(f"Install ASE, NumPy and spglib in {runtime['python']}: {check.stderr.strip()}")
    return problems


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("action", choices=("doctor", "prepare", "run"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="Directory for generated inputs, job logs and results")
    parser.add_argument("--material", action="append", metavar="ID")
    parser.add_argument("--property", action="append", choices=STAGES)
    parser.add_argument("--workers", type=int, help="Number of materials to run in parallel")
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve()
    runtime, items = load_manifest(manifest)
    if args.workers is not None:
        runtime["workers"] = positive_integer(args.workers, "--workers")
    unknown = set(args.material or ()) - items.keys()
    if unknown:
        raise ValueError(f"Unknown material ids: {sorted(unknown)}")
    selected = {}
    for name, (structure, props) in items.items():
        if args.material and name not in args.material:
            continue
        properties = set(args.property or props) & props.keys()
        for prop in list(properties):
            if source := props[prop].get("structure_from"):
                properties.add(source)
        if properties:
            selected[name] = structure, {key: props[key] for key in props if key in properties}
    if not selected:
        raise ValueError("No material/property combination selected")
    cpus = sorted(os.sched_getaffinity(0))
    cores_per_worker = runtime["ranks"] * runtime["threads"]
    workers = min(runtime["workers"], len(selected))
    if workers * cores_per_worker > len(cpus):
        raise ValueError(f"{workers} workers × {cores_per_worker} logical CPUs exceeds {len(cpus)} available")
    problems = check_runtime(runtime, {p for _, props in selected.values() for p in props})
    if problems:
        raise ValueError("\n".join(problems))
    if args.action == "doctor":
        print(json.dumps({"status": "ready", "available_logical_cpus": len(cpus), **runtime}, indent=2))
        return 0
    if args.out is None:
        parser.error("--out is required for run and prepare")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cancel = threading.Event()
    def stop_batch(signum, frame):
        cancel.set()
    signal.signal(signal.SIGINT, stop_batch)
    signal.signal(signal.SIGTERM, stop_batch)
    with (out / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another pipeline is using {out}") from None
        slots = SimpleQueue()
        for offset in range(0, workers * cores_per_worker, cores_per_worker):
            slots.put(tuple(cpus[offset:offset + cores_per_worker]))
        records = []
        summary_lock = threading.Lock()
        write_summary(out, manifest, args.action, records)

        def publish(record):
            with summary_lock:
                records.append(record)
                write_summary(out, manifest, args.action, records)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_material, name, structure, props, manifest.parent,
                                       out, runtime, slots, args.action == "prepare", cancel, publish)
                       for name, (structure, props) in selected.items()]
            for future in as_completed(futures):
                future.result()
    if cancel.is_set():
        return 130
    return int(any(row["status"] in {"failed", "blocked"} for row in records))


def run_cli(argv=None, *, prog=None):
    try:
        return main(argv, prog=prog)
    except (OSError, ValueError) as error:
        print(f"pipeline: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(run_cli())
