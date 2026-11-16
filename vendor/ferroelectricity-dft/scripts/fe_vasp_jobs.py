"""Load licensed PAWs and run independent VASP images in private scratch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Sequence
from uuid import uuid4
from xml.etree import ElementTree


INPUT_NAMES = ("INCAR", "POSCAR", "KPOINTS")
OUTPUT_NAMES = ("OUTCAR", "CONTCAR", "EIGENVAL", "OSZICAR", "vasprun.xml", "vasp.log")
MARKER_NAME = "_complete.json"
ITERATION = re.compile(r"\bIteration\s+(\d+)\(\s*(\d+)\s*\)", re.IGNORECASE)
END_DATASET = re.compile(rb"^[ \t]*End of Dataset[ \t]*(?:\r?\n|\Z)", re.MULTILINE)
TITLE = re.compile(rb"^[ \t]*TITEL\s*=\s*([^\r\n]+)", re.MULTILINE)
ENMAX = re.compile(rb"\bENMAX\s*=\s*([+\-\d.EeDd]+)")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_potcar(path: Path) -> bytes:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as compressed:
            return compressed.read()
    return path.read_bytes()


def _datasets(data: bytes) -> list[bytes]:
    datasets = []
    start = 0
    for end in END_DATASET.finditer(data):
        datasets.append(data[start:end.end()])
        start = end.end()
    if not datasets or data[start:].strip():
        raise ValueError("POTCAR contains an incomplete dataset or lacks End of Dataset")
    return datasets


def _potential_metadata(dataset: bytes, element: str, variant: str | None) -> dict:
    titles = TITLE.findall(dataset)
    energy = ENMAX.search(dataset)
    if len(titles) != 1 or energy is None:
        raise ValueError(f"POTCAR dataset for {element} needs one TITEL and a valid ENMAX")
    title = titles[0].decode("ascii").strip()
    fields = title.split()
    if len(fields) < 2 or fields[0] != "PAW_PBE":
        raise ValueError(f"POTCAR dataset for {element} is not PAW_PBE")
    used_variant = fields[1]
    if used_variant.split("_", 1)[0] != element or (variant and variant != used_variant):
        raise ValueError(f"POTCAR expected {variant or element}, found {used_variant}")
    enmax = float(energy[1].decode("ascii").replace("D", "E").replace("d", "e"))
    if not math.isfinite(enmax) or enmax <= 0:
        raise ValueError(f"POTCAR ENMAX for {element} must be positive and finite")
    return {
        "element": element,
        "variant": used_variant,
        "titel": title,
        "enmax_ev": enmax,
        "sha256": _sha_bytes(dataset),
    }


def _potential_file(paw_dir: Path, element: str, override: str | None) -> tuple[str, Path]:
    candidates = [override] if override else [element, f"{element}_sv", f"{element}_pv", f"{element}_d"]
    for variant in candidates:
        for filename in ("POTCAR", "POTCAR.gz"):
            path = paw_dir / variant / filename
            if path.is_file():
                return variant, path
    raise FileNotFoundError(f"No local POTCAR or POTCAR.gz for {', '.join(candidates)} under {paw_dir}")


def load_potentials(
    species: Sequence[str], paw_dir: Path | None, variants: dict[str, str], potcar: Path | None
) -> tuple[bytes, float, list[dict]]:
    """Read uncompressed POTCAR bytes, maximum ENMAX, and dataset provenance in POSCAR order."""
    if (paw_dir is None) == (potcar is None):
        raise ValueError("Specify exactly one PAW library directory or combined POTCAR")
    if not species or any(not re.fullmatch(r"[A-Z][a-z]?", element) for element in species):
        raise ValueError("Species must be chemical element symbols in POSCAR order")
    if set(variants) - set(species):
        raise ValueError("PAW variants contain a species missing from the POSCAR")
    for element, variant in variants.items():
        if not re.fullmatch(rf"{element}(?:_[A-Za-z0-9]+)*", variant):
            raise ValueError(f"Invalid PAW variant for {element}: {variant!r}")

    if potcar is not None:
        data = _read_potcar(potcar)
        datasets = _datasets(data)
        if len(datasets) != len(species):
            raise ValueError(f"POTCAR has {len(datasets)} datasets for {len(species)} POSCAR species")
        metadata = [
            _potential_metadata(dataset, element, variants.get(element))
            for element, dataset in zip(species, datasets)
        ]
    else:
        blocks, metadata = [], []
        for element in species:
            variant, path = _potential_file(paw_dir, element, variants.get(element))
            dataset = _read_potcar(path)
            pieces = _datasets(dataset)
            if len(pieces) != 1:
                raise ValueError(f"Expected a single potential dataset in {path}")
            metadata.append(_potential_metadata(pieces[0], element, variant))
            blocks.append(dataset if dataset.endswith(b"\n") else dataset + b"\n")
        data = b"".join(blocks)
    return data, max(entry["enmax_ev"] for entry in metadata), metadata


def _incar_tags(data: bytes) -> dict[str, str]:
    tags = {}
    for line in data.decode("utf-8-sig").splitlines():
        for field in re.split(r"[#!]", line, maxsplit=1)[0].split(";"):
            key, separator, value = field.partition("=")
            if separator:
                tags[key.strip().upper()] = value.strip()
    return tags


def _validation_error(directory: Path, incar: dict[str, str]) -> str | None:
    for name in OUTPUT_NAMES:
        target = directory / name
        if not target.is_file() or target.stat().st_size == 0:
            return f"missing or empty {name}"

    timing = False
    ionic_converged = False
    electronic_converged = False
    electronic_started = False
    with (directory / "OUTCAR").open(encoding="utf-8", errors="replace") as outcar:
        for line in outcar:
            iteration = ITERATION.search(line)
            if iteration and iteration[2] == "1":
                if electronic_started and not electronic_converged:
                    return "an electronic minimization ended before reaching EDIFF"
                electronic_started = True
                electronic_converged = False
            normalized = line.lower()
            if "ediff is reached" in normalized:
                electronic_converged = True
            if "general timing and accounting" in normalized:
                timing = True
            if "reached required accuracy" in normalized:
                ionic_converged = True
            if "hard stop was set" in normalized or "soft stop was set" in normalized:
                return "VASP terminated on a stop request"
    if not timing or not electronic_started or not electronic_converged:
        return "OUTCAR lacks normal termination or its last electronic step did not reach EDIFF"
    is_relaxation = int(incar.get("NSW", "0")) > 0 and int(incar.get("IBRION", "-1")) in (1, 2, 3)
    if is_relaxation and not ionic_converged:
        return "ionic minimization did not reach EDIFFG"

    final_position = False
    try:
        for _, element in ElementTree.iterparse(directory / "vasprun.xml"):
            if element.tag == "structure" and element.get("name") == "finalpos":
                final_position = True
            root = element
            element.clear()
    except ElementTree.ParseError:
        return "vasprun.xml is incomplete or malformed"
    if root.tag != "modeling" or not final_position:
        return "vasprun.xml lacks the final structure"
    return None


@dataclass(frozen=True)
class _Run:
    work_root: Path
    potcar: bytes
    potential_species: tuple[str, ...]
    command: tuple[str, ...]
    environment: dict[str, str]
    provenance: dict
    stop: threading.Event
    reuse: bool


@contextmanager
def _image_lock(image: Path):
    # VASP runs under Linux. flock releases the lock if the runner exits.
    import fcntl

    with (image / ".vasp-job.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another runner holds the lock for {image}") from None
        yield


def _input_hashes(inputs: dict[str, bytes]) -> dict[str, str]:
    return {name: _sha_bytes(data) for name, data in inputs.items()}


def _checked_fingerprint(image: Path, input_hashes: dict[str, str], incar: dict[str, str]) -> dict:
    marker = json.loads((image / MARKER_NAME).read_text(encoding="utf-8"))
    fingerprint = marker["fingerprint"]
    if marker["schema"] != 1 or fingerprint["inputs"] != input_hashes:
        raise ValueError("completion marker does not match the current inputs")
    for name in OUTPUT_NAMES:
        if marker["outputs"][name] != _sha_file(image / name):
            raise ValueError(f"{name} does not match the completion marker")
    error = _validation_error(image, incar)
    if error is not None:
        raise ValueError(error)
    return fingerprint


def _reusable(image: Path, fingerprint: dict, incar: dict[str, str]) -> bool:
    try:
        return _checked_fingerprint(image, fingerprint["inputs"], incar) == fingerprint
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _archive_outputs(image: Path) -> None:
    old_files = [image / name for name in (MARKER_NAME, *OUTPUT_NAMES) if (image / name).exists()]
    if old_files:
        archive = image / ".vasp-attempts" / f"previous-{uuid4().hex[:12]}"
        archive.mkdir(parents=True)
        for old_file in old_files:
            old_file.replace(archive / old_file.name)


def _publish(work: Path, image: Path, fingerprint: dict, elapsed: float) -> None:
    with tempfile.TemporaryDirectory(prefix=".vasp-publish-", dir=image) as staging_dir:
        staging = Path(staging_dir)
        output_hashes = {}
        for name in OUTPUT_NAMES:
            shutil.copyfile(work / name, staging / name)
            output_hashes[name] = _sha_file(staging / name)
        marker = {"schema": 1, "fingerprint": fingerprint, "outputs": output_hashes, "seconds": elapsed}
        (staging / MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        for name in (*OUTPUT_NAMES, MARKER_NAME):
            (staging / name).replace(image / name)


def _preserve_failure(work: Path, image: Path) -> Path:
    failed = image / ".vasp-attempts" / f"failed-{uuid4().hex[:12]}"
    failed.mkdir(parents=True)
    for name in (*INPUT_NAMES, *OUTPUT_NAMES):
        if (work / name).is_file():
            shutil.copyfile(work / name, failed / name)
    return failed


def _execute(run: _Run, work: Path) -> int:
    with (work / "vasp.log").open("wb") as log:
        with subprocess.Popen(
            run.command, cwd=work, env=run.environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        ) as process:
            while not run.stop.is_set():
                try:
                    return process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise RuntimeError("VASP run interrupted")


def _run_image(run: _Run, image: Path) -> None:
    with _image_lock(image):
        if run.stop.is_set():
            return
        inputs = {name: (image / name).read_bytes() for name in INPUT_NAMES}
        if any(not data.strip() for data in inputs.values()):
            raise ValueError(f"Empty INCAR, POSCAR, or KPOINTS in {image}")
        poscar = inputs["POSCAR"].decode("utf-8-sig").splitlines()
        if len(poscar) < 8 or tuple(poscar[5].split()) != run.potential_species:
            raise ValueError(f"POSCAR species order differs from the loaded POTCAR in {image}")
        incar = _incar_tags(inputs["INCAR"])
        fingerprint = {**run.provenance, "inputs": _input_hashes(inputs)}
        if run.reuse and _reusable(image, fingerprint, incar):
            print(f"Reusing {image}", flush=True)
            return

        _archive_outputs(image)
        with tempfile.TemporaryDirectory(prefix="image-", dir=run.work_root) as work_dir:
            work = Path(work_dir)
            for name, data in inputs.items():
                (work / name).write_bytes(data)
            (work / "POTCAR").write_bytes(run.potcar)
            started = time.monotonic()
            print(f"Starting {image}; live log: {work / 'vasp.log'}", flush=True)
            try:
                status = _execute(run, work)
                if status != 0:
                    raise RuntimeError(f"VASP exited with status {status}")
                error = _validation_error(work, incar)
                if error is not None:
                    raise RuntimeError(error)
                if any((image / name).read_bytes() != inputs[name] for name in INPUT_NAMES):
                    raise RuntimeError("Inputs changed while VASP was running")
                elapsed = round(time.monotonic() - started, 3)
                _publish(work, image, fingerprint, elapsed)
            except (OSError, ValueError, RuntimeError) as error:
                failed = _preserve_failure(work, image)
                raise RuntimeError(f"{image}: {error}. Attempt files: {failed}") from error
        print(f"Completed {image} in {elapsed:.1f} s", flush=True)


def _command_path(name: str) -> Path:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Executable not found or not executable: {name}")
    return Path(path).absolute()


def _image_paths(run_root: Path, image_names: list[str]) -> list[Path]:
    if not image_names or len(set(image_names)) != len(image_names):
        raise ValueError("Specify at least one image and no duplicate image names")
    root = run_root.resolve(strict=True)
    images = []
    for name in image_names:
        if name != "." and (not name or Path(name).name != name or name == ".."):
            raise ValueError(f"Expected a single image directory name, got {name!r}")
        image = root / name
        if not image.is_dir() or image.is_symlink():
            raise ValueError(f"Image must be a directory inside the run root: {image}")
        images.append(image)
    return images


def verify_results(run_root: Path, image_names: list[str]) -> None:
    """Verify that saved VASP outputs are complete, paired to their inputs, and from the same setup."""
    common_provenance = None
    for image in _image_paths(run_root, image_names):
        try:
            inputs = {name: (image / name).read_bytes() for name in INPUT_NAMES}
            incar = _incar_tags(inputs["INCAR"])
            fingerprint = _checked_fingerprint(image, _input_hashes(inputs), incar)
            provenance = {key: value for key, value in fingerprint.items() if key != "inputs"}
            if common_provenance is not None and common_provenance != provenance:
                raise ValueError("executable, launcher, parallel settings, or POTCAR differ across images")
            common_provenance = provenance
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RuntimeError(f"{image}: {error}. Run the affected images with the same setup before analysis.") from error


def run_jobs(
    run_root: Path, image_names: list[str], potcar_data: bytes, vasp_exe: str,
    launcher: list[str], ranks: int, jobs: int, reuse: bool = True,
) -> None:
    """Run named image directories under WSL, using an atomic completion marker for each image."""
    if ranks < 1 or jobs < 1:
        raise ValueError("Specify positive ranks and jobs")
    images = _image_paths(run_root, image_names)

    potential_info = []
    for dataset in _datasets(potcar_data):
        title = TITLE.search(dataset)
        fields = title[1].split() if title is not None else []
        if len(fields) < 2:
            raise ValueError("POTCAR dataset lacks a usable TITEL")
        element = fields[1].decode("ascii").split("_", 1)[0]
        potential_info.append(_potential_metadata(dataset, element, None))

    vasp = _command_path(vasp_exe)
    launch = [_command_path(launcher[0])] if launcher else []
    command = [str(path) for path in launch] + [token.replace("{ranks}", str(ranks)) for token in launcher[1:]]
    command.append(str(vasp))
    environment = os.environ.copy()
    thread_variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    for variable in thread_variables:
        environment.setdefault(variable, "1")
    provenance = {
        "command": command,
        "ranks": ranks,
        "executables": {str(path): _sha_file(path) for path in [*launch, vasp]},
        "threads": {variable: environment[variable] for variable in thread_variables},
        "potcar_sha256": _sha_bytes(potcar_data),
    }
    stop = threading.Event()
    failures = []
    with tempfile.TemporaryDirectory(prefix="fe-vasp-", dir="/tmp") as scratch:
        run = _Run(
            work_root=Path(scratch),
            potcar=potcar_data,
            potential_species=tuple(item["element"] for item in potential_info),
            command=tuple(command),
            environment=environment,
            provenance=provenance,
            stop=stop,
            reuse=reuse,
        )
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = [pool.submit(_run_image, run, image) for image in images]
            try:
                for result in as_completed(pending):
                    try:
                        result.result()
                    except (OSError, ValueError, RuntimeError) as error:
                        failures.append(str(error))
            except KeyboardInterrupt:
                stop.set()
                raise
    if failures:
        raise RuntimeError("\n".join(failures))
