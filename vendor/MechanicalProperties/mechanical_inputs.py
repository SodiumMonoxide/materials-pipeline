"""Prepare shared VASP inputs for a three-dimensional elastic calculation."""

from collections.abc import Mapping
import hashlib
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from ase.constraints import FixAtoms, FixScaled
from ase.io import iread
from ase.io.formats import filetype
from ase.io.vasp import write_vasp


TITLE = re.compile(rb"(?m)^[ \t]*TITEL[ \t]*=[ \t]*([^\r\n]+)")
ENMAX = re.compile(rb"\bENMAX[ \t]*=[ \t]*([0-9.+EeDd-]+)")
INCAR_TAG = re.compile(r"[A-Z][A-Z0-9_]*")


def read_structure(source, structure_format=None):
    """Load a single ordered, periodic ASE structure with a usable VASP cell."""
    source = Path(source)
    if structure_format is None:
        name = source.name.upper()
        if name in {"POSCAR", "CONTCAR"} or name.startswith(("POSCAR.", "CONTCAR.")):
            structure_format = "vasp"
        else:
            structure_format = filetype(str(source))
    options = {}
    if structure_format == "cif":
        options = {"store_tags": True, "fractional_occupancies": True}
    images = iread(source, index=":", format=structure_format, **options)
    try:
        atoms = next(images, None)
        multiple = next(images, None) is not None
    finally:
        images.close()
    if atoms is None or len(atoms) == 0 or multiple:
        raise ValueError(f"{source}: expected exactly one nonempty structure")
    cell = np.asarray(atoms.cell)
    if (
        not np.all(atoms.pbc)
        or not np.isfinite(cell).all()
        or np.linalg.matrix_rank(cell) != 3
        or np.linalg.det(cell) <= 0
    ):
        raise ValueError(f"{source}: expected a right-handed, three-dimensionally periodic cell")
    if not np.isfinite(atoms.positions).all() or np.any(atoms.numbers <= 0):
        raise ValueError(f"{source}: expected finite positions and real elements")

    occupancies = atoms.info.get("occupancy", {})
    if not isinstance(occupancies, Mapping) or any(
        not isinstance(composition, Mapping)
        or len(composition) != 1
        or not _fully_occupied(composition.values())
        for composition in occupancies.values()
    ):
        raise ValueError(f"{source}: partially occupied or mixed sites require an ordered model")
    for values in (
        atoms.info.get("_atom_site_occupancy"),
        atoms.arrays.get("occupancy"),
        atoms.arrays.get("occupancies"),
    ):
        if values is not None and not _fully_occupied(np.ravel(values)):
            raise ValueError(f"{source}: partially occupied sites require an ordered model")
    if any(not isinstance(lock, (FixAtoms, FixScaled)) for lock in atoms.constraints):
        raise ValueError(f"{source}: only whole-site and fractional-axis constraints can be preserved")
    return atoms


def _fully_occupied(values):
    try:
        return all(math.isclose(float(value), 1.0, rel_tol=0, abs_tol=1e-8) for value in values)
    except (TypeError, ValueError):
        return False


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite number greater than zero")
    return float(value)


def gamma_mesh(atoms, k_spacing):
    """Choose a Gamma mesh from reciprocal row lengths and lattice symmetry."""
    spacing = _positive_number(k_spacing, "k_spacing")
    try:
        import spglib
    except ImportError as error:
        raise ValueError("automatic k meshes require spglib, or specify an explicit k_mesh") from error

    lengths = 2 * np.pi * np.linalg.norm(atoms.cell.reciprocal(), axis=1)
    divisions = [max(1, math.ceil(length / spacing)) for length in lengths]
    lattice = (atoms.cell.array, [[0.0, 0.0, 0.0]], [1])
    symmetry = spglib.get_symmetry(lattice, symprec=1e-5)
    if symmetry is None:
        raise ValueError("spglib could not determine lattice symmetry, specify an explicit k_mesh")

    # Direct fractional rotations W induce reciprocal W^(-T). The rotation
    # group contains all inverses, so its transposes contain the same couplings.
    coupled = np.any(symmetry["rotations"] != 0, axis=0)
    coupled |= coupled.T
    np.fill_diagonal(coupled, True)
    for axis in range(3):
        coupled |= coupled[:, axis, None] & coupled[None, axis, :]
    # Equal subdivisions for coupled axes preserve every reciprocal operation
    # while meeting the spacing bound from the densest axis in each group.
    return tuple(max(divisions[axis] for axis in np.flatnonzero(group)) for group in coupled)


def _potential_data(root, element, override):
    selected = Path(element if override is None else override).expanduser()
    if not selected.is_absolute():
        selected = root / selected
    if selected.is_dir():
        selected /= "POTCAR"
    if not selected.is_file():
        raise FileNotFoundError(f"no PBE POTCAR for {element} at {selected}")
    selected = selected.resolve()
    data = selected.read_bytes()
    titles = TITLE.findall(data)
    cutoffs = ENMAX.findall(data)
    if len(titles) != 1 or len(cutoffs) != 1:
        raise ValueError(f"{selected}: expected one POTCAR dataset with TITEL and ENMAX")
    title = titles[0].decode("ascii", errors="replace").strip()
    words = title.split()
    if len(words) < 2 or not words[0].startswith("PAW_PBE") or words[1].split("_")[0] != element:
        raise ValueError(f"{selected}: expected a PAW PBE POTCAR for {element}, found {title!r}")
    try:
        cutoff = _positive_number(float(cutoffs[0].upper().replace(b"D", b"E")), "POTCAR ENMAX")
    except ValueError as error:
        raise ValueError(f"{selected}: invalid ENMAX") from error
    return data, {
        "element": element,
        "path": str(selected),
        "titel": title,
        "ENMAX_eV": cutoff,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _incar_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value):
        return float(value)
    if isinstance(value, str) and value.strip() and not any(char in value for char in "\r\n;#!"):
        return value.strip()
    raise ValueError(f"invalid INCAR value {value!r}")


def normalize_incar(settings):
    """Normalize INCAR tags and values before writing them or recording metadata."""
    normalized = {}
    for key, value in settings.items():
        if not isinstance(key, str) or not INCAR_TAG.fullmatch(key.strip().upper()):
            raise ValueError(f"invalid INCAR tag {key!r}")
        tag = key.strip().upper()
        if tag in normalized:
            raise ValueError(f"repeated INCAR tag {tag}")
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"INCAR list for {tag} must be nonempty")
            normalized[tag] = [_incar_scalar(item) for item in value]
        else:
            normalized[tag] = _incar_scalar(value)
    return normalized


def _vasp_value(value):
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, list):
        return " ".join(_vasp_value(item) for item in value)
    return str(value)


def _check_input(path, content):
    if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
        raise ValueError(f"{path}: existing input differs or is missing, use a new run directory")


def write_incar(path, settings):
    """Write a stage INCAR, accepting an identical prior file on restart."""
    path = Path(path)
    settings = normalize_incar(settings)
    content = "".join(f"{tag} = {_vasp_value(settings[tag])}\n" for tag in sorted(settings)).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".incar-") as temporary:
        staged = Path(temporary) / "INCAR"
        staged.write_bytes(content)
        try:
            os.link(staged, path)
        except FileExistsError:
            _check_input(path, content)


def _publish_inputs(directory, files):
    if directory.is_symlink():
        raise ValueError(f"{directory}: shared inputs directory must not be a symlink")
    if directory.is_dir() and any(directory.iterdir()):
        for name, content in files.items():
            _check_input(directory / name, content)
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".vasp-inputs-") as temporary:
        staged = Path(temporary) / "inputs"
        staged.mkdir()
        for name, content in files.items():
            (staged / name).write_bytes(content)
        try:
            staged.rename(directory)
        except OSError:
            if not directory.is_dir() or directory.is_symlink():
                raise
            if not any(directory.iterdir()):
                # Windows will not rename over an empty existing directory.
                directory.rmdir()
                staged.rename(directory)
            else:
                for name, content in files.items():
                    _check_input(directory / name, content)


def prepare_vasp_inputs(
    source,
    directory,
    *,
    potcar_root,
    potcar_overrides=None,
    k_spacing=0.25,
    k_mesh=None,
    encut=None,
    encut_factor=1.3,
    incar_overrides=None,
    structure_format=None,
):
    """Write immutable POSCAR, POTCAR, KPOINTS and inputs.json. Stages own INCAR."""
    source = Path(source).expanduser().resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    atoms = read_structure(source, structure_format)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f"{source}: structure changed while reading")
    symbols = atoms.get_chemical_symbols()
    groups = {}
    for site_index, symbol in enumerate(symbols):
        groups.setdefault(symbol, []).append(site_index)
    species = list(groups)
    site_indices = [index for group in groups.values() for index in group]
    overrides = {} if potcar_overrides is None else potcar_overrides
    unused = set(overrides) - set(species)
    if unused:
        raise ValueError(f"POTCAR overrides for absent elements: {', '.join(sorted(unused))}")
    records = []
    chunks = []
    pseudo = Path(potcar_root).expanduser()
    for element in species:
        data, record = _potential_data(pseudo, element, overrides.get(element))
        chunks.append(data if data.endswith(b"\n") else data + b"\n")
        records.append(record)
    potcar = b"".join(chunks)
    factor = _positive_number(encut_factor, "encut_factor")
    max_enmax = max(record["ENMAX_eV"] for record in records)
    cutoff = _positive_number(encut if encut is not None else factor * max_enmax, "ENCUT")
    spacing = None
    if k_mesh is None:
        spacing = _positive_number(k_spacing, "k_spacing")
        mesh = gamma_mesh(atoms, spacing)
    else:
        mesh = tuple(k_mesh)
        if len(mesh) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in mesh):
            raise ValueError("k_mesh must contain three positive integers")
        mesh = tuple(map(int, mesh))
    kpoints = f"Gamma-centered mesh\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    poscar = io.StringIO()
    write_vasp(poscar, atoms[site_indices], direct=True, sort=False, vasp5=True)
    metadata = {
        "structure_source": str(source),
        "structure_sha256": source_hash,
        "cell_A": atoms.cell.tolist(),
        "species": species,
        "species_counts": {symbol: len(groups[symbol]) for symbol in species},
        "site_indices": site_indices,
        "potentials": records,
        "potcar_sha256": hashlib.sha256(potcar).hexdigest(),
        "max_ENMAX_eV": max_enmax,
        "encut_eV": cutoff,
        "encut_factor": factor if encut is None else None,
        "k_mesh": list(mesh),
        "k_spacing_inv_A": spacing,
        "incar_overrides": normalize_incar({} if incar_overrides is None else incar_overrides),
    }
    _publish_inputs(Path(directory), {
        "POSCAR": poscar.getvalue().encode("utf-8"),
        "POTCAR": potcar,
        "KPOINTS": kpoints.encode("ascii"),
        "inputs.json": (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    })
    return metadata
