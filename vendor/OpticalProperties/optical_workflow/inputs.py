"""Build fixed-geometry VASP independent-particle optics inputs."""

from __future__ import annotations

import gzip
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from zlib import error as ZlibError


_PROFILES = {"screen": (0.36, 0.20), "standard": (0.24, 0.15), "dense": (0.14, 0.10)}
_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn "
    "Nh Fl Mc Lv Ts Og".split()
)
_SETTINGS = {
    "profile", "kspacing", "kmesh", "encut", "nbands", "cshift", "sigma", "omegamax",
    "nedos", "ediff", "nelm", "dimension", "vacuum_axes", "metal", "drude_gamma",
    "kpar", "ncore", "component", "potentials", "incar",
}
_POTCAR_END = re.compile(rb"^[ \t]*End of Dataset[ \t]*(?:\r?\n|$)", re.M | re.I)
_VASP_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?"


@dataclass
class InputSet:
    poscar: bytes
    potcar: bytes
    incar: str
    kpoints: str
    metadata: dict


@dataclass
class _Structure:
    species: list[str]
    counts: list[int]
    lattice: list[list[float]]
    volume: float


def _number(name: str, value: object, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    return result


def _positive(name: str, value: object) -> float:
    result = _number(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _tokens(line: str) -> list[str]:
    return re.split(r"[!#]", line, maxsplit=1)[0].split()


def _real_tokens(tokens: list[str], label: str) -> list[float]:
    if any(not re.fullmatch(_VASP_NUMBER, token) for token in tokens):
        raise ValueError(f"{label} contains an invalid number")
    values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} contains a non-finite number")
    return values


def _cross(u: list[float], v: list[float]) -> list[float]:
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def _cell_det(lattice: list[list[float]]) -> float:
    return sum(x * y for x, y in zip(lattice[0], _cross(lattice[1], lattice[2])))


def _parse_poscar(content: bytes) -> _Structure:
    lines = content.decode("utf-8-sig", errors="replace").splitlines()
    if len(lines) < 9:
        raise ValueError("POSCAR is incomplete")
    scale = _real_tokens(_tokens(lines[1]), "POSCAR scale")
    if len(scale) not in (1, 3) or (len(scale) == 3 and min(scale) <= 0):
        raise ValueError("POSCAR scale requires one nonzero number or three positive numbers")
    if len(scale) == 1 and scale[0] == 0:
        raise ValueError("POSCAR scale cannot be zero")
    lattice = [_real_tokens(_tokens(line), "POSCAR lattice") for line in lines[2:5]]
    if any(len(vector) != 3 for vector in lattice):
        raise ValueError("Each POSCAR lattice vector must have three coordinates")
    unscaled_volume = _cell_det(lattice)
    if not math.isfinite(unscaled_volume) or unscaled_volume <= 0:
        raise ValueError("POSCAR lattice must be right-handed with a finite positive volume")
    if len(scale) == 1:
        factor = scale[0]
        if factor < 0:
            factor = (-factor / unscaled_volume) ** (1 / 3)
        scale = [factor] * 3
    lattice = [[x * s for x, s in zip(row, scale)] for row in lattice]
    volume = _cell_det(lattice)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("Scaled POSCAR lattice must have a finite positive volume")

    species = _tokens(lines[5])
    count_tokens = _tokens(lines[6])
    if not species or any(symbol not in _ELEMENTS for symbol in species):
        raise ValueError("POSCAR must list element symbols in VASP 5/6 format")
    if len(species) != len(count_tokens) or any(
        not re.fullmatch(r"[0-9]+", token) for token in count_tokens
    ):
        raise ValueError("POSCAR species and ion-count lists must match")
    counts = [int(token) for token in count_tokens]
    if not all(counts):
        raise ValueError("POSCAR ion counts must be positive")

    mode_index = 7
    selective = lines[mode_index].strip().lower().startswith("s")
    if selective:
        mode_index += 1
    mode = lines[mode_index].strip().lower()
    if not mode.startswith(("d", "c", "k")):
        raise ValueError("POSCAR requires a Direct or Cartesian coordinate mode")
    if len(lines) < mode_index + 1 + sum(counts):
        raise ValueError("POSCAR contains fewer coordinate rows than its ion counts")
    for index, line in enumerate(lines[mode_index + 1 : mode_index + 1 + sum(counts)], 1):
        fields = _tokens(line)
        if len(fields) < 3:
            raise ValueError(f"POSCAR ion {index} has fewer than three coordinates")
        _real_tokens(fields[:3], f"POSCAR ion {index}")
        if selective and (len(fields) < 6 or any(flag.upper() not in {"T", "F"} for flag in fields[3:6])):
            raise ValueError(f"POSCAR ion {index} needs three selective-dynamics flags")
    return _Structure(species, counts, lattice, volume)


def _read_potcar(path: Path) -> bytes:
    contents = path.read_bytes()
    if contents.startswith(b"\x1f\x8b") or path.suffix.lower() == ".gz":
        try:
            return gzip.decompress(contents)
        except (OSError, EOFError, ZlibError) as exc:
            raise ValueError(f"Invalid gzip POTCAR at {path}") from exc
    return contents


def _potential_path(root: Path, variant: str) -> Path:
    directory = root / variant
    for candidate in (directory / "POTCAR", directory / "POTCAR.gz"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No POTCAR or POTCAR.gz in {directory}")


def _datasets(contents: bytes) -> list[bytes]:
    datasets = []
    offset = 0
    for match in _POTCAR_END.finditer(contents):
        datasets.append(contents[offset : match.end()])
        offset = match.end()
    if not datasets or contents[offset:].strip():
        raise ValueError("POTCAR must contain complete datasets ending in 'End of Dataset'")
    return datasets


def _potential_metadata(
    dataset: bytes, expected_species: str, expected_variant: str | None,
) -> dict:
    header = dataset.decode("latin-1")
    title = re.search(r"^[ \t]*TITEL[ \t]*=[ \t]*(.+)$", header, re.M)
    symbol = re.search(r"^[ \t]*VRHFIN[ \t]*=[ \t]*([A-Z][a-z]?)[ \t]*:", header, re.M)
    lexch = re.search(r"^[ \t]*LEXCH[ \t]*=[ \t]*([^\s;]+)", header, re.M)
    if title is None or symbol is None or lexch is None:
        raise ValueError("Each POTCAR dataset needs TITEL, VRHFIN and LEXCH")
    titel = title.group(1).strip()
    title_fields = titel.split()
    if len(title_fields) < 2 or title_fields[0] != "PAW_PBE" or lexch.group(1) != "PE":
        raise ValueError(f"POTCAR for {expected_species} is not PAW-PBE with LEXCH = PE")
    variant = title_fields[1]
    if symbol.group(1) != expected_species or variant.split("_", 1)[0] != expected_species:
        raise ValueError(
            f"POTCAR species/order mismatch: expected {expected_species}, "
            f"found TITEL {variant} and VRHFIN {symbol.group(1)}"
        )
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"POTCAR for {expected_species} has variant {variant}, expected {expected_variant}")
    parameters = {}
    for tag in ("ZVAL", "ENMAX"):
        match = re.search(rf"\b{tag}[ \t]*=[ \t]*({_VASP_NUMBER})(?=[ \t;\r\n]|$)", header)
        if match is None:
            raise ValueError(f"POTCAR for {variant} is missing {tag}")
        parameters[tag.lower()] = _positive(f"{variant} {tag}", _real_tokens([match.group(1)], tag)[0])
    return {
        "species": expected_species, "variant": variant, "titel": titel, "lexch": "PE",
        **parameters, "sha256": hashlib.sha256(dataset).hexdigest(),
    }


def _assemble_potcar(
    species: list[str], variants: dict, potcar: Path | None, potcar_root: Path | None,
) -> tuple[bytes, list[dict]]:
    if (potcar is None) == (potcar_root is None):
        raise ValueError("Supply exactly one of potcar or potcar_root")
    if not isinstance(variants, dict):
        raise ValueError("potentials must map element symbols to POTCAR variants")
    for element, variant in variants.items():
        if element not in species:
            raise ValueError(f"Unknown POSCAR element in potentials: {element}")
        if not isinstance(variant, str) or not re.fullmatch(
            rf"{element}(?:_[A-Za-z0-9][A-Za-z0-9_+-]*)?", variant
        ):
            raise ValueError(f"Invalid POTCAR variant for {element}")
    if potcar is not None:
        contents = _read_potcar(potcar)
        data = _datasets(contents)
        if len(data) != len(species):
            raise ValueError("POTCAR dataset count does not match the POSCAR species list")
        return contents, [
            _potential_metadata(block, element, variants.get(element))
            for block, element in zip(data, species)
        ]

    assert potcar_root is not None
    blocks = []
    metadata = []
    for element in species:
        variant = variants.get(element, element)
        block = _read_potcar(_potential_path(potcar_root, variant))
        data = _datasets(block)
        if len(data) != 1:
            raise ValueError(f"Expected exactly one POTCAR dataset for {variant}")
        metadata.append(_potential_metadata(data[0], element, variant))
        blocks.append(block if block.endswith(b"\n") else block + b"\n")
    return b"".join(blocks), metadata


def _vacuum_axes(dimension: int, axes: object) -> list[int]:
    if not 0 <= dimension <= 3:
        raise ValueError("dimension must be 0, 1, 2 or 3")
    if axes is None:
        if dimension == 3:
            return []
        if dimension == 0:
            return [1, 2, 3]
        raise ValueError("vacuum_axes are required for dimension 1 or 2")
    if not isinstance(axes, (tuple, list)):
        raise ValueError("vacuum_axes must be a list of 1-based lattice axes")
    values = [_integer("vacuum axis", axis) for axis in axes]
    if set(values) - {1, 2, 3} or len(values) != 3 - dimension or len(values) != len(set(values)):
        raise ValueError(f"dimension {dimension} requires {3 - dimension} distinct vacuum axes from 1, 2, 3")
    return sorted(values)


def _kmesh(
    structure: _Structure, settings: dict, spacing: float, vacuum_axes: list[int],
) -> list[int]:
    if "kmesh" in settings:
        mesh = settings["kmesh"]
        if not isinstance(mesh, (tuple, list)) or len(mesh) != 3:
            raise ValueError("kmesh requires three positive integer mesh sizes")
        result = [_integer("kmesh size", value, minimum=1) for value in mesh]
        if any(result[axis - 1] != 1 for axis in vacuum_axes):
            raise ValueError("Each vacuum axis requires a kmesh size of 1")
        return result
    lattice = structure.lattice
    result = []
    for axis in range(3):
        if axis + 1 in vacuum_axes:
            result.append(1)
            continue
        area = math.hypot(*_cross(lattice[(axis + 1) % 3], lattice[(axis + 2) % 3]))
        ratio = (2 * math.pi / spacing) * (area / structure.volume)
        if not math.isfinite(ratio):
            raise ValueError("kspacing and POSCAR lattice produce a non-finite k-point mesh")
        result.append(max(1, math.ceil(ratio)))
    return result


def _parallel(ranks: int, full_kpoints: int, nions: int, settings: dict) -> tuple[int, int]:
    ncore = _integer("ncore", settings.get("ncore", 1), minimum=1)
    if ranks % ncore:
        raise ValueError("ncore must divide ranks")
    if "kpar" in settings:
        kpar = _integer("kpar", settings["kpar"], minimum=1)
    else:
        # Small cells benefit from more independent k-point groups. Keep more
        # ranks per group for larger cells and reduce KPAR on coarse meshes.
        rank_limit = ranks if nions <= 8 else max(1, ranks // 4)
        limit = min(8, rank_limit, max(1, full_kpoints // 16))
        kpar = next(value for value in range(limit, 0, -1) if ranks % (value * ncore) == 0)
    if ranks % (kpar * ncore) or kpar > full_kpoints:
        raise ValueError("kpar*ncore must divide ranks, and kpar cannot exceed the full k-point count")
    return kpar, ncore


def _ion_values(tag: str, values: object, nions: int) -> str:
    if isinstance(values, str):
        values = values.split()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{tag} requires a list of values")
    emitted = []
    count = 0
    for value in values:
        if isinstance(value, str):
            match = re.fullmatch(rf"(?:(\d+)\*)?({_VASP_NUMBER})", value)
            if match is None or (match.group(1) and int(match.group(1)) < 1):
                raise ValueError(f"{tag} contains an invalid value: {value!r}")
            repeat = int(match.group(1) or "1")
            if tag == "LDAUL" and not re.fullmatch(r"[+-]?\d+", match.group(2)):
                raise ValueError("LDAUL values must be integers")
            number = _real_tokens([match.group(2)], tag)[0]
        else:
            repeat = 1
            number = _integer(tag, value) if tag == "LDAUL" else _number(tag, value)
        count += repeat
        value_string = str(int(number)) if tag == "LDAUL" else str(number)
        emitted.append(f"{repeat}*{value_string}" if repeat > 1 else value_string)
    if count != nions:
        raise ValueError(f"{tag} requires {nions} values, including any repetitions")
    return " ".join(emitted)


def _extra_incar(settings: object, structure: _Structure) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("incar must be an object mapping INCAR tags to values")
    extra = {}
    for name, value in settings.items():
        if not isinstance(name, str):
            raise ValueError("Extra INCAR tag names must be strings")
        tag = name.upper()
        if tag in extra:
            raise ValueError(f"Duplicate extra INCAR tag: {tag}")
        if tag in {"LASPH", "ADDGRID", "LDAU", "GGA_COMPAT"}:
            value = _boolean(tag, value)
        elif tag == "LREAL":
            if isinstance(value, str) and value.lower() == "auto":
                value = "Auto"
            else:
                value = _boolean(tag, value)
        elif tag == "ALGO":
            if not isinstance(value, str) or value.lower() not in {"normal", "fast"}:
                raise ValueError("ALGO must be Normal or Fast for a ground-state calculation")
            value = value.capitalize()
        elif tag in {"ISPIN", "ISYM", "LDAUTYPE", "LDAUPRINT"}:
            allowed = {
                "ISPIN": {1, 2}, "ISYM": {-1, 0, 1, 2, 3},
                "LDAUTYPE": {1, 2, 3, 4}, "LDAUPRINT": {0, 1, 2},
            }[tag]
            value = _integer(tag, value)
            if value not in allowed:
                raise ValueError(f"{tag} must be one of {sorted(allowed)}")
        elif tag in {"LMAXMIX", "NELMIN"}:
            value = _integer(tag, value, minimum=1)
        elif tag == "SYMPREC":
            value = _positive(tag, value)
        elif tag in {"AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG"}:
            value = _number(tag, value, minimum=0)
        elif tag == "MAGMOM":
            value = _ion_values(tag, value, sum(structure.counts))
        elif tag in {"LDAUL", "LDAUU", "LDAUJ"}:
            value = _ion_values(tag, value, len(structure.species))
        else:
            raise ValueError(f"Unsupported extra INCAR tag: {name}")
        extra[tag] = value
    return extra


def _incar_value(value: object) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    return str(value)


def build_inputs(
    poscar: Path, settings: dict, potcar: Path | None, potcar_root: Path | None, ranks: int,
) -> InputSet:
    """Build one static, scalar-relativistic PBE/PAW optics run from an ordered POSCAR."""
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    unknown = settings.keys() - _SETTINGS
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(map(str, unknown)))}")
    ranks = _integer("ranks", ranks, minimum=1)
    poscar_bytes = poscar.read_bytes()
    structure = _parse_poscar(poscar_bytes)
    extra_incar = _extra_incar(settings.get("incar", {}), structure)
    potentials = settings.get("potentials", {})
    potcar_bytes, potential_metadata = _assemble_potcar(
        structure.species, potentials, potcar, potcar_root,
    )

    profile = settings.get("profile", "screen")
    if not isinstance(profile, str) or profile not in _PROFILES:
        raise ValueError("profile must be screen, standard or dense")
    default_spacing, default_cshift = _PROFILES[profile]
    dimension = _integer("dimension", settings.get("dimension", 3))
    axes = _vacuum_axes(dimension, settings.get("vacuum_axes"))
    component = settings.get("component")
    if component is not None and component not in ("isotropic", "xx", "yy", "zz"):
        raise ValueError("component must be null, isotropic, xx, yy or zz")
    if dimension < 3 and component is not None:
        raise ValueError("Scalar optical constants require dimension 3")
    if "kmesh" in settings and "kspacing" in settings:
        raise ValueError("Supply kmesh or kspacing, rather than both")
    spacing = _positive("kspacing", settings.get("kspacing", default_spacing))
    mesh = _kmesh(structure, settings, spacing, axes)
    kpar, ncore = _parallel(ranks, math.prod(mesh), sum(structure.counts), settings)

    metal = _boolean("metal", settings.get("metal", False))
    gamma = settings.get("drude_gamma")
    if gamma is not None:
        if not metal:
            raise ValueError("drude_gamma requires metal=true")
        gamma = _positive("drude_gamma", gamma)
    cshift = _positive("cshift", settings.get("cshift", default_cshift))
    sigma = _positive("sigma", settings.get("sigma", 0.1 if metal else 0.01))
    omegamax = _positive("omegamax", settings.get("omegamax", 20.0))
    ediff = _positive("ediff", settings.get("ediff", 1e-8))
    nedos = _integer("nedos", settings.get("nedos", 2000), minimum=2)
    nelm = _integer("nelm", settings.get("nelm", 100), minimum=1)
    if extra_incar.get("NELMIN", 1) > nelm:
        raise ValueError("NELMIN cannot exceed nelm")
    enmax = max(entry["enmax"] for entry in potential_metadata)
    encut = _positive("encut", settings.get("encut", math.ceil(1.3 * enmax)))
    nelect = sum(
        count * entry["zval"] for count, entry in zip(structure.counts, potential_metadata)
    )
    occupied = math.ceil(nelect / 2)
    ispin = extra_incar.get("ISPIN", 1)
    default_bands = math.ceil(nelect) + 16 if ispin == 2 else occupied + max(16, occupied)
    nbands = _integer("nbands", settings.get("nbands", default_bands), minimum=occupied + 1)

    incar_fields = {
        "SYSTEM": "Independent-particle optics", "GGA": "PE", "PREC": "Accurate",
        "ENCUT": encut, "ALGO": "Normal", "EDIFF": ediff, "NELM": nelm,
        "ISMEAR": 0, "SIGMA": sigma, "NSW": 0, "ISTART": 0, "ICHARG": 2,
        "LOPTICS": True, "NBANDS": nbands, "CSHIFT": cshift, "NEDOS": nedos,
        "OMEGAMAX": omegamax, "LWAVE": False, "LCHARG": False,
        "KPAR": kpar, "NCORE": ncore,
    }
    if gamma is not None:
        incar_fields["WPLASMAI"] = gamma
    incar_fields.update(extra_incar)
    incar = "".join(f"{tag} = {_incar_value(value)}\n" for tag, value in incar_fields.items())
    kpoints = f"Gamma-centered optics mesh\n0\nGamma\n{' '.join(map(str, mesh))}\n0 0 0\n"
    metadata = {
        "species": structure.species, "counts": structure.counts, "nelect": nelect,
        "volume_angstrom3": structure.volume, "lattice_angstrom": structure.lattice,
        "profile": profile, "kspacing": None if "kmesh" in settings else spacing, "kmesh": mesh,
        "dimension": dimension, "vacuum_axes": axes, "metal": metal, "component": component,
        "encut": encut, "nbands": nbands, "ncore": ncore, "kpar": kpar, "ranks": ranks,
        "ispin": ispin, "cshift": cshift, "sigma": sigma, "drude_gamma": gamma,
        "omegamax": omegamax, "nedos": nedos, "ediff": ediff, "nelm": nelm,
        "potcar_sha256": hashlib.sha256(potcar_bytes).hexdigest(), "potentials": potential_metadata,
        "incar": incar_fields,
    }
    return InputSet(poscar_bytes, potcar_bytes, incar, kpoints, metadata)
