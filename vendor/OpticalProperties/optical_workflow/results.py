"""Validate and collect the output of a static VASP LOPTICS job."""

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import optics


EXTRA_SETTINGS = {"NELECT", "WPLASMAI", "LNONCOLLINEAR", "NCORE", "KPAR"}
OCCUPATION_TOLERANCE = 1e-3
NEGATIVE_EPSILON_TOLERANCE = 1e-4
DRUDE_XML_CAP = 99999.0  # Observed in VASP 6.5.0 LOPTICS Al output.
ITERATION = re.compile(r"^\s*-+\s*Iteration\s+\d+\s*\(")


def finished_scf(path):
    converged = False
    footer = False
    elapsed = None
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            if ITERATION.match(line):
                converged = False
                elapsed = None
                footer = False
            elif "aborting loop because EDIFF is reached" in line:
                converged = True
            elif "The electronic self-consistency was not achieved" in line:
                converged = False
            elif "General timing and accounting" in line:
                footer = True
            elif footer and "Elapsed time (sec):" in line:
                elapsed = float(line.rsplit(":", 1)[-1])
    if elapsed is None or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("OUTCAR is missing its final elapsed time")
    if not converged:
        raise ValueError("OUTCAR does not confirm EDIFF was reached in the final SCF")
    return elapsed


def read_electronic_info(path):
    """Keep one k-point's eigenvalue rows at a time."""
    settings = {}
    groups = []
    stack = []
    kpoint = None
    with path.open("rb") as source:
        for event, node in ET.iterparse(source, events=("start", "end")):
            if event == "start":
                if node.tag == "set" and node.get("comment", "").startswith("kpoint "):
                    parents = [parent.tag for parent in stack]
                    if parents == ["modeling", "calculation", "eigenvalues", "array", "set", "set"]:
                        kpoint = node
                if node.tag == "calculation" and len(stack) == 1:
                    groups = []
                stack.append(node)
                continue

            stack.pop()
            if kpoint is not None:
                if node is not kpoint:
                    continue
                groups.append(band_extrema(node))
                kpoint = None
            elif node.tag == "i" and len(stack) > 1 and stack[1].tag == "parameters":
                if node.get("name") in EXTRA_SETTINGS and node.text:
                    settings[node.get("name")] = node.text.strip()
            if stack:
                stack[-1].remove(node)
    return settings, groups


def band_extrema(kpoint):
    occupied = []
    empty = []
    partial = 0
    highest_occupation = None
    for row in kpoint.findall("r"):
        try:
            energy, occupation = map(float, (row.text or "").split())
        except ValueError:
            return None
        if not math.isfinite(energy) or not 0.0 <= occupation <= 1.0:
            return None
        if OCCUPATION_TOLERANCE < occupation < 1 - OCCUPATION_TOLERANCE:
            partial += 1
        (occupied if occupation >= 0.5 else empty).append(energy)
        highest_occupation = occupation
    if highest_occupation is None:
        return None
    vbm = max(occupied, default=None)
    cbm = min(empty, default=None)
    ceiling = max(empty) - vbm if vbm is not None and empty else None
    return len(occupied) + len(empty), vbm, cbm, ceiling, partial, highest_occupation


def electronic_diagnostics(groups, settings, warnings):
    try:
        expected_groups = int(settings["nk_written"]) * int(settings["ISPIN"])
        expected_bands = int(settings["NBANDS"])
    except (KeyError, TypeError, ValueError):
        expected_groups = expected_bands = None
    if (not groups or len(groups) != expected_groups
            or any(group is None or group[0] != expected_bands for group in groups)):
        warnings.append("Final XML eigenvalue table did not support occupation and empty-band checks.")
        return {}

    _, occupied, empty, ceilings, partial, highest = zip(*groups)
    valence_max = max((value for value in occupied if value is not None), default=None)
    conduction_min = min((value for value in empty if value is not None), default=None)
    gap = None
    if not sum(partial) and valence_max is not None and conduction_min is not None:
        gap = conduction_min - valence_max
    coverage = min((value for value in ceilings if value is not None), default=None)
    if sum(partial):
        warnings.append("Fractional occupations were found on the sampled mesh. Check the smearing and possible metallic response.")
    elif gap is not None and gap <= 0:
        warnings.append("Occupied and empty Kohn–Sham energies overlap on the sampled mesh.")
    if max(highest) > OCCUPATION_TOLERANCE:
        warnings.append("At least one highest computed band remains occupied. Increase NBANDS and recheck.")
    return {
        "partial_occupation_count": sum(partial),
        "max_highest_band_occupation": max(highest),
        "sampled_ks_gap_eV": gap,
        "min_empty_transition_ceiling_eV": coverage,
    }


def drude_table(block, part):
    array = block.find(f"{part}/array")
    if array is None:
        raise ValueError(f"Missing {part}/array")
    fields = [(field.text or "").strip().lower() for field in array.findall("field")]
    table = array.find("set")
    if table is None or fields.count("energy") != 1:
        raise ValueError(f"Missing {part} dielectric table or energy field")
    energy_index = fields.index("energy")
    samples = []
    for number, row in enumerate(table.findall("r"), 1):
        columns = (row.text or "").split()
        if len(columns) != len(fields):
            raise ValueError(f"Invalid {part} dielectric row {number}")
        try:
            energy = float(columns.pop(energy_index))
            values = [math.inf if set(token) == {"*"} else float(token) for token in columns]
        except ValueError as exc:
            raise ValueError(f"Invalid {part} dielectric row {number}") from exc
        if not math.isfinite(energy) or energy < 0 or (samples and energy <= samples[-1][1]):
            raise ValueError(f"The {part} energy grid must be nonnegative and strictly increasing")
        unusable = any(not math.isfinite(value) or abs(value) == DRUDE_XML_CAP for value in values)
        samples.append((row, energy, unusable))
    return table, samples


def omit_unusable_drude_prefix(block):
    """Omit a paired, contiguous overflow/cap prefix at the Drude singularity."""
    real_table, real = drude_table(block, "real")
    imag_table, imag = drude_table(block, "imag")
    if not real or not imag or len(real) != len(imag) or any(
        not math.isclose(r[1], i[1], rel_tol=0, abs_tol=1e-6) for r, i in zip(real, imag)
    ):
        raise ValueError("Real and imaginary energy grids differ or are empty")
    unusable = [r[2] or i[2] for r, i in zip(real, imag)]
    has_origin = real[0][1] == 0 and imag[0][1] == 0
    if has_origin:
        unusable[0] = True
    if not any(unusable):
        return 0, None
    if not has_origin:
        raise ValueError("Drude dielectric overflow or cap occurs outside the origin prefix")
    stop = next((index for index, invalid in enumerate(unusable) if not invalid), len(unusable))
    if any(unusable[stop:]):
        raise ValueError("Drude dielectric overflow or cap occurs after finite-energy output has resumed")
    if stop == len(unusable):
        raise ValueError("No usable dielectric rows remain after the Drude prefix")
    for real_row, imag_row in zip(real[:stop], imag[:stop]):
        real_table.remove(real_row[0])
        imag_table.remove(imag_row[0])
    return stop, real[stop - 1][1]


def collect_result(run_dir: Path, metadata: dict) -> dict:
    """Export spectrum.csv and return actual settings with numeric diagnostics."""
    elapsed = finished_scf(run_dir / "OUTCAR")
    path = run_dir / "vasprun.xml"
    try:
        settings, blocks = optics.read_run(path)
        extras, bands = read_electronic_info(path)
    except ET.ParseError as exc:
        raise ValueError(f"Incomplete or malformed {path.name}: {exc}") from exc
    settings.update(extras)
    if settings.get("LOPTICS", "").strip(".").lower() not in {"t", "true"}:
        raise ValueError("vasprun.xml does not confirm LOPTICS was enabled")
    index, block = optics.select_block(blocks, None)
    warnings = []
    drude = float(settings.get("WPLASMAI", 0)) > 0
    omitted_rows, omitted_energy = omit_unusable_drude_prefix(block) if drude else (0, None)
    real = optics.read_part(block, "real")
    imag = optics.read_part(block, "imag")
    if len(real) != len(imag) or any(not math.isclose(r[0], i[0], rel_tol=0, abs_tol=1e-6)
                                     for r, i in zip(real, imag)):
        raise ValueError("Real and imaginary energy grids differ")
    if omitted_rows:
        warnings.append(
            f"Omitted {omitted_rows} leading Drude dielectric rows from 0 to {omitted_energy:g} eV "
            f"at the singular origin or with overflow, nonfinite, or capped output. "
            f"Export starts at {real[0][0]:g} eV."
        )

    diagnostics = electronic_diagnostics(bands, settings, warnings)
    gap = diagnostics.get("sampled_ks_gap_eV")
    if not drude and (metadata.get("metal") or diagnostics.get("partial_occupation_count")
                      or (gap is not None and gap <= 0)):
        warnings.append("For low-frequency metallic response, assess the contribution from an explicit Drude term (WPLASMAI).")
    ceiling = diagnostics.get("min_empty_transition_ceiling_eV")
    if ceiling is not None and ceiling < real[-1][0]:
        warnings.append("Some sampled k/spin points have limited empty-band coverage over the output energy window. Check NBANDS for the target range.")
    negative_diagonal = sum(min(row[1:4]) < -NEGATIVE_EPSILON_TOLERANCE for row in imag)
    if negative_diagonal:
        warnings.append(f"{negative_diagonal} rows have an imaginary diagonal component below −{NEGATIVE_EPSILON_TOLERANCE:g}.")
    component = metadata.get("component") if metadata.get("dimension", 3) == 3 else None
    output = run_dir / "spectrum.csv"
    with TemporaryDirectory(prefix=".spectrum-", dir=run_dir) as directory:
        temporary = Path(directory) / output.name
        negative = optics.write_spectrum(temporary, real, imag, component)
        temporary.replace(output)
    return {
        **settings,
        **diagnostics,
        "spectrum": output.name,
        "electronic_converged": True,
        "elapsed_seconds": elapsed,
        "dielectric_block": index,
        "block_comment": block.get("comment", ""),
        "n_energy": len(real),
        "energy_min_eV": real[0][0],
        "energy_max_eV": real[-1][0],
        "derived_component": component or "",
        "negative_eps2_rows": negative if component else None,
        "negative_eps2_diagonal_rows": negative_diagonal,
        "drude_included": drude,
        "omitted_drude_origin": bool(omitted_rows),
        "omitted_drude_rows": omitted_rows,
        "omitted_drude_max_energy_eV": omitted_energy,
        "warnings": warnings,
    }
