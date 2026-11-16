"""Read a completed VASP static, ionic relaxation, or elastic run."""

import gzip
import math
import re
from pathlib import Path


VOIGT_ORDER = ("XX", "YY", "ZZ", "YZ", "XZ", "XY")
VASP_STRESS_INDICES = (0, 1, 2, 4, 5, 3)
INTEGER_PARAMETER = re.compile(r"^\s*(IBRION|ISIF|NSW)\s*=\s*([+-]?\d+)\b")
NIONS = re.compile(r"\bNIONS\s*=\s*(\d+)\b")
ENERGY = re.compile(r"energy\(sigma->0\)\s*=\s*(\S+)")
ENCUT = re.compile(r"^\s*ENCUT\s*=\s*(\S+)")
VOLUME = re.compile(r"volume of cell\s*:\s*(\S+)")
SCF_START = re.compile(r"\bIteration\s+\d+\s*\(\s*1\s*\)")
FORCE_HEADER = re.compile(r"^\s*POSITION\s+TOTAL-FORCE\b")
STRESS = re.compile(r"^\s*in\s+kB\b(.*)$")
ELAPSED = re.compile(r"^\s*Elapsed time \(sec\)\s*:\s*(\S+)")


def vasp_float(value):
    number = float(value.replace("D", "E").replace("d", "e"))
    if not math.isfinite(number):
        raise ValueError(f"non-finite OUTCAR number: {value}")
    return number


def stress_gpa(fields):
    if len(fields) != 6:
        raise ValueError("expected six components in the OUTCAR 'in kB' stress row")
    # The OUTCAR order is XX YY ZZ XY YZ ZX, in kbar with compression positive.
    return [-0.1 * vasp_float(fields[index]) for index in VASP_STRESS_INDICES]


def read_forces(stream, nions):
    if set(next(stream, "").strip()) != {"-"}:
        raise ValueError("missing OUTCAR TOTAL-FORCE opening separator")
    max_force = 0.0
    for row_index in range(1, nions + 1):
        values = next(stream, "").split()
        if len(values) != 6:
            raise ValueError(f"invalid OUTCAR TOTAL-FORCE row for atom {row_index}")
        position_and_force = [vasp_float(value) for value in values]
        max_force = max(max_force, math.hypot(*position_and_force[3:]))
    if set(next(stream, "").strip()) != {"-"}:
        raise ValueError("missing OUTCAR TOTAL-FORCE closing separator")
    return max_force


def read_completed_run(stream, *, ionic):
    parameters = {}
    evaluation = {}
    encut = volume = elapsed = None
    footer_seen = ionic_converged = False
    scf_started = scf_converged = 0
    converged_current_scf = False
    elastic_block_seen = False
    for line in stream:
        if footer_seen:
            match = ELAPSED.match(line)
            if match:
                elapsed = vasp_float(match[1])
            continue

        match = INTEGER_PARAMETER.match(line)
        if match:
            parameters[match[1].lower()] = int(match[2])
        match = NIONS.search(line)
        if match:
            parameters["nions"] = int(match[1])
        match = ENCUT.match(line)
        if match:
            encut = vasp_float(match[1])
        match = VOLUME.search(line)
        if match:
            volume = vasp_float(match[1])

        if SCF_START.search(line):
            scf_started += 1
            converged_current_scf = False
            evaluation = {}
        if "aborting loop because EDIFF is reached" in line:
            if not scf_started or converged_current_scf:
                raise ValueError("EDIFF marker did not match an OUTCAR electronic loop")
            converged_current_scf = True
            scf_converged += 1

        match = ENERGY.search(line)
        if match:
            evaluation["energy_eV"] = vasp_float(match[1])
        match = STRESS.match(line)
        if match:
            evaluation["stress_GPa"] = stress_gpa(match[1].split())
        if FORCE_HEADER.match(line):
            nions = parameters.get("nions", 0)
            if nions <= 0:
                raise ValueError("NIONS is required to read the OUTCAR TOTAL-FORCE table")
            evaluation["max_force_eV_A"] = read_forces(stream, nions)

        ionic_converged |= "reached required accuracy" in line
        elastic_block_seen |= line.strip().casefold() == "total elastic moduli (kbar)"
        footer_seen = "General timing and accounting" in line

    if not footer_seen or elapsed is None or elapsed < 0:
        raise ValueError("OUTCAR is incomplete. The VASP timing footer or elapsed time is missing")
    if any(name not in parameters for name in ("ibrion", "isif", "nsw", "nions")):
        raise ValueError("OUTCAR is missing IBRION, ISIF, NSW, or NIONS")
    if not scf_started:
        raise ValueError("OUTCAR is missing the electronic iteration records")
    if encut is None or encut <= 0 or volume is None or volume <= 0:
        raise ValueError("OUTCAR contains no positive cutoff or cell volume")
    required = ("energy_eV", "max_force_eV_A", "stress_GPa")
    if any(field not in evaluation for field in required):
        raise ValueError("OUTCAR is missing energy, forces, or stress for the final evaluation")

    ibrion = parameters["ibrion"]
    isif = parameters["isif"]
    if isif < 2:
        raise ValueError("ISIF >= 2 is required for full stress")
    if ibrion == -1 and parameters["nsw"] == 0:
        stage = "static"
    elif ibrion in (1, 2, 3):
        stage = "relax"
    elif ibrion in (5, 6) and isif >= 3 and elastic_block_seen:
        stage = "elastic"
    else:
        raise ValueError(f"OUTCAR is not a supported stress run (IBRION={ibrion}, ISIF={isif})")
    if ionic and (stage != "relax" or not ionic_converged):
        raise ValueError("OUTCAR has no successful ionic relaxation marker")

    evaluation["volume_A3"] = volume
    result = {
        **parameters,
        "stage": stage,
        "encut_eV": encut,
        "elapsed_s": elapsed,
        "scf_started": scf_started,
        "scf_converged": scf_converged,
        "scf_all_converged": scf_started == scf_converged,
        "ionic_converged": ionic_converged if stage == "relax" else None,
    }
    if stage == "elastic":
        # The last finite difference evaluation is displaced from the reference.
        result.update(dict.fromkeys(evaluation))
        result["last_evaluation"] = evaluation
    else:
        result.update(evaluation)
    return result


def read_run(path, *, ionic=False):
    """Return phase, convergence diagnostics and the final evaluation from an OUTCAR.

    A finite-difference run places the displaced outputs in `last_evaluation`;
    its top-level energy, force, stress and volume are null. The elastic tensor
    itself can be read using `mech.analyze_outcar`.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
            return read_completed_run(stream, ionic=ionic)
    except (ValueError, EOFError, gzip.BadGzipFile) as error:
        raise ValueError(f"{path}: {error}") from error
