"""Analyze static, closed-shell VASP images along a fixed-cell polar path."""

import json
import math
import os
import tempfile
from pathlib import Path

import polarization_path as pol
import vasp_collect_batio3 as vasp


POLARIZATION_TOL = 0.01  # C/m²
ENERGY_TOL = 0.001  # eV/cell
GAP_SYMMETRY_TOL = 0.01  # eV
COORDINATE_TOL = 1e-9


class AnalysisError(ValueError):
    """An analysis failure with a code suitable for a case status record."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def read_outputs(run_root, manifest, lattice, min_gap, occ_tol):
    raw = []
    for name, coordinate in manifest:
        image_dir = run_root / name
        try:
            axes, _, polarization, energy, nelect = vasp.read_outcar(image_dir / "OUTCAR")
            if not all(
                math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                for out_axis, input_axis in zip(axes, lattice)
                for a, b in zip(out_axis, input_axis)
            ):
                raise ValueError(f"{image_dir / 'OUTCAR'}: lattice differs from case lattice")
            gap = vasp.read_gap(image_dir / "EIGENVAL", nelect, occ_tol)
        except ValueError as error:
            message = str(error)
            if "sampled fundamental gap must be positive" in message:
                raise AnalysisError("nonpositive_gap", message) from error
            if ": occupation " in message or "first occupied band is not full" in message:
                raise AnalysisError(
                    "unsupported_occupations", f"{message}. Check smearing, k-point mesh and insulation"
                ) from error
            raise AnalysisError("invalid_output", message) from error
        if gap <= min_gap:
            raise AnalysisError(
                "gap_below_threshold",
                f"{image_dir}: sampled gap {gap:g} eV must exceed {min_gap:g} eV at lambda={coordinate:g}",
            )
        raw.append((coordinate, gap, *polarization, energy))
    return raw


def branch_metrics(manifest, unwrapped, basis, warnings):
    dual = pol.dual_basis(basis)
    shortest = pol.shortest_quantum(basis, dual)
    steps = []
    for (start, _), (end, _), left, right in zip(manifest, manifest[1:], unwrapped, unwrapped[1:]):
        target = pol.subtract(left.polarization, right.image.polarization)
        (step, _), (runner_up, _) = pol.closest_branches(target, basis, dual)
        steps.append({
            "from_image": start,
            "to_image": end,
            "branch_shift": right.branch,
            "raw_step_c_m2": math.hypot(*pol.subtract(right.image.polarization, left.image.polarization)),
            "step_c_m2": step,
            "runner_up_step_c_m2": runner_up,
        })
    return {
        "quantum_basis_c_m2": basis,
        "shortest_quantum_c_m2": shortest,
        "maximum_step_c_m2": max(s["step_c_m2"] for s in steps),
        "warnings": warnings,
        "steps": steps,
    }


def polarization_metrics(manifest, unwrapped, physical_basis, centrosymmetric, inversion_pair):
    first, last = unwrapped[0], unwrapped[-1]
    delta = pol.subtract(last.polarization, first.polarization)
    reference = min(
        ((name, row) for (name, coord), row in zip(manifest, unwrapped) if abs(coord) <= COORDINATE_TOL),
        key=lambda item: abs(item[1].image.coordinate),
        default=None,
    )
    observables = {
        "endpoint_delta_c_m2": delta,
        "reference": None,
        "inversion_switch_delta_c_m2": None,
    }
    symmetry = {
        "centrosymmetric_reference_declared": centrosymmetric,
        "inversion_pair_declared": inversion_pair,
    }
    warnings = []
    dual = pol.dual_basis(physical_basis)
    if reference is not None:
        name, center = reference
        observables["reference"] = {
            "image": name,
            "lambda": center.image.coordinate,
            "first_minus_ref_c_m2": pol.subtract(first.polarization, center.polarization),
            "last_minus_ref_c_m2": pol.subtract(last.polarization, center.polarization),
        }
        if centrosymmetric:
            target = tuple(2 * x for x in center.polarization)
            (residual, _), _ = pol.closest_branches(target, physical_basis, dual)
            symmetry["reference_twice_polarization_mod_quantum_residual_c_m2"] = residual
            symmetry["centrosymmetric_reference_passed"] = residual <= POLARIZATION_TOL
            if residual > POLARIZATION_TOL:
                warnings.append("The declared centrosymmetric reference has 2P off the polarization lattice.")
    elif centrosymmetric:
        warnings.append("The declared centrosymmetric reference needs a lambda=0 image.")

    if inversion_pair:
        target = tuple(x + y for x, y in zip(first.polarization, last.polarization))
        (residual, _), _ = pol.closest_branches(target, physical_basis, dual)
        energy_diff = abs(first.image.energy - last.image.energy)
        gap_diff = abs(first.image.gap - last.image.gap)
        endpoints_passed = (
            residual <= POLARIZATION_TOL and energy_diff <= ENERGY_TOL and gap_diff <= GAP_SYMMETRY_TOL
        )
        symmetry.update({
            "endpoints_polarization_sum_mod_quantum_residual_c_m2": residual,
            "endpoints_energy_difference_ev_cell": energy_diff,
            "endpoints_gap_difference_ev": gap_diff,
            "inversion_pair_passed": endpoints_passed,
        })
        if not endpoints_passed:
            warnings.append("The declared inversion endpoints differ beyond the polarization, energy or gap tolerances.")

        if reference is not None:
            _, center = reference
            centered_residual = math.hypot(*pol.subtract(target, tuple(2 * x for x in center.polarization)))
            symmetry["endpoints_centered_polarization_sum_norm_c_m2"] = centered_residual
            if centered_residual > POLARIZATION_TOL:
                warnings.append("Endpoint polarizations do not cancel relative to the lambda=0 branch.")
            if (
                symmetry.get("centrosymmetric_reference_passed")
                and endpoints_passed
                and centered_residual <= POLARIZATION_TOL
                and first.image.coordinate < center.image.coordinate < last.image.coordinate
            ):
                observables["inversion_switch_delta_c_m2"] = delta
    return observables, symmetry, warnings


def save_summary(path: Path, summary: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def analyze(
    run_root: Path,
    lattice,
    *,
    min_gap: float = 0.0,
    occ_tol: float = 0.001,
    quantum_multiple: int = 1,
    centrosymmetric: bool = False,
    inversion_pair: bool = False,
) -> dict:
    """Write path.csv, unwrapped.csv and summary.json for an image directory.

    The supplied lattice contains three Cartesian real-space vectors in angstroms.
    Flags for structural symmetry declare hypotheses to check numerically.
    """
    summary_path = run_root / "summary.json"
    summary_path.unlink(missing_ok=True)
    lattice = tuple(tuple(float(x) for x in axis) for axis in lattice)
    if len(lattice) != 3 or any(len(axis) != 3 for axis in lattice):
        raise ValueError("lattice must contain three Cartesian lattice vectors")
    if not all(math.isfinite(x) for axis in lattice for x in axis):
        raise ValueError("lattice vectors must be finite")
    if not math.isfinite(min_gap) or min_gap < 0 or not 0 <= occ_tol < 0.5:
        raise ValueError("min_gap must be finite and nonnegative, occ_tol must be in [0, 0.5)")
    if type(quantum_multiple) is not int or quantum_multiple < 1:
        raise ValueError("quantum_multiple must be a positive integer")

    manifest = vasp.read_manifest(run_root / "images.csv")
    raw = read_outputs(run_root, manifest, lattice, min_gap, occ_tol)
    images = [pol.Image(coordinate, gap, (px, py, pz), energy) for coordinate, gap, px, py, pz, energy in raw]
    physical_basis = pol.quantum_basis(lattice, 1)
    branch_basis = pol.quantum_basis(lattice, quantum_multiple)
    try:
        continuous, branch_warnings = pol.unwrap(images, branch_basis)
        branch = branch_metrics(manifest, continuous, branch_basis, branch_warnings)
    except (ArithmeticError, ValueError) as error:
        raise AnalysisError("branch_unresolved", str(error)) from error
    polarization, symmetry, symmetry_warnings = polarization_metrics(
        manifest, continuous, physical_basis, centrosymmetric, inversion_pair
    )
    warnings = branch_warnings + symmetry_warnings
    if branch_warnings:
        polarization["inversion_switch_delta_c_m2"] = None

    minimum_image, minimum = min(zip(manifest, images), key=lambda pair: pair[1].gap)
    maximum_image, maximum = max(zip(manifest, images), key=lambda pair: pair[1].energy)
    first_energy, last_energy = images[0].energy, images[-1].energy
    summary = {
        "status": "review" if warnings else "passed",
        "image_count": len(images),
        "lattice_angstrom": lattice,
        "warnings": warnings,
        "checks": {
            "static_and_electronically_converged": True,
            "fixed_lattice": True,
            "occupations_passed": True,
            "occupation_tolerance": occ_tol,
            "gap_threshold_passed": True,
            "branch_warnings_absent": not branch_warnings,
            "symmetry": symmetry,
            "symmetry_tolerances": {
                "polarization_c_m2": POLARIZATION_TOL,
                "energy_ev_cell": ENERGY_TOL,
                "gap_ev": GAP_SYMMETRY_TOL,
            },
        },
        "gap": {
            "minimum_image": minimum_image[0],
            "minimum_lambda": minimum.coordinate,
            "minimum_ev": minimum.gap,
            "required_above_ev": min_gap,
        },
        "branch": {"quantum_multiple": quantum_multiple, **branch},
        "polarization": polarization,
        "energy": {
            "maximum_image": maximum_image[0],
            "maximum_lambda": maximum.coordinate,
            "maximum_energy_ev_cell": maximum.energy,
            "above_first_ev_cell": maximum.energy - first_energy,
            "above_last_ev_cell": maximum.energy - last_energy,
            "above_endpoint_mean_ev_cell": maximum.energy - (first_energy + last_energy) / 2,
        },
    }
    vasp.save_csv(run_root / "path.csv", raw)
    pol.output_csv(run_root / "unwrapped.csv", continuous)
    save_summary(summary_path, summary)
    return summary
