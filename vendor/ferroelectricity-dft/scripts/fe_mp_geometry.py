"""Classify MP structures and align plausible polar/reference pairs in one cell."""

import math
from collections import Counter
from dataclasses import replace
from itertools import permutations, product

import numpy as np
import spglib

from fe_structures import MappingError, Structure, _match, _periodic_shift, _volume
from polarization_path import dual_basis, subtract


POLAR_POINT_GROUPS = frozenset({"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"})
INVERSION = -np.eye(3, dtype=int)


def _signed_bases():
    for order in permutations(range(3)):
        permuted = np.eye(3, dtype=int)[list(order)]
        for signs in product((1, -1), repeat=3):
            change = permuted * np.array(signs)[:, None]
            if round(np.linalg.det(change)) == 1:
                yield change


SIGNED_BASES = tuple(_signed_bases())


def _structure(lattice, elements, positions):
    lattice = np.array(lattice, dtype=float, copy=True)
    positions = np.array(positions, dtype=float, copy=True)
    if lattice.shape != (3, 3) or positions.shape != (len(elements), 3):
        raise ValueError("expected a 3x3 lattice and three fractional coordinates per site")
    if not np.isfinite(lattice).all() or not np.isfinite(positions).all() or not len(elements):
        raise ValueError("expected finite geometry and at least one occupied atom")
    _volume(tuple(map(tuple, lattice.tolist())))
    if np.linalg.det(lattice) < 0:
        lattice[2] *= -1
        positions[:, 2] *= -1
    return Structure(tuple(map(tuple, lattice.tolist())), tuple(elements),
                     tuple(map(tuple, (positions % 1).tolist())))


def load_mp_structure(record: dict) -> Structure:
    """Read a fully occupied ordered structure from a Materials Project summary."""
    raw = record["structure"]
    elements, positions = [], []
    for site in raw["sites"]:
        species = site["species"]
        if len(species) != 1 or not math.isclose(float(species[0]["occu"]), 1.0, abs_tol=1e-6):
            raise ValueError("partial occupancy or disorder requires an explicit ordered model")
        elements.append(species[0]["element"])
        positions.append(site["abc"])
    return _structure(raw["lattice"]["matrix"], elements, positions)


def _spg_cell(structure):
    types = {name: index for index, name in enumerate(dict.fromkeys(structure.elements), 1)}
    return (structure.lattice, structure.positions, [types[name] for name in structure.elements]), types


def crystal_symmetry(structure: Structure, symprec: float = 0.03):
    cell, _ = _spg_cell(structure)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not resolve the crystal symmetry")
    return dataset


def _centrosymmetric(dataset):
    return np.any(np.all(dataset.rotations == INVERSION, axis=(1, 2)))


def poscar_text(structure: Structure, title: str) -> str:
    species = tuple(dict.fromkeys(structure.elements))
    counts = Counter(structure.elements)
    rows = [title, "1.0", *(" ".join(f"{x:.15g}" for x in row) for row in structure.lattice),
            " ".join(species), " ".join(str(counts[name]) for name in species), "Direct"]
    rows.extend(" ".join(f"{round(x % 1, 14) % 1:.14f}" for x in r)
                for name in species for name_at_site, r in zip(structure.elements, structure.positions)
                if name_at_site == name)
    return "\n".join(rows) + "\n"


def classify(records: list[dict], *, symprec: float = 0.03) -> list[dict]:
    """Classify ordered endpoint structures without inferring switchability."""
    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be a positive finite distance in Å")
    outcomes = []
    for record in records:
        outcome = {"record": record, "classification": "ineligible", "reason": "",
                   "structure": None, "spacegroup": None, "point_group": None}
        try:
            structure = load_mp_structure(record)
            symmetry = crystal_symmetry(structure, symprec)
            outcome.update(structure=structure, spacegroup=symmetry.international,
                           point_group=symmetry.pointgroup)
            gap = record.get("band_gap")
            if record.get("deprecated") is True:
                outcome["reason"] = "record is deprecated"
            elif record.get("is_metal") is True or gap is not None and float(gap) <= 0:
                outcome["reason"] = "MP reports a metal or nonpositive band gap"
            elif record.get("is_magnetic") is True:
                outcome["reason"] = "MP reports magnetism; this workflow uses nonmagnetic inputs"
            elif _centrosymmetric(symmetry):
                outcome.update(classification="centrosymmetric", reason="contains inversion")
            elif symmetry.pointgroup in POLAR_POINT_GROUPS:
                outcome.update(classification="polar", reason="polar crystallographic point group")
            else:
                outcome["reason"] = "point group is nonpolar without inversion"
        except (KeyError, TypeError, ValueError, ArithmeticError) as error:
            outcome["reason"] = f"invalid or unresolved geometry: {error}"
        outcomes.append(outcome)
    return outcomes


def _variants(structure, symprec, reference=False):
    cell, types = _spg_cell(structure)
    names = {number: element for element, number in types.items()}
    variants = [] if reference else [("input", structure)]
    for name, primitive in (("primitive", True), ("conventional", False)):
        standardized = spglib.standardize_cell(
            cell, to_primitive=primitive, no_idealize=not reference, symprec=symprec,
        )
        if standardized is None:
            continue
        lattice, positions, numbers = standardized
        candidate = _structure(lattice, [names[n] for n in numbers], positions)
        if not any(candidate == previous for _, previous in variants):
            variants.append((name, candidate))
    if reference and not variants:
        variants.append(("input", structure))
    return variants


def _retains_operations(polar_symmetry, reference_symmetry, lattice, dual, symprec):
    by_rotation = {}
    for rotation, translation in zip(reference_symmetry.rotations, reference_symmetry.translations):
        by_rotation.setdefault(tuple(rotation.flat), []).append(translation)
    return all(
        any(_periodic_shift(subtract(translation, other), lattice, dual, symprec) is not None
            for other in by_rotation.get(tuple(rotation.flat), ()))
        for rotation, translation in zip(polar_symmetry.rotations, polar_symmetry.translations)
    )


def _align_pair(polar, reference, symprec, max_strain, max_displacement):
    polar_variants = _variants(polar, symprec)
    ref_variants = _variants(reference, symprec, reference=True)
    pairs = [(pname, p, rname, r) for pname, p in polar_variants for rname, r in ref_variants
             if Counter(p.elements) == Counter(r.elements)]
    if not pairs:
        return [], "no primitive or conventional representations have matching site counts", {}

    # Use one smallest common polar representation, trying another only if it
    # produces no verified match. Alternative bases of that same cell are not
    # rival physical alignments for the user to choose among.
    pairs.sort(key=lambda pair: len(pair[1].elements))
    smallest_strain = math.inf
    best_displacement = (math.inf, math.inf)
    alignments = []
    chosen_polar_cell = None
    for polar_name, child, reference_name, parent in pairs:
        if chosen_polar_cell is not None and child != chosen_polar_cell:
            continue
        lattice = child.lattice
        dual = dual_basis(lattice)
        child_symmetry = crystal_symmetry(child, symprec)
        if child_symmetry.pointgroup not in POLAR_POINT_GROUPS:
            continue
        rare = min(Counter(child.elements).items(), key=lambda item: item[1])[0]
        anchor = child.positions[child.elements.index(rare)]

        for change in SIGNED_BASES:
            parent_basis = change @ np.asarray(parent.lattice)
            strain = np.linalg.svd(np.linalg.solve(parent_basis, lattice), compute_uv=False) - 1
            max_abs_strain = float(np.max(np.abs(strain)))
            smallest_strain = min(smallest_strain, max_abs_strain)
            if max_abs_strain > max_strain:
                continue
            oriented = np.asarray(parent.positions) @ change.T
            for index, element in enumerate(parent.elements):
                if element != rare:
                    continue
                origin = (np.asarray(anchor) - oriented[index]) % 1
                preliminary = _structure(lattice, parent.elements, oriented + origin)
                try:
                    # Aligning one anchor can shift the provisional origin by
                    # one more site's displacement. Apply the limit after fitting.
                    matches = _match(child, preliminary, dual, 2 * max_displacement)
                    origin = (origin - np.mean([pair.shift for pair in matches], axis=0)) % 1
                    shifted = _structure(lattice, parent.elements, oriented + origin)
                    matches = _match(child, shifted, dual, max_displacement)
                except MappingError:
                    continue
                distances = [pair.distance for pair in matches]
                rms = math.sqrt(math.fsum(d**2 for d in distances) / len(distances))
                largest = max(distances)
                best_displacement = min(best_displacement, (rms, largest))
                fitted = replace(child, positions=tuple(shifted.positions[pair.index] for pair in matches))
                if any(_same_alignment(fitted, previous["reference"], dual, symprec)
                       for previous in alignments):
                    continue
                try:
                    fitted_symmetry = crystal_symmetry(fitted, symprec)
                except ValueError:
                    continue
                if not _centrosymmetric(fitted_symmetry) or not _retains_operations(
                    child_symmetry, fitted_symmetry, lattice, dual, symprec,
                ):
                    continue
                alignments.append({
                    "reference": fitted,
                    "polar": child,
                    "reference_matched_spacegroup": fitted_symmetry.international,
                    "polar_matched_spacegroup": child_symmetry.international,
                    "principal_strains": strain.tolist(),
                    "max_abs_strain": max_abs_strain,
                    "rms_displacement_angstrom": rms,
                    "max_displacement_angstrom": largest,
                    "matching_cell": {"polar": polar_name, "reference": reference_name,
                                      "atoms": len(child.elements)},
                    "reference_basis_transform": change.tolist(),
                    "reference_origin_offset_fractional": origin.tolist(),
                    "reference_idealized": reference_name != "input",
                })
                chosen_polar_cell = child

    diagnostics = {}
    if math.isfinite(smallest_strain):
        diagnostics["minimum_trial_strain"] = smallest_strain
    if math.isfinite(best_displacement[0]):
        diagnostics["minimum_trial_rms_displacement_angstrom"] = best_displacement[0]
        diagnostics["max_displacement_at_minimum_rms_angstrom"] = best_displacement[1]
    if not alignments:
        if smallest_strain > max_strain:
            reason = f"best basis requires {smallest_strain:.3%} principal strain, limit {max_strain:.3%}"
        elif not math.isfinite(best_displacement[0]):
            reason = f"no same-element assignment satisfies the {max_displacement:g} Å limit"
        else:
            reason = "matched reference lost inversion or did not contain the polar endpoint symmetry"
        return [], reason, diagnostics
    return sorted(alignments, key=lambda row: (row["rms_displacement_angstrom"], row["max_abs_strain"])), "", diagnostics


def _same_alignment(left, right, dual, symprec):
    try:
        _match(left, right, dual, symprec)
    except MappingError:
        return False
    return True


def _row(polar, reference, status, reason):
    row = {"status": status, "reason": reason, "reference_centrosymmetric": False,
           "alignment_id": None, "alignment_ambiguous": False}
    for role, outcome in (("polar", polar), ("reference", reference)):
        record = outcome["record"] if outcome is not None else {}
        row.update({
            f"{role}_id": record.get("material_id"),
            f"{role}_energy_above_hull": record.get("energy_above_hull"),
            f"{role}_band_gap": record.get("band_gap"),
            f"{role}_spacegroup": outcome["spacegroup"] if outcome is not None else None,
            f"{role}_point_group": outcome["point_group"] if outcome is not None else None,
            f"requested_{role}": record.get("requested_id") is True,
        })
    return row


def discover(
    records: list[dict], *, symprec: float = 0.03,
    max_strain: float = 0.08, max_displacement: float = 1.5,
) -> list[dict]:
    """Return checked fixed-cell database pair candidates and rejection reasons.

    Strain is the maximum absolute principal stretch minus one. Structural
    validation establishes an inversion-related distortion, not switchability.
    """
    if not all(math.isfinite(x) and x > 0 for x in (max_strain, max_displacement)):
        raise ValueError("max_strain and max_displacement must be positive and finite")
    outcomes = classify(records, symprec=symprec)
    requested = [outcome for outcome in outcomes if outcome["record"].get("requested_id") is True]
    rows = [_row(outcome, None, "rejected", outcome["reason"])
            for outcome in requested if outcome["classification"] == "ineligible"]
    if requested and all(outcome["classification"] == "ineligible" for outcome in requested):
        return rows

    polar = [outcome for outcome in outcomes if outcome["classification"] == "polar"]
    references = [outcome for outcome in outcomes if outcome["classification"] == "centrosymmetric"]
    polar = [item for item in polar if item in requested] or polar
    references = [item for item in references if item in requested] or references
    if not polar or not references:
        rows.extend(_row(item, None, "rejected", "no eligible centrosymmetric reference in the MP records")
                    for item in polar)
        rows.extend(_row(None, item, "rejected", "no eligible polar structure in the MP records")
                    for item in references)
        if not rows:
            rows = [_row(item, None, "rejected", item["reason"]) for item in outcomes]
        return rows or [_row(None, None, "rejected", "no MP records to classify")]

    for child, parent in product(polar, references):
        row = _row(child, parent, "accepted", "matched in a common polar lattice and verified reference inversion")
        try:
            alignments, reason, diagnostics = _align_pair(child["structure"], parent["structure"], symprec,
                                                          max_strain, max_displacement)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            alignments, reason, diagnostics = [], f"unable to validate geometry: {error}", {}
        if not alignments:
            rows.append({**row, **diagnostics, "status": "rejected", "reason": reason})
            continue
        best = alignments[0]
        comparable = [entry for entry in alignments
                      if entry["rms_displacement_angstrom"] <= best["rms_displacement_angstrom"] + symprec
                      and entry["max_abs_strain"] <= best["max_abs_strain"] + .01]
        for index, alignment in enumerate(comparable, 1):
            candidate = dict(alignment)
            aligned_reference = candidate.pop("reference")
            aligned_polar = candidate.pop("polar")
            rows.append({
                **row, **candidate,
                "reference_centrosymmetric": True,
                "alignment_id": str(index),
                "alignment_ambiguous": len(comparable) > 1,
                "alignment_count": len(alignments),
                "reference_poscar": poscar_text(aligned_reference, f"{row['reference_id']} strained reference"),
                "polar_poscar": poscar_text(aligned_polar, f"{row['polar_id']} polar endpoint"),
            })
    return rows
