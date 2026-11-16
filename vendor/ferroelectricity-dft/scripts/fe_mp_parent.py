"""Propose inversion projections for manual review when a parent is unavailable."""

import math
from collections import Counter
from dataclasses import replace

from fe_structures import MappingError, Structure, _invert, _inversions, _match, _periodic_shift, _volume
from polarization_path import dual_basis, shortest_quantum, subtract


GEOMETRIC_TOLERANCE = 1e-7  # angstrom


def separation(positions, lattice, dual):
    for i, position in enumerate(positions):
        for j in range(i):
            _, distance = _periodic_shift(
                subtract(position, positions[j]), lattice, dual, math.inf
            )
            yield i, j, distance


def project(polar, trial, dual, limit, half_image_length, polar_distances):
    lattice = polar.lattice
    # An anchor can itself be distorted. Refit the operation to all sites
    # before applying the final limit on the individual displacements.
    anchor_matches = _match(_invert(polar, trial), polar, dual, 4 * limit)
    translation = tuple(
        (origin + math.fsum(match.shift[i] for match in anchor_matches) / len(anchor_matches)) % 1
        for i, origin in enumerate(trial)
    )
    inverted = _invert(polar, translation)
    pairs = _match(inverted, polar, dual, 2 * limit)
    if max(pair.distance for pair in pairs) >= half_image_length - GEOMETRIC_TOLERANCE:
        raise MappingError("the inversion residual is too large to disambiguate its periodic image")

    for i, pair in enumerate(pairs):
        if pairs[pair.index].index != i:
            raise MappingError("the proposed inversion pairing is not an involution")
        if any(
            _periodic_shift(subtract(candidate, inverted.positions[i]), lattice, dual, 2 * limit)
            is not None
            for j, (element, candidate) in enumerate(zip(polar.elements, polar.positions))
            if element == polar.elements[i] and j != pair.index
        ):
            raise MappingError("more than one same-element inversion partner lies within the distance limit")

    positions = tuple(
        tuple((value - residual / 2) % 1 for value, residual in zip(position, pair.shift))
        for position, pair in zip(polar.positions, pairs)
    )
    distances = [pair.distance / 2 for pair in pairs]
    if any(initial - distances[i] - distances[j] <= GEOMETRIC_TOLERANCE
           for i, j, initial in polar_distances):
        raise MappingError("the proposed motion does not guarantee separation of distinct sites")
    min_separation = min(distance for _, _, distance in separation(positions, lattice, dual))
    if min_separation <= GEOMETRIC_TOLERANCE:
        raise MappingError("the projected reference contains overlapping sites")

    reference = replace(polar, positions=positions)
    inverted_reference = _invert(reference, translation)
    residuals = [
        _periodic_shift(
            subtract(positions[pair.index], inverted_reference.positions[i]),
            lattice, dual, GEOMETRIC_TOLERANCE,
        )
        for i, pair in enumerate(pairs)
    ]
    if any(residual is None for residual in residuals):
        raise MappingError("the projected reference did not recover the intended inversion")
    return {
        "structure": reference,
        "inversion_translation_fractional": translation,
        "inversion_center_fractional": tuple(x / 2 for x in translation),
        "inversion_partners": tuple(pair.index for pair in pairs),
        "max_displacement_angstrom": max(distances),
        "rms_displacement_angstrom": math.sqrt(math.fsum(x * x for x in distances) / len(distances)),
        "inversion_residual_angstrom": max(residual[1] for residual in residuals),
        "minimum_reference_separation_angstrom": min_separation,
    }


def synthesize_parent(
    polar: Structure, *, max_displacement_angstrom: float = 0.3, symprec: float = 0.03,
) -> dict:
    """Return provisional same-cell structures obtained by inversion projection.

    Candidate centers come from the least-populated element's inversion
    partners. Distances and site checks use the actual Cartesian cell metric.
    Every proposal requires manual review of the parent phase and path.
    """
    if not all(math.isfinite(x) and x > 0 for x in (max_displacement_angstrom, symprec)):
        raise ValueError("max_displacement_angstrom and symprec must be finite positive lengths")
    if not polar.positions or len(polar.positions) != len(polar.elements):
        raise ValueError("expected a populated structure with one element label per position")
    if len(polar.lattice) != 3 or any(
        len(vector) != 3 or not all(map(math.isfinite, vector))
        for vector in (*polar.lattice, *polar.positions)
    ):
        raise ValueError("expected finite three-dimensional lattice vectors and positions")
    result = {
        "requires_human_confirmation": True,
        "method": "fixed-cell inversion projection from the polar structure",
        "candidates": [],
        "rejections": [],
    }
    lattice = polar.lattice
    _volume(lattice)
    dual = dual_basis(lattice)
    half_image_length = shortest_quantum(lattice, dual) / 2
    polar_distances = list(separation(polar.positions, lattice, dual))
    if any(distance <= GEOMETRIC_TOLERANCE for _, _, distance in polar_distances):
        result["rejections"].append("The supplied structure contains overlapping sites.")
        return result
    if _inversions(polar, polar, dual, symprec):
        result["rejections"].append("The supplied structure already has inversion within symprec.")
        return result

    rare = min(Counter(polar.elements).items(), key=lambda item: item[1])[0]
    anchor = polar.elements.index(rare)
    for i, element in enumerate(polar.elements):
        if element != rare:
            continue
        trial = tuple((r + s) % 1 for r, s in zip(polar.positions[anchor], polar.positions[i]))
        try:
            option = project(polar, trial, dual, max_displacement_angstrom, half_image_length, polar_distances)
        except MappingError as error:
            result["rejections"].append(f"Inversion anchor sites {anchor + 1}, {i + 1}: {error}")
            continue
        option["anchor_sites"] = (anchor, i)
        center = option["inversion_translation_fractional"]
        if any(
            _periodic_shift(
                subtract(center, candidate["inversion_translation_fractional"]),
                lattice, dual, GEOMETRIC_TOLERANCE,
            ) is not None
            for candidate in result["candidates"]
        ):
            continue
        result["candidates"].append(option)
    result["candidates"].sort(key=lambda candidate: candidate["rms_displacement_angstrom"])
    return result
