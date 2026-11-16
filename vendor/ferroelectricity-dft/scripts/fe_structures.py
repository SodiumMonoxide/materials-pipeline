"""Read fixed-cell POSCARs and match periodic sites for VASP polarization paths."""

import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from polarization_path import Basis, Vector, add_branch, candidates, cross, dot, dual_basis, subtract


@dataclass(frozen=True)
class Structure:
    lattice: Basis
    elements: tuple[str, ...]
    positions: tuple[Vector, ...]


@dataclass(frozen=True)
class SiteMatch:
    index: int
    shift: Vector
    distance: float


@dataclass(frozen=True)
class Inversion:
    translation: Vector
    matches: tuple[SiteMatch, ...]
    max_error: float


class MappingError(ValueError):
    """There is no bijective assignment satisfying the displacement limit."""


def _volume(lattice: Basis) -> float:
    volume = abs(dot(lattice[0], cross(lattice[1], lattice[2])))
    scale = math.prod(math.hypot(*row) for row in lattice)
    if not math.isfinite(scale) or volume <= scale * 1e-10:
        raise ValueError("lattice vectors must form a numerically nonsingular cell")
    return volume


def _numbers(tokens, name):
    values = tuple(float(x.replace("D", "E").replace("d", "e")) for x in tokens)
    if not all(map(math.isfinite, values)):
        raise ValueError(f"{name} must contain finite numbers")
    return values


def read_poscar(path: Path) -> Structure:
    """Read the positions, their element labels, and the physical lattice in Å."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    def tokens(index):
        return re.split(r"[!#]", lines[index], maxsplit=1)[0].split()

    try:
        factors = _numbers(tokens(1), "scale")
        unscaled = tuple(_numbers(tokens(i), "lattice") for i in range(2, 5))
        if any(len(row) != 3 for row in unscaled):
            raise ValueError("each lattice vector must have three Cartesian components")
        if len(factors) == 1 and factors[0] != 0:
            factor = factors[0]
            if factor < 0:
                factor = (-factor / _volume(unscaled)) ** (1 / 3)
            factors = (factor,) * 3
        elif len(factors) != 3 or min(factors) <= 0:
            raise ValueError("expected one nonzero scale or three positive scales")
        lattice = tuple(tuple(x * s for x, s in zip(row, factors)) for row in unscaled)
        _volume(lattice)

        species = tokens(5)
        if not species or any(re.fullmatch(r"[A-Z][a-z]?", name) is None for name in species):
            raise ValueError("expected a VASP 5+ element-symbol line before the atom counts")
        counts = tuple(map(int, tokens(6)))
        if len(species) != len(counts) or any(n < 0 for n in counts) or sum(counts) == 0:
            raise ValueError("atom counts must match the element-symbol line and include at least one ion")

        index = 7
        if lines[index].strip().lower().startswith("s"):
            index += 1
        mode = lines[index].strip().lower()[:1]
        if mode not in ("d", "c", "k"):
            raise ValueError("expected Direct or Cartesian coordinates")
        dual = dual_basis(lattice) if mode != "d" else None
        sites = []
        for i in range(index + 1, index + 1 + sum(counts)):
            vector = _numbers(tokens(i)[:3], f"site on line {i + 1}")
            if len(vector) != 3:
                raise ValueError(f"expected three coordinates on line {i + 1}")
            if dual is not None:
                physical = tuple(x * s for x, s in zip(vector, factors))
                vector = tuple(dot(row, physical) for row in dual)
            if not all(map(math.isfinite, vector)):
                raise ValueError(f"scaled coordinates on line {i + 1} must be finite")
            sites.append(tuple(x % 1 for x in vector))
    except IndexError:
        raise ValueError(f"{path}: incomplete POSCAR") from None
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{path}: {error}") from None
    elements = tuple(element for element, count in zip(species, counts) for _ in range(count))
    return Structure(lattice, elements, tuple(sites))


def _periodic_shift(delta: Vector, lattice: Basis, dual: Basis, limit: float):
    near = tuple(-round(x) for x in delta)
    cartesian = add_branch((0.0, 0.0, 0.0), delta, lattice)
    radius = min(limit, math.hypot(*add_branch(cartesian, near, lattice)))
    target = tuple(-x for x in cartesian)
    best = (math.inf, near)
    for branch in candidates(target, radius, dual):
        distance = math.hypot(*add_branch(cartesian, branch, lattice))
        best = min(best, (distance, branch))
    distance, offset = best
    if distance > limit + 1e-10:
        return None
    return tuple(x + n for x, n in zip(delta, offset)), distance


def _assignment(costs):
    """Hungarian algorithm, with infinite costs representing forbidden pairs."""
    size = len(costs)
    left = [0.0] * (size + 1)
    right = [0.0] * (size + 1)
    assigned = [0] * (size + 1)
    previous = [0] * (size + 1)
    for row in range(1, size + 1):
        assigned[0] = row
        best = [math.inf] * (size + 1)
        visited = [False] * (size + 1)
        column = 0
        while True:
            visited[column] = True
            current = assigned[column]
            step = math.inf
            following = 0
            for candidate in range(1, size + 1):
                if visited[candidate]:
                    continue
                residual = costs[current - 1][candidate - 1] - left[current] - right[candidate]
                if residual < best[candidate]:
                    best[candidate] = residual
                    previous[candidate] = column
                if best[candidate] < step:
                    step, following = best[candidate], candidate
            if not math.isfinite(step):
                raise MappingError("no bijective same-element match lies within the distance limit")
            for candidate, used in enumerate(visited):
                if used:
                    left[assigned[candidate]] += step
                    right[candidate] -= step
                else:
                    best[candidate] -= step
            column = following
            if assigned[column] == 0:
                break
        while column:
            previous_column = previous[column]
            assigned[column] = assigned[previous_column]
            column = previous_column

    result = [0] * size
    for column, row in enumerate(assigned[1:]):
        result[row - 1] = column
    return result


def _match(source, target, dual, limit, ordered=False):
    lattice = source.lattice

    def move(i, j):
        delta = subtract(target.positions[j], source.positions[i])
        return _periodic_shift(delta, lattice, dual, limit)

    if ordered:
        if source.elements != target.elements:
            raise MappingError("ordered matching requires the same element at each index")
        result = []
        for i in range(len(source.positions)):
            pair = move(i, i)
            if pair is None:
                raise MappingError(f"ordered site {i + 1} exceeds {limit:g} Å")
            result.append(SiteMatch(i, *pair))
        return tuple(result)

    result = {}
    for element in dict.fromkeys(source.elements):
        rows = [i for i, name in enumerate(source.elements) if name == element]
        columns = [j for j, name in enumerate(target.elements) if name == element]
        pairs = [[move(i, j) for j in columns] for i in rows]
        costs = [[pair[1] ** 2 if pair is not None else math.inf for pair in row] for row in pairs]
        try:
            assignment = _assignment(costs)
        except MappingError as error:
            raise MappingError(f"{element}: {error} ({limit:g} Å)") from None
        for local, column in enumerate(assignment):
            result[rows[local]] = SiteMatch(columns[column], *pairs[local][column])
    return tuple(result[i] for i in range(len(source.positions)))


def _invert(structure, translation):
    return replace(structure, positions=tuple(subtract(translation, r) for r in structure.positions))


def _inversions(source, target, dual, tolerance):
    rare = min(Counter(source.elements).items(), key=lambda item: item[1])[0]
    anchor = source.positions[source.elements.index(rare)]
    operations = []
    for element, position in zip(target.elements, target.positions):
        if element != rare:
            continue
        translation = tuple((r + s) % 1 for r, s in zip(anchor, position))
        # An anchor-derived translation may be offset by one site's residual.
        # Fit the translation to all matched positions, then enforce the requested limit.
        try:
            tentative = _match(_invert(source, translation), target, dual, 2 * tolerance)
            fit = tuple(
                (r + math.fsum(pair.shift[i] for pair in tentative) / len(tentative)) % 1
                for i, r in enumerate(translation)
            )
            final = _match(_invert(source, fit), target, dual, tolerance)
        except MappingError:
            continue
        operations.append(Inversion(fit, final, max(pair.distance for pair in final)))
    return sorted(operations, key=lambda operation: operation.max_error)


def _write_poscar(structure, species, coordinate, shifts):
    counts = Counter(structure.elements)
    rows = [f"Polarization path lambda={coordinate:+.8g}", "1.0"]
    rows.extend(" ".join(f"{x:.15g}" for x in row) for row in structure.lattice)
    rows.extend((" ".join(species), " ".join(str(counts[name]) for name in species), "Direct"))
    ordered_indices = (i for name in species for i, element in enumerate(structure.elements) if element == name)
    for i in ordered_indices:
        position = structure.positions[i]
        shift = shifts[i]
        fraction = tuple(round((r + abs(coordinate) * d) % 1, 15) % 1 for r, d in zip(position, shift))
        rows.append(" ".join(f"{x:.15f}" for x in fraction))
    return "\n".join(rows) + "\n"


def build_path(
    reference: Path,
    polar: Path,
    opposite: Path | None,
    images: int,
    mirror: bool = False,
    ordered: bool = False,
    max_displacement: float = 1.5,
    symprec: float = 0.01,
):
    """Return lattice, ordered species, (lambda, POSCAR) images, and metadata.

    Match atoms by the minimum sum of squared Cartesian displacements within
    each species. With ordered=True, retain the supplied site correspondence.
    """
    if mirror and opposite is not None:
        raise ValueError("supply an opposite structure or mirror the polar endpoint, not both")
    switch = mirror or opposite is not None
    if not isinstance(images, int) or images < 2 or (switch and (images < 3 or images % 2 == 0)):
        raise ValueError("need at least two images for a half path or an odd count of at least three for a switch path")
    if not all(math.isfinite(x) and x > 0 for x in (max_displacement, symprec)):
        raise ValueError("max_displacement and symprec must be positive finite lengths in Å")

    source = read_poscar(reference)
    species = tuple(dict.fromkeys(source.elements))
    lattice = source.lattice
    dual = dual_basis(lattice)

    def endpoint(path):
        target = read_poscar(path)
        if any(abs(a - b) > 1e-6 for row, other in zip(lattice, target.lattice) for a, b in zip(row, other)):
            raise ValueError(f"{path}: lattice vectors must match the reference basis within 1e-6 Å")
        if Counter(source.elements) != Counter(target.elements):
            raise ValueError(f"{path}: composition or atom counts differ from the reference")
        target = replace(target, lattice=lattice)
        try:
            matched = _match(source, target, dual, max_displacement, ordered)
        except MappingError as error:
            raise ValueError(f"{path}: {error}") from None
        return target, tuple(pair.shift for pair in matched)

    positive, forward = endpoint(polar)
    operations = _inversions(source, source, dual, symprec)
    inversion = operations[0] if operations else None
    backward = None
    opposites_inverted = share_inversion = None
    if mirror:
        if inversion is None:
            raise ValueError("mirror requires a centrosymmetric reference; supply an explicit opposite endpoint")
        reversed_shifts = {}
        for i, pair in enumerate(inversion.matches):
            # Match the inverted trajectory to the reference partner. Include
            # its small residual if the reference is symmetric within tolerance.
            reversed_shifts[pair.index] = tuple(-d - s for d, s in zip(forward[i], pair.shift))
        backward = tuple(reversed_shifts[i] for i in range(len(forward)))
        opposites_inverted = share_inversion = True
    elif opposite is not None:
        negative, backward = endpoint(opposite)
        for operation in operations:
            try:
                _match(_invert(positive, operation.translation), negative, dual, symprec)
            except MappingError:
                continue
            inversion = operation
            share_inversion = True
            break
        else:
            share_inversion = False
        opposites_inverted = share_inversion or bool(_inversions(positive, negative, dual, symprec))

    displacements = (*forward, *backward) if backward is not None else forward
    largest = max(math.hypot(*add_branch((0.0, 0.0, 0.0), shift, lattice)) for shift in displacements)
    if largest > max_displacement + 1e-10:
        raise ValueError(f"mirrored path exceeds {max_displacement:g} Å maximum displacement")
    if switch:
        middle = images // 2
        coordinates = [i / middle for i in range(-middle, middle + 1)]
    else:
        coordinates = [i / (images - 1) for i in range(images)]
    path_images = [
        (lam, _write_poscar(source, species, lam, backward if lam < 0 else forward))
        for lam in coordinates
    ]
    metadata = {
        "path_kind": "mirrored" if mirror else "explicit" if switch else "half",
        "atom_mapping": "ordered" if ordered else "minimum-displacement",
        "max_displacement_angstrom": largest,
        "reference_centrosymmetric": inversion is not None,
        "reference_inversion_center_fractional": (tuple(t / 2 for t in inversion.translation)
                                                   if inversion is not None else None),
        "reference_inversion_max_error_angstrom": inversion.max_error if inversion is not None else None,
        "endpoints_inversion_equivalent": opposites_inverted,
        "endpoints_share_reference_inversion": share_inversion,
        "symprec_angstrom": symprec,
    }
    return lattice, species, path_images, metadata
