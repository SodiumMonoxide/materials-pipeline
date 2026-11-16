"""Follow a polarization branch for an ordered, fixed-cell insulating path."""

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path


ELEMENTARY_CHARGE_PER_ANGSTROM_SQUARED = 16.02176634  # C/m²
MAX_CANDIDATES = 250_000
REQUIRED_COLUMNS = ("lambda", "gap_ev", "px_c_m2", "py_c_m2", "pz_c_m2")
ENERGY_COLUMN = "energy_ev_cell"
OUTPUT_COLUMNS = (
    "branch_a", "branch_b", "branch_c",
    "px_cont_c_m2", "py_cont_c_m2", "pz_cont_c_m2",
)
Vector = tuple[float, float, float]
Basis = tuple[Vector, Vector, Vector]
Branch = tuple[int, int, int]


@dataclass(frozen=True)
class Image:
    coordinate: float
    gap: float
    polarization: Vector
    energy: float | None


@dataclass(frozen=True)
class Unwrapped:
    image: Image
    branch: Branch
    polarization: Vector


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Vector, b: Vector) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def subtract(a: Vector, b: Vector) -> Vector:
    return tuple(x - y for x, y in zip(a, b))


def add_branch(raw: Vector, branch: Branch, basis: Basis) -> Vector:
    return tuple(
        math.fsum([raw[axis], *(n * v[axis] for n, v in zip(branch, basis))])
        for axis in range(3)
    )


def dual_basis(basis: Basis) -> Basis:
    a, b, c = basis
    crosses = cross(b, c), cross(c, a), cross(a, b)
    determinant = dot(a, crosses[0])
    return tuple(tuple(coordinate / determinant for coordinate in row) for row in crosses)


def quantum_basis(lattice: Basis, quantum_multiple: int) -> Basis:
    a, b, c = lattice
    volume = abs(dot(a, cross(b, c)))
    norm_product = math.prod(math.hypot(*row) for row in lattice)
    if not math.isfinite(norm_product) or volume <= norm_product * 1e-10:
        raise ValueError("lattice vectors must form a numerically nonsingular cell")
    scale = quantum_multiple * ELEMENTARY_CHARGE_PER_ANGSTROM_SQUARED / volume
    return tuple(tuple(scale * x for x in row) for row in lattice)


def candidates(target: Vector, radius: float, dual: Basis):
    # If |B n - target| <= radius, dual row i bounds the error in n[i].
    centers = [dot(row, target) for row in dual]
    widths = [radius * math.hypot(*row) for row in dual]
    bounds = []
    for center, width in zip(centers, widths):
        margin = 32 * math.ulp(max(1.0, abs(center), width))
        bounds.append((math.ceil(center - width - margin), math.floor(center + width + margin)))
    count = math.prod(high - low + 1 for low, high in bounds)
    if count > MAX_CANDIDATES:
        raise ValueError(
            "the lattice search is too large for this cell basis. "
            "Use a less skewed equivalent basis and update the input lattice consistently"
        )
    return product(*(range(low, high + 1) for low, high in bounds))


def shortest_quantum(basis: Basis, dual: Basis) -> float:
    zero = (0.0, 0.0, 0.0)
    radius = min(math.hypot(*row) for row in basis)
    return min(
        math.hypot(*add_branch(zero, branch, basis))
        for branch in candidates(zero, radius, dual)
        if branch != (0, 0, 0)
    )


def closest_branches(target: Vector, basis: Basis, dual: Basis):
    negative_target = tuple(-x for x in target)

    def distance(branch: Branch) -> float:
        return math.hypot(*add_branch(negative_target, branch, basis))

    rounded = tuple(round(dot(row, target)) for row in dual)
    # Two distinct trial branches give an upper bound for the runner-up.
    neighbors = (
        tuple(n + (step if i == axis else 0) for i, n in enumerate(rounded))
        for axis in range(3)
        for step in (-1, 1)
    )
    radius = max(distance(rounded), min(map(distance, neighbors)))
    best = second = (math.inf, (0, 0, 0))
    for branch in candidates(target, radius, dual):
        entry = distance(branch), branch
        if entry < best:
            best, second = entry, best
        elif entry < second:
            second = entry
    if not math.isfinite(second[0]):
        raise ValueError("could not resolve the two closest polarization branches")
    return best, second


def finite(value: str | None, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def read_images(path: Path, min_gap: float) -> list[Image]:
    images = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        headers = reader.fieldnames or []
        missing = set(REQUIRED_COLUMNS).difference(headers)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        if len(set(headers)) != len(headers):
            raise ValueError("CSV has duplicate headers")
        for row in reader:
            try:
                if None in row or None in row.values():
                    raise ValueError("row has a different number of fields than the header")
                coordinate, gap, px, py, pz = (
                    finite(row[column], column) for column in REQUIRED_COLUMNS
                )
                energy = finite(row[ENERGY_COLUMN], ENERGY_COLUMN) if ENERGY_COLUMN in row else None
                if images and coordinate <= images[-1].coordinate:
                    raise ValueError("lambda must increase strictly in file order")
                if gap <= min_gap:
                    raise ValueError(f"gap {gap:g} eV must exceed {min_gap:g} eV")
                images.append(Image(coordinate, gap, (px, py, pz), energy))
            except ValueError as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from None
    if len(images) < 2:
        raise ValueError("CSV needs at least two images")
    return images


def unwrap(images: list[Image], basis: Basis) -> tuple[list[Unwrapped], list[str]]:
    dual = dual_basis(basis)
    shortest = shortest_quantum(basis, dual)
    first = images[0]
    result = [Unwrapped(first, (0, 0, 0), first.polarization)]
    warnings = []
    for image in images[1:]:
        previous = result[-1]
        target = subtract(previous.polarization, image.polarization)
        (jump, branch), (runner_up, _) = closest_branches(target, basis, dual)
        interval = f"lambda {previous.image.coordinate:g} -> {image.coordinate:g}"
        separation = runner_up - jump
        if separation <= 1e-10 * shortest:
            raise ValueError(f"{interval}: tied closest branches. Add intermediate images")
        if separation <= 0.10 * shortest or jump >= 0.45 * shortest:
            warnings.append(
                f"{interval}: jump {jump:.6g} C/m², runner-up {runner_up:.6g} C/m², "
                f"shortest quantum {shortest:.6g} C/m². Check with a denser path"
            )
        result.append(Unwrapped(image, branch, add_branch(image.polarization, branch, basis)))
    return result, warnings


def write_csv(stream, rows: list[Unwrapped]) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    include_energy = rows[0].image.energy is not None
    energy_columns = (ENERGY_COLUMN, "energy_rel_first_ev_cell") if include_energy else ()
    writer.writerow((*REQUIRED_COLUMNS, *OUTPUT_COLUMNS, *energy_columns))
    for row in rows:
        image = row.image
        values = (image.coordinate, image.gap, *image.polarization, *row.branch, *row.polarization)
        if include_energy:
            values += (image.energy, image.energy - rows[0].image.energy)
        writer.writerow(values)


def output_csv(path: Path | None, rows: list[Unwrapped]) -> None:
    if path is None:
        write_csv(sys.stdout, rows)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            write_csv(stream, rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help=f"input with {','.join(REQUIRED_COLUMNS)}")
    parser.add_argument(
        "--lattice", nargs=9, type=float, required=True,
        metavar="A", help="three lattice vectors in angstrom, a then b then c",
    )
    parser.add_argument(
        "--quantum-multiple", type=int, default=1,
        help="integer multiple of e*a_i/volume matching the input convention (default: 1)",
    )
    parser.add_argument(
        "--min-gap-ev", type=float, default=0.0,
        help="require a gap strictly above this value at every image (default: 0 eV)",
    )
    parser.add_argument("--output", type=Path, help="write CSV to this path instead of stdout")
    args = parser.parse_args()
    try:
        if args.quantum_multiple < 1:
            raise ValueError("--quantum-multiple must be a positive integer")
        if not math.isfinite(args.min_gap_ev) or args.min_gap_ev < 0:
            raise ValueError("--min-gap-ev must be finite and nonnegative")
        if not all(math.isfinite(x) for x in args.lattice):
            raise ValueError("--lattice must contain finite numbers")
        if args.output is not None and args.csv.resolve() == args.output.resolve():
            raise ValueError("--output must differ from the input CSV")
        lattice = tuple(tuple(args.lattice[i:i + 3]) for i in (0, 3, 6))
        images = read_images(args.csv, args.min_gap_ev)
        rows, warnings = unwrap(images, quantum_basis(lattice, args.quantum_multiple))
        output_csv(args.output, rows)
    except (ValueError, ArithmeticError, OSError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    delta = subtract(rows[-1].polarization, rows[0].polarization)
    print(f"Endpoint delta P (C/m²): ({', '.join(f'{x:.8g}' for x in delta)})", file=sys.stderr)
    if images[0].energy is not None:
        peak = max(images, key=lambda image: image.energy)
        height = peak.energy - images[0].energy
        print(
            f"Highest sampled energy relative to first (eV/cell): {height:.8g} "
            f"at lambda={peak.coordinate:g}", file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
