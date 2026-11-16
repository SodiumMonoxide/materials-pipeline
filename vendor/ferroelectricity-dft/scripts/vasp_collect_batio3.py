"""Collect a fixed-cell path CSV from nonmagnetic static VASP calculations."""

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from pathlib import Path


NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
DIPOLE = rf"\s*=\s*\(\s*{NUMBER}\s+{NUMBER}\s+{NUMBER}\s*\)\s*(\S+)\s+Angst"
CHARGE_PER_ANGSTROM_SQUARED = 16.02176634  # C/m²


def read_manifest(path: Path) -> list[tuple[str, float]]:
    images = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["image", "lambda"]:
            raise ValueError(f"{path}: expected the CSV header image,lambda")
        for row in reader:
            image = (row["image"] or "").strip()
            if None in row or not re.fullmatch(r"[A-Za-z0-9_-]+", image):
                raise ValueError(f"{path}:{reader.line_num}: invalid image directory or fields")
            try:
                coordinate = float(row["lambda"])
            except (ValueError, TypeError):
                coordinate = math.nan
            if not math.isfinite(coordinate) or (images and coordinate <= images[-1][1]):
                raise ValueError(f"{path}:{reader.line_num}: lambda must be finite and strictly increasing")
            if image.casefold() in seen:
                raise ValueError(f"{path}:{reader.line_num}: repeated image directory {image}")
            seen.add(image.casefold())
            images.append((image, coordinate))
    if len(images) < 2:
        raise ValueError(f"{path}: expected at least two images")
    return images


def number(value: str) -> float:
    result = float(value.upper().replace("D", "E"))
    if not math.isfinite(result):
        raise ValueError(f"number must be finite: {value}")
    return result


def final_match(pattern: str, log: str, label: str) -> re.Match:
    matches = list(re.finditer(pattern, log, re.MULTILINE | re.IGNORECASE))
    if not matches:
        raise ValueError(f"could not find {label}")
    return matches[-1]


def read_outcar(path: Path):
    log = path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        if "General timing and accounting information" not in log:
            raise ValueError("missing final timing and accounting section")
        if "aborting loop because EDIFF is reached" not in log:
            raise ValueError("no electronic convergence by EDIFF reported")
        nsw = final_match(r"\bNSW\s*=\s*(\d+)", log, "NSW")[1]
        if int(nsw) != 0:
            raise ValueError(f"expected a static calculation (NSW=0); found NSW={nsw}")
        if re.search(r"\b(?:LSORBIT|LNONCOLLINEAR)\s*=\s*\.?T(?:RUE)?\b", log, re.IGNORECASE):
            raise ValueError("expected a nonmagnetic scalar calculation without SOC")

        printed_volume = final_match(rf"volume of cell\s*:\s*{NUMBER}", log, "cell volume")
        position = final_match(r"^[ \t]*direct lattice vectors.*\n", log, "direct lattice vectors").end()
        axes = [tuple(map(number, line.split()[:3])) for line in log[position:].splitlines()[:3]]
        if len(axes) != 3 or any(len(axis) != 3 for axis in axes):
            raise ValueError("incomplete direct lattice vectors")
        a, b, c = axes
        volume = abs(
            a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if volume <= 0 or not math.isclose(volume, number(printed_volume[1]), abs_tol=0.0051):
            raise ValueError("printed volume and direct lattice vectors disagree")

        ion = final_match(r"\bp\[ion\]" + DIPOLE, log, "ionic dipole p[ion]")
        electron = final_match(r"\bp\[elc\]" + DIPOLE, log, "electronic dipole p[elc]")
        signs = {"|e|": 1, "electrons": -1, "e*": -1}
        if ion[4] != electron[4] or ion[4].lower() not in signs:
            raise ValueError(f"unrecognized or inconsistent dipole units ({ion[4]}, {electron[4]})")
        scale = signs[ion[4].lower()] * CHARGE_PER_ANGSTROM_SQUARED / volume
        polarization = tuple(scale * (number(ion[i]) + number(electron[i])) for i in range(1, 4))
        energy = final_match(rf"energy\(sigma->0\)\s*=\s*{NUMBER}", log, "energy(sigma->0)")
        nelect = final_match(rf"\bNELECT\s*=\s*{NUMBER}", log, "NELECT")
        return axes, volume, polarization, number(energy[1]), number(nelect[1])
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def read_gap(path: Path, outcar_nelect: float, occ_tol: float) -> float:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            first = stream.readline().split()
            if len(first) != 4 or first[-1] != "1":
                raise ValueError("expected the ISPIN=1 EIGENVAL format")
            for _ in range(4):
                stream.readline()
            nelect, nkpts, nbands = map(int, stream.readline().split())
            nocc = nelect // 2
            if nelect != outcar_nelect or nelect % 2 or not 0 < nocc < nbands or nkpts < 1:
                raise ValueError("expected matching closed-shell NELECT, occupied bands, and empty bands")
            records = (line.split() for line in stream if line.strip())
            return fundamental_gap(records, nkpts, nbands, nocc, occ_tol)
    except StopIteration:
        raise ValueError(f"{path}: truncated k-point or band records") from None
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def fundamental_gap(records, nkpts: int, nbands: int, nocc: int, occ_tol: float) -> float:
    vbm, cbm = -math.inf, math.inf
    filled = None
    for kpoint in range(1, nkpts + 1):
        k_row = list(map(number, next(records)))
        if len(k_row) != 4 or k_row[-1] <= 0:
            raise ValueError(f"k point {kpoint}: expected a k vector with positive integration weight")
        for band in range(1, nbands + 1):
            band_row = next(records)
            if len(band_row) != 3 or int(band_row[0]) != band:
                raise ValueError(f"k point {kpoint}: expected indexed energy and occupation for band {band}")
            energy, occupation = map(number, band_row[1:])
            if filled is None:
                # Accept either occupation convention used for a filled closed-shell band.
                filled = round(occupation)
                if filled not in (1, 2):
                    raise ValueError("the first occupied band is not full")
            target = filled if band <= nocc else 0
            if abs(occupation - target) > occ_tol:
                raise ValueError(f"k point {kpoint}, band {band}: occupation {occupation:g}, expected {target}")
            if band <= nocc:
                vbm = max(vbm, energy)
            else:
                cbm = min(cbm, energy)
    if next(records, None) is not None:
        raise ValueError("extra k-point or band records beyond the counts in the header")
    gap = cbm - vbm
    if gap <= 0:
        raise ValueError(f"sampled fundamental gap must be positive, got {gap:g} eV")
    return gap


def collect(root: Path, images, expected_volume: float | None, occ_tol: float):
    rows = []
    reference_axes = None
    for image, coordinate in images:
        outcar = root / image / "OUTCAR"
        axes, volume, polarization, energy, nelect = read_outcar(outcar)
        if expected_volume is not None and not math.isclose(volume, expected_volume, abs_tol=1e-5):
            raise ValueError(f"{outcar}: volume {volume:g} differs from {expected_volume:g} angstrom^3")
        if reference_axes is None:
            reference_axes = axes
        if axes != reference_axes:
            raise ValueError(f"{outcar}: direct lattice vectors differ from the first image")
        gap = read_gap(root / image / "EIGENVAL", nelect, occ_tol)
        rows.append((coordinate, gap, *polarization, energy))
    return rows


def write_csv(stream, rows) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("lambda", "gap_ev", "px_c_m2", "py_c_m2", "pz_c_m2", "energy_ev_cell"))
    writer.writerows(rows)


def save_csv(path: Path, rows) -> None:
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
    parser.add_argument("run_root", type=Path, help="directory containing the image subdirectories")
    parser.add_argument("--manifest", type=Path, help="image,lambda CSV (default: RUN_ROOT/images.csv)")
    parser.add_argument("--output", type=Path, help="CSV destination (default: stdout)")
    parser.add_argument("--expected-volume", type=float, help="optional volume check in angstrom^3")
    parser.add_argument("--occ-tol", type=float, default=0.001, help="occupation tolerance (default: 0.001)")
    args = parser.parse_args()
    try:
        if args.expected_volume is not None and not 0 < args.expected_volume < math.inf:
            raise ValueError("--expected-volume must be finite and positive")
        if not 0 <= args.occ_tol < 0.5:
            raise ValueError("--occ-tol must be in [0, 0.5)")
        manifest = args.manifest or args.run_root / "images.csv"
        images = read_manifest(manifest)
        rows = collect(args.run_root, images, args.expected_volume, args.occ_tol)
        if args.output is None:
            write_csv(sys.stdout, rows)
        else:
            input_paths = [manifest]
            for image, _ in images:
                input_paths.extend(args.run_root / image / name for name in ("OUTCAR", "EIGENVAL"))
            if args.output.resolve() in {path.resolve() for path in input_paths}:
                raise ValueError("--output must differ from the manifest and VASP outputs")
            save_csv(args.output, rows)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    min_gap = min(rows, key=lambda row: row[1])
    max_energy = max(rows, key=lambda row: row[-1])
    print(f"Collected {len(rows)} images.", file=sys.stderr)
    print(f"Minimum sampled gap: {min_gap[1]:.8g} eV (lambda={min_gap[0]:g})", file=sys.stderr)
    print(
        f"Maximum sampled energy above first: {max_energy[-1] - rows[0][-1]:.8g} eV/cell "
        f"(lambda={max_energy[0]:g})", file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
