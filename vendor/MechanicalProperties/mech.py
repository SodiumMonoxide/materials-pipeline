"""Extract relaxed-ion elastic constants and aggregate moduli from VASP OUTCARs."""

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


BLOCK_TITLE = "TOTAL ELASTIC MODULI (kBar)"
VOIGT_ORDER = ("XX", "YY", "ZZ", "YZ", "XZ", "XY")
TABLE_LINES = 1 + 2 + len(VOIGT_ORDER)  # Column labels, two rules, and stiffness rows.
MODULUS_FIELDS = (
    "K_V_GPa", "G_V_GPa", "K_R_GPa", "G_R_GPa", "K_H_GPa", "G_H_GPa", "E_H_GPa", "nu_H"
)


def voigt_label(label):
    return {"ZX": "XZ", "ZY": "YZ", "YX": "XY"}.get(label, label)


def read_total_elastic_moduli(path):
    """Return the last total elastic tensor in canonical Voigt order and GPa."""
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    block_line = None
    block = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if line.casefold() == BLOCK_TITLE.casefold():
                block_line = line_number
                block = []
            elif block_line is not None and line and len(block) < TABLE_LINES:
                block.append(line)

    if block_line is None:
        raise ValueError(f"No {BLOCK_TITLE} block found")

    try:
        if len(block) != TABLE_LINES:
            raise ValueError("incomplete table")
        header, top_rule, *rows, bottom_rule = block
        if set(top_rule) != {"-"} or set(bottom_rule) != {"-"}:
            raise ValueError("missing table separator")
        heading, *labels = header.split()
        columns = [voigt_label(label.upper()) for label in labels]
        if heading.casefold() != "direction" or sorted(columns) != sorted(VOIGT_ORDER):
            raise ValueError("invalid Voigt column labels")

        values_by_row = {}
        for row in rows:
            label, *values = row.split()
            label = voigt_label(label.upper())
            if label not in VOIGT_ORDER or label in values_by_row or len(values) != 6:
                raise ValueError("invalid or repeated row label or wrong number of values")
            parsed = [float(value.replace("D", "E").replace("d", "e")) for value in values]
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError("non-finite stiffness value")
            values_by_row[label] = parsed
        if len(values_by_row) != 6:
            raise ValueError("expected six stiffness rows")
    except ValueError as error:
        raise ValueError(f"Invalid {BLOCK_TITLE} block at line {block_line} ({error})") from error

    indices = [columns.index(label) for label in VOIGT_ORDER]
    raw_kbar = np.array([values_by_row[label] for label in VOIGT_ORDER])[:, indices]
    return raw_kbar * 0.1, block_line


def vrh_moduli(stiffness):
    compliance = np.linalg.inv(stiffness)
    normal = np.trace(stiffness[:3, :3])
    coupling = stiffness[0, 1] + stiffness[0, 2] + stiffness[1, 2]
    shear = np.trace(stiffness[3:, 3:])
    normal_s = np.trace(compliance[:3, :3])
    coupling_s = compliance[0, 1] + compliance[0, 2] + compliance[1, 2]
    shear_s = np.trace(compliance[3:, 3:])

    bulk_voigt = (normal + 2 * coupling) / 9
    shear_voigt = (normal - coupling + 3 * shear) / 15
    bulk_reuss = 1 / (normal_s + 2 * coupling_s)
    shear_reuss = 15 / (4 * (normal_s - coupling_s) + 3 * shear_s)
    bulk_hill = (bulk_voigt + bulk_reuss) / 2
    shear_hill = (shear_voigt + shear_reuss) / 2
    young_hill = 9 * bulk_hill * shear_hill / (3 * bulk_hill + shear_hill)
    poisson_hill = (3 * bulk_hill - 2 * shear_hill) / (2 * (3 * bulk_hill + shear_hill))
    results = (bulk_voigt, shear_voigt, bulk_reuss, shear_reuss, bulk_hill,
               shear_hill, young_hill, poisson_hill)
    if not all(math.isfinite(value) for value in results):
        raise ValueError("non-finite modulus; check stiffness values")
    return dict(zip(MODULUS_FIELDS, map(float, results)))


def analyze_stiffness(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (6, 6) or not np.all(np.isfinite(raw)):
        raise ValueError("expected a finite 6×6 stiffness matrix in GPa")
    stiffness = (raw + raw.T) / 2
    eigenvalues = np.linalg.eigvalsh(stiffness)
    tolerance = float(np.max(np.abs(eigenvalues))) * 6 * np.finfo(float).eps
    smallest = float(eigenvalues[0])
    if smallest > tolerance:
        stability = "positive_definite"
        moduli = vrh_moduli(stiffness)
    elif smallest < -tolerance:
        stability = "not_positive_definite"
        moduli = None
    else:
        stability = "numerically_singular"
        moduli = None

    return {
        "C_raw_GPa": raw.tolist(),
        "C_symmetric_GPa": stiffness.tolist(),
        "max_asymmetry_GPa": float(np.max(np.abs(raw - raw.T))),
        "eigenvalues_GPa": eigenvalues.tolist(),
        "eigenvalue_zero_tolerance_GPa": float(tolerance),
        "stability": stability,
        "vrh": moduli,
    }


def analyze_outcar(path):
    raw, block_line = read_total_elastic_moduli(path)
    return {"source_block_line": block_line, **analyze_stiffness(raw)}


def build_report(paths):
    results = []
    for path in paths:
        item = {"path": str(path)}
        try:
            item.update(analyze_outcar(path))
            item["status"] = "ok"
        except (OSError, EOFError, ValueError, np.linalg.LinAlgError) as error:
            item.update(status="error", error=str(error))
        results.append(item)

    n_errors = sum(item["status"] == "error" for item in results)
    return {
        "source_block": BLOCK_TITLE,
        "tensor": "relaxed-ion",
        "voigt_order": VOIGT_ORDER,
        "stability_scope": "3D bulk near zero stress; test on (C + C.T) / 2",
        "files": len(results),
        "errors": n_errors,
        "results": results,
    }


def write_csv(stream, report):
    fields = ("path", "status", "error", "source_block_line", "stability",
              "min_eigenvalue_GPa", "max_asymmetry_GPa", *MODULUS_FIELDS)
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in report["results"]:
        row = {key: item[key] for key in fields if key in item}
        if item["status"] == "ok":
            row["min_eigenvalue_GPa"] = item["eigenvalues_GPa"][0]
            row.update(item["vrh"] or {})
        writer.writerow(row)


def write_report(output, report):
    # Replace the destination only after a complete write, including on Windows.
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            if output.suffix.lower() == ".csv":
                write_csv(stream, report)
            else:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
            stream.close()
            temporary.replace(output)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    outcar = commands.add_parser("outcar", help="analyze a file or a calculation directory")
    outcar.add_argument("path", type=Path, help="file or directory to scan for OUTCAR and OUTCAR.gz")
    outcar.add_argument("--output", type=Path, help="write .json or summary .csv; default: JSON on stdout")
    args = parser.parse_args()

    if args.path.is_file():
        paths = [args.path]
    elif args.path.is_dir():
        paths = sorted(path for pattern in ("OUTCAR", "OUTCAR.gz")
                       for path in args.path.rglob(pattern) if path.is_file())
    else:
        parser.error(f"input path does not exist or is not a file or directory: {args.path}")
    if not paths:
        parser.error(f"no OUTCAR or OUTCAR.gz files found in {args.path}")

    if args.output is not None:
        if args.output.suffix.lower() not in {".json", ".csv"}:
            parser.error("--output requires a .json or .csv extension")
        if any(path.resolve() == args.output.resolve() for path in paths):
            parser.error("--output cannot overwrite an input file")

    report = build_report(paths)
    try:
        if args.output is None:
            json.dump(report, sys.stdout, indent=2, allow_nan=False)
            print()
        else:
            write_report(args.output, report)
    except OSError as error:
        parser.error(str(error))

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
