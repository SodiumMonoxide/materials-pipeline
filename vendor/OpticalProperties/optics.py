"""Collect symmetric bulk dielectric spectra from VASP vasprun.xml files."""

import argparse
import cmath
import csv
import gzip
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


AXES = ("xx", "yy", "zz", "xy", "yz", "zx")
FIELDS = ("energy", *AXES)
SETTINGS = ("ALGO", "GGA", "METAGGA", "LHFCALC", "ISPIN", "LSORBIT", "LOPTICS", "ENCUT",
            "NBANDS", "EDIFF", "NELM", "ISMEAR", "SIGMA", "CSHIFT", "NEDOS", "OMEGAMAX")
HC_EV_NM = 1239.8419843320026


def read_run(path):
    metadata = {}
    blocks = []
    stack = []
    scsteps = 0
    keep = {"generator", "incar", "parameters", "kpoints", "dielectricfunction"}
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rb") as source:
        for event, node in ET.iterparse(source, events=("start", "end")):
            if event == "start":
                stack.append(node)
                if len(stack) == 1 and node.tag != "modeling":
                    raise ValueError("Expected a VASP <modeling> root")
                continue

            stack.pop()
            if any(parent.tag in keep for parent in stack):
                continue
            if node.tag == "generator":
                metadata["vasp_version"] = node.findtext("i[@name='version']", "").strip()
            elif node.tag in ("incar", "parameters"):
                metadata.update({item.get("name"): item.text.strip()
                                 for item in node.iter("i")
                                 if item.get("name") in SETTINGS and item.text})
            elif node.tag == "kpoints":
                divisions = node.findtext("generation/v[@name='divisions']", "")
                metadata["kmesh"] = "x".join(divisions.split())
                metadata["nk_written"] = len(node.findall("varray[@name='kpointlist']/v")) or ""
            elif node.tag == "scstep":
                scsteps += 1
            elif node.tag == "calculation":
                metadata["last_calculation_scsteps"] = scsteps
                scsteps = 0
            elif node.tag == "dielectricfunction":
                blocks.append(node)
            if stack:
                stack[-1].remove(node)
    return metadata, blocks


def select_block(blocks, requested):
    if not blocks:
        raise ValueError("No <dielectricfunction> found")
    if requested is not None:
        if 1 <= requested <= len(blocks):
            return requested, blocks[requested - 1]
        raise ValueError(f"Block {requested} is outside the range 1–{len(blocks)}")
    if len(blocks) == 1:
        return 1, blocks[0]
    density = [(i, block) for i, block in enumerate(blocks, 1)
               if block.get("comment", "").strip().lower() == "density-density"]
    if len(density) == 1:
        return density[0]
    labels = ", ".join(f"{i}={block.get('comment') or '(unlabelled)'}"
                       for i, block in enumerate(blocks, 1))
    raise ValueError(f"Ambiguous dielectric blocks; use --block N ({labels})")


def read_part(block, part):
    array = block.find(f"{part}/array")
    if array is None:
        raise ValueError(f"Missing {part}/array")
    aliases = {"xz": "zx", "zy": "yz", "yx": "xy"}
    labels = [(field.text or "").strip().lower() for field in array.findall("field")]
    labels = [aliases.get(label, label) for label in labels]
    if len(labels) != len(FIELDS) or set(labels) != set(FIELDS):
        raise ValueError(f"Unsupported {part} dielectric fields: {labels}")
    order = [labels.index(field) for field in FIELDS]
    rows = []
    for number, row in enumerate(array.findall("set/r"), 1):
        values = [float(value) for value in (row.text or "").split()]
        if len(values) != len(FIELDS) or not all(map(math.isfinite, values)):
            raise ValueError(f"Invalid {part} dielectric row {number}")
        rows.append(tuple(values[i] for i in order))
    if not rows:
        raise ValueError(f"Empty {part} dielectric table")
    if rows[0][0] < 0 or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"The {part} energy grid must be nonnegative and strictly increasing")
    return rows


def scalar(tensor, component):
    if component == "isotropic":
        return sum(tensor[:3]) / 3
    return tensor[AXES.index(component)]


def derived(energy, eps1, eps2):
    n, k = "", ""
    alpha, reflectance = "", ""
    wavelength = HC_EV_NM / energy if energy > 0 else ""
    if eps2 >= 0:
        index = cmath.sqrt(complex(eps1, eps2))
        n, k = index.real, index.imag
        alpha = 4 * math.pi * k * energy * 1e7 / HC_EV_NM
        reflectance = abs((index - 1) / (index + 1)) ** 2
    return wavelength, n, k, alpha, reflectance


def write_spectrum(path, real, imag, component):
    columns = ["energy_eV", *(f"eps1_{axis}" for axis in AXES),
               *(f"eps2_{axis}" for axis in AXES)]
    if component == "isotropic":
        columns += ["eps1_isotropic", "eps2_isotropic"]
    if component:
        columns += ["wavelength_nm", "n", "k", "alpha_cm-1", "reflectance"]
    negative = 0
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(columns)
        for r, i in zip(real, imag):
            row = (*r, *i[1:])
            if component:
                eps1 = scalar(r[1:], component)
                eps2 = scalar(i[1:], component)
                negative += eps2 < 0
                if component == "isotropic":
                    row += (eps1, eps2)
                row += derived(r[0], eps1, eps2)
            writer.writerow(row)
    return negative


def collect(path, csv_path, args):
    metadata, blocks = read_run(path)
    index, block = select_block(blocks, args.block)
    real, imag = read_part(block, "real"), read_part(block, "imag")
    if len(real) != len(imag) or any(not math.isclose(r[0], i[0], rel_tol=0, abs_tol=1e-6)
                                     for r, i in zip(real, imag)):
        raise ValueError("Real and imaginary energy grids differ")
    negative = write_spectrum(csv_path, real, imag, args.component)
    result = {"dielectric_block": index, "block_comment": block.get("comment", ""),
              "n_energy": len(real), "energy_min_eV": real[0][0], "energy_max_eV": real[-1][0],
              "derived_component": args.component or "", "negative_eps2_rows": negative if args.component else ""}
    return {**metadata, **result}, imag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="XML/XML.gz files, or directories to search for vasprun.xml[.gz]")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--block", type=int, help="1-based dielectric block index")
    parser.add_argument("--component", choices=(*AXES[:3], "isotropic"),
                        help="derive optical constants for a bulk scalar response or eigenpolarization")
    parser.add_argument("--plot", action="store_true", help="plot an epsilon2 overlay using Matplotlib")
    parser.add_argument("--plot-window", nargs=2, metavar=("MIN_EV", "MAX_EV"), type=float)
    args = parser.parse_args()
    if args.plot and not args.component:
        parser.error("--plot requires --component")
    if args.plot_window:
        low, high = args.plot_window
        if not args.plot or not all(map(math.isfinite, (low, high))) or not (0 <= low < high):
            parser.error("--plot-window requires --plot and finite 0 <= MIN_EV < MAX_EV")
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            parser.error("--plot requires Matplotlib (python -m pip install matplotlib)")

    paths = set()
    for item in args.inputs:
        if item.is_dir():
            files = [file for file in item.rglob("vasprun.xml*")
                     if file.name in {"vasprun.xml", "vasprun.xml.gz"} and file.is_file()]
            if not files:
                parser.error(f"No vasprun.xml[.gz] in {item}")
        else:
            files = [item]
        paths.update(file.resolve() for file in files)

    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"Cannot create output directory: {exc}")

    fields = ["source", "status", "message", "spectrum", "vasp_version", "kmesh", "nk_written",
              "last_calculation_scsteps", *SETTINGS, "dielectric_block", "block_comment", "n_energy",
              "energy_min_eV", "energy_max_eV", "derived_component", "negative_eps2_rows"]
    failed = 0
    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    with (args.out / "summary.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        for number, path in enumerate(sorted(paths), 1):
            csv_file = f"spectrum_{number:04d}.csv"
            try:
                metadata, imag = collect(path, args.out / csv_file, args)
            except (OSError, EOFError, ValueError, ET.ParseError) as exc:
                failed += 1
                writer.writerow({"source": str(path), "status": "error", "message": str(exc)})
                print(f"{path}: {exc}", file=sys.stderr)
                continue
            writer.writerow({"source": str(path), "status": "parsed", "spectrum": csv_file, **metadata})
            if args.plot:
                label = f"{number:04d} {path.parent.name}"
                ax.plot([row[0] for row in imag], [scalar(row[1:], args.component) for row in imag], label=label)

    if args.plot and failed < len(paths):
        ax.set(xlabel="Photon energy (eV)", ylabel=f"Im ε ({args.component})")
        if args.plot_window:
            ax.set_xlim(*args.plot_window)
        if len(paths) - failed <= 12:
            ax.legend(fontsize="small")
        fig.savefig(args.out / "eps2.png", dpi=180)
    print(f"Parsed {len(paths) - failed}/{len(paths)} files. Summary: {args.out / 'summary.csv'}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
