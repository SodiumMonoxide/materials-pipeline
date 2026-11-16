"""Refine one material at a time and require two independent comparisons."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .convergence_metrics import load_spectrum, compare_spectra
from .convergence_report import write_reports


@dataclass(frozen=True)
class ConvergenceOptions:
    window: tuple[float, float]
    rtol: float
    atol: float
    k_growth: float
    k_steps: int
    kmeshes: tuple[tuple[int, int, int], ...]
    band_growth: float
    band_steps: int

    def validate(self):
        low, high = self.window
        if not all(map(math.isfinite, self.window)) or not 0 <= low < high:
            raise ValueError("--conv-window requires finite 0 <= MIN < MAX")
        if not all(math.isfinite(x) and x >= 0 for x in (self.rtol, self.atol)):
            raise ValueError("Convergence tolerances must be finite and nonnegative")
        if not all(math.isfinite(x) and x > 1 for x in (self.k_growth, self.band_growth)):
            raise ValueError("Convergence growth factors must be finite and greater than 1")
        if self.k_steps < 1 or self.band_steps < 1:
            raise ValueError("Convergence step limits must be positive")
        if any(value < 1 for mesh in self.kmeshes for value in mesh):
            raise ValueError("Convergence meshes require positive integers")


def mesh_levels(inputs, options):
    if options.window[1] > inputs["omegamax"]:
        raise ValueError("Convergence window exceeds OMEGAMAX")
    if inputs.get("drude_gamma") and options.window[0] == 0:
        raise ValueError("Drude convergence requires a positive lower energy inside the exported spectrum")
    baseline = tuple(inputs["kmesh"])
    vacuum = {axis - 1 for axis in inputs["vacuum_axes"]}
    levels = [baseline]
    if options.kmeshes:
        for mesh in options.kmeshes:
            last = levels[-1]
            if mesh == last or any(a < b for a, b in zip(mesh, last)) or any(mesh[i] != 1 for i in vacuum):
                raise ValueError("Each --conv-kmesh must increase the mesh, without decreasing a direction "
                                 "or changing a vacuum axis")
            levels.append(mesh)
    elif inputs["dimension"] != 0:
        for _ in range(options.k_steps):
            levels.append(tuple(value if i in vacuum else math.ceil(value * options.k_growth)
                                for i, value in enumerate(levels[-1])))
    return levels


@dataclass(frozen=True)
class Trial:
    mesh: tuple
    nbands: int
    attempt: str
    run_status: str
    results: dict
    spectrum: object


class Refinement:
    def __init__(self, output, report, options, solve, progress, baseline):
        self.output = output
        self.report = report
        self.options = options
        self.solve = solve
        self.progress = progress
        self.trials = {}
        self.comparisons = {}
        self.curves = []
        self.meshes = mesh_levels(baseline["inputs"], options)
        self.k_index = 0
        self.band_steps_used = 0
        self.lower_bands = None
        raw = baseline["inputs"]
        self.parallel = {"kpar": raw["kpar"], "ncore": raw["ncore"]}
        self.current = self.load_trial(baseline, self.meshes[0])
        self.report["mesh_ladder"] = self.meshes
        self.report["kmesh_required"] = len(self.meshes) > 1

    def load_trial(self, state, mesh):
        if state["status"] not in {"completed", "reused"}:
            raise ValueError(f"{state.get('attempt', 'VASP attempt')}: "
                             f"{state.get('message', state['status'])}")
        results = state["results"]
        if results.get("kmesh") != "x".join(map(str, mesh)):
            raise ValueError(f"{state['attempt']}: actual k mesh does not match the requested mesh")
        bands = int(results["NBANDS"])
        spectrum = load_spectrum(self.output / state["attempt"] / results["spectrum"])
        if spectrum.energy[0] > self.options.window[0] or spectrum.energy[-1] < self.options.window[1]:
            raise ValueError(f"{state['attempt']}: requested window is outside the exported "
                             f"{spectrum.energy[0]:g}–{spectrum.energy[-1]:g} eV range")
        trial = Trial(mesh, bands, state["attempt"], state["status"], results, spectrum)
        self.trials[(mesh, bands)] = trial
        label = f'{state["attempt"].split("/")[-1]} · {"×".join(map(str, mesh))} · {bands} bands'
        self.curves.append((label, spectrum))
        self.report["cases"].append({"attempt": trial.attempt, "kmesh": mesh, "nbands": bands,
                                     "run_status": trial.run_status, "warnings": results.get("warnings", [])})
        return trial

    def calculate(self, mesh, nbands, exact=True):
        if trial := self.trials.get((mesh, nbands)):
            return trial
        self.progress(f'checking {"×".join(map(str, mesh))}, requested NBANDS={nbands}')
        trial = self.load_trial(self.solve({"kmesh": list(mesh), "nbands": nbands, **self.parallel}), mesh)
        if trial.nbands < nbands or (exact and trial.nbands != nbands):
            raise ValueError(f"{trial.attempt}: expected NBANDS={nbands}, got {trial.nbands}; "
                             "adjust the parallel settings")
        return trial

    def compare(self, axis, previous):
        current = self.current
        key = (axis, previous.attempt, current.attempt)
        if key in self.comparisons:
            return self.comparisons[key]
        metrics = compare_spectra(previous.spectrum, current.spectrum, self.options.window,
                                  self.options.rtol, self.options.atol)
        row = {**metrics, "axis": axis, "previous_attempt": previous.attempt, "attempt": current.attempt,
               "kmesh": current.mesh, "nbands": current.nbands, "previous_kmesh": previous.mesh,
               "previous_nbands": previous.nbands, "run_status": current.run_status}
        if axis == "nbands":
            coverage = current.results.get("min_empty_transition_ceiling_eV")
            highest = current.results.get("max_highest_band_occupation")
            valid_bands = (coverage is not None and coverage >= self.options.window[1]
                           and highest is not None and highest <= 1e-3)
            row.update(coverage_eV=coverage, coverage_ok=valid_bands, passed=metrics["passed"] and valid_bands)
            if not valid_bands:
                row["detail"] = "Empty-band coverage cannot be verified over the requested window"
        self.report["steps"].append(row)
        self.comparisons[key] = row["passed"]
        ratio = "unbounded" if row["worst_ratio"] is None else f'{row["worst_ratio"]:.3g}× threshold'
        self.progress(f"{axis}: {previous.attempt} → {current.attempt}, {ratio}, "
                      f'{"pass" if row["passed"] else "continue"}')
        return row["passed"]

    def mesh_pass(self):
        if len(self.meshes) == 1:
            return True
        if self.k_index == 0:
            return False
        return self.compare("kmesh", self.calculate(self.meshes[self.k_index - 1], self.current.nbands))

    def advance_mesh(self):
        self.k_index += 1
        self.current = self.calculate(self.meshes[self.k_index], self.current.nbands)

    def refine_bands(self):
        if self.lower_bands is not None and self.compare(
            "nbands", self.calculate(self.current.mesh, self.lower_bands)
        ):
            return True
        while self.band_steps_used < self.options.band_steps:
            previous = self.current
            self.lower_bands = previous.nbands
            requested = math.ceil(previous.nbands * self.options.band_growth)
            self.current = self.calculate(previous.mesh, requested, exact=False)
            self.band_steps_used += 1
            if self.compare("nbands", previous):
                return True
        return False

    def run(self):
        last_mesh = len(self.meshes) - 1
        while True:
            while not self.mesh_pass() and self.k_index < last_mesh:
                self.advance_mesh()
            if not self.refine_bands():
                return "not_converged", "Band refinement limit reached before meeting the tolerance and coverage"
            if self.mesh_pass():
                return "converged", "Required same-axis comparisons passed at the final settings"
            if self.k_index == last_mesh:
                return "not_converged", "k-mesh ladder exhausted before meeting the tolerance at the final NBANDS"
            self.advance_mesh()


def run_convergence(output: Path, baseline: dict, options: ConvergenceOptions, solve, progress):
    fixed_keys = ("encut", "cshift", "sigma", "omegamax", "nedos", "ediff", "kpar", "ncore",
                  "dimension", "vacuum_axes", "drude_gamma", "potcar_sha256")
    inputs = baseline.get("inputs", {})
    report = {
        "material_id": baseline["material_id"], "status": "failed", "message": "",
        "window_eV": options.window, "rtol": options.rtol, "atol": options.atol,
        "baseline_attempt": baseline.get("attempt", ""), "final_attempt": "", "final_kmesh": None,
        "final_nbands": None, "kmesh_required": True, "steps": [], "cases": [],
        "band_growth": options.band_growth, "band_steps_limit": options.band_steps,
        "fixed_settings": {key: inputs[key] for key in fixed_keys if key in inputs},
    }
    refinement = None
    try:
        if baseline["status"] not in {"completed", "reused"}:
            raise ValueError(f'Screening failed: {baseline.get("message", baseline["status"])}')
        refinement = Refinement(output, report, options, solve, progress, baseline)
        report["status"], report["message"] = refinement.run()
    except (OSError, ValueError, OverflowError, KeyError, csv.Error) as exc:
        report["message"] = str(exc)
    if refinement is not None:
        final = refinement.current
        report.update(final_attempt=final.attempt, final_kmesh=final.mesh, final_nbands=final.nbands)
    report_dir = output / baseline["material_id"]
    if report_dir.is_symlink():
        raise ValueError(f"Material output directory is a symlink: {report_dir}")
    report_dir.mkdir(exist_ok=True)
    curves = []
    if refinement is not None:
        for case, (label, spectrum) in zip(report["cases"], refinement.curves):
            if case["attempt"] == report["baseline_attempt"]:
                label = f"Screen · {label}"
            if report["status"] == "converged" and case["attempt"] == report["final_attempt"]:
                label = f"Reference · {label}"
            curves.append((label, spectrum))
    write_reports(report_dir, report, curves)
    return report
