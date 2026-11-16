"""Compare dielectric spectra over a common, explicitly covered energy window."""

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from optics import AXES


CHANNELS = tuple(f"eps{part}_{axis}" for part in (1, 2) for axis in AXES)
VACUUM_DIAGONAL = frozenset(("eps1_xx", "eps1_yy", "eps1_zz"))


@dataclass(frozen=True)
class Spectrum:
    energy: tuple[float, ...]
    channels: dict[str, tuple[float, ...]]

    def __post_init__(self):
        if (len(self.energy) < 2 or self.energy[0] < 0
                or not all(math.isfinite(value) for value in self.energy)
                or any(left >= right for left, right in zip(self.energy, self.energy[1:]))):
            raise ValueError("Spectrum needs at least two finite, strictly increasing, nonnegative energies")
        if self.channels.keys() != set(CHANNELS):
            raise ValueError("Spectrum needs all twelve real and imaginary dielectric channels")
        for channel, values in self.channels.items():
            if len(values) != len(self.energy) or not all(math.isfinite(value) for value in values):
                raise ValueError(f"Invalid {channel} values in spectrum")


def load_spectrum(path: Path) -> Spectrum:
    """Read the dielectric columns produced by optics, with optional derived columns."""
    fields = ("energy_eV", *CHANNELS)
    column_values: dict[str, list[float]] = {name: [] for name in fields}
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or not set(fields).issubset(header):
                raise ValueError(f"Missing or duplicate spectrum columns in {path}")
            for row in reader:
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("wrong number of columns")
                    values = [float(row[name]) for name in fields]
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid spectrum row {reader.line_num} in {path}: {exc}") from exc
                for name, value in zip(fields, values):
                    column_values[name].append(value)
    except csv.Error as exc:
        raise ValueError(f"Malformed spectrum CSV in {path}: {exc}") from exc
    try:
        return Spectrum(tuple(column_values.pop("energy_eV")),
                        {name: tuple(values) for name, values in column_values.items()})
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _interpolation_weights(grid, points):
    weights = []
    for energy in points:
        right = bisect_left(grid, energy)
        if grid[right] == energy:
            weights.append((right, right, 0.0))
        else:
            left = right - 1
            fraction = (energy - grid[left]) / (grid[right] - grid[left])
            weights.append((left, right, fraction))
    return weights


def _interpolate(values, weights):
    return [(1.0 - weight) * values[left] + weight * values[right]
            for left, right, weight in weights]


def _rms(values, widths):
    scale = max(map(abs, values))
    if scale == 0:
        return 0.0
    if not math.isfinite(scale):
        raise ValueError("Spectrum difference exceeded the finite numerical range")
    scaled = [value / scale for value in values]
    # Integrate the square of a linear interpolant exactly on each interval.
    mean_square = math.fsum(
        width * (left * left + left * right + right * right) / 3
        for width, left, right in zip(widths, scaled, scaled[1:])
    )
    return scale * math.sqrt(min(1.0, mean_square))


def _ratio(numerator, denominator):
    if not denominator:
        return None
    result = numerator / denominator
    return result if math.isfinite(result) else None


def compare_spectra(previous: Spectrum, reference: Spectrum, window: tuple[float, float],
                    rtol: float, atol: float) -> dict:
    """Require every channel to satisfy RMS(change) <= atol + rtol * RMS(reference).

    The real diagonal reference uses epsilon_1 minus its vacuum value. The
    comparison integrates piecewise linear spectra on both sets of energy knots.
    """
    start, end = window
    if not all(map(math.isfinite, (start, end))) or not 0 <= start < end:
        raise ValueError("Energy window must have finite bounds with 0 <= start < end")
    if not all(map(math.isfinite, (rtol, atol))) or min(rtol, atol) < 0:
        raise ValueError("Convergence tolerances must be finite and nonnegative")
    common_start = max(previous.energy[0], reference.energy[0])
    common_end = min(previous.energy[-1], reference.energy[-1])
    if common_start > common_end:
        raise ValueError("Spectra have no common energy range")
    if start < common_start or end > common_end:
        raise ValueError(
            f"Requested window {start:g}–{end:g} eV exceeds common coverage "
            f"{common_start:g}–{common_end:g} eV"
        )
    points = sorted({start, end, *(energy for energy in previous.energy if start < energy < end),
                     *(energy for energy in reference.energy if start < energy < end)})
    widths = [(right - left) / (end - start) for left, right in zip(points, points[1:])]
    previous_weights = _interpolation_weights(previous.energy, points)
    reference_weights = _interpolation_weights(reference.energy, points)
    channels = {}
    for channel in CHANNELS:
        before = _interpolate(previous.channels[channel], previous_weights)
        after = _interpolate(reference.channels[channel], reference_weights)
        baseline = float(channel in VACUUM_DIAGONAL)
        change_rms = _rms([left - right for left, right in zip(before, after)], widths)
        reference_rms = _rms([value - baseline for value in after], widths)
        threshold = atol + rtol * reference_rms
        if not math.isfinite(threshold):
            raise ValueError(f"Convergence threshold exceeded the finite numerical range for {channel}")
        channels[channel] = {
            "difference_rms": change_rms,
            "reference_rms": reference_rms,
            "relative_error": _ratio(change_rms, reference_rms),
            "threshold": threshold,
            "ratio": _ratio(change_rms, threshold) if change_rms else 0.0,
            "passed": change_rms <= threshold,
        }
    worst_channel = max(
        channels, key=lambda name: math.inf if channels[name]["ratio"] is None else channels[name]["ratio"]
    )
    return {
        "passed": all(channel["passed"] for channel in channels.values()),
        "worst_channel": worst_channel,
        "worst_ratio": channels[worst_channel]["ratio"],
        "channels": channels,
    }
