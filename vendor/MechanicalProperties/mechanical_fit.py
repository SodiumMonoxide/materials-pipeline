"""Fit 3D elastic stiffness from paired infinitesimal strains and Cauchy stresses."""

import math

import numpy as np

from mech import VOIGT_ORDER, analyze_stiffness


COMPONENT_AXES = dict(zip(VOIGT_ORDER, ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))))
VASP_TO_VOIGT = (0, 1, 2, 4, 5, 3)  # From XX YY ZZ XY YZ ZX.


def as_finite_array(values, shape, name):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}")
    return array


def engineering_amplitude(amplitude):
    try:
        amplitude = float(amplitude)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("strain amplitude must be a finite number") from error
    # |h| < 1 keeps F nonsingular for both signs of every supported mode.
    if not math.isfinite(amplitude) or abs(amplitude) >= 1:
        raise ValueError("strain amplitude must have magnitude less than 1")
    return amplitude


def deformed_cell(lattice, component, amplitude):
    """Apply F=I+epsilon to ASE row vectors, keeping fractional coordinates fixed.

    Amplitudes are dimensionless and signed. A shear amplitude is the engineering
    shear gamma, so the two off-diagonal entries of epsilon are gamma/2 each.
    """
    lattice = as_finite_array(lattice, (3, 3), "lattice")
    if np.linalg.matrix_rank(lattice) != 3:
        raise ValueError("lattice must be a nonsingular 3D cell")
    if component not in COMPONENT_AXES:
        raise ValueError(f"unknown strain component {component!r}")
    amplitude = engineering_amplitude(amplitude)
    row, column = COMPONENT_AXES[component]
    gradient = np.eye(3)
    if row == column:
        gradient[row, column] += amplitude
    else:
        gradient[row, column] = gradient[column, row] = amplitude / 2
    return lattice @ gradient.T


def vasp_stress_to_gpa(stress_kbar):
    """Convert a VASP 'in kB' row to tensile-positive stress in canonical order."""
    raw = as_finite_array(stress_kbar, (6,), "VASP stress")
    return (-0.1 * raw[list(VASP_TO_VOIGT)]).tolist()


def paired_stresses(cases):
    indexed = {}
    for case in cases:
        try:
            component = case["component"]
            amplitude = engineering_amplitude(case["amplitude"])
            stress = as_finite_array(case["stress_GPa"], (6,), "case stress_GPa")
        except (KeyError, TypeError) as error:
            raise ValueError("each case needs component, amplitude, and stress_GPa fields") from error
        if component not in COMPONENT_AXES:
            raise ValueError(f"unknown strain component {component!r}")
        if amplitude == 0:
            raise ValueError("zero strain belongs in reference_stress")
        key = (component, amplitude)
        if key in indexed:
            raise ValueError(f"duplicate strain case {component} {amplitude:+g}")
        indexed[key] = stress

    magnitudes = sorted({abs(amplitude) for _, amplitude in indexed})
    if not magnitudes:
        raise ValueError("at least one complete set of six positive/negative strain pairs is required")
    missing = [f"{component} {sign * magnitude:+g}" for magnitude in magnitudes
               for component in VOIGT_ORDER for sign in (-1, 1)
               if (component, sign * magnitude) not in indexed]
    if missing:
        raise ValueError(f"missing strain cases: {', '.join(missing)}")

    plus = np.array([np.column_stack([indexed[component, h] for component in VOIGT_ORDER])
                     for h in magnitudes])
    minus = np.array([np.column_stack([indexed[component, -h] for component in VOIGT_ORDER])
                      for h in magnitudes])
    return np.asarray(magnitudes), plus, minus


def residual_summary(positive, negative):
    residuals = np.stack((positive, negative))
    rms = np.linalg.norm(residuals.ravel()) / math.sqrt(residuals.size)
    return {"rms_GPa": float(rms), "max_abs_GPa": float(np.max(np.abs(residuals)))}


def fit_stress_strain(cases, reference_stress):
    """Return the stiffness and fit diagnostics using balanced +/- strain pairs.

    Each case is a dictionary with 'component', signed 'amplitude', and
    'stress_GPa'. Both reference and case stresses are tensile-positive vectors
    in VOIGT_ORDER. At each magnitude, provide both signs of each of the six
    pure strain modes. The reported stability applies near zero reference stress.
    """
    reference = as_finite_array(reference_stress, (6,), "reference_stress")
    magnitudes, plus, minus = paired_stresses(cases)
    step = magnitudes[:, None, None]
    plus_delta = plus - reference[:, None]
    minus_delta = minus - reference[:, None]
    slopes = (plus_delta - minus_delta) / (2 * step)
    # Fits with a fixed intercept at the reference stress and equal observation weights.
    raw = np.average(slopes, axis=0, weights=(magnitudes / magnitudes[-1]) ** 2)
    analysis = analyze_stiffness(raw)
    symmetric = np.asarray(analysis["C_symmetric_GPa"])
    raw_residuals = (plus_delta - step * raw, minus_delta + step * raw)
    symmetric_residuals = (plus_delta - step * symmetric, minus_delta + step * symmetric)
    even_stress = (plus_delta + minus_delta) / 2
    slope_spread = float(np.max(np.ptp(slopes, axis=0))) if len(magnitudes) > 1 else None
    per_magnitude = [
        {
            "amplitude": float(magnitude),
            "C_central_GPa": slopes[i].tolist(),
            "C_positive_one_sided_GPa": (plus_delta[i] / magnitude).tolist(),
            "C_negative_one_sided_GPa": (-minus_delta[i] / magnitude).tolist(),
            "even_stress_offset_GPa": even_stress[i].tolist(),
            "positive_stress_GPa": plus[i].tolist(),
            "negative_stress_GPa": minus[i].tolist(),
            "positive_residual_raw_GPa": raw_residuals[0][i].tolist(),
            "negative_residual_raw_GPa": raw_residuals[1][i].tolist(),
        }
        for i, magnitude in enumerate(magnitudes)
    ]
    return {
        "method": "paired central slopes with an equal-weight linear stress fit",
        "voigt_order": VOIGT_ORDER,
        "strain_convention": "infinitesimal engineering strain; tensile-positive Cauchy stress",
        "stability_scope": "3D bulk near zero stress; test on (C + C.T) / 2",
        "reference_stress_GPa": reference.tolist(),
        "reference_pressure_GPa": float(-np.mean(reference[:3])),
        "reference_max_abs_stress_GPa": float(np.max(np.abs(reference))),
        **analysis,
        "fit_diagnostics": {
            "max_amplitude_slope_spread_GPa": slope_spread,
            "max_abs_even_stress_offset_GPa": float(np.max(np.abs(even_stress))),
            "raw_residual": residual_summary(*raw_residuals),
            "symmetric_residual": residual_summary(*symmetric_residuals),
            "per_magnitude": per_magnitude,
        },
    }
