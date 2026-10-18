"""EuclidEmulator2 nonlinear boost on a CLASS power-spectrum grid."""

from __future__ import annotations

import numpy as np


def class_ee2_parameters(classres) -> dict[str, float]:
    """Translate computed CLASS parameters into EuclidEmulator2 inputs.

    Omega_m includes massive neutrinos; m_ncdm is their summed mass in eV.
    The primordial amplitude is read from CLASS so sigma8-normalized runs
    use the amplitude actually computed by CLASS.
    """
    pars = classres.pars
    masses = pars.get("m_ncdm", 0.0)
    if isinstance(masses, str):
        mass_sum = sum(float(mass) for mass in masses.split(","))
    else:
        mass_sum = float(np.sum(masses))

    # Read the exact computed pivot amplitude, including sigma8-normalized runs.
    # Interpolating the primordial grid introduces a small bias which can push
    # valid amplitude inputs outside EE2 bounds near its hard-domain edges.
    amplitude = float(classres.get_current_derived_parameters(["A_s"])["A_s"])

    return {
        "h": float(classres.h()),
        "Omega_b": float(classres.Omega_b()),
        "Omega_m": float(classres.Omega_m()),
        "m_ncdm": mass_sum,
        "n_s": float(pars["n_s"]),
        "A_s": amplitude,
        "w0_fld": float(pars.get("w0_fld", -1.0)),
        "wa_fld": float(pars.get("wa_fld", 0.0)),
    }


def class_ee2_boost(classres, k_1mpc, redshifts) -> np.ndarray:
    """Return EE2 boost in (k, z) order on a CLASS grid.

    CLASS k is in 1/Mpc; EE2 k is in h/Mpc. Below EE2's native k
    range the boost is set to unity; above its range an error is raised.
    Redshifts are evaluated at their exact CLASS samples, without z
    interpolation or extrapolation. EE2 supports at most 100 z per call.
    """
    try:
        from euclidemu2 import PyEuclidEmulator
    except ImportError as exc:
        raise ImportError(
            "CLASS+EE2 requires the optional 'emulators' extra (euclidemu2)."
        ) from exc

    k = np.asarray(k_1mpc, dtype=float) / classres.h()
    z = np.asarray(redshifts, dtype=float)
    if k.ndim != 1 or z.ndim != 1 or not np.all(np.isfinite(k)) or not np.all(np.isfinite(z)):
        raise ValueError("CLASS+EE2 requires finite, one-dimensional k and redshift grids.")
    if np.any(z < 0) or np.any(z > 10):
        raise ValueError("EE2 supports redshifts from 0 to 10 (including CLASS grid padding).")

    emulator = PyEuclidEmulator()
    parameters = class_ee2_parameters(classres)
    boost = np.empty((len(k), len(z)))
    for start in range(0, len(z), 100):
        chunk = z[start : start + 100]
        native_k, values = emulator.get_boost(parameters, chunk.tolist())
        native_k = np.asarray(native_k)
        if np.any(k > native_k[-1]):
            raise ValueError(
                f"CLASS k grid exceeds EE2 maximum of {native_k[-1]:g} h/Mpc; "
                "reduce P_k_max_1/Mpc."
            )
        for index in range(len(chunk)):
            boost[:, start + index] = np.interp(k, native_k, values[index], left=1.0)
    return boost
