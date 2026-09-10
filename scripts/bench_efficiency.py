#!/usr/bin/env python
"""Benchmark lensing-efficiency integrators in photo_obs.

Measures wall-clock time for three implementations and compares outputs:
  - memo_integral_efficiency      (legacy O(N^2) with Python loop)
  - faster_integral_efficiency    (vectorized O(N^2))
  - much_faster_integral_efficiency (O(N) cumulative trapezoid)

Uses a lightweight ComputeCls instance configured similarly to the tests.

Run:
  python scripts/bench_efficiency.py
"""

from time import perf_counter

import numpy as np

from cosmicfishpie.fishermatrix import cosmicfish as cff
from cosmicfishpie.LSSsurvey.photo_obs import (
    ComputeCls,
    faster_integral_efficiency,
    memo_integral_efficiency,
    much_faster_integral_efficiency,
)


def build_computecls():
    options = {
        "accuracy": 1,
        "outroot": "bench_efficiency",
        "results_dir": "results/",
        "derivatives": "3PT",
        "ell_sampling": 25,
        "nonlinear": True,
        "feedback": 0,  # quiet
        "specs_dir": "cosmicfishpie/configs/default_survey_specifications/",
        "survey_name": "Euclid",
        "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
        "cosmo_model": "LCDM",
        "code": "symbolic",
    }

    fiducial = {"Omegam": 0.32, "h": 0.67}
    freepars = {"Omegam": 0.01, "h": 0.01}
    observables = ["WL", "GCph"]

    cosmoFM = cff.FisherMatrix(
        fiducialpars=fiducial,
        freepars=freepars,
        options=options,
        observables=observables,
        cosmoModel=options["cosmo_model"],
        surveyName=options["survey_name"],
    )

    cosmopars = {"Omegam": 0.3, "h": 0.7}
    cls = ComputeCls(cosmopars, cosmoFM.photopars, cosmoFM.IApars, cosmoFM.photobiaspars)
    return cls


def time_call(fn, *args, **kwargs):
    t0 = perf_counter()
    out = fn(*args, **kwargs)
    t1 = perf_counter()
    return out, (t1 - t0)


def compare_vectors(a, b, name_a="A", name_b="B", thresh=1e-12):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.abs(b) > thresh
    rel = np.zeros_like(a)
    rel[mask] = (a[mask] / b[mask]) - 1.0
    abs_rel_max = np.max(np.abs(rel[mask])) if np.any(mask) else 0.0
    abs_max_near0 = np.max(np.abs(a[~mask] - b[~mask])) if np.any(~mask) else 0.0
    print(f"Comparison {name_a}/{name_b}:")
    print(f"  - rel max |ratio-1| over |{name_b}|>{thresh:g}: {abs_rel_max:.3e}")
    print(f"  - abs max near zero (|{name_b}|<={thresh:g}): {abs_max_near0:.3e}")


def main():
    cls = build_computecls()

    z = cls.z
    i = cls.binrange_WL[0]

    ngal_func = cls.window.norm_ngal_photoz
    comoving_func = cls.cosmo.comoving

    # Prepare memo_integral inputs (legacy path)
    zsamp = cls.zsamp
    zint_mat = np.linspace(z, z[-1], zsamp).T  # rows: z_k..z_max
    dx = float(np.mean(np.diff(z)))

    print("Benchmarking lensing efficiency integrators on WL bin", i)

    # memo_integral_efficiency (legacy, O(N^2) with Python loop)
    f_memo, t_memo = time_call(
        memo_integral_efficiency, i, ngal_func, comoving_func, z, zint_mat, dx
    )
    eff_memo, t_eval_memo = time_call(f_memo, z)

    # faster_integral_efficiency (vectorized O(N^2))
    f_fast, t_fast = time_call(faster_integral_efficiency, i, ngal_func, comoving_func, z)
    eff_fast, t_eval_fast = time_call(f_fast, z)

    # much_faster_integral_efficiency (O(N))
    f_much, t_much = time_call(much_faster_integral_efficiency, i, ngal_func, comoving_func, z)
    eff_much, t_eval_much = time_call(f_much, z)

    # Report timings
    print("\nTimings (build + evaluate on z grid):")
    print(f"  memo_integral_efficiency        : {t_memo:.4f}s + {t_eval_memo:.4f}s")
    print(f"  faster_integral_efficiency      : {t_fast:.4f}s + {t_eval_fast:.4f}s")
    print(f"  much_faster_integral_efficiency : {t_much:.4f}s + {t_eval_much:.4f}s")

    # Speedups relative to vectorized O(N^2)
    print("\nSpeedup vs faster_integral_efficiency (vectorized O(N^2)):")
    print(
        f"  much_faster is {(t_fast / t_much):.1f}x faster (constructor only), {( (t_fast + t_eval_fast) / (t_much + t_eval_much) ):.1f}x including evaluation"
    )
    print(f"  memo (legacy) is {(t_memo / t_fast):.1f}x slower than vectorized (constructor only)")

    # Output comparisons: ratios
    print("\nOutput agreement (ratios):")
    compare_vectors(eff_fast, eff_memo, name_a="fast", name_b="memo")
    compare_vectors(eff_much, eff_fast, name_a="much", name_b="fast")


if __name__ == "__main__":
    main()
