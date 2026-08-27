#!/usr/bin/env python
# coding: utf-8

"""Full 7-parameter Case 01 re-run: CLASS-DP (original) vs CAMB (forked Halofit tolerance).

Thin wrapper around ``run_fisher_compare_w0wa_fast.py``. Despite its name, that
script is a fully generic fast-Fisher-comparison engine driven by ``--params``:
passing all 7 cosmological parameters from Case 01 (Omegam, Omegab, h, ns,
sigma8, w0, wa) reproduces the exact original validation-suite comparison
(scripts/validation_configs/compare_run_config.env_01_class_camb_photo_mpvalidation_w0waCDM),
just invoked directly in Python instead of via the shell/env-file wrapper.

This variant pairs:
  - CLASS with the ORIGINAL default-precision (DP) YAML, unchanged:
    cosmicfishpie/configs/default_boltzmann_yaml_files/class/mpvalidation.yaml
  - CAMB with the P3 YAML:
    cosmicfishpie/configs/default_boltzmann_yaml_files/camb/mpvalidation.yaml

The purpose of this DP variant is to isolate how much of Case 01's original
26.67%-level discrepancy was due to CLASS precision (DP vs HP) versus CAMB's
Halofit numerical noise (PyPI-exposed halofit_tol_sigma vs the fork's
hardcoded 1e-6 in fortran/halofit.f90). Compare its PASS/FAIL result against
``run_case01_full7_class_hp_vs_camb.py`` (same CAMB, HP-instead-of-DP CLASS)
to disentangle the two effects.

CAMB is expected to be the local fork with a hardcoded Halofit tolerance of
1e-6 in fortran/halofit.f90 (commit c52551a4c4cc9bf8d6fa64b4cb829365c89242b5,
remote git@github.com:santiagocasas/CAMB.git, installed editable via
``uv pip install -e /path/to/CAMB --python .venv/bin/python``), reproducing
the precision setting used in Casas et al. 2023 (arXiv:2303.09451). The
underlying engine's ``run_metadata.json`` automatically records whichever
CAMB backend is actually installed at run time -- including its git commit,
remote URL, and dirty flag when it is an editable local-path install -- via
its provenance-detection code (see
``run_fisher_compare_w0wa_fast.py::_package_provenance``). No manual
bookkeeping of the fork commit is required, but the CAMB fork must actually
be installed in the venv *before* running this script (and must be
reinstalled after any ``uv run ...`` invocation, since ``uv run`` resyncs the
venv to the ``camb==1.6.4`` PyPI pin in pyproject.toml). Always invoke this
script with the venv's python directly (``.venv/bin/python``), never via
``uv run python``, or the fork will be silently reverted mid-run.

Any extra CLI args are forwarded to the underlying engine, so you can
override --accuracy, --feedback, --omp-threads, --outdir, --threshold, etc.

Example
-------
.venv/bin/python scripts/run_case01_full7_class_dp_vs_camb.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent / "run_fisher_compare_w0wa_fast.py"
CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "cosmicfishpie"
    / "configs"
    / "default_boltzmann_yaml_files"
)

FULL7_PARAMS = "Omegam,Omegab,h,ns,sigma8,w0,wa"
CLASS_DP_YAML = str(CFG_DIR / "class" / "mpvalidation.yaml")
CAMB_P3_YAML = str(CFG_DIR / "camb" / "mpvalidation.yaml")


def main() -> int:
    cmd = [
        sys.executable,
        str(ENGINE),
        "--params",
        FULL7_PARAMS,
        "--code-a",
        "class",
        "--code-b",
        "camb",
        "--yaml-a",
        CLASS_DP_YAML,
        "--yaml-b",
        CAMB_P3_YAML,
        "--compare",
    ] + sys.argv[1:]
    print("[case01-full7-class-dp-vs-camb] Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
