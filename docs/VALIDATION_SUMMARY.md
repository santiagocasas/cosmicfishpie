# Phase 2 Validation Summary

**Branch:** `phase-1-validation-integration` (commit: 73ace12)  
**Date:** 2026-08-17  
**Status:** ✓ All 6 validation cases completed  
**CLASS version:** 3.3.4.0 (installed via pip)

---

## Overview

Ran 6 backend comparison cases across photometric and spectroscopic observables using validation configurations. While all runs executed successfully, **parameter-level sigma deviations reveal significant backend differences in extended dark energy models (w0waCDM)**, particularly in photometric observations.

---

## Validation Cases Summary

| # | Backends | Observable | Model | Max Sigma Dev | Status | Notes |
|---|----------|-----------|-------|--------------|--------|-------|
| 01 | CLASS ↔ CAMB | Photo (GCph, WL) | w0waCDM | 26.7% Omegab | ⚠️ **CAUTION** | 6 params >10% (w0, wa, Omegab, h, ns, sigma8) |
| 02 | CLASS ↔ CAMB | Spectro (GCsp) | w0waCDM | 0.7% | ✓ PASS | Excellent agreement |
| 03 | CAMB ↔ CLASS | Photo (GCph, WL) | nuCDM | 8.2% mnu | ✓ PASS | mnu and h only above 5% |
| 04 | CAMB ↔ CLASS | Spectro (GCsp) | nuCDM | 1.8% | ✓ PASS | Very good agreement |
| 05 | Symbolic ↔ CAMB | Spectro (GCsp) | ΛCDM | 27.7% h | ℹ️ EXPECTED | Symbolic is analytic; 8 params >10% |
| 06 | Symbolic ↔ CAMB | Photo (GCph, WL) | ΛCDM | 29.9% Omegab | ℹ️ EXPECTED | Symbolic is analytic; 3 params >10% |

---

## Detailed Findings

### Case 01: CLASS vs CAMB — Photometric w0waCDM (**ALERT**)

**Parameters with >10% deviation:**
- Omegab: **26.67%** ← Most problematic
- h: **24.20%**
- ns: **16.09%**
- sigma8: **14.02%**
- wa: **12.94%** ← Dark energy parameter
- w0: **10.07%** ← Dark energy parameter

**Likely causes:**
- Nonlinear power spectrum (halofit) derivatives for extended dark energy are computed differently between CLASS and CAMB
- CLASS uses `non linear: halofit` with specific accuracy settings
- CAMB uses `halofit_version: takahashi` with `dark_energy_model: ppf` (PPF = parametrized post-Friedmann)
- Photometric observables couple nonlinear scales where backend differences amplify

**Matrix metrics:**
- Entrywise rel. max: 34.2% (very high)
- Frobenius relative diff: 8.5%
- Diag ratio range: [0.83, 1.03]

**Recommendation:** This case requires detailed investigation before using either backend for w0waCDM photometric forecasts. Consider documenting backend choice and applying systematic uncertainty estimates.

---

### Case 02: CLASS vs CAMB — Spectroscopic w0waCDM (**PASS**)

All parameters within 10% deviation; spectroscopic (linear-scale dominated) observables show excellent backend consistency.

---

### Case 03: CAMB vs CLASS — Photometric nuCDM (**PASS**)

Minimal deviation except:
- mnu: 8.15% (neutrino mass handling, expected)
- h: 6.07% (minor)

---

### Case 04: CAMB vs CLASS — Spectroscopic nuCDM (**PASS**)

All parameters within 10%; nuCDM spectroscopic forecasts are robust across backends.

---

### Cases 05–06: Symbolic vs CAMB (**EXPECTED DIVERGENCE**)

Symbolic backend uses analytic power spectrum approximations (no halofit), so nonlinear parameters and precision cosmology show large deviations. These cases validate that the framework correctly runs all three backends; deviations are expected and not a regression.

---

## Test Suite Baseline

- **Unit tests:** 217 passed, 1 skipped, 3 failed (doctest docstrings; non-critical)
- **Code coverage:** 57%
- **Total runtime:** ~8.5 hours (includes class initialization and derivatives)

---

## Recommendations for Consolidation

1. **Document Case 01 findings** in a DECISION/SPEC file:
   - Flag w0waCDM + photometric as requiring backend-specific validation
   - Record maximum acceptable sigma deviation thresholds
   - Recommend using CAMB as default for w0waCDM photometric (or vice versa with clear justification)

2. **Add validation gate** in CI/CD:
   - Periodically re-run validation suite (e.g., monthly or per major CLASS/CAMB release)
   - Alert if sigma deviations exceed documented thresholds

3. **Investigate nonlinear halofit differences:**
   - Compare PPF vs halofit parameterizations in CLASS
   - Check if CLASS has alternative dark energy implementations

4. **Consider splitting acceptance criteria:**
   - Spectroscopic: <5% threshold (tight)
   - Photometric LCDM/nuCDM: <10% threshold (moderate)
   - Photometric w0waCDM: <20% threshold (loose) with documented caveats

---

## Artifacts

- **Combined report:** `scripts/benchmark_results/validate_many_20260817_132902/report_many_20260817_133043.html`
- **Per-case reports:** `scripts/benchmark_results/validate_many_20260817_132902/{01..06}_*/report_single.html`
- **Comparison JSON:** `scripts/benchmark_results/validate_many_20260817_132902/*/compare_fishers_*.json`
- **Fisher matrices:** `.txt`, `.paramnames`, `_specs.json` for each case
