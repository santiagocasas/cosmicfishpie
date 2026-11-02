# Historical snapshot: CLASS vs CAMB w0waCDM deviation investigation

> **Historical validation-tracking note (2026-08-17).** This page is a point-in-time
> investigation, not current guidance. Use the
> [backend validation guide](backend_validation.md) for the maintained workflow and
> dashboard, and the [validation configuration directory](https://github.com/santiagocasas/cosmicfishpie/tree/main/scripts/validation/configs)
> for current case definitions and results. Commands below are illustrative of the
> diagnostic approach, not a guaranteed-current CLI.

**Case investigated:** Case 01 (CLASS vs CAMB, photometric w0waCDM)  
**Reference:** Casas et al., "Euclid: Validation of the MontePython forecasting tools," arXiv:[2303.09451](https://arxiv.org/abs/2303.09451) (S. Casas, J. Lesgourgues, N. Schöneberg et al., March 2023)

---

## Summary

The snapshot run showed a 26.67% deviation in Ωb and >10% deviations in five other parameters between CLASS and CAMB for photometric w0waCDM forecasts (Case 01) — far above the Euclid paper's <3% for comparable configurations. The paper itself reaches <1% CLASS/CAMB agreement when both solvers use matched high-precision settings, pointing to a configuration mismatch in that run rather than a fundamental backend incompatibility.

## Paper benchmarks (Casas et al., Section 4, optimistic setting)

| Comparison | Max deviation | Probe |
|---|---:|---|
| CF/int/CAMB vs CF/int/CLASS (matched HP/P3) | 0.55% | Photometric w0waCDM |
| CF/int/CAMB vs CF/ext/CAMB | 1.3% | Photometric w0waCDM |
| CosmicFish vs MontePython | 4.3% | Photometric w0waCDM |
| All methods vs IST:F median | 9% max | Photometric w0waCDM |
| CosmicFish/CLASS vs CosmicFish/CAMB | <0.5% | Spectroscopic w0waCDM |
| MontePython/Fisher vs CosmicFish | 2.5% | Spectroscopic w0waCDM |

At matched precision — CLASS high-precision ("HP", halofit tolerance ~10⁻⁸) and CAMB P3 (`accuracy_boost=3`, `l_accuracy_boost=3`, `halofit_tol_sigma=1.e-6`) — linear-power derivatives agree to 0.07%. Section 6.3 reports CLASS DP vs HP differs by 11% in marginalised errors (CAMB P2 vs P3 by ~1%), concluding: *"the CLASS user should employ the 'high-precision' settings when performing Fisher matrix forecasts with MontePython."*

## Snapshot results vs. paper expectation

| Case | Backends | Observable | Deviation | Paper expectation |
|---|---|---|---:|---|
| 01 | CLASS vs CAMB | Photometric w0waCDM | 26.67% | <1% (HP/P3) or ~5-11% max (DP) |
| 02 | CLASS vs CAMB | Spectroscopic w0waCDM | 0.7% | <1% — matches paper |
| 03 | CAMB vs CLASS | Photometric nuCDM | 8.2% | <3% |

Spectroscopic agreement (Case 02) matched the paper; photometric w0waCDM (Case 01) and photometric nuCDM (Case 03) did not — consistent with the paper's finding that nonlinear scales and extended dark-energy models need strict, matched precision settings to keep CLASS/CAMB agreement tight.

### Case 01, parameter by parameter

| Parameter | Snapshot deviation | Paper w0waCDM (HP/P3) | Paper w0waCDM (DP) |
|---|---:|---|---|
| Omegab | 26.67% | <1% | ~5% max |
| h | 24.20% | <1% | ~3% max |
| ns | 16.09% | <1% | ~2% max |
| sigma8 | 14.02% | <1% | ~4% max |
| wa | 12.94% | <1% | ~1% max |
| w0 | 10.07% | <1% | <1% typical |

All six parameters exceeding 10% points to a global accuracy mismatch between the two solver configurations in that run, not an isolated parameter issue.

## Likely configuration gaps identified at the time

- **CLASS precision:** default/DP settings instead of high-precision (HP) — `non_linear: halofit` with a tightened halofit tolerance (~10⁻⁸) and integration accuracy; the exact knobs depend on the CLASS version (see the CLASS docs for its "high precision" settings).
- **CAMB precision:** default P0/P1 instead of P3 (`accuracy_boost=3`, `l_accuracy_boost=3`, `halofit_tol_sigma=1.e-6`).
- **Halofit mismatch:** different halofit versions/tolerances between CLASS and CAMB couple into nonlinear-scale observables, which is where photometric forecasts are most sensitive — consistent with Case 02 (linear-scale dominated) passing while Case 01/03 did not.

## Reproducing and fixing a mismatch like this today

1. Compare the run's CLASS and CAMB config YAMLs under `cosmicfishpie/configs/default_boltzmann_yaml_files/` against Casas et al. Appendix A (CLASS HP, Table A.5; CAMB P3).
2. Re-run the comparison with matched high-precision settings using the maintained [backend validation workflow](backend_validation.md) — `bash scripts/validation/run_selected_validations.sh --help` prints the currently configured case IDs, or run `scripts/validation/run_fisher_compare_backends.py` directly (see the [scripts reference](scripts_reference.md) for its current CLI options).
3. Expect <1% deviation at matched HP/P3 settings; document any larger residual explicitly (different solver versions, different survey cuts, etc.).

## References (Casas et al., arXiv:2303.09451)

- Section 2.3.2: MontePython/Fisher derivative method.
- Section 4.2.2: photometric w0waCDM, <1% agreement at matched precision.
- Sections 6.1-6.3: precision-setting requirements and their impact.
- Appendix A: CAMB P3 and CLASS HP configuration tables.

## Conclusion

Casas et al. (2023) show CLASS and CAMB can agree to <1% on w0waCDM photometric forecasts at matched precision. The >10% deviation seen in this snapshot's Case 01 is evidence of a configuration asymmetry between the two solver setups at the time, not a fundamental incompatibility — Appendix A and Section 6 of the paper give the settings needed to close the gap.

---

**Snapshot date:** 2026-08-17 · Reference: https://arxiv.org/html/2303.09451v1
