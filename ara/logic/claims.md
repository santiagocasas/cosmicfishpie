# Claims

## C01: Spawn-safe worker-local likelihood execution
- **Statement**: For the symbolic backend, a two-worker explicit spawn pool initialized from plain configuration and precomputed NumPy cells performs repeated Nautilus likelihood evaluations without the prior `mappingproxy` pickling failure or fork deadlock.
- **Status**: supported
- **Provenance**: user
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Job 15629768 raises a serialization error, stalls with workers blocked before likelihood progress, or fails before completing the smoke chain for a multiprocessing-specific reason.
- **Proof**: [jobs_punch/logs/cfp-symbolic-spawn-2x1-smoke.15629768.out, jobs_punch/logs/cfp-symbolic-spawn-2x1-smoke.15629768.err]
- **Dependencies**: []
- **Tags**: nautilus, multiprocessing, spawn, symbolic
- **From staging**: O02

## V-C01: P_cb reduces neutrino Fisher amplification
- **Statement**: For the tested Euclid photometric massive-neutrino validation, using P_cb for galaxy clustering makes the mnu response larger and monotonic and reduces the marginalized CAMB-vs-CLASS mnu deviation from 10.06% to 5.53%, primarily by improving Fisher conditioning rather than unmarginalized backend agreement.
- **Status**: supported
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Falsification criteria**: A reproducible derivative calculation with the recorded solver profiles and 0.006-eV step fails to reproduce the P_mm zero crossing/P_cb monotonic response, or a controlled Fisher comparison shows no reduction in marginalization amplification when only the tracer changes.
- **Proof**: [`scripts/validation_configs/PCB_NEUTRINO_DERIVATIVE_ANALYSIS.md`, `scripts/benchmark_results/compare_photo_camb_vs_class_cfg_a5eb631459/`]
- **Dependencies**: []
- **Tags**: massive-neutrinos, P_cb, P_mm, Fisher-conditioning, CAMB, CLASS
- **From staging**: V-O02

## C02: Padded CLASS HMcode samples can corrupt photometric splines
- **Statement**: With the matched CLASS HMcode 2020 profile, the explicit array API can sporadically return non-finite nonlinear total-matter or cb power at CLASS-added samples beyond `P_k_max_1/Mpc` or `z_max_pk`; fitting those samples can propagate NaNs through the photometric spectra.
- **Status**: supported
- **Provenance**: ai-executed
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A captured failing run places every non-finite sample inside the configured nonlinear domain, or preserving the padded samples while excluding non-finite values fails to remove the downstream NaNs.
- **Proof**: [`trace/exploration_tree.yaml:N41`, `/tmp/opencode/probe_w0_nan.py`, `cosmicfishpie/cosmology/cosmology.py:_class_nonlinear_pk_grid`]
- **Dependencies**: []
- **Tags**: CLASS, HMcode-2020, nonlinear-power, interpolation, photometric-Fisher
- **From staging**: O10

## C03: SYREN nonlinear tilt response contributes to the Fisher discrepancy
- **Statement**: At the recorded fiducial, SYREN-NEW has a substantially suppressed low-redshift nonlinear ns response relative to EE2, with strong cancellation from explicit ns dependence in term3. Fixing ns reduces several saved LCDM marginalized-error differences, supporting an important ns-degeneracy contribution without establishing sole causality or a transcription bug.
- **Status**: supported
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Falsification criteria**: Reproducing the documented fiducial calculations fails to recover the response deficit, differentiated term3 cancellation, or reduction in saved-Fisher error differences after fixing ns.
- **Proof**: [`ara/evidence/tables/syren_ns_response_2026-09-29.yaml`, `docs/source/syren_ns_response.md`, `docs/source/_static/ns_response/ns_response.npz`]
- **Dependencies**: []
- **Tags**: SYREN-NEW, ns, nonlinear-response, Fisher-degeneracies
- **From staging**: O11
