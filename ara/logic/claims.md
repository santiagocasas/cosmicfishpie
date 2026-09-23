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
