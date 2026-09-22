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
