# Photometric Pipeline Optimization Results

## Scope

This document records the performance comparison between `main` and `perf-v2`
for the photometric Fisher pipeline using the shared comparison tools:

- `shared-tools/compare_photo_fisher.py`
- `shared-tools/compare_photo_fisher.sh`

The calculation uses Euclid photometric specifications, `WL` and `GCph`
observables, 18 free parameters, three-point derivatives, and four OpenMP
threads. Fisher snapshots were compared numerically after each run.

## Optimizations

The `perf-v2` photometric path includes:

1. Row-wise vectorization of `sqrtP_limber`, replacing scalar power-spectrum
   evaluations for every `(ell, z)` pair with vector evaluations.
2. Reuse of cached comoving distances in the lensing-efficiency calculation.
3. Vectorized photometric redshift distribution and normalization evaluation.
4. One lensing-kernel evaluation per WL bin instead of two.
5. Cached per-observable/bin window factors in `clsintegral`.

The window cache stores only Hubble-independent factors. The Hubble factor is
applied on every `clsintegral` call, so calls with different Hubble arrays
remain correct.

## Focused Per-Point Benchmark

For the symbolic backend, GCph+WL, accuracy 1, and two varied parameters, the
average cost of one repeated `ComputeCls` point decreased from approximately
`3.69 s` to `1.05 s`, a `3.5x` speedup.

| Component | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| `sqrtP_limber` | 1.00 s | 0.042 s | 24x |
| `compute_kernels` | 1.04 s | 0.11 s | 9.5x |
| `computecls_vectorized` | 0.69 s | 0.16 s | 4.4x |
| Total per point | 3.69 s | 1.05 s | 3.5x |

The remaining dominant component is symbolic cosmology construction and power
spectrum generation, which is outside the photometric pipeline optimization
scope.

## End-to-End CLASS Comparison

Command configuration:

```text
OMP_NUM_THREADS=4
observables = WL, GCph
Fisher matrix shape = 18 x 18
derivatives = 3PT
feedback = 0
```

| Section | `main` | `perf-v2` | Change |
| --- | ---: | ---: | ---: |
| Photometric covariance | 0.51 s | 0.20 s | 2.55x |
| Photometric derivatives | 76.42 s | 65.06 s | 1.17x, 14.9% faster |
| Fisher assembly | 0.01 s | 0.01 s | negligible |
| `FisherMatrix.compute` | 83.64 s | 71.39 s | 1.17x, 14.6% faster |
| Wall time | 89.96 s | 79.59 s | 1.13x, 11.5% faster |

Numerical comparison:

```text
Fisher matrix max relative difference:        1.482969e-10
Inverse Fisher matrix max relative difference: 1.116915e-09
Parameter names: identical
```

## End-to-End Symbolic Comparison

The second run used the symbolic backend with the same observable and Fisher
configuration.

| Section | `main` | `perf-v2` | Change |
| --- | ---: | ---: | --- |
| Photometric covariance | 0.53 s | 0.20 s | 2.65x |
| Photometric derivatives | 28.19 s | 4.56 s | 6.18x, 83.8% faster |
| Fisher assembly | 0.01 s | 0.01 s | negligible |
| `FisherMatrix.compute` | 29.56 s | 4.95 s | 5.97x, 83.3% faster |
| Wall time | 31.44 s | 6.83 s | 4.60x, 78.3% faster |

Numerical comparison:

```text
Fisher matrix max relative difference:        3.307426e-11
Inverse Fisher matrix max relative difference: 7.832543e-10
Parameter names: identical
```

## Interpretation

The derivative phase dominates both backends, but it behaves differently:

- With the symbolic backend, the optimized photometric implementation reduces
  repeated per-point array and window-function work substantially. Derivative
  time falls from `28.19 s` to `4.56 s`.
- With CLASS, derivative time falls only from `76.42 s` to `65.06 s`. The
  derivative calls repeatedly invoke CLASS cosmology calculations, which
  dominate the remaining runtime.
- Covariance and Fisher assembly are already small compared with derivatives.

Therefore, CLASS is the bottleneck for the CLASS configuration, specifically
the backend work performed during each derivative evaluation. The result does
not imply that the remaining bottleneck is in covariance construction or Fisher
matrix assembly.

The next non-backend optimization target is parallelizing independent derivative
stencil points. Optimizing CLASS itself is deliberately out of scope for this
pipeline work.

## Reproduction

The shared wrapper defaults to `main` versus `perf-v2`:

```bash
bash /home/casas/Cosmo/dev-cosmicfishpie/shared-tools/compare_photo_fisher.sh
```

The wrapper's logs and snapshots are written to a timestamped directory under
`/tmp/photo-fisher-compare-*`. The recorded runs used the backend selected in
the shared Python script.
