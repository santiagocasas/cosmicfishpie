# Photometric `mnu`/`sigma8` gate failures: unmarginalized diagnosis

Status: historical diagnosis for the photometric gate failures in cases `06.2.0`, `03.2.0`,
`08.2`, and `07.2` (formerly numbered "case 09", "case 11", "case 13", "case 15").
Supersedes the "nonlinear HMcode2020 baseline difference" reading in
`CASE13_W0WA_MNU_INVESTIGATION.md`, which was directionally right but wrong
by an order of magnitude about the size of the backend effect. Note also that `08.2`
is a stress test explicitly excluded by the paper's own Sec. 6 validation scope
(arXiv:2405.06047v1) -- see `PAPER_NEUTRINO_VALIDATION.md`.

> **Configuration correction:** the matrices analyzed below used P_mm for galaxies and
> varied `betaIA`, contrary to the paper definitions. Canonical cases now use P_cb and
> fix `betaIA=2.17`. The final model-2 photometric cases pass with maximum deviations of
> 5.53% (pessimistic `03.2.0`) and 2.60% (optimistic `03.2.1`); those final survey
> definitions are the authoritative comparisons.

Source data: case `07.2` (`compare_photo_camb_vs_class_cfg_ea069c683b`) and case `07.1`
(`compare_spectro_camb_vs_class_cfg_86d26edc54`), run at commit `57e0757`
(dirty), `OMP_NUM_THREADS=6`.

## Summary

The photometric CAMB-vs-CLASS Fisher matrices agree element-by-element to a
median of **0.046%**. Every derivative, including the fully nonlinear GCph/WL
ones, matches to better than 0.12% in the unmarginalized errors. The single
genuine backend-level difference in the whole 21x21 matrix is `mnu`, at
**1.55%** unmarginalized.

The reported 10-15% marginalized deviations are not a measurement of backend
agreement. They are the amplification of sub-0.1% input differences by a
near-singular marginalization in the `{sigma8, mnu, b1..b10}` amplitude
subspace. The current 10% marginalized gate is therefore measuring the
conditioning of the photometric Fisher matrix, not CAMB-vs-CLASS physics.

## Case `07.2` unmarginalized versus marginalized errors

Unmarginalized (conditional) errors are `1/sqrt(F_ii)`; the amplification
column is `sigma_marginalized / sigma_unmarginalized` for CAMB.

| Parameter | unmarg. dev. | marg. dev. | amplification (CAMB) |
|-----------|-------------:|-----------:|---------------------:|
| `Omegam`  | 0.00%        | 7.31%      | 16.4 |
| `Omegab`  | 0.07%        | 1.05%      | 5.8  |
| `h`       | 0.12%        | 0.41%      | 12.5 |
| `ns`      | 0.09%        | 1.14%      | 7.7  |
| `sigma8`  | 0.03%        | 10.13%     | 30.5 |
| `mnu`     | 1.55%        | 14.71%     | 7.9  |
| `w0`      | 0.00%        | 0.60%      | 25.3 |
| `wa`      | 0.01%        | 4.58%      | 23.5 |
| `b1`-`b10`| 0.00-0.01%   | 9.2-9.8%   | 6.8-12.6 |
| `AIA`, `betaIA`, `etaIA` | 0.00% | 0.01% | 9.2-64.5 |

## Why `sigma8` "failed"

`sigma8` did not fail independently. `sigma8`, `mnu`, `Omegam`, and all ten
`b_i` deviate by the same 7-15% because they are projections of one near-flat
eigenmode of the photometric Fisher matrix.

- 93% of the marginalized `sigma8` variance comes from a single eigenmode of
  the correlation-normalized Fisher matrix with eigenvalue `4.1e-4`; 67% of the
  marginalized `mnu` variance comes from the same mode.
- `corr(sigma8, b_i)` is -0.94 to -0.95 for every bin, `corr(mnu, b_i)` is
  +0.87 to +0.90, and `corr(mnu, sigma8)` is -0.91.

GCph constrains the product `b_i * sigma8` per redshift bin. With all ten bias
amplitudes free, the overall amplitude direction is broken only by WL, which
carries no galaxy bias, and by the residual ell/z shape information. That
leaves one direction that is nearly unconstrained, and `mnu` lies almost
entirely inside it.

## The degeneracy cliff

Marginalizing case `07.2`'s own matrices over nested subsets of free parameters,
with all remaining parameters held fixed:

```
free parameters kept                 sigma8 dev.   mnu dev.
8 cosmological only (bias+IA fixed)      1.17%       5.67%
cosmological + b1                        1.01%       5.63%
cosmological + b1..b5                    1.05%       5.42%
cosmological + b1..b9                    1.37%       5.83%
cosmological + b1..b10                   8.93%      13.30%   <-- cliff
all 21 parameters                       10.13%      14.71%
```

The jump is a rank effect, not a bad redshift bin. Removing **any** single bias
bin from the full set of ten collapses the deviation back to the ~1% / ~5.7%
plateau:

```
drop b1  : sigma8 1.17%  mnu 5.99%      drop b6  : sigma8 1.03%  mnu 5.60%
drop b2  : sigma8 0.93%  mnu 5.64%      drop b7  : sigma8 1.02%  mnu 5.60%
drop b3  : sigma8 1.23%  mnu 5.86%      drop b8  : sigma8 1.08%  mnu 5.61%
drop b4  : sigma8 1.32%  mnu 5.77%      drop b9  : sigma8 1.18%  mnu 5.67%
drop b5  : sigma8 0.87%  mnu 5.52%      drop b10 : sigma8 1.37%  mnu 5.83%
```

Adding each bias bin alone to the eight cosmological parameters also stays on
the plateau (1.01-1.26% for `sigma8`, 5.63-5.71% for `mnu`). Only the complete
set of ten closes the amplitude degeneracy.

## Noise-amplification check

Injecting random symmetric relative noise into the CAMB Fisher matrix and
recomputing the marginalized errors, 300 draws per level:

| injected relative noise | non-positive-definite | median `sigma8` dev. | p90 | median `mnu` dev. | p90 |
|---|---|---|---|---|---|
| 0.02% | 63/300 | 9.43%  | 29.4% | 6.83%  | 21.9% |
| 0.05% | 133/300 | 18.56% | 69.4% | 15.61% | 54.8% |
| 0.10% | 210/300 | 31.07% | 83.2% | 24.02% | 88.6% |

A 0.02% perturbation, well below the observed 0.046% median CAMB-vs-CLASS
element difference, already reproduces the observed deviations. At 0.05% noise
nearly half the perturbed matrices are no longer positive definite.

## Contrast with the spectroscopic case `07.1`

Case `07.1` is *more* ill-conditioned than case `07.2`, yet agrees far better:

| quantity | case `07.2` photo | case `07.1` spectro |
|---|---:|---:|
| scale-invariant condition number of the normalized Fisher | 3.50e4 | 7.06e4 |
| amplification of `sigma8` | 30.5 | 45.7 |
| amplification of `w0` / `wa` | 25.3 / 23.5 | 81.4 / 60.4 |
| unmarginalized deviations | 0.00-1.55% | 0.05-0.52% |
| marginalized deviations | 0.41-14.71% | 0.01-1.44% |

The spectroscopic unmarginalized deviations are actually *worse* than the
photometric ones, and its amplification factors are larger. It nevertheless
passes, because RSD and Alcock-Paczynski information breaks the amplitude
direction and only four `lnbg_i` plus four `Ps_i` nuisances are marginalized,
instead of ten free bias amplitudes. There is no rank cliff.

The 3809.9s CLASS runtime for case `07.1` is the price of that agreement:
`class/paper_mnuvalidation_spectro.yaml` uses `ncdm_fluid_approximation: 3`
(exact ncdm hierarchy) and `l_max_ncdm: 40`, while
`class/paper_mnuvalidation_photo.yaml` uses the CLASS default fluid
approximation and `l_max_ncdm: 25` despite requesting `P_k_max_1/Mpc: 50`.
That asymmetry is the most plausible origin of the residual 1.55%
unmarginalized `mnu` difference, which is the only real backend signal in the
photometric matrix.

## Consequences for the earlier interpretation

The mechanism proposed in `CASE13_W0WA_MNU_INVESTIGATION.md` and
`PAPER_NEUTRINO_VALIDATION.md` (small backend differences amplified by
marginalization over a correlated block) is confirmed, with two corrections:

1. The dominant degenerate block is **not** `mnu`-`Neff`-`w0`-`wa`. Case `07.2` has
   `Neff` fixed and is *worse* than case `08.2` (14.71% versus 12.82%). The
   dominant block is the amplitude subspace `{sigma8, mnu, b1..b10}`.
2. The backend contribution is 1.55%, not 5-13%. Roughly a factor of ten of the
   quoted deviation is conditioning, and would appear identically if both codes
   agreed to 0.05% rounding.

This also explains the previously unexplained monotonic escalation across cases
`06.2.0` (6.82%), the old `03.2.0` precursor (10.06%), `08.2` (12.82%), and `07.2`
(14.71%): every additional
free parameter that projects onto the amplitude direction moves the matrix closer to
the cliff, independently of any change in backend physics.

## Reproduction

Both matrices and their `.paramnames` sidecars are already on disk. The numbers
above come from `cosmicfishpie.analysis.fisher_matrix.fisher_matrix`:

```python
import numpy as np
from cosmicfishpie.analysis.fisher_matrix import fisher_matrix

d = "scripts/benchmark_results/compare_photo_camb_vs_class_cfg_ea069c683b/"
a = fisher_matrix(file_name=d + "CosmicFish_v1.3.0_compare_photo_20260828_123040_A__GCphWL_FM.txt")
b = fisher_matrix(file_name=d + "CosmicFish_v1.3.0_compare_photo_20260828_123040_B__GCphWL_FM.txt")

names = a.get_param_names()
marg_a = a.get_confidence_bounds(confidence_level=0.6827)
unmarg_a = a.get_confidence_bounds(confidence_level=0.6827, marginal=False)

# nested-subset test: keep only a chosen index set free, fix the rest
FA, FB = a.get_fisher_matrix(), b.get_fisher_matrix()
idx = [names.index(p) for p in ("Omegam", "Omegab", "h", "ns", "sigma8", "mnu", "w0", "wa")]
sa = np.sqrt(np.diag(np.linalg.inv(FA[np.ix_(idx, idx)])))
sb = np.sqrt(np.diag(np.linalg.inv(FB[np.ix_(idx, idx)])))
```

Note that `get_confidence_bounds(..., marginal=False)` is the unmarginalized
branch; the default `marginal=True` is what the dashboard currently reports.

## Recommended follow-up

1. Report unmarginalized errors alongside marginalized ones in
   `scripts/compare_reference_fishers.py`, in `compare_fishers_*.json`, and on
   the per-case dashboard pages. This is the physically meaningful
   backend-comparison metric.
2. Add a degeneracy-diagnostic block per case: scale-invariant condition
   number, per-parameter amplification factor, and the nested-subset curve, so
   a rank cliff is visible instead of being misread as a physics failure.
3. Make the blocking gate act on the unmarginalized deviation, keeping the
   marginalized deviation as informational. Under that criterion case `07.2` passes
   everything except `mnu` at 1.55%.
4. Test the residual 1.55% with a photometric run using the spectroscopic CLASS
   neutrino precision (`l_max_ncdm: 40`, `ncdm_fluid_approximation: 3`, keeping
   `P_k_max_1/Mpc: 50`). Expect a CLASS runtime comparable to case `07.1`. (Case
   alternative `01.4` already implements a related precision variant, but with
   `ncdm_fluid_trigger_tau_over_tau_k=90` rather than
   `ncdm_fluid_approximation=3` -- this follow-up would need a distinct YAML.)
5. If the 1.55% survives that, isolate HMcode2020 with a photometric control at
   `nonlinear: false` and `nonlinear_photo: false`.
