# Why P_cb improves the massive-neutrino validation

## Summary

Switching the galaxy observables from the total-matter power spectrum P_mm to the
CDM+baryon power spectrum P_cb substantially improves the marginalized CAMB-vs-CLASS
agreement for `mnu`. This is both the physically correct prescription for galaxy tracers
in a massive-neutrino cosmology and a numerically better-conditioned Fisher observable.

The main improvement is not a large change in the fiducial power-spectrum agreement or
in the unmarginalized `mnu` error. P_cb instead removes a cancellation in the P_mm
neutrino response and makes the `mnu` signature more distinct from the `sigma8` and
galaxy-bias amplitude directions. Consequently, Fisher inversion amplifies the residual
backend differences much less.

## Physical reason

Galaxies trace the clustered CDM+baryon density field. Massive neutrinos free-stream on
the scales relevant to the Euclid galaxy observables and do not cluster like CDM and
baryons. Using P_mm for galaxy clustering mixes the clustered component with the smooth
neutrino component and makes the galaxy bias artificially scale dependent.

The paper therefore prescribes P_cc, where `c` denotes CDM+baryons. CosmicFishPie calls
this spectrum P_cb and selects it with:

```json
"GCsp_Tracer": "clustering",
"GCph_Tracer": "clustering"
```

Weak lensing remains correctly evaluated with P_mm because lensing responds to the total
gravitating matter distribution.

## Direct derivative diagnostic

A focused three-point derivative diagnostic compared CAMB and CLASS at:

- `mnu = 0.054, 0.060, 0.066 eV`, corresponding to the production 10% derivative step;
- redshifts `z = 0.5, 1.0, 1.5, 2.0`;
- 60 logarithmically spaced modes over `k = 1e-3 ... 10 1/Mpc`;
- nonlinear P_mm and P_cb using the paper CAMB HP and CLASS photo HP profiles.

The diagnostic evaluated

```text
d ln P / d mnu = [ln P(mnu + 0.006 eV) - ln P(mnu - 0.006 eV)] / 0.012 eV.
```

### Aggregate results

| Quantity | P_mm | P_cb |
|---|---:|---:|
| Median `abs(dlnP/dmnu)` in CAMB | 0.1095 | 0.2470 |
| Median absolute CAMB/CLASS derivative difference | 0.00705 | 0.00574 |
| Median relative derivative difference | 5.071% | 1.822% |
| Maximum relative derivative difference | 431.66% | 4.82% |
| Fiducial nonlinear P(k,z) median difference | 0.023% | 0.019% |
| Fiducial nonlinear P(k,z) maximum difference | 0.04% | 0.07% |

At nonlinear scales (`k > 0.1 1/Mpc`), the median relative derivative differences were:

| Redshift | P_mm | P_cb |
|---:|---:|---:|
| 0.5 | 11.452% | 2.184% |
| 1.0 | 11.719% | 2.937% |
| 1.5 | 10.848% | 3.504% |
| 2.0 | 9.528% | 3.733% |

The fiducial spectra agree at approximately 0.02% for both tracers. The discrepancy is
therefore in the neutrino response, not in the baseline P(k,z).

## Why P_mm is numerically delicate

Under the validation's fixed-`sigma8` parameterization, changing `mnu` produces competing
effects in P_mm:

1. neutrino free-streaming suppresses clustered power;
2. changing the smooth-neutrino fraction changes the composition of total matter;
3. the internal `sigma8 -> A_s` rescaling modifies the overall amplitude response.

These effects can cancel, causing `d ln P_mm / d mnu` to cross zero. A small absolute
CAMB-vs-CLASS difference divided by a derivative close to zero produces the apparent
431.66% maximum relative discrepancy. This is a zero-crossing artifact, not a comparably
large disagreement in P(k,z).

The P_cb derivative is larger, monotonic over the tested range, and has no corresponding
zero crossing. Its median signal is about 2.3 times larger while its absolute backend
difference is slightly smaller. The relative response is therefore substantially more
stable.

Both backends also construct nonlinear P_cb using the same neutrino-subtraction form,
applied to their respective nonlinear total-matter spectra. This provides a more closely
matched definition of the galaxy-tracer response than comparing the neutrino-sensitive
P_mm response directly.

## Fisher-matrix consequence

The original model-2 photometric comparison using P_mm gave:

| `mnu` comparison | P_mm | P_cb precursor |
|---|---:|---:|
| Marginalized CAMB/CLASS deviation | 10.06% | 5.53% |
| Unmarginalized CAMB/CLASS deviation | 1.55% | 1.70% |
| Marginalized/unmarginalized amplification | about 5.3 | about 3.3 |

The unmarginalized discrepancy did not improve. The marginalized discrepancy nearly
halved because P_cb changed the geometry and conditioning of the Fisher matrix. Its
more distinctive `mnu` response is less degenerate with `sigma8` and the redshift-binned
galaxy biases, so inversion does not magnify small backend differences as strongly.

The final paper-faithful batch, which uses P_cb and fixes `betaIA=2.17`, passed every
formal and stress-test gate. Relevant photometric maxima were:

| Case | Model/scenario | Maximum deviation |
|---|---|---:|
| `03.2.0` | LCDM + `mnu` + `Neff`, pessimistic | 5.53% (`mnu`) |
| `03.2.1` | LCDM + `mnu` + `Neff`, optimistic | 2.60% (`b10`) |
| `04.2.0` | w0CDM + `mnu`, pessimistic | 5.49% (`mnu`) |
| `04.2.1` | w0CDM + `mnu`, optimistic | 1.98% (`mnu`) |
| `07.2` | 8-parameter stress test | 6.34% (`mnu`) |
| `08.2` | 9-parameter stress test | 5.10% (`mnu`) |

P_cb and fixed `betaIA` were introduced together relative to the original stress-test
matrices, so the stress-test improvement cannot be assigned to either change in
isolation. However, the P_cb/free-`betaIA` model-2 precursor and the final P_cb/fixed-
`betaIA` case both give approximately 5.53%. Combined with the direct derivative test,
this identifies P_cb as the dominant improvement for the model-2 `mnu` agreement.

## Future visualization

A notebook or marimo application should regenerate the derivative arrays rather than
copying the summary numbers from this document. Recommended panels are:

1. `d ln P / d mnu` versus k for CAMB and CLASS, with rows for redshift and columns for
   P_mm/P_cb;
2. the CAMB/CLASS derivative ratio or symmetric relative difference, with zero-crossing
   regions clearly masked or annotated;
3. the fiducial P(k) backend ratio, demonstrating that the baseline spectra agree while
   their derivatives differ;
4. the P_cb/P_mm derivative-amplitude ratio;
5. marginalized and unmarginalized `mnu` deviations for the P_mm, P_cb/free-`betaIA`,
   and P_cb/fixed-`betaIA` configurations;
6. selected Fisher correlation coefficients involving `mnu`, `sigma8`, and the galaxy
   biases, before and after the tracer change.

The visualization should persist the numerical arrays, solver YAML/profile hashes,
common-specs hash, git commit, backend versions, derivative stencil, step size, redshift
grid, and k grid alongside the plots. Relative differences should not be shown without
also plotting absolute differences and marking derivative zero crossings.

## Interpretation boundary

This diagnostic establishes why P_cb improves this CAMB-vs-CLASS Fisher validation. It
does not imply that P_cb should replace P_mm for every observable: galaxy clustering uses
P_cb, while weak lensing and other total-matter observables must continue to use P_mm.
