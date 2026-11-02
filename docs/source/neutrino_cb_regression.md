# Photometric neutrino-response regression

## Summary for collaborators

**The large new photometric neutrino discrepancies are explained, for the tested
case, by a change in the CLASS nonlinear cold-dark-matter-plus-baryon (cb)
prescription.** CLASS was switched to its native nonlinear cb spectrum, while
CAMB continued to use a reconstruction from total nonlinear matter power and
linear neutrino terms. This was a change of model convention, not just a solver
accuracy setting.

Small differences in fiducial power do not guarantee small differences in its
neutrino-mass derivative. In our local control, fiducial spectra differ by only
about `0.02–0.08%` in median, but native CLASS cb has substantially different
mass responses on nonlinear scales. Those responses enter the Fisher matrix
and change parameter degeneracies.

Restoring the matched reconstruction in CLASS, together with its previous
mass-to-density convention, reduces the maximum marginalized-error discrepancy
in case `03.2.0` from **21.38% to 5.5266%**, reproducing the historical passing
forecast. We retained the new supported-grid checks throughout this test.

**Interpretation:** the comparison must use a declared cb prescription on both
sides. The experiment does not establish that native CLASS cb is physically
incorrect or that the reconstruction is more accurate. It establishes why the
new forecast no longer reproduced the earlier matched-backend validation.
The remaining affected Fisher cases still need confirmation on Frey.

## What are the three spectra being compared?

Let $f_{cb}=\Omega_{cb}/\Omega_m$ and $f_\nu=\Omega_\nu/\Omega_m$. The historical
reconstruction is

$$
P_{cb}^{\mathrm{rec}}(k,z) =
\frac{P_{mm}^{\mathrm{NL}}(k,z)
 - 2 f_{cb} f_\nu P_{cb,\nu}^{\mathrm{L}}(k,z)
 - f_\nu^2 P_{\nu,\nu}^{\mathrm{L}}(k,z)}{f_{cb}^2}.
$$

The superscripts L and NL denote linear and nonlinear quantities. The linear
cross and neutrino-auto terms are obtained from each solver's own outputs.
This reconstruction approximates those neutrino terms as linear; it is not
the same prescription as directly requesting native nonlinear cb.

| Curve | Total matter / neutrino inputs | cb prescription |
|---|---|---|
| CAMB reconstruction (reference) | CAMB outputs | Equation above |
| CLASS reconstruction | CLASS outputs | Same equation |
| CLASS native cb | CLASS nonlinear cb API | `get_pk_cb_array(..., nonlinear=True)` |

The main three-way comparison uses density factor `94.07` for both solvers to
isolate the prescription change. A separate control repeats CLASS with `93.14`.
All comparisons hold the input **total-matter sigma8 fixed at 0.81**: they are
not derivatives at fixed primordial amplitude As.

The [interactive comparison notebook](../../notebooks/Neutrino-investigation_native-cb-vs-reconstruction.ipynb)
plots fiducial spectra, percent residuals, $\partial P_{cb}/\partial M_\nu$,
$\partial\ln P_{cb}/\partial M_\nu$, and derivative differences at four redshifts.
It uses the saved nine-run spectrum control by default and offers explicit
solver regeneration from the committed case configuration.

### Sharing the notebook

Open `notebooks/Neutrino-investigation_native-cb-vs-reconstruction.ipynb` to see
the executed figures and tables immediately. The default `RUN_SOLVERS = False`
replots the saved local spectrum control. For a fresh checkout without that
cache, select the repository Python environment and set `RUN_SOLVERS = True`;
this regenerates only nine Boltzmann runs, not Fishers. The notebook saves
PNG/PDF figures alongside the selected cache for use in presentations.

## Local diagnosis (12 October 2026)

The October CAMB/CLASS campaign changed passing photometric neutrino cases into
large discrepancies, while CAMB errors remained consistent with the historical
dashboard. Two CLASS changes in commit `399ae0a` were isolated:

- native nonlinear cb spectra replaced the matched linear-neutrino subtraction
  approximation still used by CAMB;
- the default mass-to-density factor changed from `94.07` to `93.14`.

These are distinct numerical/physical conventions, not interchangeable ways of
computing the same derivative. The diagnosis does not establish that native
CLASS cb is physically wrong; it establishes that it is not the matched recipe
used by the passing comparison.

### Spectrum control

Using saved `03.2.0` settings, fixed sigma8 and masses `0.054, 0.060, 0.066` eV,
the local experiment sampled four redshifts (`0.5, 1, 1.5, 2`) and 60 physical
wavenumbers from `0.001` to `10` inverse Mpc. Both CLASS prescriptions were
evaluated from each solver run. The reference is CAMB's subtraction prescription.
The central response is the finite difference of log power with respect to mass.
Here $M_\nu$ denotes the summed neutrino mass, represented by the `mnu` parameter:

$$
\left.\frac{\partial\ln P}{\partial M_\nu}\right|_{\sigma_8}
\simeq \frac{\ln P(0.066)-\ln P(0.054)}{0.012\ \mathrm{eV}}.
$$

The notebook also computes the direct finite difference of P, without taking
its logarithm. This is one finite-difference step, not a step-convergence study.
Both k and power use **physical units**: inverse Mpc and cubic Mpc, respectively.

| CLASS prescription | Density factor | Median fiducial difference | Median response difference | Maximum response difference |
|---|---:|---:|---:|---:|
| Native cb | 93.14 | 0.07482% | 3.31424% | 100.69628% |
| Subtraction | 93.14 | 0.02405% | 2.78695% | 5.52321% |
| Native cb | 94.07 | 0.03983% | 3.10270% | 100.35201% |
| Subtraction | 94.07 | 0.01940% | 1.82190% | 4.82212% |

Relative derivative differences can amplify near a small reference response.
However, the discrepancy is not just one such point: with factor `94.07`, the
native prescription has median response differences of `18.83%` over
`0.1–1` inverse Mpc and `55.23%` over `1–10` inverse Mpc. The corresponding
subtraction differences are `3.02%` and `3.51%`. The restored result reproduces
the earlier spectrum-response diagnostic.

### One-Fisher confirmation

Only CLASS case `03.2.0` was rerun, preserving the saved fiducial parameters,
free parameters, solver profile and survey settings. Machine-specific paths
were relocated. Its CAMB reference is the existing October run, not a new CAMB
calculation. The corrected CLASS run took approximately 738 seconds with four
OpenMP threads.

- Maximum marginalized error deviation: **21.38% before → 5.5266% after**.
- Sigma8 error deviation after correction: **2.2720%**.
- The largest remaining deviation is mnu (**5.5266%**).
- Corrected CLASS errors reproduce the historical passing dashboard to displayed
  precision; this case is below its 10% acceptance threshold.

The supported nonlinear-grid hardening was retained in this confirmation.
The remaining affected cases have **not** yet been rerun with the correction.

## Backend convention

For CLASS-native nonlinear models, the default `NONLINEAR.cb_prescription` is
`neutrino_subtraction`: total nonlinear matter power minus linear neutrino cross
and auto terms, divided by the squared cb fraction. This matches CAMB's existing
approximation. To deliberately request native CLASS nonlinear cb instead:

```yaml
NONLINEAR:
  model: class_native
  cb_prescription: class_native
```

The default CLASS `PARAMETER_TRANSLATION.neutrino_mass_fac` is again `94.07`,
matching CAMB. Explicit profile overrides (including `93.14` in matched
SYREN/EE2 profiles) are preserved. The EE2 cb boost approximation is unchanged.

## Reproduction and remaining campaign

The isolated diagnostic is `scripts/validation/diagnose_neutrino_response.py`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run python scripts/validation/diagnose_neutrino_response.py \
  --specs /path/to/saved_CLASS_FM_specs.json \
  --outdir scripts/benchmark_results/neutrino_isolation
```

It saves spectra, translated CLASS parameters, the input hash and numerical
summary. Local evidence is under `scripts/benchmark_results/` in
`neutrino_isolation_20261012b/` and `neutrino_fixed_03.2.0b/`; these generated
files are ignored by Git and have not been uploaded to the DVC archive.
The archived reference is
`compare_photo_camb_vs_class_cfg_13fddbddf6_20261007_175905` in `cjfp-data`.

After transferring the corrected package code to Frey, rerun the affected cases
and the fixed-neutrino/Pmm controls before claiming campaign-wide resolution:

```bash
bash scripts/validation/run_parallel_validations.sh \
  --cases 03.2.0,03.2.1,04.2.0,04.2.1,05.2.0,06.2.0,07.2.0,07.2.0.1,08.2.0 \
  --jobs 2 --omp-threads 4 --force
```

Case `07.2.0.1` is the deliberate Pmm teaching control; it is not expected to
pass merely because the matched cb prescription is restored. Review reports
before updating the DVC manifest and uploading results.
