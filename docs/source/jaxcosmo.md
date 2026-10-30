# JAXcosmo and automatic differentiation

The optional `jaxcosmo` backend supplies differentiable cosmological theory from
[`jax-cosmo`](https://github.com/DifferentiableUniverseInitiative/jax_cosmo).
Use `options["code"] = "jaxcosmo"` to select it, and pass a derivative provider to
`FisherMatrix` to choose automatic differentiation (AD) or finite differences (FD)
of the JAX-native observables.

## Installation

From a source checkout containing the JAXcosmo implementation:

```bash
uv sync --extra jaxcosmo
# Alternatively:
pip install -e '.[jaxcosmo]'
```

The extra installs JAX, `jaxlib`, `jax-cosmo` and its compatibility dependencies.
The shipped backend profile enables JAX float64 (`NUMERICS.x64: True`), which is
important for small finite-difference steps and Fisher inversions. Initial JIT
compilation contributes to the first evaluation's runtime; subsequent evaluations
with the same shapes reuse compiled functions.

## A photometric forecast

This example keeps both the neutrino mass and the spectral index fixed. Parameters
omitted from `freepars` retain their fiducial values; the usual survey nuisance
parameters are added by CosmicFishPie.

```python
from cosmicfishpie.fishermatrix.cosmicfish import FisherMatrix
from cosmicfishpie.fishermatrix.jaxcosmo_derivatives import JaxAutodiffProvider

fiducial = {
    "10^9As": 2.1,
    "Omegam": 0.32,
    "Omegab": 0.05,
    "h": 0.67,
    "ns": 0.96,
    "mnu": 0.0,
}
free = {name: 0.01 for name in ("10^9As", "Omegam", "Omegab", "h")}
options = {
    "code": "jaxcosmo",
    "accuracy": 1,
    "nonlinear": True,
    "derivatives": "3PT",
    "survey_name_photo": "Euclid-Photometric-ISTF-Pessimistic",
    "results_dir": "results/jaxcosmo/",
    "outroot": "photo_ad",
}

forecast = FisherMatrix(
    fiducialpars=fiducial,
    freepars=free,
    options=options,
    observables=["GCph", "WL"],
    cosmoModel="LCDM",
    surveyName="Euclid",
    derivative_provider=JaxAutodiffProvider(),
)
fisher = forecast.compute()
```

`JaxAutodiffProvider` uses `jax.jacfwd` to differentiate the implemented observable;
the values in `freepars` select the parameters but do not set an AD step size.
For finite differences they are relative steps (absolute steps when the fiducial
value is zero), as described in {doc}`derivatives`.

## Three routes with the same theory

| Route | `derivative_provider` | Observable used for derivatives |
| --- | --- | --- |
| JAX AD | `JaxAutodiffProvider()` | JAX-native CosmicFishPie observable |
| JAX FD | `JaxFiniteDifferenceProvider()` | The same JAX-native observable |
| Standard FD | Omitted, or `None` | Existing NumPy/SciPy observable |

Import both providers from
`cosmicfishpie.fishermatrix.jaxcosmo_derivatives`. All three routes can use
`options["code"] = "jaxcosmo"`.

**AD versus JAX FD isolates the derivative method.** Decrease the FD step to check
convergence. **JAX FD versus standard FD checks the observable port**, including its
interpolation, integration and boundary choices. The JAX observables reproduce
CosmicFishPie's photometric Limber and spectroscopic galaxy-power calculations;
they do not call `jax_cosmo.angular_cl`. The existing covariance and Fisher
assembly are retained; selecting AD does not add covariance-derivative terms.

The normal backend adapter returns NumPy/SciPy tables for the existing pipeline.
Those tables do not preserve JAX gradients, which is why choosing the backend
alone does not enable autodiff. Use a fresh options dictionary for each forecast:
configuration initialisation fills defaults into it.

## Comparison notebook

Download {download}`07-JAXcosmo-autodiff-vs-finite-differences.ipynb
<../../notebooks/07-JAXcosmo-autodiff-vs-finite-differences.ipynb>` and run it from
the repository checkout (or open it directly in `notebooks/`). Select that
checkout's Python environment, for example `.venv/bin/python`, as the Jupyter
kernel. Install `jupyterlab` and `ipykernel` into that environment if needed.

The notebook compares the three JAXcosmo routes above with the standard
CosmicFishPie SYREN-NEW forecast. It includes:

- linear and nonlinear matter power, and the response to the spectral index;
- angular spectra from the JAX and standard observable implementations;
- FD step-size convergence, marginalized errors and a triangle overlay;
- an optional CLASS reference, disabled by default.

The fiducial model is flat LCDM with massless neutrinos. **The Fisher runs keep
`ns = 0.96` fixed**, varying `10^9As`, `Omegam`, `Omegab` and `h`, along with the
default photometric nuisance parameters. This avoids using the known SYREN-NEW
nonlinear spectral-index response discrepancy in the forecast comparison; see
{doc}`syren_ns_response`. The diagnostic response plot still varies `ns` to show
the theory response, independently of the Fisher parameter list. Fixing `ns`
does not make the two backends' predictions identical.

The notebook writes figures and its summary into
`results/jaxcosmo_ad_vs_fd/`. These are local outputs. Re-run all cells after
changing the free parameters so the displayed tables and figures share the same
configuration.

### Why SYREN is a separate theory comparison

JAXcosmo uses an Eisenstein–Hu transfer function (with baryon wiggles by default)
and Halofit nonlinear power. SYREN-NEW uses an Eisenstein–Hu-based linear fit with
its own corrections and growth model, and a separate nonlinear emulator. A shared
Eisenstein–Hu ingredient does not make their power spectra or derivatives equal.
Compare AD against FD with the **same JAX observable** to assess differentiation;
compare SYREN against the standard JAXcosmo route to assess theory dependence.

### Where the SYREN amplitude difference comes from

With `ns` and `mnu` fixed, the marginalized `10^9As` error differs by about
10% between SYREN-NEW and JAXcosmo. The autodiff and finite-difference routes
through the same JAX observable agree to `1.8e-5` in the Fisher matrix, so the
difference is not a differentiation problem. The following decomposition
locates it.

| `10^9As` error, fiducial 1 | SYREN-NEW | JAXcosmo AD | AD/SYREN |
| --- | --- | --- | --- |
| unmarginalized (`1/sqrt(F_AsAs)`) | 0.00191324 | 0.00186387 | -2.6% |
| cosmology only | 0.0223482 | 0.0237162 | +6.1% |
| cosmology + 10 galaxy biases | 0.0294102 | 0.0263472 | -10.4% |
| cosmology + biases + IA (all) | 0.0294396 | 0.0264081 | -10.3% |

The two models agree closely on the unmarginalized amplitude error and on
`dlnP_nl/dlnAs` (ratio 0.98-1.00 over `k = 10^-3` to `1 h/Mpc`). The gap appears
only once the galaxy-bias parameters are marginalized: SYREN's amplitude error
grows by a factor 1.32, JAXcosmo's by 1.11. The amplitude-bias correlations
differ in shape, for example `r(10^9As, b8) = -0.37` for SYREN and `-0.28` for
JAXcosmo, while the amplitude response of the spectra themselves is nearly
identical.

The decisive test replaces the matter power spectrum inside the *JAX* observable
with SYREN-NEW's, keeping the JAX kernels, integration, covariance and Fisher
assembly. That hybrid reproduces the SYREN result:

| route | marginalized `10^9As` error | ratio to SYREN |
| --- | --- | --- |
| SYREN-NEW, standard pipeline | 0.0294396 | 1.00000 |
| JAXcosmo, autodiff | 0.0264081 | 0.89703 |
| SYREN-NEW power inside the JAX observable | 0.0294160 | 0.99920 |

The hybrid's fiducial angular spectra match the standard SYREN ones to
`max|dC|/max|C| = 8.3e-4`. The same comparison at a second fiducial gives
`-10.3%` and `-11.3%`, so the effect is stable rather than a single-point
accident. The difference is therefore a genuine matter-power-model difference,
not the observable implementation, the covariance, the Fisher assembly or the
autodiff. It cannot be removed without changing the theory.

### Both models are "Eisenstein-Hu", so why do they differ?

A common objection is that both backends are Eisenstein-Hu based, so the spectra
should agree. They are not the same formula:

* `jax_cosmo` evaluates the Eisenstein-Hu 1998 fit *including* baryon wiggles
  (`eisenhu_osc`) and normalises it to `sigma8`.
* SYREN-NEW starts from the Eisenstein-Hu *no-wiggle* fit
  (`get_eisensteinhu_nw`) and multiplies it by `exp(logF)`, an emulated
  correction of the true linear spectrum relative to the Eisenstein-Hu fit
  (Bartlett et al. 2023), a growth correction `R` and a further `S(k)`
  correction with its own coefficients. In other words SYREN-NEW's "linear"
  model is calibrated to reproduce a full Boltzmann spectrum with Eisenstein-Hu
  only as a backbone.

At equal `As` the two linear spectra differ by a few percent with a
`k`-dependent sign, while the growth factors agree to `4e-4`:

| band [`h/Mpc`] | `P_lin`(SYREN)/`P_lin`(EH), z=0 |
| --- | --- |
| 0.001-0.005 | 1.0032 |
| 0.005-0.02 | 1.0164 |
| 0.02-0.05 | 1.0093 |
| 0.05-0.1 | 0.9872 |
| 0.1-0.2 | 0.9914 |
| 0.2-0.5 | 1.0093 |
| 0.5-1 | 1.0161 |

Those percent-level shape changes sit exactly where the galaxy-bias
marginalization operates. Feeding Halofit with SYREN-NEW's *linear* shape, while
keeping the JAX observable and the Halofit nonlinear prescription, gives a
marginalized `10^9As` error of `0.0301323`, i.e. `+2.4%` with respect to SYREN
(`0.0294396`) and `+14%` with respect to plain Eisenstein-Hu plus Halofit
(`0.0264081`). The linear input therefore carries essentially all of the
difference: the two nonlinear prescriptions (Halofit emulator versus the
SYREN-NEW emulator) contribute at the percent level in the opposite direction.
This is why the gap cannot be removed by changing the derivative method,
the grids or the marginalization; only a common power-spectrum model can.

Reproduce the decomposition with:

```bash
uv run --extra jaxcosmo python scripts/validation/compare_theory_amplitude_response.py \
    --outdir results/theory_amplitude_response
```

The script prints the theory response table, the nuisance-marginalization
decomposition, the hybrid test, the linear-shape comparison, the
Halofit-on-SYREN-linear bridge, both fiducials and the AD-versus-FD control, and
writes `summary.json`. The response-optimised SYREN coefficient sets from the
`ns`-response study are not a fix here: they increase this amplitude error.

## Theory and supported configurations

| Component | Current implementation |
| --- | --- |
| Cosmology | Flat `LCDM` and `w0waCDM`, massless neutrinos (`mnu = 0`) |
| Matter tracer | Total matter; clustering/CB tracers are unsupported |
| Background and growth | `jax-cosmo` distances, expansion and scale-independent growth |
| Linear power | Eisenstein–Hu, with or without baryon wiggles |
| Nonlinear power | Halofit, `takahashi2012` or `smith2003` |
| Amplitude | JAX port of the `symbolic_pofk` maximum-precision `As`–`sigma8` fit |
| JAX observables | Photometric `GCph`/`WL`; spectroscopic `GCsp` default modelling path |
| Outside this implementation | CMB, intensity-mapping autodiff, massive neutrinos and curvature |

The amplitude fit is re-evaluated inside the traced function, retaining its
cosmology dependence. Supply one amplitude coordinate, for example `10^9As` or
`sigma8`, rather than both. The AD parameter mapping currently exposes
`10^9As`, `Omegam`, `Omegab`, `h`, `ns`, `w0` and `wa`; an accepted backend input
alias is not necessarily an available AD coordinate. Use the coordinates in the
example for a controlled comparison.

The GCsp JAX path requires `fix_cosmo_nl_terms=True`, `bfs8terms=False` and a
supported bias model (`linear`, `linear_log` or `linear_Qbias`). Photometry
supports the implemented photo-z distributions, galaxy-bias models and eNLA
intrinsic alignments. Unsupported modelling choices raise explicit errors.

Backend configuration lives in
`cosmicfishpie/configs/default_boltzmann_yaml_files/jaxcosmo/default.yaml`.
To customise it, copy the profile and set
`options["jaxcosmo_config_yaml"]` to the copy's path. Key controls are
`ACCURACY.transfer`, `ACCURACY.halofit`, `ACCURACY.distance_steps`, and the
`NUMERICS` power/background grids and `x64` flag. Public CosmicFishPie power uses
`k` in inverse Mpc and `P(k)` in cubic Mpc; the theory adapter converts the
internal `jax-cosmo` h-scaled units.

### Smooth Halofit and table boundaries

The theory layer includes a locally maintained Halofit nonlinear-scale solver.
It uses a converged Newton solve with an implicit derivative to avoid the
fixed-grid root-interpolation derivative artefacts in `jax-cosmo` 0.1.0. It is
used by both the standard backend adapter and the JAX-native theory path. The
upstream proposal and numerical comparisons are recorded in {doc}`jaxcosmo_review`.

The GCsp JAX observable evaluates power directly by default. Set
`NUMERICS.legacy_pk_clamp: True` in a custom profile to reproduce the standard
spline's clamp at its tabulated `k` boundaries for parity studies. This introduces
a derivative kink at the boundary. Spectroscopic Savitzky–Golay window changes
can also affect finite differences at large steps; inspect step convergence
rather than relying on a single step size.

## Reproducible parity checks and development

For a straight-line terminal example timing SYREN finite differences against
JAXcosmo autodiff with `ns` fixed:

```bash
uv run --extra jaxcosmo python scripts/validation/time_syren_vs_jaxcosmo.py
```

It prints setup, fiducial spectra, covariance, derivatives and Fisher assembly
times, followed by marginalized errors. The first AD derivative call includes
compilation and execution; a second call measures the compiled Jacobian's warm
execution. The two backends use different theory, so this timing comparison is
distinct from the same-theory parity checks below.

From the checkout, with the extra installed:

```bash
uv run --extra jaxcosmo python scripts/validation/compare_jax_photo_fishers.py \
    --outdir results/jax_photo_parity
uv run --extra jaxcosmo python scripts/validation/compare_jax_spectro_fishers.py \
    --outdir results/jax_spectro_parity --fd-step 0.003
```

These scripts compare the three same-theory routes and save reports and numerical
summaries. They are separate from the SYREN comparison notebook. Existing parity
measurements cover a limited Euclid, massless-neutrino LCDM setup; the
{doc}`jaxcosmo_review` records metrics, normalisations and remaining review items.

The backend-neutral contract in `cosmology/jax_theory.py` separates the traceable
theory from observables. A future differentiable backend can implement that
contract; a backend returning NumPy arrays alone cannot supply cosmological AD.
The standard likelihood/context interface remains available to sampling code.
Derivative providers cache compiled observables locally and discard that cache
when pickled; create them within workers. Nautilus serial-versus-spawn parity
with JAX remains a separate validation item, rather than a demonstrated result
of these Fisher comparisons.
