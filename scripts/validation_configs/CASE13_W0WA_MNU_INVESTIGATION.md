# Case `08.2` (`w0waCDM + mnu + Neff`, formerly "case 13") investigation

> **Superseded in part.** Cases `07.2` and `07.1` (formerly "case 15"/"case 16") have
> since been run and analysed. The hypothesis below is directionally correct but wrong
> by about an order of magnitude on the size of the backend contribution, and it
> identifies the wrong degenerate block. See `PHOTO_MARGINALIZATION_DIAGNOSIS.md` for
> the resolved, quantitative diagnosis. Note also that `08.*` is a stress test
> explicitly excluded by the paper's own Sec. 6 validation scope (arXiv:2405.06047v1)
> -- see `PAPER_NEUTRINO_VALIDATION.md`.
> The numeric results also predate the paper-faithful P_cb and fixed-`betaIA`
> correction. They remain useful conditioning diagnostics, not current validation
> outcomes.

## Finding

Case `08.2` is a photometric CAMB-vs-CLASS comparison with nonlinear HMcode2020
power spectra. It fails the current 10% sigma gate for `mnu`:

| Parameter | CAMB sigma | CLASS sigma | Deviation |
|-----------|-----------:|------------:|----------:|
| `mnu`     | 0.171009   | 0.149084    | 12.82%    |

The other reported cosmological deviations are smaller but nonzero: `Omegam`
3.69%, `Omegab` 2.75%, `h` 3.94%, `ns` 1.65%, `sigma8` 4.91%, `Neff` 3.74%,
`w0` 0.76%, and `wa` 2.28%. The run used `nonlinear=true`,
`nonlinear_photo=true`, and `ShareDeltaNeff=false` for both backends.

This is not currently indicated to be a derivative-stencil or Python-layer
`Neff` mapping bug:

- Photometric observables request nonlinear `P(k)` through HMcode2020, while
  spectroscopic observables use linear `P(k)` by default.
- CAMB and CLASS use symmetric `mnu`/`Neff` mass-mapping formulas in the
  CosmicFishPie wrappers.
- Both probes use the same 3PT derivative method and the same relative `mnu`
  step of 0.1, which is 6 meV at the 0.06 eV fiducial.
- The spectroscopic counterpart (case `08.1`) has only a 0.89% `mnu` deviation,
  consistent with the nonlinear photo path being the important difference.

The most likely explanation is a small CAMB-vs-CLASS nonlinear/HMcode2020
baseline difference across the photometric derivatives, amplified when the
Fisher matrix is marginalized over the correlated `mnu`-`Neff`-`w0`-`wa`
block. This extends the mechanism already documented for cases `05.4` and
`03.2.0` (formerly "case 09" and "case 11").
A secondary numerical factor is that the photo CLASS profile uses lower
massive-neutrino perturbation settings than the spectro profile
(`l_max_ncdm=25` and the default fluid approximation versus
`l_max_ncdm=40` and `ncdm_fluid_approximation=3`).

## Controlled follow-up

Case `07.2` (formerly "case 15") isolates the effect of freeing `Neff` without
changing the `08.2` photo settings. It uses the same fiducial parameters, options,
CAMB YAML, CLASS photo YAML, nonlinear settings, derivative method, and 10% gate. The
only numerical-specification change is that `Neff` is retained as a fixed
fiducial value (`3.044`) and removed from `freepars`.

Run only this case with:

```bash
uv run bash scripts/run_selected_validations.sh --cases 07.2 --omp-threads 8 --force
```

The runner also includes `07.2` in `--all`. The run is intentionally not
started by this change because it is an expensive Fisher validation.

## Outcome

Case `07.2` was run and gives a marginalized `mnu` deviation of 14.71%, which is
*worse* than `08.2` despite `Neff` being fixed, and a `sigma8` deviation of
10.13%. Option 2 above is therefore ruled out together with option 1: freeing
`Neff` is not the driver, and neither is the `w0`/`wa` block on its own.

The unmarginalized errors resolve it. Case `07.2`'s CAMB and CLASS Fisher matrices
agree to a median of 0.046% element-by-element, and every unmarginalized error
matches to better than 0.12% except `mnu` at 1.55%. The reported 10-15%
deviations are amplification of those sub-0.1% differences by a near-singular
marginalization over the amplitude subspace `{sigma8, mnu, b1..b10}`; removing
any single one of the ten bias bins collapses the deviation to ~1% for `sigma8`
and ~5.7% for `mnu`.

Full numbers, the nested-subset cliff, the noise-amplification check, the
contrast with spectroscopic case `07.1`, and the recommended changes to the gate
are in `PHOTO_MARGINALIZATION_DIAGNOSIS.md`.
