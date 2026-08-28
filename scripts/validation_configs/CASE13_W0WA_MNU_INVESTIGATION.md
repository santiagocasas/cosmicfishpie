# Case 13 `w0waCDM + mnu + Neff` investigation

## Finding

Case 13 is a photometric CAMB-vs-CLASS comparison with nonlinear HMcode2020
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
- The spectroscopic counterpart (case 14) has only a 0.89% `mnu` deviation,
  consistent with the nonlinear photo path being the important difference.

The most likely explanation is a small CAMB-vs-CLASS nonlinear/HMcode2020
baseline difference across the photometric derivatives, amplified when the
Fisher matrix is marginalized over the correlated `mnu`-`Neff`-`w0`-`wa`
block. This extends the mechanism already documented for cases 09 and 11.
A secondary numerical factor is that the photo CLASS profile uses lower
massive-neutrino perturbation settings than the spectro profile
(`l_max_ncdm=25` and the default fluid approximation versus
`l_max_ncdm=40` and `ncdm_fluid_approximation=3`).

## Controlled follow-up

Case 15 isolates the effect of freeing `Neff` without changing the case 13
photo settings. It uses the same fiducial parameters, options, CAMB YAML,
CLASS photo YAML, nonlinear settings, derivative method, and 10% gate. The
only numerical-specification change is that `Neff` is retained as a fixed
fiducial value (`3.044`) and removed from `freepars`.

Run only this new case with:

```bash
uv run bash scripts/run_selected_validations.sh --cases 15 --omp-threads 8 --force
```

The runner also includes case 15 in `--all`. The run is intentionally not
started by this change because it is an expensive Fisher validation.

## Interpretation plan

Compare case 15 with case 13, especially the marginalized `mnu` sigma:

1. If the deviation drops below 10%, freeing `Neff` is a major contributor to
   the marginalization amplification.
2. If it remains high, the `w0`/`wa` degeneracy and nonlinear HMcode2020
   baseline difference remain sufficient without a free `Neff` axis.
3. If the deviation is still accompanied by broad 2-5% shifts in other
   parameters, test a linear-photo control (`nonlinear_photo=false`) to
   isolate the nonlinear backend contribution.

Further diagnostics, if needed, are to compare marginalized versus diagonal
(`unmarginalized`) `mnu` errors, then rerun with the higher-precision CLASS
massive-neutrino settings from the spectro YAML.
