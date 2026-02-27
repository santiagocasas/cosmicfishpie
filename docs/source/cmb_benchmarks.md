# CMB Benchmarks

CosmicFishPie includes a basic CMB Fisher path for the observables:

- `CMB_T`
- `CMB_E`
- `CMB_B`

The CMB implementation uses a single ell range (`lmin_CMB`, `lmax_CMB`) for all
spectra. Separate TT/TE/EE ranges and lensing spectra (e.g. `phi-phi`) are not
implemented yet.

## Survey Spec YAMLs

The following example CMB experiment specifications are shipped as survey spec
YAMLs:

- `cosmicfishpie/configs/default_survey_specifications/Planck.yaml`
- `cosmicfishpie/configs/default_survey_specifications/Simons-Observatory-PlanckLowEll.yaml`
- `cosmicfishpie/configs/default_survey_specifications/CMB-Stage4-PlanckLowEll.yaml`

For Simons Observatory and Stage-IV, we use a conservative common `lmax_CMB=3000`
as an approximation.

## Run A Single CMB Fisher (Smoke)

Use the CAMB backend and a spec YAML:

```bash
uv run python scripts/run_cmb_fisher_smoke.py \
  --code camb \
  --spec-yaml cosmicfishpie/configs/default_survey_specifications/Planck.yaml \
  --observables CMB_T,CMB_E \
  --outdir tmp/cmb_planck_smoke \
  --write-enabled-yaml
```

Tips:
- Use `--lmax 120` for a fast end-to-end smoke.
- `--write-enabled-yaml` writes a copy of the CAMB YAML with CMB outputs enabled
  into the output directory for provenance.

## Run The Standard Presets

```bash
uv run python scripts/run_cmb_benchmarks.py \
  --outdir tmp/cmb_bench \
  --which planck,so,s4
```
