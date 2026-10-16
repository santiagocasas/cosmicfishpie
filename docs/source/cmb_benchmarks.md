# CMB benchmarks

See {ref}`Supported observables and survey settings <observables>` for how to
pass CMB observables and the Planck survey name to `FisherMatrix`. This page
covers the shipped experiment specs and the smoke-test/benchmark scripts.

CosmicFishPie includes a basic CMB Fisher path for the observables:

- `CMB_T`
- `CMB_E`
- `CMB_B`

The CMB implementation uses a single ell range (`lmin_CMB`, `lmax_CMB`) for all
spectra. Separate TT/TE/EE ranges and lensing spectra (e.g. `phi-phi`) are not
implemented yet.

## Survey spec YAMLs

The following example CMB experiment specifications are shipped as survey spec
YAMLs:

- `cosmicfishpie/configs/default_survey_specifications/Planck.yaml`
- `cosmicfishpie/configs/default_survey_specifications/Simons-Observatory-PlanckLowEll.yaml`
- `cosmicfishpie/configs/default_survey_specifications/CMB-Stage4-PlanckLowEll.yaml`

For Simons Observatory and Stage-IV, we use a conservative common `lmax_CMB=3000`
as an approximation.

## Run a single CMB Fisher (smoke test)

The CMB scripts live in `scripts/archive/cmb/`, where historical tools are kept. They
compute their repository root from their own location, so pass the solver YAML
explicitly. Run from the repository root:

```bash
uv run python scripts/archive/cmb/run_cmb_fisher_smoke.py \
  --code camb \
  --boltzmann-yaml cosmicfishpie/configs/default_boltzmann_yaml_files/camb/default.yaml \
  --spec-yaml cosmicfishpie/configs/default_survey_specifications/Planck.yaml \
  --observables CMB_T,CMB_E \
  --lmax 120 \
  --outdir tmp/cmb_planck_smoke \
  --write-enabled-yaml
```

With `--lmax 120` this takes under a minute. Tips:

- Drop `--lmax` to use the `lmax_CMB` of the spec YAML.
- `--write-enabled-yaml` writes a copy of the CAMB YAML with CMB outputs enabled
  into the output directory for provenance.
- Without `--boltzmann-yaml` the script fails with "Boltzmann YAML not found".

## Run the standard presets

`scripts/archive/cmb/run_cmb_benchmarks.py` is meant to run the Planck, Simons
Observatory and Stage-IV presets in one go, but it does not currently work: it looks
for the spec files under `scripts/archive/`, fails with "Missing spec YAML", and has
no option to point it elsewhere. It also calls a smoke script path
(`scripts/run_cmb_fisher_smoke.py`) that no longer exists. Until it is fixed, run
the smoke script once per spec:

```bash
for spec in Planck Simons-Observatory-PlanckLowEll CMB-Stage4-PlanckLowEll; do
  uv run python scripts/archive/cmb/run_cmb_fisher_smoke.py \
    --code camb \
    --boltzmann-yaml cosmicfishpie/configs/default_boltzmann_yaml_files/camb/default.yaml \
    --spec-yaml cosmicfishpie/configs/default_survey_specifications/${spec}.yaml \
    --observables CMB_T,CMB_E \
    --outdir tmp/cmb_bench/${spec}
done
```

See also the [scripts reference](scripts_reference.md).
