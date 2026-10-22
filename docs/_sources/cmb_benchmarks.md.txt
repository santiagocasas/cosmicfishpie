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

The CMB scripts live in `scripts/archive/cmb/`, where historical tools are kept.
Run from the repository root:

```bash
uv run python scripts/archive/cmb/run_cmb_fisher_smoke.py \
  --code camb \
  --spec-yaml cosmicfishpie/configs/default_survey_specifications/Planck.yaml \
  --observables CMB_T,CMB_E \
  --lmax 120 \
  --outdir tmp/cmb_planck_smoke \
  --write-enabled-yaml
```

With `--lmax 120` this takes under a minute. Tips:

- Drop `--lmax` to use the `lmax_CMB` of the spec YAML.
- Keep `--write-enabled-yaml`: it writes a copy of the solver YAML with CMB outputs
  enabled into the output directory and uses it for the run. The shipped default
  YAML does not enable them, so the run fails without the flag.
- `--boltzmann-yaml` selects another CAMB or CLASS configuration; the default is the
  package's `default.yaml` for the chosen `--code`.

## Run the standard presets

`scripts/archive/cmb/run_cmb_benchmarks.py` runs the Planck, Simons Observatory and
Stage-IV specs in turn, each through the smoke script with CAMB:

```bash
uv run python scripts/archive/cmb/run_cmb_benchmarks.py \
  --which planck,so,s4 --observables CMB_T,CMB_E --outdir tmp/cmb_bench
```

Add `--lmax 120` for a quick test. Each preset writes its Fisher matrix to
`<outdir>/<preset>/`. Without `--outdir`, results go to
`scripts/benchmark_results/cmb_bench_<timestamp>/`, which is git-ignored.

See also the [scripts reference](scripts_reference.md).
