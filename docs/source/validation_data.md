# Validation data and examples (cjfp-data)

[cjfp-data](https://dagshub.com/santiagocasas/cjfp-data) is the public companion
repository for CosmicJellyfish validation artifacts, posterior chains and HPC
workflows. The source repository keeps runnable tools and small configuration
examples; the data repository holds published outputs. A real metadata-only
example is shown in {doc}`likelihood_sampling`.

## Get the repository and selected data

Keep the checkouts next to one another:

```bash
git clone https://github.com/santiagocasas/cosmicjellyfish.git
git clone https://dagshub.com/santiagocasas/cjfp-data.git
cd cjfp-data
python -m pip install -r requirements.txt
```

The repository is public. Downloading DVC-backed artifacts may require a DagsHub
account and authentication; follow the repository's access instructions if
prompted. Git contains scripts and metadata; large arrays and chains need a
separate DVC download. Choose a target rather than downloading every suite:

```bash
dvc pull validation/fisher.dvc
# Alternatively, fetch only the ns-response bundle:
dvc pull scripts/validation_results/ns_derivative_lhs.dvc
```

| Location in cjfp-data | Contents and instructions |
| --- | --- |
| `chains/` | Selected completed Nautilus runs (`chains.dvc`). |
| `validation/fisher/` | CAMB-versus-CLASS Fisher comparisons, logs and dashboard. |
| `validation/syren_deaglan_lhs_opt*/` | SYREN comparisons; [targeted-download guide](https://dagshub.com/santiagocasas/cjfp-data/src/main/validation/SYREN_COMPARISONS.md). |
| `scripts/validation_results/ns_derivative_lhs/` | ns-response arrays, figures and provenance; [sharing guide](https://dagshub.com/santiagocasas/cjfp-data/src/main/scripts/NS_RESPONSE_SHARING.md). |
| `jobs_punch/` | JURECA launchers, sampler YAMLs and campaign notes. |

## Run a CAMB-versus-CLASS Fisher campaign

Install the source environment first, then run the data-side preflight from the
source checkout:

```bash
cd ../cosmicjellyfish
uv sync --locked
bash ../cjfp-data/scripts/run-fisher-validation.sh --source "$PWD" --check
```

For a fresh 25-case campaign (adjust concurrency to your CPU allocation):

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash ../cjfp-data/scripts/run-fisher-validation.sh --source "$PWD" \
  --run --all --jobs 2 --omp-threads 4
```

The launcher records the source revision, package versions and inputs, and writes
to a new `cjfp-data/validation/fisher/numpy2_<revision>_<timestamp>_<pid>/`
directory. Keep the checkout unchanged until it finishes. The validation compares
CAMB and CLASS; consult the per-case comparison reports and
`dashboard/index.html` inside that campaign. For setup and troubleshooting see the
[workstation runbook](https://dagshub.com/santiagocasas/cjfp-data/src/main/validation/NUMPY2_VALIDATION.md).

## Nautilus and cluster workflows

From the source root, preview the maintained Euclid sampling campaign with:

```bash
bash scripts/likelihood/nautilus/run_euclid_mp_nautilus.sh --dry-run
```

Remove `--dry-run` to sample after reviewing the resource settings in
{doc}`scripts_reference`. This campaign defaults to timestamped outputs under
`../cjfp-data/chains/`. `CFP_DATA_REPO` selects another data checkout for maintained
source runners; pass `--source` explicitly to the data-side Fisher launcher when
the source checkout is elsewhere. The generic YAML sampler still defaults to
`./chains/chains_<name>/` unless `output_dir` is set.

For Slurm production runs, use the configurations and site-specific instructions
in [jobs_punch](https://dagshub.com/santiagocasas/cjfp-data/src/main/jobs_punch)
and its [Nautilus findings](https://dagshub.com/santiagocasas/cjfp-data/src/main/jobs_punch/nautilus_jureca_findings.md).
Review scratch paths, environment activation and CPU allocation for your site.

## Citation and reproducibility

When publishing results, link to cjfp-data and record both the source and data
Git revisions plus the selected campaign directory. See the source repository's
`CITATION.cff` for software citation metadata. A Zenodo DOI is planned for the
CosmicJellyfish GitHub release; a separate data DOI or mirror is deferred.
