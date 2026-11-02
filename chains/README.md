# Chain artifacts

Generated Nautilus chains and HDF5 checkpoints are not source-repository
artifacts. JURECA jobs write complete runs to
`/p/scratch/punch_astro/cosmicfish/chains`; intentionally selected suites can
then be copied to the validation repository with
`cosmicfishpie-validation/jobs_punch/sync_selected_chains.sh`.

`example_symbolic_2param/` is a deliberately tiny documentation fixture from a
completed symbolic LCDM validation: its text chain contains the header and the
first 100 rows only, and its metadata marks it as truncated. It has no HDF5
checkpoint and must not be used to judge convergence or resume a sampler.
