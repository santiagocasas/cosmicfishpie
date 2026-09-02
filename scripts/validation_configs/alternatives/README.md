# Validation alternatives

These configurations are sensitivity diagnostics, not canonical paper-validation cases.
They are deliberately outside the primary `compare_run_config.env_*` discovery glob and
must be launched directly with `scripts/compare_backends_report.sh --config <path>`.

The `01.*` alternatives isolate the choices that differed from the definitions in
arXiv:2405.06047v1 for the photometric LCDM + `mnu` + `Neff` model:

| ID | Galaxy tracer | `betaIA` | CLASS precision |
|----|---------------|----------|-----------------|
| `01.1` | P_mm | fixed | paper HP |
| `01.2` | P_cb | free | paper HP |
| `01.3` | P_mm | free | paper HP (legacy behavior) |
| `01.4` | P_cb | fixed | stricter non-paper profile |

The canonical primary case `03.2.0` uses P_cb, fixed `betaIA`, and the paper HP CLASS
profile. A future alternatives dashboard can discover `alternative_run_config.env_*`
without mixing these scenarios into the primary validation gate.
