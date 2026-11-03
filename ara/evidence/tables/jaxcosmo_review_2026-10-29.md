# jaxcosmo branch review: evidence summary (2026-10-29)

Full log: cfp-jaxcosmo/docs/source/jaxcosmo_review.md (findings F1-F10, fix log, limits of evidence).

| item | before | after |
|---|---|---|
| Savitzky-Golay operator (GCsp) fiducial lnP, max rel vs legacy | 7.9e-5 | 2.2e-6 |
| GCsp Fisher max abs diff / max, JAX-FD vs legacy (0.3% step) | 1.5e-6 | 1.5e-6 |
| GCsp dlnP/dOmegam JAX-FD vs legacy, max / max-significant | 1.1e-1 / 2.4e-1 | 5.9e-5 / 1.7e-3 with legacy_pk_clamp |
| AP_effect=False, perturbed cosmology, max abs dlnP | 3.8e-2 | 2.5e-5 with legacy_pk_clamp |
| Halofit Newton residual over 24 box cosmologies x 4 z | n/a | 4.4e-16 |
| Halofit vs jax-cosmo, max rel over the box | n/a | 0.59% (0.05% at fiducial) |
| Halofit Jacobian cost (6z x 4k x 3 params) | 24.5 s | 8.8 s |
| photometric AD Fisher end-to-end | 37.6 s | 18.3 s (identical sigmas) |
| jax-cosmo master vs patch, 1% step AD-vs-FD error in P (scale-relative) | 2.4e-1 | 1.0e-3 |
| CFP test suite | 443 passed | 474 passed, 1 skipped |

Commits on jaxcosmo-autodiff: c2da941, 4212e0d, 180d2f8, 043b8fd, bd77f26, bfa8f72, 756e52a, b9d9af6.
jax_cosmo local branch smooth-halofit-nonlinear-scale: d193ebf (not pushed, no PR).
