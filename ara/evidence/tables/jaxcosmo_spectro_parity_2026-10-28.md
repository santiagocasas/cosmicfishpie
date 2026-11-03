# jaxcosmo spectroscopic parity

Fiducial: `{"10^9As": 2.1, "Omegam": 0.32, "Omegab": 0.05, "h": 0.67, "ns": 0.96, "mnu": 0.0}`

## ln P(k, mu) at the fiducial point
- 4 redshift bins; max |dlnP|/max|lnP| = 7.926e-05, mean = 6.937e-07; max relative where |lnP| > 1% of max = 8.426e-05
- worst bins: `0` 7.93e-05, `1` 6.78e-05, `2` 5.45e-05, `3` 4.66e-05

## Derivatives
- **jax-fd_vs_legacy-fd**: worst parameter `Omegam` with max |dP|/max|dP| = 1.074e-01
  - `10^9As`: max 1.656e-04, max-significant 1.656e-04
  - `Omegam`: max 1.074e-01, max-significant 2.407e-01
  - `Omegab`: max 1.683e-05, max-significant 1.818e-04
  - `h`: max 5.380e-04, max-significant 2.606e-02
  - `ns`: max 1.476e-04, max-significant 2.112e-04
  - `lnbg_1`: max 7.803e-13, max-significant 1.250e-12
  - `lnbg_2`: max 9.210e-07, max-significant 1.440e-06
  - `lnbg_3`: max 2.393e-06, max-significant 3.661e-06
  - `lnbg_4`: max 1.566e-07, max-significant 2.343e-07
- **jax-ad_vs_jax-fd**: worst parameter `h` with max |dP|/max|dP| = 6.909e-03
  - `10^9As`: max 3.000e-06, max-significant 3.000e-06
  - `Omegam`: max 5.512e-03, max-significant 7.726e-03
  - `Omegab`: max 1.961e-06, max-significant 3.476e-05
  - `h`: max 6.909e-03, max-significant 1.004e-02
  - `ns`: max 9.100e-08, max-significant 8.867e-06
  - `lnbg_1`: max 2.075e-08, max-significant 2.699e-08
  - `lnbg_2`: max 3.240e-08, max-significant 4.203e-08
  - `lnbg_3`: max 4.478e-08, max-significant 5.822e-08
  - `lnbg_4`: max 5.930e-08, max-significant 7.709e-08
- **jax-ad_vs_legacy-fd**: worst parameter `Omegam` with max |dP|/max|dP| = 1.074e-01
  - `10^9As`: max 1.626e-04, max-significant 1.626e-04
  - `Omegam`: max 1.074e-01, max-significant 2.407e-01
  - `Omegab`: max 1.798e-05, max-significant 2.068e-04
  - `h`: max 6.912e-03, max-significant 2.611e-02
  - `ns`: max 1.477e-04, max-significant 2.113e-04
  - `lnbg_1`: max 2.075e-08, max-significant 2.699e-08
  - `lnbg_2`: max 8.993e-07, max-significant 1.406e-06
  - `lnbg_3`: max 2.426e-06, max-significant 3.710e-06
  - `lnbg_4`: max 2.026e-07, max-significant 3.032e-07

## Fisher matrices

| variant | seconds | 10^9As | Omegam | Omegab | h | ns | lnbg_1 | lnbg_2 | lnbg_3 | lnbg_4 |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy-fd | 1.6 | 0.0234364 | 0.0047303 | 0.00124459 | 0.0106187 | 0.00985228 | 0.00930391 | 0.0093226 | 0.00935341 | 0.00936088 |
| jax-fd | 5.5 | 0.0234369 | 0.00473048 | 0.0012446 | 0.0106187 | 0.00985398 | 0.00930395 | 0.00932265 | 0.00935351 | 0.00936086 |
| jax-ad | 17.8 | 0.0234312 | 0.00472608 | 0.00124375 | 0.0105974 | 0.00985449 | 0.00930105 | 0.0093199 | 0.00935086 | 0.00935829 |

| parameter | jax-fd sigma vs legacy-fd | jax-ad sigma vs legacy-fd |
|---|---|---|
| `10^9As` | +0.002% | -0.022% |
| `Omegam` | +0.004% | -0.089% |
| `Omegab` | +0.001% | -0.067% |
| `h` | -0.001% | -0.201% |
| `ns` | +0.017% | +0.022% |
| `lnbg_1` | +0.000% | -0.031% |
| `lnbg_2` | +0.001% | -0.029% |
| `lnbg_3` | +0.001% | -0.027% |
| `lnbg_4` | -0.000% | -0.028% |

- `jax-fd` vs `legacy-fd`: Fisher max |dF|/max|F| = 1.495e-06, mean = 4.965e-08
- `jax-ad` vs `legacy-fd`: Fisher max |dF|/max|F| = 1.388e-05, mean = 5.274e-07
- `jax-ad` vs `jax-fd`: Fisher max |dF|/max|F| = 1.427e-05, mean = 5.139e-07

