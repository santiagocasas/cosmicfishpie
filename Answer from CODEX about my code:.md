Answer from CODEX about my code:

Short answer: you’re good — the flags benchmarks still behave correctly.

Flags benchmark and fast-only: Unaffected. The script explicitly sets the env flags (OFF for SLOW, ON for FAST) and reloads the modules before each run. This overrides the new defaults reliably.

Regular Fisher benchmark: Now runs with FAST paths by default (since defaults are ON). If you ever want the old “slow-by-default” behavior for a single run, either:

Use the flags benchmark (it gives you both SLOW and FAST in one go), or
Temporarily disable via env for that one run: COSMICFISH_FAST_EFF=0 COSMICFISH_FAST_P=0 COSMICFISH_FAST_KERNEL=0 python scripts/photometric_benchmark.py …
Tests: Unaffected. They monkeypatch the module flags directly (True/False) and don’t rely on defaults.

So: SLOW/FAST comparisons in the script remain accurate; only the “plain” run becomes FAST by default.

I profiled the script using py-spy, can you read this file and tell me the slowest functions that are still bottlenecks?


Here are the hottest remaining bottlenecks from your py-spy capture, ordered by impact:

comoving distance inside WL efficiency

Stack: scripts/photometric_benchmark.py → Fisher.compute → PhotoCov.compute_derivs → Derivatives.derivative_3pt → PhotoCov.getcls → ComputeCls.compute_all → compute_kernels → lensing_efficiency → integral_efficiency → much_faster_integral_efficiency → cosmology.comoving → colossus.cosmology.comovingDistance → scipy.integrate.quad
File refs: cosmicfishpie/LSSsurvey/photo_obs.py:639, cosmicfishpie/cosmology/cosmology.py:1786
Notes: This is the clear top hotspot in your sample. comoving is computed via Colossus integrals under the hood; even when vectorized, it integrates per z. It dominates the efficiency path.
Symbolic power spectrum pipeline (halofit + linear Pk)

Stack: symbolic_results → vectorized_halofit → run_halofit → linear.plin_emulated → pk_EisensteinHu_zb → scipy.integrate.simpson/logF_max_precision
File refs: cosmicfishpie/cosmology/cosmology.py:1119, symbolic_pofk/linear.py:52/79/302/361/366
Notes: Shows up on initialization and when P(k,z) is evaluated; smaller than comoving but non-trivial.
sqrtP_limber → matpow/Pmm

Stack: ComputeCls.sqrtP_limber → cosmology.matpow → Pmm
File refs: cosmicfishpie/LSSsurvey/photo_obs.py:488, cosmicfishpie/cosmology/cosmology.py:1670–1691
Notes: Present but lighter in this capture compared to comoving.
Photo window normalization and n(z)

Stack: PhotoWindow.init → norm → ngal_photoz
File refs: cosmicfishpie/LSSsurvey/photo_window.py:170/212–213
Notes: One-off cost on initialization; small compared to the above.
Quick wins to consider (if you want to trim comoving cost further)

Reuse cached chi in efficiency: in integral_efficiency you already have self._chi_z. For the “much faster” path, skip comoving_func(z) entirely and use the cached array to avoid re-invoking Colossus; that removes the profiled comoving stack in this function.
Ensure no redundant cosmology/comoving calls sneak in elsewhere (the profile suggests most load is in efficiency).
If you want, I can patch integral_efficiency to inject the cached chi into much_faster_integral_efficiency to avoid any fresh comoving calls and re-run the profile to quantify the gain.

now read rawprof_GC.txt, this is for the observable GCph, what do you notice differently?


Key differences vs the WL profile and current GCph bottlenecks

What changed

The heavy WL lensing-efficiency hotspot is gone. In the WL run, the top cost was comoving distance inside much_faster_integral_efficiency; that entire stack disappears for GCph-only.
For GCph, time shifts to power-spectrum evaluation and interpolation inside sqrtP_limber and cosmology.matpow/Pmm.
Top hotspots (from rawprof_GC.txt)

SciPy spline interpolation during P(k, z) evaluation

Frames: scipy/interpolate/_fitpack2.py:1089 and 1079 called from cosmology.Pmm via cosmology.matpow during ComputeCls.sqrtP_limber.
Shows up 29 times at 1089 and 2 times at 1079 → strong indicator that repeated 1D spline evaluations dominate GCph compute.
Interpretation: matpow/Pmm is hitting spline interpolants of P(k,z) many times across the kappa grid.
sqrtP_limber for GCph

Frames: photo_obs.py:492, 496, 503 → multiple visits while filling sqrtPell["GCph"].
Calls cosmology.matpow → Pmm and spline evaluations (see above).
Initialization/comoving (minor now)

One call path through photo_obs.init → cosmology.comoving (to precompute self._chi_z), but it’s no longer the dominant cost as in the WL case.
Window normalization/ngal (minor)

photo_window.norm/ngal_photoz list comprehensions are visible (e.g., hit 12 times), but they are a smaller fraction compared to the interpolation heavy path.
Symbolic power spectrum setup (smaller share)

vectorized_halofit → run_halofit → pk_EisensteinHu_zb and some Simpson integration/logF steps pop up, but they rank behind the spline calls.
Why this differs from the WL profile

WL spent big in lensing_efficiency due to repeated (now optimized) comoving calculations; GCph doesn’t use that integral and instead leans heavily on P(k,z) interpolation while building sqrtPell["GCph"].
Suggestions to reduce GCph time further

Batch or cache spline evaluations in matpow/Pmm:
If Pmm currently evaluates splines per scalar k or small arrays, change to evaluate entire k rows per z in one call and avoid Python loops.
Consider constructing a single RegularGridInterpolator over (z, k) for P(k,z) and do vectorized queries on the whole kappa grid.
Reuse interpolation objects:
If new interp1d/splines are created repeatedly inside loops, hoist them out (create once per z or globally) and reuse.
Precompute P(k,z) on the required kappa grid:
Build the kappa grid (ell+0.5)/chi[z] once, clip to kmax mask, evaluate Pmm in a vectorized chunk, then index into the result — minimizes Python overhead and splines’ call count.
Minor: vectorize photo_window.ngal normalization to eliminate small list comps.
If you want, I can inspect cosmology.matpow/Pmm and sqrtP_limber to propose a concrete vectorization patch that reduces the number of scipy.interpolate calls made during GCph.

I really want to bring the computations of the photometric C_ls in less than one second, actually 0.1seconds each or so, is that poissible?


Short answer: getting a full photometric Cls build down to ~0.1 s is only realistic if you (a) aggressively cache/precompute and (b) accept coarser grids or simplified physics. With your current fidelity (symbolic backend, halofit, ~200 z samples and ~100 ℓ samples), sub‑second is feasible; ~0.1 s is ambitious unless you pretabulate P(k,z) or switch to a very light model.

What still costs time

WL: comoving distance inside lensing efficiency was the top hotspot; we reduced that to O(N), but it still calls comoving(z). We can remove those calls entirely by passing the cached χ(z) into the efficiency helper.
GCph+WL: spline interpolation inside Pmm/matpow while filling sqrtP_limber is the main remaining bottleneck (fitpack call). Even with broadcasting, repeated spline calls dominate.
What would make 0.1 s possible

Precompute P(k,z) once per (z,k) grid and interpolate via a fast, vectorized 2D interpolator (RegularGridInterpolator) or even bilinear with numpy if the grid is regular. Then reuse it for Pell and sqrtP_limber. This avoids thousands of spline __call__s.
Remove SciPy splines from the hot path. Where you currently create/use splines per z or per column, replace with:
global/tabulated arrays + numpy.interp (monotonic) or a small custom linear/PCHIP in pure NumPy
or a single RegularGridInterpolator over (z,k)
Lower the workload:
Reduce zsamp and ellsamp (or lmin/lmax window) and do a second pass upsampling via interpolation in ℓ.
Skip halofit in FAST mode (linear P only) or use a lightweight emulator pretabulated for the survey’s z,k box.
Caching:
Cache P(k,z) on the κ grid for the target ℓs (κ=(ℓ+0.5)/χ). Compute once and reuse for WL/GCph sqrtP and Pell.
Cache n_i(z)/n̄(z) per bin and hubble(z), etc. across code paths (you already do some of this).
Two concrete, surgical changes I can implement now

WL efficiency: eliminate comoving calls inside much_faster_integral_efficiency by feeding cached χ(z).
Result: removes the comovingDistance→quad stack from that function entirely.
sqrtP_limber: batch evaluation for GCph and WL
Build κ array for all (z,ℓ) with a mask for kmax
Evaluate Pmm(z, κ) in one or a few large vector calls
Fill both sqrtPell["GCph"] and sqrtPell["WL"/"WL_IA"] from that, avoiding per‑element spline calls
Expected gains

WL: small but measurable (the O(N) integrator becomes “no SciPy/Colossus inside”).
GCph+WL: biggest win; typically 2–5× speedup vs per‑column/per‑element spline calling. With coarser grids (e.g., zsamp100, ellsamp50) this can push total compute well under a second.
What won’t meet 0.1 s without larger tradeoffs

Keeping halofit + fine grids + Python interpolation in the loop. To hit ~0.1 s per full compute, you likely need pretabulated P(k,z) or a compiled path (Numba/Cython/JAX).
Next steps

Want me to implement the two patches above (cached-χ lensing efficiency and vectorized sqrtP_limber) and re‑run your benchmark/profiles? If yes, I’ll do it now and report the new timings.