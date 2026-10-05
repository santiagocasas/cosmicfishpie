"""Symbolic check of the GCsp wedge likelihood versus Fisher integrands.

Run: python scripts/verify_gcsp_covariance_sympy.py (requires sympy).
No cosmology runs or sampling. This checks algebra, not runtime model equality.
Assumptions: galaxy auto-spectrum, fixed noise and covariance, matched signal
and its derivatives, identical grids/volumes, zero residual at the fiducial.
"""

import sympy as s


def show(title, expression):
    print(f"\n{title}")
    s.pprint(s.simplify(expression))


P, n, k, V = s.symbols("P n k V", positive=True)
Pi, Pj = s.symbols("P_i P_j", real=True)
N = 1 / n
Q = P + N

# spectro_cov.py:223-225 and cosmicfish.py:597-603.
veff = (n * P / (1 + n * P)) ** 2 / (8 * s.pi**2)
fisher = 2 * k**2 * V * veff * (Pi / P) * (Pj / P)
show("1. Fisher integrand, with d(log P)/d(theta_i) = P_i/P:", fisher)

# noisy_P_ij adds 1/n BEFORE observable_Pgg enters the covariance.
# spectro_cov.py:361-370; spectro_like.py:37-40, 180, 186.
covariance = 8 * s.pi**2 * Q**2 / (k**2 * V)
show("2. Likelihood covariance kernel, Q = P + 1/n:", covariance)

# A local two-parameter signal expansion suffices for the exact Hessian at
# zero residual; second signal derivatives multiply the vanishing residual.
x, y = s.symbols("delta_theta_i delta_theta_j", real=True)
Q_theory = Q + Pi * x + Pj * y
chi2 = 2 * (Q_theory - Q) ** 2 / covariance  # factor 2: mu in [0, 1]
nll = chi2 / 2  # negative log likelihood, NOT chi2 itself
hessian = s.diff(nll, x, y).subs({x: 0, y: 0})
show("3. Mixed Hessian of -log L = chi2/2 at the fiducial:", hessian)
show("4. Hessian minus Fisher (must be zero):", hessian - fisher)
assert s.simplify(hessian - fisher) == 0

# Demonstrate why inserting an additional effective-volume suppression would
# count the same Poisson noise a second time.
suppression = (n * P / (1 + n * P)) ** 2
show("5. Ratio after incorrectly adding another noise suppression:", suppression)

# Equality above depends on the SAME signal derivative in both code paths.
# Allow the likelihood's signal to have derivatives R_i,R_j at the same
# fiducial value, and/or its noise to vary with parameters.
Ri, Rj, Ni, Nj = s.symbols("R_i R_j N_i N_j", real=True)
different_theory = Q + (Ri + Ni) * x + (Rj + Nj) * y
different_nll = (different_theory - Q) ** 2 / covariance
different_hessian = s.diff(different_nll, x, y).subs({x: 0, y: 0})
show(
    "6. General mismatch: different signal/noise derivatives:", s.factor(different_hessian - fisher)
)
assert s.simplify((different_hessian - fisher).subs({Ri: Pi, Rj: Pj, Ni: 0, Nj: 0})) == 0

print("\nPASS: fixed-noise, matched-signal wedge Fisher and likelihood weights agree.")
print("This does not verify actual signal derivatives, multipoles, or sampler convergence.")
