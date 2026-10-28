"""Smooth Halofit for the ``jaxcosmo`` theory layer.

``jax_cosmo.power.halofit`` locates the nonlinear scale by interpolating
``sigma(R) = 1`` linearly on a fixed 256-node grid in ``log R``.  The resulting
nonlinear spectrum is only piecewise smooth in the cosmological parameters:
whenever the root moves between two grid nodes the derivative jumps.  For a 1%
parameter step those jumps reach ~13% in individual redshift rows, which is
fatal both for autodiff (it differentiates one linear piece) and for finite
differences (they average over several).

This module keeps the Halofit prescription and its coefficients - taken from
``jax_cosmo.power`` (MIT licence, Copyright (c) 2022 Differentiable Universe
Initiative) - and only replaces the root find with a fixed-count Newton
iteration on the smooth integral

.. math::

    \\sigma^2(R) = \\int d\\ln k \\, \\frac{k^3 P_{\\rm lin}(k, a)}{2\\pi^2}
                  e^{-k^2 R^2},

which converges to machine precision and is differentiable.  Everything else
(``n_eff``, ``C``, the ``a_n ... nu_n`` polynomials and equation C2) follows
jax-cosmo so that the two implementations agree to the accuracy of their
respective root finds.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax_cosmo import background as jc_background
from jax_cosmo import power as jc_power

__all__ = ["halofit", "nonlinear_scale"]

# Quadrature limits used by jax-cosmo's Halofit (in ln k).
_LNK_MIN = float(jnp.log(1e-4))
_LNK_MAX = float(jnp.log(1e4))
_LNK_STEPS = 256

# Newton iteration on ln R (in Mpc/h): start, bracket, step limit, iterations.
_LNR_START = 1.0
_LNR_MIN = -20.0
_LNR_MAX = 10.0
_MAX_STEP = 2.0
_NEWTON_ITERATIONS = 40


def _lnk_grid(steps: int = _LNK_STEPS) -> jnp.ndarray:
    """Uniform grid in ``ln k`` (``steps + 1`` nodes) used for every integral."""
    return jnp.linspace(_LNK_MIN, _LNK_MAX, steps + 1)


def _simpson(values: jnp.ndarray, lnk: jnp.ndarray) -> jnp.ndarray:
    """Simpson integration of ``values`` (last axis) over the uniform grid ``lnk``."""
    weights = jnp.ones_like(lnk).at[1:-1:2].set(4.0).at[2:-1:2].set(2.0)
    dlnk = lnk[1] - lnk[0]
    return jnp.sum(values * weights, axis=-1) * dlnk / 3.0


def _sigma2(logr: jnp.ndarray, lnk: jnp.ndarray, pk3: jnp.ndarray, want_derivative: bool = True):
    """``sigma^2(R)`` and its derivative with respect to ``ln R``.

    ``pk3`` holds ``k^3 P_lin(k, a)`` on the ``lnk`` grid; ``logr`` is any shape.
    """
    r = jnp.exp(logr)
    k = jnp.exp(lnk)
    y2 = (r[..., None] * k) ** 2
    kernel = jnp.exp(-y2) * pk3 / (2.0 * jnp.pi**2)
    sigma2 = _simpson(kernel, lnk)
    if not want_derivative:
        return sigma2, None
    return sigma2, _simpson(-2.0 * y2 * kernel, lnk)


def _solve_logr(target, lnk, pk3):
    """Solve ``sigma^2(R) = target`` for ``ln R`` with a fixed-count Newton iteration.

    A converged Newton fixed point differentiates to the implicit derivative,
    unlike a bisection, whose branch conditions carry the parameter dependence.
    ``sigma^2`` is smooth and monotonically decreasing in ``R``, so the iteration
    converges quadratically from a generous starting guess; the step is clamped
    so a bad start cannot throw it out of the bracket.
    """
    lnr = jnp.asarray(_LNR_START)
    for _ in range(_NEWTON_ITERATIONS):
        sigma2, dsigma2 = _sigma2(lnr, lnk, pk3)
        step = jnp.clip((sigma2 - target) / dsigma2, -_MAX_STEP, _MAX_STEP)
        lnr = jnp.clip(lnr - step, _LNR_MIN, _LNR_MAX)
    return lnr


def nonlinear_scale(cosmo, a, transfer_fn):
    """``(k_nl, n_eff, C)`` in ``h/Mpc`` and dimensionless, smooth in the parameters."""
    lnk = _lnk_grid()
    k = jnp.exp(lnk)
    growth = jc_background.growth_factor(cosmo, jnp.atleast_1d(a))[0]
    pk3 = jc_power.linear_matter_power(cosmo, k, 1.0, transfer_fn) * k**3

    lnr = _solve_logr(1.0 / growth**2, lnk, pk3)
    k_nl = 1.0 / jnp.exp(lnr)

    y2 = (k / k_nl) ** 2
    res = pk3 * growth**2 * jnp.exp(-y2) / (2.0 * jnp.pi**2)
    stack = jnp.stack([2.0 * res * y2, 4.0 * res * (y2 - y2**2)], axis=0)
    integrals = _simpson(stack, lnk)
    n = integrals[0] - 3.0
    C = integrals[0] ** 2 + integrals[1]
    return k_nl, n, C


def halofit(cosmo, k, a, transfer_fn, prescription="takahashi2012"):
    """Nonlinear matter power spectrum ``P(k)`` in ``(Mpc/h)^3``.

    Mirrors ``jax_cosmo.power.halofit`` but with the smooth nonlinear scale of
    :func:`nonlinear_scale`.
    """
    a = jnp.atleast_1d(a)[0]
    k = jnp.atleast_1d(k)

    pklin = jc_power.linear_matter_power(cosmo, k, a, transfer_fn)
    k_nl, n, C = nonlinear_scale(cosmo, a, transfer_fn)

    om_m = jc_background.Omega_m_a(cosmo, a)
    om_de = jc_background.Omega_de_a(cosmo, a)
    w = jc_background.w(cosmo, a)
    frac = om_de / (1.0 - om_m)

    if prescription == "smith2003":
        a_n = 10 ** (
            1.4861
            + 1.8369 * n
            + 1.6762 * n**2
            + 0.7940 * n**3
            + 0.1670 * n**4
            - 0.6206 * C
        )
        b_n = 10 ** (0.9463 + 0.9466 * n + 0.3084 * n**2 - 0.9400 * C)
        c_n = 10 ** (-0.2807 + 0.6669 * n + 0.3214 * n**2 - 0.0793 * C)
        gamma_n = 0.8649 + 0.2989 * n + 0.1631 * C
        alpha_n = 1.3884 + 0.3700 * n - 0.1452 * n**2
        beta_n = 0.8291 + 0.9854 * n + 0.3401 * n**2
        mu_n = 10 ** (-3.5442 + 0.1908 * n)
        nu_n = 10 ** (0.9585 + 1.2857 * n)
    elif prescription == "takahashi2012":
        a_n = 10 ** (
            1.5222
            + 2.8553 * n
            + 2.3706 * n**2
            + 0.9903 * n**3
            + 0.2250 * n**4
            - 0.6038 * C
            + 0.1749 * om_de * (1 + w)
        )
        b_n = 10 ** (
            -0.5642
            + 0.5864 * n
            + 0.5716 * n**2
            - 1.5474 * C
            + 0.2279 * om_de * (1 + w)
        )
        c_n = 10 ** (0.3698 + 2.0404 * n + 0.8161 * n**2 + 0.5869 * C)
        gamma_n = 0.1971 - 0.0843 * n + 0.8460 * C
        alpha_n = jnp.abs(6.0835 + 1.3373 * n - 0.1959 * n**2 - 5.5274 * C)
        beta_n = (
            2.0379
            - 0.7354 * n
            + 0.3157 * n**2
            + 1.2490 * n**3
            + 0.3980 * n**4
            - 0.1682 * C
        )
        mu_n = 0.0
        nu_n = 10 ** (5.2105 + 3.6902 * n)
    else:
        raise NotImplementedError(f"Unknown Halofit prescription {prescription!r}")

    if prescription == "takahashi2012":
        f1 = om_m ** (-0.0307)
        f2 = om_m ** (-0.0585)
        f3 = om_m**0.0743
    else:
        f1a, f2a, f3a = om_m ** (-0.0732), om_m ** (-0.1423), om_m**0.0725
        f1b, f2b, f3b = om_m ** (-0.0307), om_m ** (-0.0585), om_m**0.0743
        f1 = frac * f1b + (1 - frac) * f1a
        f2 = frac * f2b + (1 - frac) * f2a
        f3 = frac * f3b + (1 - frac) * f3a

    f = lambda x: x / 4.0 + x**2 / 8.0  # noqa: E731

    d2l = k**3 * pklin / (2.0 * jnp.pi**2)
    y = k / k_nl

    d2q = d2l * ((1.0 + d2l) ** beta_n / (1 + alpha_n * d2l)) * jnp.exp(-f(y))
    d2hprime = a_n * y ** (3 * f1) / (1.0 + b_n * y**f2 + (c_n * f3 * y) ** (3.0 - gamma_n))
    d2h = d2hprime / (1.0 + mu_n / y + nu_n / y**2)
    pk_nl = 2.0 * jnp.pi**2 / k**3 * (d2q + d2h)
    return pk_nl.squeeze()
