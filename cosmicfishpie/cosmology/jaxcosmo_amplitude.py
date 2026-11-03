r"""JAX port of the SYREN-NEW ``As <-> sigma8`` conversion.

``jax-cosmo`` normalises its linear power spectrum with ``sigma8`` and has no
primordial amplitude ``A_s`` input.  CosmicFishPie parameterises cosmologies
with ``10^9 A_s``, so the ``jaxcosmo`` backend needs a differentiable map
``(10^9 A_s, Omega_m, Omega_b, h, n_s, m_nu, w0, w_a) -> sigma8`` that is
evaluated *inside* the traced function.  Treating ``sigma8`` as a constant
would silently give wrong derivatives with respect to ``A_s`` and to every
parameter on which the conversion depends.

The expressions below are the "max precision" symbolic fits of
``symbolic_pofk.linear_new`` (Bartlett et al.), ported verbatim to
``jax.numpy`` so that they can be traced, jitted and differentiated.  The
coefficients are copied from the ``symbolic_pofk`` revision pinned in
``pyproject.toml``; ``tests`` should compare against that package.

All functions are pure and accept scalars or arrays.  ``As`` is ``10^9 A_s``.
The fit is valid for the symbolic_pofk training domain; callers are
responsible for staying inside it.
"""

from __future__ import annotations

import jax.numpy as jnp

# Coefficients of the max-precision fit (symbolic_pofk.linear_new, 37 numbers).
_B = (
    0.0246, 2.1062, 2.9355, 0.7626, 0.2962, 0.5096,
    4.4025, 3.6495, 0.4144, 0.8615, 0.6188, 0.1751,
    0.824, 0.5466, 0.5519, 0.3689, 0.3261, 0.2002,
    0.8892, 0.4462, 1.215, 3.4829, 2.5852, 0.0242,
    0.0051, 0.1614, 1.2991, 4.1426, 3.3055, 0.5716,
    6.0094, 1.9569, 2.1477, 1.1902, 0.128, 0.6931,
    0.2661,
)  # fmt: skip


def _sigma8_over_sqrt_as(Om, Ob, h, ns, mnu, w0, wa):
    """Return the amplitude-independent factor ``sigma8 / sqrt(10^9 A_s)``."""
    b = _B

    term1_inner = Om * b[1] + (b[2] * mnu - b[3] * ns + jnp.log(b[4] * h - b[5] * mnu)) * (
        b[6] * h + b[7] * mnu - b[8] * ns + 1
    )
    term1 = b[0] * term1_inner

    term2 = b[9] * h - mnu

    term3_inner1 = (b[12] * w0 - b[13] * wa - jnp.log(Om * b[14])) * (
        Om * b[15] + b[16] * w0 + b[17] * wa + jnp.log(-b[18] * w0 - b[19] * wa)
    )
    term3_inner2 = jnp.log(Om * b[20] + jnp.log(-b[21] * w0 - b[22] * wa))
    term3 = (
        b[10] * w0 - b[11] * mnu - term3_inner1 - term3_inner2 + jnp.log(-b[23] * w0 - b[24] * wa)
    )

    term4_inner1 = Ob * b[30] - b[31] * h - jnp.log(Om * b[32])
    term4_inner2 = Om * b[33] - b[34] * h - b[35] * mnu - b[36] * ns
    term4 = (
        b[25] * mnu
        - jnp.sqrt(Ob) * b[26]
        - Ob * b[27]
        + Om * b[28]
        - b[29] * h
        + 1
        + term4_inner1 * term4_inner2
    )

    return term1 * term2 * term3 * term4


def As_to_sigma8(As, Om, Ob, h, ns, mnu, w0, wa):
    """Emulated ``10^9 A_s -> sigma8`` (differentiable, JAX).

    Parameters follow ``symbolic_pofk.linear_new.As_to_sigma8_max_precision``:
    ``As`` is ``10^9 A_s``, ``Om``/``Ob`` the z=0 total-matter and baryon
    density parameters, ``h`` the reduced Hubble constant, ``mnu`` the sum of
    neutrino masses in eV.
    """
    return _sigma8_over_sqrt_as(Om, Ob, h, ns, mnu, w0, wa) * jnp.sqrt(As)


def sigma8_to_As(sigma8, Om, Ob, h, ns, mnu, w0, wa):
    """Emulated ``sigma8 -> 10^9 A_s`` (differentiable, JAX); inverse of :func:`As_to_sigma8`."""
    return (sigma8 / _sigma8_over_sqrt_as(Om, Ob, h, ns, mnu, w0, wa)) ** 2
