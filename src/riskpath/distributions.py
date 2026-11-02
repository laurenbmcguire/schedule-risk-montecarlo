"""PERT-beta sampling with an optional Gaussian copula.

PERT-beta (modified PERT with shape parameter lambda, default 4):

    alpha = 1 + lambda * (m - a) / (b - a)
    beta  = 1 + lambda * (b - m) / (b - a)
    X     = a + (b - a) * Beta(alpha, beta)
    E[X]  = (a + lambda * m + b) / (lambda + 2)

Correlation is introduced with a one-factor Gaussian copula: every task's
latent normal is Z_i = sqrt(rho) * F + sqrt(1 - rho) * e_i, so any two tasks
have latent correlation rho. Uniforms U_i = Phi(Z_i) are mapped through each
task's PERT-beta inverse CDF, which keeps every marginal exactly PERT-beta.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def pert_params(a, m, b, lam: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    a, m, b = (np.asarray(x, dtype=float) for x in (a, m, b))
    span = b - a
    safe = np.where(span > 0, span, 1.0)
    alpha = np.where(span > 0, 1.0 + lam * (m - a) / safe, 1.0)
    beta = np.where(span > 0, 1.0 + lam * (b - m) / safe, 1.0)
    return alpha, beta


def pert_mean(a, m, b, lam: float = 4.0) -> np.ndarray:
    a, m, b = (np.asarray(x, dtype=float) for x in (a, m, b))
    return (a + lam * m + b) / (lam + 2.0)


def pert_ppf(u, a, m, b, lam: float = 4.0) -> np.ndarray:
    """Inverse CDF of PERT-beta. Broadcasting: a, m, b shape (k, 1), u shape (k, n)."""
    a, m, b = (np.asarray(x, dtype=float) for x in (a, m, b))
    alpha, beta = pert_params(a, m, b, lam)
    x = stats.beta.ppf(u, alpha, beta)
    return a + (b - a) * x


def correlated_uniforms(k: int, n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """(k, n) uniforms from a one-factor Gaussian copula with pairwise latent correlation rho."""
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0, 1)")
    # Always draw the common factor so that rho=0 and rho>0 runs share
    # random numbers (common random numbers make comparisons cleaner).
    factor = rng.standard_normal(n)
    idio = rng.standard_normal((k, n))
    z = np.sqrt(rho) * factor[None, :] + np.sqrt(1.0 - rho) * idio
    u = stats.norm.cdf(z)
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def sample_pert(
    a, m, b, n: int, rng: np.random.Generator, lam: float = 4.0, rho: float = 0.0
) -> np.ndarray:
    """Sample PERT-beta durations for k tasks, n iterations. Returns (k, n)."""
    a = np.atleast_1d(np.asarray(a, dtype=float))[:, None]
    m = np.atleast_1d(np.asarray(m, dtype=float))[:, None]
    b = np.atleast_1d(np.asarray(b, dtype=float))[:, None]
    u = correlated_uniforms(a.shape[0], n, rho, rng)
    x = pert_ppf(u, a, m, b, lam)
    # degenerate (a == b) tasks are fixed durations
    return np.where(b > a, x, a)
