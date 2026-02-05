from __future__ import annotations

import numpy as np

from ..api.types import ProjectedTransitionsCache


def compute_heff(
    t: np.ndarray,
    G_fock: np.ndarray,
    cache_np1: ProjectedTransitionsCache,
    cache_nm1: ProjectedTransitionsCache,
    E0: float,
) -> np.ndarray:
    """Compute the second-order effective Hamiltonian.

    Returns Heff of shape (d*d, d*d) where d is the ground-manifold dimension.
    """
    if cache_np1.which != "np1" or cache_nm1.which != "nm1":
        raise ValueError("cache_np1 must be 'np1' and cache_nm1 must be 'nm1'")
    if t.shape != (14, 14):
        raise ValueError("t must have shape (14, 14)")

    d = G_fock.shape[1]
    Aplus = np.stack([T @ G_fock for T in cache_np1.T], axis=0)  # (14, N_np1, d)
    Aminus = np.stack([T @ G_fock for T in cache_nm1.T], axis=0)  # (14, N_nm1, d)

    X = np.einsum("pq,pnc,qmd->nmcd", t, Aplus, Aminus, optimize=True)

    E_np1 = cache_np1.E_mid
    E_nm1 = cache_nm1.E_mid
    denom = E0 - (E_np1[:, None] + E_nm1[None, :])
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Encountered zero energy denominators in SOPT contraction")
    weight = 1.0 / denom

    Heff = np.einsum("nmcd,nmab,nm->cdab", X, X.conjugate(), weight, optimize=True)
    return Heff.reshape(d * d, d * d)
