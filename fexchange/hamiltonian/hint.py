"""
Coulomb interaction term H_int via Slater-Condon parameters.

Spec reference: 02-01-HINT_FORM.
Uses sympy.physics.wigner for 3j symbols.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sympy.physics.wigner import wigner_3j

from fexchange.utils.constants import ELL, N_ORB
from fexchange.core.fock import dim_sector
from fexchange.core.space_ls import orbital_index
from fexchange.core.fermion import two_body_operator_matrix
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_ZERO
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError

# Cache C^(k)(ma, mb) coefficients
_CK_CACHE: dict[tuple[int, int, int], float] = {}
_HINT_RANKS: tuple[int, ...] = (0, 2, 4, 6)
_HINT_RANK_COEF_CACHE: dict[tuple[int, int], NDArray[np.complexfloating]] = {}

__all__ = [
    "build_hint_rank_coefficients",
    "build_hint_rank_matrix",
    "build_hint_matrix",
]


def _ck(k: int, ma: int, mb: int) -> float:
    """
    Gaunt coefficient C^(k)(ma, mb) for l=3 shell (02-01 §4).

    C^(k)(ma,mb) = (-1)^ma * (2l+1) * 3j(l,k,l; 0,0,0) * 3j(l,k,l; -ma, q, mb)
    where q = ma - mb.
    """
    key = (k, ma, mb)
    if key in _CK_CACHE:
        return _CK_CACHE[key]

    ell = ELL
    q = ma - mb
    if abs(q) > k:
        _CK_CACHE[key] = 0.0
        return 0.0

    # 3j symbols via sympy (exact rational arithmetic)
    winger_3j_000 = float(wigner_3j(ell, k, ell, 0, 0, 0))
    winger_3j_mqm = float(wigner_3j(ell, k, ell, -ma, q, mb))

    val = ((-1) ** ma) * (2 * ell + 1) * winger_3j_000 * winger_3j_mqm
    _CK_CACHE[key] = val
    return val


def _build_rank_tensor(k: int, n_orb: int = N_ORB) -> NDArray[np.complexfloating]:
    """Build one rank-resolved two-body coefficient tensor V_k[p,q,r,s]."""
    if k not in _HINT_RANKS:
        raise PhysError(
            "FXE-PHYS-001",
            f"Unsupported Coulomb rank k={k}; expected one of {_HINT_RANKS}",
            actual={"k": k, "allowed": list(_HINT_RANKS)},
        )
    ell = ELL
    m_range = range(-ell, ell + 1)
    V = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=DTYPE_COMPLEX)
    for m1 in m_range:
        for m2 in m_range:
            for m3 in m_range:
                m4 = m1 + m2 - m3
                if abs(m4) > ell:
                    continue
                ck_14 = _ck(k, m1, m4)
                ck_32 = _ck(k, m3, m2)
                coeff = ck_14 * ck_32
                if abs(coeff) < EPS_ZERO:
                    continue
                for s1 in (-0.5, 0.5):
                    for s2 in (-0.5, 0.5):
                        p = orbital_index(m1, s1)
                        q = orbital_index(m2, s2)
                        r = orbital_index(m3, s2)
                        s = orbital_index(m4, s1)
                        V[p, q, r, s] += coeff
    return V


def build_hint_rank_coefficients(
    k: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """Return cached rank-k coefficient tensor V_k[p,q,r,s] for given n_orb."""
    if k not in _HINT_RANKS:
        raise PhysError(
            "FXE-PHYS-001",
            f"Unsupported Coulomb rank k={k}; expected one of {_HINT_RANKS}",
            actual={"k": k, "allowed": list(_HINT_RANKS)},
        )
    cache_key = (k, n_orb)
    cached = _HINT_RANK_COEF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    V_k = _build_rank_tensor(k, n_orb)
    V_k.setflags(write=False)
    _HINT_RANK_COEF_CACHE[cache_key] = V_k
    return V_k


def build_hint_rank_matrix(
    k: int,
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build one rank-resolved many-body matrix O_k for k in {0,2,4,6}.

    Coefficient tensor V_k is cached by (k, n_orb).
    """
    V_k = build_hint_rank_coefficients(k, n_orb)
    return two_body_operator_matrix(V_k, n_ele, n_orb)


def build_hint_matrix(
    n_ele: int,
    F: dict[int, float] | None = None,
    *,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build the full H_int many-body matrix in sector n_ele.

    Parameters
    ----------
    n_ele : int
        Electron count.
    F : dict mapping k -> F^k value (k in {0,2,4,6}).
        If None, returns rank-resolved operator sum with F^k = 1 for each k.
    """
    F_values = F if F is not None else {0: 1.0, 2: 1.0, 4: 1.0, 6: 1.0}

    dim = dim_sector(n_ele, n_orb)
    H = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)

    for k in _HINT_RANKS:
        coeff_k = float(F_values.get(k, 0.0))
        if abs(coeff_k) < EPS_ZERO:
            continue
        V_k = build_hint_rank_coefficients(k, n_orb)
        H_k = two_body_operator_matrix(V_k, n_ele, n_orb)
        H += coeff_k * H_k

    check_hermitian(H, label="H_int", module="hint")
    return H
