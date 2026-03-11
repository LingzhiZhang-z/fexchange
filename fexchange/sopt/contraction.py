"""
SOPT runtime contraction: Level L2 (route factors), L3 (denominator sum), L4 (W projection).

Spec reference: 04-02-RUNTIME_CONTRACTION.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_ZERO
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError, NumError, BindError

logger = logging.getLogger("fexchange")


# ---------------------------------------------------------------------------
# Level L2: Route factors M_A, M_B (04-02 §1)
# ---------------------------------------------------------------------------

def build_L2(
    l1_result: dict[str, Any],
    t_mu: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Build route factors M_A and M_B (04-02 §1).

    M_A^R_{uv; j1,j2} = sum_{p,q} t_mu[p,q] A^{i,p}_{u,j1} conj(B^{j,q}_{j2,v})
    M_B^R_{rs; j1,j2} = sum_{p',q'} conj(t_mu[p',q']) conj(B^{i,p'}_{j1,r}) A^{j,q'}_{s,j2}

    Parameters
    ----------
    l1_result : output from build_L1.
    t_mu : hopping matrix (n_orb, n_orb). t_mu[p,q] = hopping from site-j orbital q to site-i orbital p.

    Returns dict with M_A, M_B arrays.
    M_A shape: (n_u * n_v, n_j * n_j)  or equivalently (n_u, n_v, n_j, n_j)
    M_B shape: (n_v * n_u, n_j * n_j)  similarly
    """
    A = l1_result["A"]  # (n_orb, n_u, n_j)
    B = l1_result["B"]  # (n_orb, n_j, n_v)
    n_orb = l1_result["n_orb"]
    n_u = l1_result["n_u"]
    n_j = l1_result["n_j"]
    n_v = l1_result["n_v"]

    if t_mu.shape != (n_orb, n_orb):
        raise BindError(
            "FXE-BIND-003",
            f"t_mu shape {t_mu.shape} != ({n_orb},{n_orb})",
            actual={"lhs_shape": list(t_mu.shape), "rhs_shape": [n_orb, n_orb]},
        )

    logger.info("L2: Building route factors")

    # Route A: M_A^R[u,v,j1,j2] = sum_p sum_q t[p,q] * A[p,u,j1] * conj(B[q,j2,v])
    # A[kappa] shape (n_u, n_j), site-i binding: kappa=p
    # B[kappa] shape (n_j, n_v), site-j binding: kappa=q

    # Efficient einsum:
    # M_A_R[u,v,j1,j2] = sum_{p,q} t[p,q] * A[p,u,j1] * conj(B[q,j2,v])
    M_A = np.zeros((n_u, n_v, n_j, n_j), dtype=DTYPE_COMPLEX)
    for p in range(n_orb):
        for q in range(n_orb):
            if abs(t_mu[p, q]) < EPS_ZERO:
                continue
            # A[p] shape (n_u, n_j), B[q] shape (n_j, n_v)
            # outer: A[p][:,j1] * conj(B[q][j2,:])
            # M_A[:,:,j1,j2] += t[p,q] * A[p][:,j1] * conj(B[q][j2,:]).T
            contrib = t_mu[p, q] * np.einsum("uj,kv->uvjk", A[p], B[q].conj())
            M_A += contrib

    # Route B: M_B^R[r,s,j1,j2] = sum_{p',q'} conj(t[p',q']) * conj(B[p',j1,r]) * A[q',s,j2]
    # site-i binding for B: kappa=p', site-j binding for A: kappa=q'
    M_B = np.zeros((n_v, n_u, n_j, n_j), dtype=DTYPE_COMPLEX)
    for pp in range(n_orb):
        for qp in range(n_orb):
            t_conj = t_mu[pp, qp].conj()
            if abs(t_conj) < EPS_ZERO:
                continue
            # B[pp] shape (n_j, n_v) -> conj(B[pp][j1, r]) for site-i
            # A[qp] shape (n_u, n_j) -> A[qp][s, j2] for site-j
            # conj(B[pp])[j1,r] * A[qp][s,j2] -> M_B[r,s,j1,j2]
            M_B += t_conj * np.einsum("ar,sb->rsab", B[pp].conj(), A[qp])

    logger.info("L2 complete: M_A shape=%s, M_B shape=%s", M_A.shape, M_B.shape)

    return {
        "M_A": M_A,
        "M_B": M_B,
        "n_u": n_u,
        "n_v": n_v,
        "n_j": n_j,
    }


# ---------------------------------------------------------------------------
# Level L3: Denominator summation (04-02 §2)
# ---------------------------------------------------------------------------

def build_L3(
    l2_result: dict[str, Any],
    E_u_np1: NDArray[np.floating],
    E_u_nm1: NDArray[np.floating],
    n_ele: int,
) -> dict[str, Any]:
    """
    Build h_pre_j_mu by denominator-weighted summation (04-02 §2).

    h_pre[j3,j4,j1,j2] = sum_{u,v} conj(M_A[u,v,j3,j4]) * M_A[u,v,j1,j2] / Delta_uv
                        + sum_{r,s} conj(M_B[r,s,j3,j4]) * M_B[r,s,j1,j2] / Delta_rs

    Parameters
    ----------
    l2_result : output from build_L2.
    E_u_np1 : intermediate energies for n+1 sector (referenced to E0=0).
    E_u_nm1 : intermediate energies for n-1 sector (referenced to E0=0).
    """
    M_A = l2_result["M_A"]  # (n_u, n_v, n_j, n_j)
    M_B = l2_result["M_B"]  # (n_v, n_u, n_j, n_j) actually (r=n_v_nm1, s=n_u_np1)
    n_j = l2_result["n_j"]

    logger.info("L3: Denominator summation")

    # Route A denominators: Delta_uv = E0 - E_uv = -(E_u^{n+1}[u] + E_v^{n-1}[v])
    n_u = M_A.shape[0]
    n_v = M_A.shape[1]

    # Flatten M_A for matrix contraction (04-02 §2 recommended form)
    J2 = n_j * n_j
    YA = M_A.reshape(n_u * n_v, J2)

    # Build denominator vector for route A
    denom_A = np.zeros(n_u * n_v)
    for u in range(n_u):
        for v in range(n_v):
            E_uv = E_u_np1[u] + E_u_nm1[v]
            Delta_uv = -E_uv  # E0 - E_uv with E0=0
            if (not np.isfinite(Delta_uv)) or (abs(Delta_uv) < EPS_ZERO):
                raise NumError(
                    "FXE-NUM-002",
                    f"Invalid denominator Delta_uv at u={u}, v={v}",
                    actual={"Delta_uv": float(Delta_uv)},
                )
            denom_A[u * n_v + v] = Delta_uv

    w_A = 1.0 / denom_A
    hA = YA.conj().T @ (w_A[:, None] * YA)

    # Route B denominators: Delta_rs = E0 - E_rs = -(E_r^{n-1}[r] + E_s^{n+1}[s])
    n_r = M_B.shape[0]  # n_v from nm1 sector
    n_s = M_B.shape[1]  # n_u from np1 sector

    YB = M_B.reshape(n_r * n_s, J2)

    denom_B = np.zeros(n_r * n_s)
    for r in range(n_r):
        for s in range(n_s):
            E_rs = E_u_nm1[r] + E_u_np1[s]
            Delta_rs = -E_rs
            if (not np.isfinite(Delta_rs)) or (abs(Delta_rs) < EPS_ZERO):
                raise NumError(
                    "FXE-NUM-002",
                    f"Invalid denominator Delta_rs at r={r}, s={s}",
                    actual={"Delta_rs": float(Delta_rs)},
                )
            denom_B[r * n_s + s] = Delta_rs

    w_B = 1.0 / denom_B
    hB = YB.conj().T @ (w_B[:, None] * YB)

    h_pre_j_mu = (hA + hB).reshape(n_j, n_j, n_j, n_j)

    logger.info("L3 complete: h_pre_j_mu shape=%s", h_pre_j_mu.shape)

    return {
        "h_pre_j_mu": h_pre_j_mu,
        "n_j": n_j,
    }


# ---------------------------------------------------------------------------
# Level L4: W projection (04-02 §3)
# ---------------------------------------------------------------------------

def build_L4(
    l3_result: dict[str, Any],
    W: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Apply W projection to get final Heff (04-02 §3).

    h_pre[c,d,a,b] = sum_{j3,j4,j1,j2} conj(W[j3,c]) conj(W[j4,d]) h_pre_j[j3,j4,j1,j2] W[j1,a] W[j2,b]

    Parameters
    ----------
    l3_result : output from build_L3.
    W : projector from SOC-lowest subspace to target basis. Shape (n_j, n_k).
    """
    h_pre_j = l3_result["h_pre_j_mu"]
    n_j = l3_result["n_j"]

    if W.shape[0] != n_j:
        raise BindError(
            "FXE-BIND-003",
            f"W.shape[0]={W.shape[0]} != n_j={n_j}",
            actual={"W_shape": list(W.shape), "n_j": n_j},
        )

    n_k = W.shape[1]
    logger.info("L4: W projection, n_j=%d -> n_k=%d", n_j, n_k)

    # Use einsum for the 4-index projection
    # h[c,d,a,b] = conj(W)[j3,c] conj(W)[j4,d] h[j3,j4,j1,j2] W[j1,a] W[j2,b]
    Wc = W.conj()
    h_mu = np.einsum(
        "ic,jd,ijkl,ka,lb->cdab",
        Wc, Wc, h_pre_j, W, W,
        optimize=True,
    )

    Heff = h_mu.copy()

    # Hermiticity check: reshape to (n_k^2, n_k^2) and check
    Heff_flat = Heff.reshape(n_k * n_k, n_k * n_k)
    check_hermitian(Heff_flat, label="Heff", module="contraction")

    logger.info("L4 complete: Heff shape=%s", Heff.shape)

    return {
        "h_mu_abcd": h_mu,
        "Heff_mu_abcd": Heff,
        "n_k": n_k,
    }
