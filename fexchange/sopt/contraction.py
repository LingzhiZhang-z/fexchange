"""
SOPT runtime contraction: Level L2 (route factors), L3 (denominator sum), L4 (W projection).

Canonical runtime path: L2 -> ``build_L4_fast``, which fuses the L3
denominator summation with the L4 W projection (04-02 §3).

``build_L3`` and ``build_L4`` are the reference (materialized) implementations.
They are not on the runtime path; they exist to (a) emit ``h_pre_j_mu`` for
standalone-L3 execution, and (b) serve as the equivalence oracle that pins
``build_L4_fast`` at 1e-12 (see ``tests/test_contraction.py``).

Spec reference: 04-02-RUNTIME_CONTRACTION.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.numerics import EPS_ZERO
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError, NumError, BindError

logger = logging.getLogger("fexchange")


def _denominator_vector(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    left_label: str,
    right_label: str,
    delta_label: str,
    expected_left: int,
    expected_right: int,
) -> NDArray[np.float64]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.ndim != 1 or right_arr.ndim != 1:
        raise BindError(
            "FXE-BIND-003",
            f"{delta_label} energies must be one-dimensional",
            actual={
                left_label: list(left_arr.shape),
                right_label: list(right_arr.shape),
            },
        )
    if left_arr.shape[0] != expected_left or right_arr.shape[0] != expected_right:
        raise BindError(
            "FXE-BIND-003",
            f"{delta_label} energy lengths do not match route dimensions",
            expected={left_label: expected_left, right_label: expected_right},
            actual={left_label: int(left_arr.shape[0]), right_label: int(right_arr.shape[0])},
        )

    denom = -(left_arr[:, None] + right_arr[None, :]).ravel()
    bad = ~np.isfinite(denom) | (np.abs(denom) < EPS_ZERO)
    if np.any(bad):
        idx = int(np.argmax(bad))
        i, j = divmod(idx, expected_right)
        raise NumError(
            "FXE-NUM-002",
            f"Invalid denominator {delta_label} at {left_label}={i}, {right_label}={j}",
            actual={delta_label: float(denom[idx])},
        )
    return denom


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

    # Route A: M_A[u,v,j1,j2] = sum_{p,q} t[p,q] * A[p,u,j1] * conj(B[q,j2,v])
    # Optimal contraction path (optimize=True): first contract t with A -> (n_orb,n_u,n_j),
    # then contract with conj(B) -> (n_u,n_v,n_j,n_j). Single BLAS ZGEMM for the heavy step.
    M_A = np.einsum("pq,puj,qkv->uvjk", t_mu, A, B.conj(), optimize=True)

    # Route B: M_B[r,s,j1,j2] = sum_{p,q} conj(t[p,q]) * conj(B[p,j1,r]) * A[q,s,j2]
    M_B = np.einsum("pq,par,qsb->rsab", t_mu.conj(), B.conj(), A, optimize=True)

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
#
# Reference (materialized) implementation. Off the runtime path: used for
# standalone-L3 `h_pre_j_mu` emission and as the equivalence oracle for
# `build_L4_fast`. The runtime path uses `build_L4_fast` instead.
# ---------------------------------------------------------------------------

def build_L3(
    l2_result: dict[str, Any],
    E_u_np1: NDArray[np.floating],
    E_u_nm1: NDArray[np.floating],
    n_ele: int,
) -> dict[str, Any]:
    """
    Build h_pre_j_mu by denominator-weighted summation (04-02 §2).

    Reference (materialized) L3. Materializes the (n_j, n_j, n_j, n_j)
    `h_pre_j_mu` tensor. Standalone-L3 execution must emit this; the runtime
    L3->L4 path does not call it (see `build_L4_fast`).

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
    YA = M_A.reshape(n_u * n_v, n_j * n_j)

    denom_A = _denominator_vector(
        E_u_np1,
        E_u_nm1,
        left_label="u",
        right_label="v",
        delta_label="Delta_uv",
        expected_left=n_u,
        expected_right=n_v,
    )

    w_A = 1.0 / denom_A
    hA = YA.conj().T @ (w_A[:, None] * YA)

    # Route B denominators: Delta_rs = -(E_nm1[r] + E_np1[s])
    n_r = M_B.shape[0]  # n_v from nm1 sector
    n_s = M_B.shape[1]  # n_u from np1 sector

    YB = M_B.reshape(n_r * n_s, n_j * n_j)

    denom_B = _denominator_vector(
        E_u_nm1,
        E_u_np1,
        left_label="r",
        right_label="s",
        delta_label="Delta_rs",
        expected_left=n_r,
        expected_right=n_s,
    )

    w_B = 1.0 / denom_B
    hB = YB.conj().T @ (w_B[:, None] * YB)

    h_pre_j_mu = (hA + hB).reshape(n_j, n_j, n_j, n_j)

    logger.info("L3 complete: h_pre_j_mu shape=%s", h_pre_j_mu.shape)

    return {
        "h_pre_j_mu": h_pre_j_mu,
        "n_j": n_j,
    }


def build_L4_fast(
    l2_result: dict[str, Any],
    E_u_np1: NDArray[np.floating],
    E_u_nm1: NDArray[np.floating],
    W: NDArray[np.complexfloating],
    n_ele: int,
) -> dict[str, Any]:
    """
    Canonical runtime L3->L4 path: fused denominator sum + W projection (04-02 §3).

    This is the implementation the pipeline runs. It projects the external
    (j1,j2) legs of each route factor to the target Kramers/non-Kramers basis
    *first*, then does the denominator-weighted Gram contraction directly in the
    n_k space -- so the (n_j, n_j, n_j, n_j) `h_pre_j_mu` tensor is never
    materialized. Intermediate-state sums and denominators are unchanged.

    Correctness contract: algebraically equal to `build_L4(build_L3(...), W)`
    (the reference oracle), pinned at 1e-12 by
    `tests/test_contraction.py::test_build_l4_fast_matches_materialized_l4_for_projector`.
    The equality holds because W acts only on the external j-legs and the
    denominator is a pure (u,v)/(r,s) weight, so projection commutes with the
    intermediate-state sum.

    Applicability: a large memory/compute win when n_k << n_j (the production
    case, e.g. n_k=2 lowest Kramers doublet vs n_j=2J+1); roughly neutral, or
    slightly slower, when n_k ~= n_j (W near-square), since the projection is
    then applied per intermediate-state pair rather than once on the aggregate.
    """
    M_A = l2_result["M_A"]  # (n_u, n_v, n_j, n_j)
    M_B = l2_result["M_B"]  # (n_r, n_s, n_j, n_j)
    n_j = l2_result["n_j"]

    if W.shape[0] != n_j:
        raise BindError(
            "FXE-BIND-003",
            f"W.shape[0]={W.shape[0]} != n_j={n_j}",
            actual={"W_shape": list(W.shape), "n_j": n_j},
        )

    n_k = W.shape[1]
    logger.info("L4 fast: fused denominator summation and W projection, n_j=%d -> n_k=%d", n_j, n_k)

    n_u = M_A.shape[0]
    n_v = M_A.shape[1]
    denom_A = _denominator_vector(
        E_u_np1,
        E_u_nm1,
        left_label="u",
        right_label="v",
        delta_label="Delta_uv",
        expected_left=n_u,
        expected_right=n_v,
    )

    n_r = M_B.shape[0]
    n_s = M_B.shape[1]
    denom_B = _denominator_vector(
        E_u_nm1,
        E_u_np1,
        left_label="r",
        right_label="s",
        delta_label="Delta_rs",
        expected_left=n_r,
        expected_right=n_s,
    )

    # Project ket-side route amplitudes once: M_K[m,a,b] = M[m,j1,j2] W[j1,a] W[j2,b].
    M_A_k = np.einsum("uvij,ia,jb->uvab", M_A, W, W, optimize=True)
    M_B_k = np.einsum("rsij,ia,jb->rsab", M_B, W, W, optimize=True)

    YA = M_A_k.reshape(n_u * n_v, n_k * n_k)
    YB = M_B_k.reshape(n_r * n_s, n_k * n_k)

    hA = YA.conj().T @ ((1.0 / denom_A)[:, None] * YA)
    hB = YB.conj().T @ ((1.0 / denom_B)[:, None] * YB)

    h_mu = (hA + hB).reshape(n_k, n_k, n_k, n_k)
    Heff = h_mu.copy()

    Heff_flat = Heff.reshape(n_k * n_k, n_k * n_k)
    check_hermitian(Heff_flat, label="Heff", module="contraction")

    logger.info("L4 fast complete: Heff shape=%s", Heff.shape)

    return {
        "h_mu_abcd": h_mu,
        "Heff_mu_abcd": Heff,
        "n_k": n_k,
    }


# ---------------------------------------------------------------------------
# Level L4: W projection (04-02 §3)
#
# Reference (materialized) implementation. Off the runtime path: it consumes
# the materialized `h_pre_j_mu` from `build_L3`. Kept as the equivalence oracle
# for `build_L4_fast` (the runtime path) and for explicit L3-then-L4 inspection.
# ---------------------------------------------------------------------------

def build_L4(
    l3_result: dict[str, Any],
    W: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Apply W projection to get final Heff (04-02 §3).

    Reference (materialized) L4. The runtime L3->L4 path is `build_L4_fast`;
    this path is the oracle that pins it (see `build_L4_fast`).

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
