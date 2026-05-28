"""
SOPT runtime contraction for external levels L2 (projected route factors) and L3
(denominator-weighted final output).

Canonical runtime path: L2 -> ``build_L3``. L2 consumes the low-energy
projector W and stores route factors directly in the projected basis; L3 then
only performs denominator-weighted Gram contractions.

``build_L3_legacy`` and ``build_L4_legacy`` retain the algebraic split used by
the reference implementation. They are not on the runtime path; they serve as
the equivalence oracle that pins ``build_L3`` at 1e-12
(see ``tests/test_contraction.py``).

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
    W: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Build projected route factors M_A and M_B (04-02 §1).

    M_A^R_{uv; a,b} = sum_{p,q,j1,j2} t_mu[p,q] A^{i,p}_{u,j1} W[j1,a]
                      conj(B^{j,q}_{j2,v}) W[j2,b]
    M_B^R_{rs; a,b} = sum_{p,q,j1,j2} conj(t_mu[p,q]) conj(B^{i,p}_{j1,r}) W[j1,a]
                      A^{j,q}_{s,j2} W[j2,b]

    Parameters
    ----------
    l1_result : output from build_L1.
    t_mu : hopping matrix (n_orb, n_orb). t_mu[p,q] = hopping from site-j orbital q to site-i orbital p.
    W : projector from SOC-lowest f^n LSJM subspace to the target local basis.

    Returns dict with M_A, M_B arrays.
    M_A shape: (n_u, n_v, n_k, n_k)
    M_B shape: (n_v, n_u, n_k, n_k)
    """
    A = l1_result["A"]  # (n_orb, n_u, n_j)
    B = l1_result["B"]  # (n_orb, n_j, n_v)
    n_orb = l1_result["n_orb"]
    n_u = l1_result["n_u"]
    n_j = l1_result["n_j"]
    n_v = l1_result["n_v"]
    W_arr = np.asarray(W, dtype=np.complex128)

    if t_mu.shape != (n_orb, n_orb):
        raise BindError(
            "FXE-BIND-003",
            f"t_mu shape {t_mu.shape} != ({n_orb},{n_orb})",
            actual={"lhs_shape": list(t_mu.shape), "rhs_shape": [n_orb, n_orb]},
        )
    if W_arr.ndim != 2 or W_arr.shape[0] != n_j:
        raise BindError(
            "FXE-BIND-003",
            f"W shape {tuple(W_arr.shape)} incompatible with n_j={n_j}",
            expected={"W_shape": [n_j, "n_k"]},
            actual={"W_shape": list(W_arr.shape), "n_j": n_j},
        )

    n_k = W_arr.shape[1]
    logger.info("L2: Building projected route factors, n_j=%d -> n_k=%d", n_j, n_k)

    A_k = np.einsum("puj,ja->pua", A, W_arr, optimize=True)
    Bc_k = np.einsum("pjv,ja->pav", B.conj(), W_arr, optimize=True)

    # Route A: M_A[u,v,a,b] = sum_{p,q} t[p,q] * A_k[p,u,a] * conj(B)_k[q,b,v]
    M_A = np.einsum("pq,pua,qbv->uvab", t_mu, A_k, Bc_k, optimize=True)

    # Route B: M_B[r,s,a,b] = sum_{p,q} conj(t[p,q]) * conj(B)_k[p,a,r] * A_k[q,s,b]
    M_B = np.einsum("pq,par,qsb->rsab", t_mu.conj(), Bc_k, A_k, optimize=True)

    logger.info("L2 complete: M_A shape=%s, M_B shape=%s", M_A.shape, M_B.shape)

    return {
        "M_A": M_A,
        "M_B": M_B,
        "n_u": n_u,
        "n_v": n_v,
        "n_j": n_j,
        "n_k": n_k,
    }


# ---------------------------------------------------------------------------
# Level L3: Denominator summation (04-02 §2)
#
# Reference (materialized) implementation. Off the runtime path: used as the
# equivalence oracle for `build_L3`. The runtime path uses `build_L3` instead.
# ---------------------------------------------------------------------------

def build_L3_legacy(
    l2_result: dict[str, Any],
    E_u_np1: NDArray[np.floating],
    E_u_nm1: NDArray[np.floating],
    n_ele: int,
) -> dict[str, Any]:
    """
    Build h_pre_j_mu by denominator-weighted summation (04-02 §2).

    Reference (materialized) L3. Materializes the (n_j, n_j, n_j, n_j)
    `h_pre_j_mu` tensor for oracle/debug use. The runtime fused final-L3 path
    does not call it (see `build_L3`).

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


def build_L3(
    l2_result: dict[str, Any],
    E_u_np1: NDArray[np.floating],
    E_u_nm1: NDArray[np.floating],
    n_ele: int,
) -> dict[str, Any]:
    """
    Canonical runtime final-L3 path: denominator sum over projected L2 factors.

    This is the implementation the pipeline runs. L2 has already projected the
    external legs to the target Kramers/non-Kramers basis, so L3 only does the
    denominator-weighted Gram contraction in n_k space. The
    (n_j, n_j, n_j, n_j) `h_pre_j_mu` tensor is never materialized.

    Correctness contract: algebraically equal to
    `build_L4_legacy(build_L3_legacy(raw_l2, ...), W)` when `l2_result` is
    built from the same raw L1/hopping/projector inputs.
    (the reference oracle), pinned at 1e-12 by
    `tests/test_contraction.py::test_build_l3_matches_legacy_materialized_projector`.
    The equality holds because W acts only on the external j-legs and the
    denominator is a pure (u,v)/(r,s) weight, so projection commutes with the
    intermediate-state sum.

    Applicability: a large memory/compute win when n_k << n_j (the production
    case, e.g. n_k=2 lowest Kramers doublet vs n_j=2J+1); roughly neutral, or
    slightly slower, when n_k ~= n_j (W near-square), since the projection is
    then applied per intermediate-state pair rather than once on the aggregate.
    """
    M_A = l2_result["M_A"]  # (n_u, n_v, n_k, n_k)
    M_B = l2_result["M_B"]  # (n_r, n_s, n_k, n_k)
    n_k = int(l2_result.get("n_k", M_A.shape[2]))
    if M_A.ndim != 4 or M_B.ndim != 4 or M_A.shape[2:] != (n_k, n_k) or M_B.shape[2:] != (n_k, n_k):
        raise BindError(
            "FXE-BIND-003",
            "L3 requires projected L2 tensors with trailing axes (n_k,n_k)",
            actual={"M_A_shape": list(M_A.shape), "M_B_shape": list(M_B.shape), "n_k": n_k},
        )
    logger.info("L3: denominator summation over projected route factors, n_k=%d", n_k)

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

    YA = M_A.reshape(n_u * n_v, n_k * n_k)
    YB = M_B.reshape(n_r * n_s, n_k * n_k)

    hA = YA.conj().T @ ((1.0 / denom_A)[:, None] * YA)
    hB = YB.conj().T @ ((1.0 / denom_B)[:, None] * YB)

    h_mu = (hA + hB).reshape(n_k, n_k, n_k, n_k)
    Heff = h_mu.copy()

    Heff_flat = Heff.reshape(n_k * n_k, n_k * n_k)
    check_hermitian(Heff_flat, label="Heff", module="contraction")

    logger.info("L3 complete: Heff shape=%s", Heff.shape)

    return {
        "h_mu_abcd": h_mu,
        "Heff_mu_abcd": Heff,
        "n_k": n_k,
    }


# ---------------------------------------------------------------------------
# Reference W projection (04-02 §3)
#
# Reference (materialized) implementation. Off the runtime path: it consumes
# the materialized `h_pre_j_mu` from `build_L3_legacy`. Kept as the equivalence
# oracle for `build_L3` (the runtime path).
# ---------------------------------------------------------------------------

def build_L4_legacy(
    l3_result: dict[str, Any],
    W: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Apply W projection to get final Heff (04-02 §3).

    Reference materialized projection. The runtime final-L3 path is `build_L3`;
    this path is the oracle that pins it.

    h_pre[c,d,a,b] = sum_{j3,j4,j1,j2} conj(W[j3,c]) conj(W[j4,d]) h_pre_j[j3,j4,j1,j2] W[j1,a] W[j2,b]

    Parameters
    ----------
    l3_result : output from build_L3_legacy.
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
    logger.info("reference W projection, n_j=%d -> n_k=%d", n_j, n_k)

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

    logger.info("reference W projection complete: Heff shape=%s", Heff.shape)

    return {
        "h_mu_abcd": h_mu,
        "Heff_mu_abcd": Heff,
        "n_k": n_k,
    }
