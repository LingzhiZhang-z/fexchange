"""
Fermionic creation / annihilation operators and operator algebra.

Spec reference: 01-02-OPERATOR_IMPLEMENTATION.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from fexchange.core.fock_basis import (
    N_ORB,
    occ,
    count_below,
    set_bit,
    clear_bit,
    enumerate_dets,
    det_index,
    dim_sector,
)
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_ZERO
from fexchange.utils.errors import PhysError

logger = logging.getLogger("fexchange")


# ---------------------------------------------------------------------------
# Primitive action on determinants (01-02 §4, 01-00 §7)
# ---------------------------------------------------------------------------

def apply_cdag(det: int, p: int) -> tuple[int, int] | None:
    """
    Apply c^dag_p to |det>.

    Returns (sign, det_new) or None if orbital p is occupied.
    """
    if occ(det, p):
        return None
    sign = (-1) ** count_below(det, p)
    return sign, set_bit(det, p)


def apply_c(det: int, p: int) -> tuple[int, int] | None:
    """
    Apply c_p to |det>.

    Returns (sign, det_new) or None if orbital p is empty.
    """
    if not occ(det, p):
        return None
    sign = (-1) ** count_below(det, p)
    return sign, clear_bit(det, p)


def apply_cc(det: int, r: int, s: int) -> tuple[int, int] | None:
    """
    Apply c_r c_s to |det> (right-to-left: c_s acts first, then c_r).

    No ordering constraint on r, s.
    Returns (sign, det_new) or None if the action is forbidden (Pauli).
    """
    res_s = apply_c(det, s)
    if res_s is None:
        return None
    sign_s, det1 = res_s
    res_r = apply_c(det1, r)
    if res_r is None:
        return None
    sign_r, det2 = res_r
    return sign_s * sign_r, det2


def apply_cdagcdag(det: int, p: int, q: int) -> tuple[int, int] | None:
    """
    Apply c^dag_p c^dag_q to |det> (right-to-left: c^dag_q acts first, then c^dag_p).

    No ordering constraint on p, q.
    Returns (sign, det_new) or None if the action is forbidden (Pauli).
    """
    res_q = apply_cdag(det, q)
    if res_q is None:
        return None
    sign_q, det1 = res_q
    res_p = apply_cdag(det1, p)
    if res_p is None:
        return None
    sign_p, det2 = res_p
    return sign_q * sign_p, det2


# ---------------------------------------------------------------------------
# Matrix representations of c^dag_p and c_p  (01-02 §4-5)
# ---------------------------------------------------------------------------

def cdag_matrix(
    p: int,
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build the matrix representation of c^dag_p: sector n -> sector n+1.

    Shape: (dim(n+1), dim(n)).
    """
    if not 0 <= p < n_orb:
        raise PhysError("FXE-PHYS-001", f"Orbital p={p} out of range", actual={"p": p})
    dets_from = enumerate_dets(n_ele, n_orb)
    dets_to = enumerate_dets(n_ele + 1, n_orb)
    d_from = len(dets_from)
    d_to = len(dets_to)
    mat = np.zeros((d_to, d_from), dtype=DTYPE_COMPLEX)

    for j, det in enumerate(dets_from):
        result = apply_cdag(int(det), p)
        if result is not None:
            sign, det_new = result
            i = det_index(det_new, dets_to)
            mat[i, j] = sign
    return mat


def c_matrix(
    p: int,
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build the matrix representation of c_p: sector n -> sector n-1.

    Shape: (dim(n-1), dim(n)).
    """
    if not 0 <= p < n_orb:
        raise PhysError("FXE-PHYS-001", f"Orbital p={p} out of range", actual={"p": p})
    dets_from = enumerate_dets(n_ele, n_orb)
    dets_to = enumerate_dets(n_ele - 1, n_orb)
    d_from = len(dets_from)
    d_to = len(dets_to)
    mat = np.zeros((d_to, d_from), dtype=DTYPE_COMPLEX)

    for j, det in enumerate(dets_from):
        result = apply_c(int(det), p)
        if result is not None:
            sign, det_new = result
            i = det_index(det_new, dets_to)
            mat[i, j] = sign
    return mat


# ---------------------------------------------------------------------------
# One-body operator in fixed sector (01-02 §4.1)
# ---------------------------------------------------------------------------

def one_body_operator_matrix(
    h_pq: NDArray[np.complexfloating],
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build many-body matrix for one-body operator sum_{p,q} h[p,q] c^dag_p c_q.

    Operates within sector *n_ele* (number-conserving).
    Shape: (dim(n_ele), dim(n_ele)).
    """
    if h_pq.shape != (n_orb, n_orb):
        raise PhysError(
            "FXE-PHYS-001",
            f"h_pq shape {h_pq.shape} != ({n_orb},{n_orb})",
            actual={"shape": list(h_pq.shape)},
        )
    dets = enumerate_dets(n_ele, n_orb)
    dim = len(dets)
    H = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)

    nz_idx = np.argwhere(np.abs(h_pq) > EPS_ZERO)
    if len(nz_idx) == 0:
        return H

    det_to_idx = {int(det): i for i, det in enumerate(dets)}

    for j_col, det_in in enumerate(dets):
        det_in = int(det_in)
        for p_raw, q_raw in nz_idx:
            p, q = int(p_raw), int(q_raw)

            res_ann = apply_c(det_in, q)
            if res_ann is None:
                continue
            sign_ann, det_mid = res_ann

            res_cre = apply_cdag(det_mid, p)
            if res_cre is None:
                continue
            sign_cre, det_out = res_cre

            i_row = det_to_idx.get(det_out)
            if i_row is None:
                continue
            H[i_row, j_col] += complex(h_pq[p, q]) * sign_ann * sign_cre
    return H


# ---------------------------------------------------------------------------
# Two-body operator in fixed sector (01-02 §3.1, 02-01)
# ---------------------------------------------------------------------------

def two_body_operator_matrix(
    V_pqrs: NDArray[np.complexfloating],
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build many-body matrix for two-body operator:
        H = (1/2) * sum_{p,q,r,s} V[p,q,r,s] c^dag_p c^dag_q c_r c_s

    where the antisymmetrised coefficient tensor V[p,q,r,s] is provided.
    Operates within sector *n_ele* (number-conserving).
    """
    if V_pqrs.shape != (n_orb, n_orb, n_orb, n_orb):
        raise PhysError(
            "FXE-PHYS-001",
            f"V_pqrs shape {V_pqrs.shape} != expected",
            actual={"shape": list(V_pqrs.shape)},
        )
    dets = enumerate_dets(n_ele, n_orb)
    dim = len(dets)
    H = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)

    nz_idx = np.argwhere(np.abs(V_pqrs) > EPS_ZERO)
    if len(nz_idx) == 0:
        return H

    det_to_idx = {int(det): i for i, det in enumerate(dets)}

    for j_col, det_in in enumerate(dets):
        det_in = int(det_in)
        for p_raw, q_raw, r_raw, s_raw in nz_idx:
            p, q, r, s = int(p_raw), int(q_raw), int(r_raw), int(s_raw)

            res_ann = apply_cc(det_in, r, s)
            if res_ann is None:
                continue
            sign_ann, det_mid = res_ann

            res_cre = apply_cdagcdag(det_mid, p, q)
            if res_cre is None:
                continue
            sign_cre, det_out = res_cre

            i_row = det_to_idx.get(det_out)
            if i_row is None:
                continue
            H[i_row, j_col] += 0.5 * complex(V_pqrs[p, q, r, s]) * sign_ann * sign_cre

    return H
