"""fopt L0/L1: f-shell + p-shell Fock primitives and rotated vertices."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.constants import N_ORB
from fexchange.core.fermion import cdag_matrix
from fexchange.core.fock import make_basis_id
from fexchange.utils.numerics import DTYPE_COMPLEX
from fexchange.sopt.precompute import build_L0 as _sopt_build_L0
from fexchange.sopt.precompute import build_L1 as _sopt_build_L1

logger = logging.getLogger("fexchange")

N_ORB_F: int = N_ORB
N_ORB_P: int = 6


def compute_L0_f(n_ele: int, n_orb: int = N_ORB_F) -> dict[str, Any]:
    """f-shell Fock primitives. Wraps sopt.precompute.build_L0."""
    return _sopt_build_L0(n_ele, n_orb)


def compute_L0_p() -> dict[str, Any]:
    """p-shell raw Fock-basis creation tensors at fixed sectors {4, 5, 6}.

    P_raw_5_6[kappa, 1, 6]  : creation 5 -> 6.
    P_raw_4_5[kappa, 6, 15] : creation 4 -> 5.
    """
    P_raw_5_6 = np.zeros((N_ORB_P, 1, 6), dtype=DTYPE_COMPLEX)
    P_raw_4_5 = np.zeros((N_ORB_P, 6, 15), dtype=DTYPE_COMPLEX)
    for kappa in range(N_ORB_P):
        P_raw_5_6[kappa] = cdag_matrix(kappa, 5, N_ORB_P)
        P_raw_4_5[kappa] = cdag_matrix(kappa, 4, N_ORB_P)

    return {
        "P_raw_5_6": P_raw_5_6,
        "P_raw_4_5": P_raw_4_5,
        "n_orb_p": N_ORB_P,
        "basis_id_p_6": make_basis_id(6, N_ORB_P),
        "basis_id_p_5": make_basis_id(5, N_ORB_P),
        "basis_id_p_4": make_basis_id(4, N_ORB_P),
    }


def compute_L1_f(
    l0_f: dict[str, Any],
    U_np1: NDArray[np.complexfloating],
    U_n_soc0: NDArray[np.complexfloating],
    U_nm1: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """f-shell LSJM-rotated vertices. Wraps sopt build_L1 and renames A/B for fopt clarity.

    F_n_np1 : f-creation n -> n+1   (replaces sopt's "A")
    F_nm1_n : f-creation n-1 -> n   (replaces sopt's "B")
    """
    out = _sopt_build_L1(l0_f, U_np1, U_n_soc0, U_nm1)
    return {
        "F_n_np1": out["A"],
        "F_nm1_n": out["B"],
        "n_orb_f": out.get("n_orb"),
        "n_u":     out.get("n_u"),
        "n_j":     out.get("n_j"),
        "n_v":     out.get("n_v"),
        "n_ele":   out.get("n_ele"),
    }


def compute_L1_p(
    l0_p: dict[str, Any],
    U_p_6: NDArray[np.complexfloating] | None = None,
    U_p_5: NDArray[np.complexfloating] | None = None,
    U_p_4: NDArray[np.complexfloating] | None = None,
) -> dict[str, Any]:
    """p-shell rotated creation tensors at fixed sectors {4, 5, 6}.

    P_5_6[kappa, n_u_6, n_j_5] = U_p_6^dag @ P_raw_5_6[kappa] @ U_p_5
    P_4_5[kappa, n_j_5, n_v_4] = U_p_5^dag @ P_raw_4_5[kappa] @ U_p_4

    Default U_p_* = identity (no SOC/CEF on ligand p).
    """
    P_raw_5_6 = l0_p["P_raw_5_6"]
    P_raw_4_5 = l0_p["P_raw_4_5"]
    n_orb_p = l0_p["n_orb_p"]
    dim_6, dim_5 = P_raw_5_6.shape[1], P_raw_5_6.shape[2]
    dim_4 = P_raw_4_5.shape[2]

    if U_p_6 is None:
        U_p_6 = np.eye(dim_6, dtype=DTYPE_COMPLEX)
    if U_p_5 is None:
        U_p_5 = np.eye(dim_5, dtype=DTYPE_COMPLEX)
    if U_p_4 is None:
        U_p_4 = np.eye(dim_4, dtype=DTYPE_COMPLEX)

    n_u_6 = U_p_6.shape[1]
    n_j_5 = U_p_5.shape[1]
    n_v_4 = U_p_4.shape[1]

    P_5_6 = np.zeros((n_orb_p, n_u_6, n_j_5), dtype=DTYPE_COMPLEX)
    P_4_5 = np.zeros((n_orb_p, n_j_5, n_v_4), dtype=DTYPE_COMPLEX)
    for kappa in range(n_orb_p):
        P_5_6[kappa] = U_p_6.conj().T @ P_raw_5_6[kappa] @ U_p_5
        P_4_5[kappa] = U_p_5.conj().T @ P_raw_4_5[kappa] @ U_p_4

    return {
        "P_5_6": P_5_6,
        "P_4_5": P_4_5,
        "n_orb_p": n_orb_p,
        "n_u_6": n_u_6,
        "n_j_5": n_j_5,
        "n_v_4": n_v_4,
    }
