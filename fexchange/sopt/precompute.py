"""
SOPT precompute: Level L0 (Fock transition tensors) and Level L1 (rotated vertices).

Spec reference: 04-01-PRECOMPUTE_PIPELINE.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.constants import N_ORB
from fexchange.core.fock import enumerate_dets, dim_sector, make_basis_id
from fexchange.core.fermion import cdag_matrix
from fexchange.utils.numerics import DTYPE_COMPLEX
from fexchange.utils.errors import PhysError, BindError

logger = logging.getLogger("fexchange")


# ---------------------------------------------------------------------------
# Level L0: Fock-basis primitive transitions (04-01 §1)
# ---------------------------------------------------------------------------

def build_L0(
    n_ele: int,
    n_orb: int = N_ORB,
) -> dict[str, Any]:
    """
    Build Fock-basis transition tensors X and Y (04-01 §1).

    X^{kappa,n}_{alpha,beta} = <alpha^{n+1} | f_kappa^dag | beta^n>
    Y^{kappa,n-1}_{beta,gamma} = <beta^n | f_kappa^dag | gamma^{n-1}>

    Returns dict with X, Y arrays and metadata.
    X shape: (n_orb, dim(n+1), dim(n))
    Y shape: (n_orb, dim(n), dim(n-1))
    """
    if n_ele < 1 or n_ele > n_orb - 1:
        raise PhysError(
            "FXE-PHYS-001",
            f"L0 requires 1 <= n_ele <= {n_orb-1}, got {n_ele}",
            actual={"n_ele": n_ele},
        )

    logger.info("L0: Building transition tensors for n=%d", n_ele)

    dim_n = dim_sector(n_ele, n_orb)
    dim_np1 = dim_sector(n_ele + 1, n_orb)
    dim_nm1 = dim_sector(n_ele - 1, n_orb)

    # X^{kappa,n}: f^dag from sector n to sector n+1
    X = np.zeros((n_orb, dim_np1, dim_n), dtype=DTYPE_COMPLEX)
    for kappa in range(n_orb):
        X[kappa] = cdag_matrix(kappa, n_ele, n_orb)

    # Y^{kappa,n-1}: f^dag from sector n-1 to sector n
    Y = np.zeros((n_orb, dim_n, dim_nm1), dtype=DTYPE_COMPLEX)
    for kappa in range(n_orb):
        Y[kappa] = cdag_matrix(kappa, n_ele - 1, n_orb)

    logger.info(
        "L0 complete: X shape=%s, Y shape=%s", X.shape, Y.shape
    )

    return {
        "X": X,
        "Y": Y,
        "n_ele": n_ele,
        "n_orb": n_orb,
        "basis_id_n": make_basis_id(n_ele, n_orb),
        "basis_id_np1": make_basis_id(n_ele + 1, n_orb),
        "basis_id_nm1": make_basis_id(n_ele - 1, n_orb),
    }


# ---------------------------------------------------------------------------
# Level L1: Local transition vertices (04-01 §2)
# ---------------------------------------------------------------------------

def build_L1(
    l0_result: dict[str, Any],
    U_np1: NDArray[np.complexfloating],
    U_n_soc0: NDArray[np.complexfloating],
    U_nm1: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Build LSJM-rotated transition vertices A and B (04-01 §2).

    A^{kappa,n}_{u,j} = sum_{alpha,beta} conj(U_np1[alpha,u]) X[kappa,alpha,beta] U_n_soc0[beta,j]
    B^{kappa,n-1}_{j,v} = sum_{beta,gamma} conj(U_n_soc0[beta,j]) Y[kappa,beta,gamma] U_nm1[gamma,v]

    Parameters
    ----------
    l0_result : output from build_L0.
    U_np1 : Fock->LSJM transform for n+1 sector. Shape (dim_np1, n_states_np1).
    U_n_soc0 : Fock->SOC-lowest-subspace for n sector. Shape (dim_n, n_j).
    U_nm1 : Fock->LSJM transform for n-1 sector. Shape (dim_nm1, n_states_nm1).

    Returns dict with A, B vertex tensors.
    A shape: (n_orb, n_states_np1, n_j)
    B shape: (n_orb, n_j, n_states_nm1)
    """
    X = l0_result["X"]
    Y = l0_result["Y"]
    n_orb = l0_result["n_orb"]

    logger.info("L1: Building transition vertices")

    n_u = U_np1.shape[1]
    n_j = U_n_soc0.shape[1]
    n_v = U_nm1.shape[1]

    # A^{kappa}_{u,j} = U_np1^dag @ X[kappa] @ U_n_soc0  (04-01 §2)
    A = np.zeros((n_orb, n_u, n_j), dtype=DTYPE_COMPLEX)
    for kappa in range(n_orb):
        A[kappa] = U_np1.conj().T @ X[kappa] @ U_n_soc0

    # B^{kappa}_{j,v} = U_n_soc0^dag @ Y[kappa] @ U_nm1
    B = np.zeros((n_orb, n_j, n_v), dtype=DTYPE_COMPLEX)
    for kappa in range(n_orb):
        B[kappa] = U_n_soc0.conj().T @ Y[kappa] @ U_nm1

    logger.info("L1 complete: A shape=%s, B shape=%s", A.shape, B.shape)

    return {
        "A": A,
        "B": B,
        "n_orb": n_orb,
        "n_u": n_u,
        "n_j": n_j,
        "n_v": n_v,
        "n_ele": l0_result["n_ele"],
    }
