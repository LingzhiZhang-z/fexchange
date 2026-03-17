"""
Post-SOPT spin-1/2 mapping: Pauli decomposition and exchange extraction.

Spec reference: 04-03-SPIN12_MAPPING.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.constants import SIGMA_0, SIGMA_X, SIGMA_Y, SIGMA_Z
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_MAP, EPS_ZERO
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import NumError

logger = logging.getLogger("fexchange")

_SIGMA = {"0": SIGMA_0, "x": SIGMA_X, "y": SIGMA_Y, "z": SIGMA_Z}
_SIGMA_LIST = [SIGMA_0, SIGMA_X, SIGMA_Y, SIGMA_Z]
_LABELS = ["0", "x", "y", "z"]


def spin12_map(
    Heff: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """
    Map 4x4 Heff to spin-1/2 exchange model (04-03).

    Parameters
    ----------
    Heff : effective Hamiltonian matrix, shape (2,2,2,2) or (4,4).

    Returns dict with J_mu (3x3) and mapping_residual.
    """
    if Heff.shape == (2, 2, 2, 2):
        Heff_flat = Heff.reshape(4, 4)
    elif Heff.shape == (4, 4):
        Heff_flat = Heff.copy()
    else:
        raise NumError(
            "FXE-NUM-001",
            f"Heff shape {Heff.shape} not (2,2,2,2) or (4,4)",
            actual={"shape": list(Heff.shape)},
        )

    check_hermitian(Heff_flat, label="Heff_input", module="spin12")

    # Pauli decomposition (04-03 §2)
    C = np.zeros((4, 4), dtype=DTYPE_COMPLEX)
    for eta_idx, eta in enumerate(_LABELS):
        for nu_idx, nu in enumerate(_LABELS):
            basis_op = np.kron(_SIGMA_LIST[eta_idx], _SIGMA_LIST[nu_idx])
            C[eta_idx, nu_idx] = 0.25 * np.trace(basis_op @ Heff_flat)

    # Exchange matrix J_ab = 4 * C_{ab} for a,b in {x,y,z} (04-03 §3)
    J_mu = np.zeros((3, 3), dtype=DTYPE_COMPLEX)
    for a in range(3):
        for b in range(3):
            J_mu[a, b] = 4 * C[a + 1, b + 1]

    # Take real part (should be real for physical couplings)
    J_real = J_mu.real

    # Reconstruction check (04-03 §5)
    H_reconstructed = np.zeros((4, 4), dtype=DTYPE_COMPLEX)
    for eta_idx in range(4):
        for nu_idx in range(4):
            H_reconstructed += C[eta_idx, nu_idx] * np.kron(
                _SIGMA_LIST[eta_idx], _SIGMA_LIST[nu_idx]
            )

    norm_H = np.linalg.norm(Heff_flat, "fro")
    residual = np.linalg.norm(H_reconstructed - Heff_flat, "fro") / max(norm_H, EPS_ZERO)

    if residual > EPS_MAP:
        raise NumError(
            "FXE-NUM-001",
            f"Spin-1/2 reconstruction residual {residual:.3e} > eps_map={EPS_MAP}",
            actual={"residual_name": "r_spin12", "residual_value": float(residual), "threshold": EPS_MAP},
        )

    logger.info("Spin-1/2 mapping: residual=%.2e", residual)

    return {
        "J_mu": J_real,
        "mapping_residual": float(residual),
    }
