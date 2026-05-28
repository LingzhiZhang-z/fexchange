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
        Heff_flat = np.asarray(Heff, dtype=DTYPE_COMPLEX).reshape(4, 4)
    elif Heff.shape == (4, 4):
        Heff_flat = np.asarray(Heff, dtype=DTYPE_COMPLEX).copy()
    else:
        raise NumError(
            "FXE-NUM-001",
            f"Heff shape {Heff.shape} not (2,2,2,2) or (4,4)",
            actual={"shape": list(Heff.shape)},
        )
    if not np.all(np.isfinite(Heff_flat)):
        raise NumError(
            "FXE-NUM-001",
            "Heff_input contains non-finite values",
            module="spin12",
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

    imag_leakage = np.linalg.norm(J_mu.imag, "fro") / max(np.linalg.norm(J_mu, "fro"), EPS_ZERO)
    if imag_leakage > EPS_MAP:
        raise NumError(
            "FXE-NUM-001",
            f"Spin-1/2 exchange imaginary leakage {imag_leakage:.3e} > eps_map={EPS_MAP}",
            module="spin12",
            actual={
                "residual_name": "r_spin12_imag",
                "residual_value": float(imag_leakage),
                "threshold": EPS_MAP,
            },
        )

    J_real = J_mu.real

    # Reconstruction diagnostic (04-03 §5): scalar shifts are retained for the
    # residual, while local fields and other non-exchange terms are hard failures.
    H_reconstructed = C[0, 0].real * np.kron(SIGMA_0, SIGMA_0)
    for a in range(3):
        for b in range(3):
            H_reconstructed += (J_real[a, b] / 4.0) * np.kron(
                _SIGMA_LIST[a + 1], _SIGMA_LIST[b + 1]
            )

    norm_H = np.linalg.norm(Heff_flat, "fro")
    residual = np.linalg.norm(H_reconstructed - Heff_flat, "fro") / max(norm_H, EPS_ZERO)

    if residual > EPS_MAP:
        raise NumError(
            "FXE-NUM-001",
            f"Spin-1/2 exchange residual={residual:.3e} > eps_map={EPS_MAP:.2e}",
            module="spin12",
            actual={
                "residual_name": "r_spin12_map",
                "residual_value": float(residual),
                "threshold": EPS_MAP,
            },
        )
    logger.info("Spin-1/2 mapping: residual=%.2e", residual)

    return {
        "J_mu": J_real,
        "mapping_residual": float(residual),
    }
