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

    Returns dict with J_mu (3x3), J_iso, K, D (3,), Gamma (3,3),
    const, h_i (3,), h_j (3,), residual.
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

    # Derived decomposition (04-03 §3.2)
    J_iso = (J_real[0, 0] + J_real[1, 1] + J_real[2, 2]) / 3.0
    K = J_real[2, 2] - J_iso

    D = np.array([
        0.5 * (J_real[1, 2] - J_real[2, 1]),
        0.5 * (J_real[2, 0] - J_real[0, 2]),
        0.5 * (J_real[0, 1] - J_real[1, 0]),
    ])

    J_sym = 0.5 * (J_real + J_real.T)
    J_diag = np.diag(np.diag(J_real))
    Gamma = J_sym - J_diag

    # Non-exchange terms (04-03 §3.3)
    const = C[0, 0].real
    h_i = np.array([2 * C[a + 1, 0].real for a in range(3)])
    h_j = np.array([2 * C[0, b + 1].real for b in range(3)])

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

    logger.info("Spin-1/2 mapping: J_iso=%.4f, K=%.4f, |D|=%.4f, residual=%.2e",
                J_iso, K, np.linalg.norm(D), residual)

    return {
        "J_mu": J_real,
        "J_iso": float(J_iso),
        "K": float(K),
        "D": D,
        "Gamma": Gamma,
        "const": float(const),
        "h_i": h_i,
        "h_j": h_j,
        "residual": float(residual),
    }
