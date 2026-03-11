"""
Runtime validation checks: Hermiticity, orthonormality, unitarity.

Spec reference: 00-02 §6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.numerics import (
    EPS_HERM,
    EPS_ORTH,
    EPS_ZERO,
    EPS_UNITARY,
)
from fexchange.utils.errors import NumError


def check_hermitian(
    H: NDArray[np.complexfloating],
    *,
    label: str = "H",
    eps: float = EPS_HERM,
    module: str = "",
    level: str = "",
) -> float:
    """
    Verify that *H* is Hermitian using normalised Frobenius residual (00-02 §6).

    Returns the residual.  Raises NumError on failure.
    """
    diff = H - H.conj().T
    norm_H = np.linalg.norm(H, "fro")
    if norm_H < EPS_ZERO:
        return 0.0
    r_herm = np.linalg.norm(diff, "fro") / max(norm_H, EPS_ZERO)
    if r_herm > eps:
        raise NumError(
            "FXE-NUM-001",
            f"Hermiticity check failed for {label}: r_herm={r_herm:.3e} > {eps:.1e}",
            module=module,
            level=level,
            actual={"residual_name": "r_herm", "residual_value": float(r_herm), "threshold": eps},
        )
    return float(r_herm)


def check_orthonormal(
    V: NDArray[np.complexfloating],
    *,
    label: str = "V",
    eps: float = EPS_ORTH,
    module: str = "",
    level: str = "",
) -> float:
    """
    Verify V^dag V = I using Frobenius residual (00-02 §6).

    Returns the residual.  Raises NumError on failure.
    """
    n_cols = V.shape[1]
    gram = V.conj().T @ V
    r_orth = np.linalg.norm(gram - np.eye(n_cols), "fro")
    if r_orth > eps:
        raise NumError(
            "FXE-NUM-001",
            f"Orthonormality check failed for {label}: r_orth={r_orth:.3e} > {eps:.1e}",
            module=module,
            level=level,
            actual={"residual_name": "r_orth", "residual_value": float(r_orth), "threshold": eps},
        )
    return float(r_orth)


def check_unitary(
    U: NDArray[np.complexfloating],
    *,
    label: str = "U",
    eps: float = EPS_UNITARY,
    module: str = "",
    level: str = "",
) -> float:
    """
    Verify U^dag U = I (unitarity) using Frobenius residual.

    Returns the residual.  Raises NumError on failure.
    """
    n = U.shape[1]
    gram = U.conj().T @ U
    r = np.linalg.norm(gram - np.eye(n), "fro")
    if r > eps:
        raise NumError(
            "FXE-NUM-001",
            f"Unitarity check failed for {label}: residual={r:.3e} > {eps:.1e}",
            module=module,
            level=level,
            actual={"residual_name": "r_unitary", "residual_value": float(r), "threshold": eps},
        )
    return float(r)
