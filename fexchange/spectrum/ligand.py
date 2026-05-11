"""Ligand p-shell {4, 5, 6} sector spectrum under optional atomic SOC.

Mirrors `spectrum/lsjm.py` structure: build the single-particle L*S operator
on the p shell, lift to the N-particle Fock sector via
`one_body_operator_matrix`, diagonalise. Three lines for the SOC path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.core.fermion import one_body_operator_matrix
from fexchange.core.fock import dim_sector, make_basis_id
from fexchange.utils.numerics import DTYPE_COMPLEX
from fexchange.utils.errors import PhysError

P_ELL: int = 1
P_N_ORB: int = 6
_SUPPORTED_SECTORS: tuple[int, ...] = (4, 5, 6)


def _build_single_particle_LS() -> NDArray[np.complexfloating]:
    """Single-particle L*S on the p shell (l=1, s=1/2), 6x6.

    Fock orbital ordering: kappa = 2*(m_l + ell) + s_idx, s_idx=0 -> sigma=-1/2.
    """
    Lz = np.zeros((P_N_ORB, P_N_ORB), dtype=DTYPE_COMPLEX)
    Sz = np.zeros((P_N_ORB, P_N_ORB), dtype=DTYPE_COMPLEX)
    Lp = np.zeros((P_N_ORB, P_N_ORB), dtype=DTYPE_COMPLEX)
    Sp = np.zeros((P_N_ORB, P_N_ORB), dtype=DTYPE_COMPLEX)
    for m in range(-P_ELL, P_ELL + 1):
        for s_idx in range(2):
            kappa = 2 * (m + P_ELL) + s_idx
            Lz[kappa, kappa] = m
            Sz[kappa, kappa] = -0.5 + s_idx
            if m < P_ELL:
                Lp[2 * (m + P_ELL + 1) + s_idx, kappa] = float(np.sqrt(P_ELL * (P_ELL + 1) - m * (m + 1)))
            if s_idx == 0:
                Sp[kappa + 1, kappa] = 1.0
    return Lz @ Sz + 0.5 * (Lp @ Sp.conj().T + Lp.conj().T @ Sp)


def build_ligand_spectrum(
    n_ele: int,
    n_orb_p: int = P_N_ORB,
    *,
    soc: bool = True,
) -> dict[str, Any]:
    """Payload for E_p^N = n_hole*Delta + C(n_hole,2)*U_p + lambda_p*coef_lambda_p."""
    if n_orb_p != P_N_ORB:
        raise PhysError(
            "FXE-PHYS-001",
            f"Only ligand p shell (n_orb_p={P_N_ORB}) is supported, got {n_orb_p}",
            actual={"n_orb_p": n_orb_p},
        )
    if n_ele not in _SUPPORTED_SECTORS:
        raise PhysError(
            "FXE-PHYS-001",
            f"build_ligand_spectrum supports n_ele in {_SUPPORTED_SECTORS}, got {n_ele}",
            actual={"n_ele": n_ele},
        )

    n_hole = n_orb_p - n_ele
    dim_N = dim_sector(n_ele, n_orb_p)

    payload: dict[str, Any] = {
        "coef_U_p":   (n_hole * (n_hole - 1) / 2.0) * np.ones(dim_N, dtype=np.float64),
        "coef_Delta": float(n_hole) * np.ones(dim_N, dtype=np.float64),
        "soc":        bool(soc),
        "n_ele":      int(n_ele),
        "n_orb_p":    int(n_orb_p),
        "dim":        int(dim_N),
        "basis_id":   make_basis_id(n_ele, n_orb_p),
    }

    if n_ele == n_orb_p:
        payload["V_fock"]        = np.eye(1, dtype=DTYPE_COMPLEX)
        payload["coef_lambda_p"] = np.zeros(1, dtype=np.float64)
        payload["coef_U_p"]      = np.zeros(1, dtype=np.float64)
        payload["coef_Delta"]    = np.zeros(1, dtype=np.float64)
        return payload

    if not soc:
        payload["V_fock"]        = np.eye(dim_N, dtype=DTYPE_COMPLEX)
        payload["coef_lambda_p"] = np.zeros(dim_N, dtype=np.float64)
        return payload

    H_N = one_body_operator_matrix(_build_single_particle_LS(), n_ele, n_orb=P_N_ORB)
    eigvals, eigvecs = np.linalg.eigh(H_N)
    payload["V_fock"]        = eigvecs.astype(DTYPE_COMPLEX)
    payload["coef_lambda_p"] = eigvals.astype(np.float64)
    return payload
