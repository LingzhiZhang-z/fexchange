"""
Single-particle LS-space angular matrices.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.errors import PhysError
from fexchange.utils.constants import ELL, N_ORB
from fexchange.utils.numerics import DTYPE_COMPLEX
from fexchange.utils.checks import check_hermitian

logger = logging.getLogger("fexchange")


def orbital_map(p: int) -> tuple[int, float]:
    """
    Map spin-orbital index p -> (m, sigma).

    Convention (01-00 §2): m = -3..3, sigma order = [-1/2, +1/2] per m.
    p = 2*(m + ell) + s_idx where s_idx=0 -> sigma=-1/2, s_idx=1 -> sigma=+1/2.
    """
    if not 0 <= p < N_ORB:
        raise PhysError(
            "FXE-PHYS-001",
            f"Orbital index p={p} out of range [0, {N_ORB})",
            actual={"p": p},
        )
    m_block = p // 2
    s_idx = p % 2
    m = m_block - ELL
    sigma = -0.5 + s_idx
    return m, sigma


def orbital_index(m: int, sigma: float) -> int:
    """Inverse of orbital_map: (m, sigma) -> p."""
    if not (-ELL <= m <= ELL):
        raise PhysError(
            "FXE-PHYS-001", f"m={m} out of range [-{ELL}, {ELL}]", actual={"m": m}
        )
    s_idx = 0 if sigma < 0 else 1
    return 2 * (m + ELL) + s_idx


def _build_lz_operator() -> NDArray[np.complexfloating]:
    h = np.zeros((N_ORB, N_ORB), dtype=DTYPE_COMPLEX)
    for p in range(N_ORB):
        m_p, _ = orbital_map(p)
        h[p, p] = m_p
    return h


def _build_sz_operator() -> NDArray[np.complexfloating]:
    h = np.zeros((N_ORB, N_ORB), dtype=DTYPE_COMPLEX)
    for p in range(N_ORB):
        _, sigma_p = orbital_map(p)
        h[p, p] = sigma_p
    return h


def _build_lp_operator() -> NDArray[np.complexfloating]:
    h = np.zeros((N_ORB, N_ORB), dtype=DTYPE_COMPLEX)
    ell = ELL
    for m in range(-ell, ell):
        coeff = np.sqrt(ell * (ell + 1) - m * (m + 1))
        for s_idx in range(2):
            sigma = -0.5 + s_idx
            p_from = orbital_index(m, sigma)
            p_to = orbital_index(m + 1, sigma)
            h[p_to, p_from] = coeff
    return h


def _build_sp_operator() -> NDArray[np.complexfloating]:
    h = np.zeros((N_ORB, N_ORB), dtype=DTYPE_COMPLEX)
    for m in range(-ELL, ELL + 1):
        p_from = orbital_index(m, -0.5)
        p_to = orbital_index(m, +0.5)
        h[p_to, p_from] = 1.0
    return h


def build_space_ls_operator() -> dict[str, NDArray[np.complexfloating]]:
    """
    Build LS/J operators in the local spin-orbital space.

    Returned keys follow existing convention:
    Lz, Sz, Lp, Lm, Sp, Sm, Lx, Ly, Sx, Sy, Jz, Jp, Jm, Jx, Jy.
    """
    Lz = _build_lz_operator()
    Sz = _build_sz_operator()
    Lp = _build_lp_operator()
    Lm = Lp.conj().T
    Sp = _build_sp_operator()
    Sm = Sp.conj().T

    Lx = 0.5 * (Lp + Lm)
    Ly = -0.5j * (Lp - Lm)
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    Jz = Lz + Sz
    Jp = Lp + Sp
    Jm = Lm + Sm
    Jx = Lx + Sx
    Jy = Ly + Sy

    return {
        "Lz": Lz,
        "Sz": Sz,
        "Lp": Lp,
        "Lm": Lm,
        "Sp": Sp,
        "Sm": Sm,
        "Lx": Lx,
        "Ly": Ly,
        "Sx": Sx,
        "Sy": Sy,
        "Jz": Jz,
        "Jp": Jp,
        "Jm": Jm,
        "Jx": Jx,
        "Jy": Jy,
    }


def build_space_ls_matrix(
    n_ele: int,
    n_orb: int = N_ORB,
) -> dict[str, NDArray[np.complexfloating]]:
    """
    Build many-body LS/J matrices in Fock space for the sector n_ele.

    Returns keys:
    Lz, Sz, Lp, Lm, Sp, Sm, Lx, Ly, Sx, Sy, Jz, Jp, Jm, Jx, Jy, L2, S2, J2.
    """
    # Lazy import keeps this module importable in environments without scipy.
    from fexchange.core.fermion import one_body_operator_matrix

    one = build_space_ls_operator()
    out: dict[str, NDArray[np.complexfloating]] = {}

    for name, h_pq in one.items():
        out[name] = one_body_operator_matrix(h_pq, n_ele, n_orb)

    out["L2"] = out["Lx"] @ out["Lx"] + out["Ly"] @ out["Ly"] + out["Lz"] @ out["Lz"]
    out["S2"] = out["Sx"] @ out["Sx"] + out["Sy"] @ out["Sy"] + out["Sz"] @ out["Sz"]
    out["J2"] = out["Jx"] @ out["Jx"] + out["Jy"] @ out["Jy"] + out["Jz"] @ out["Jz"]

    for name in ("L2", "S2", "J2", "Lz", "Sz", "Jz"):
        check_hermitian(out[name], label=name, module="space_ls")

    for plus_name, minus_name in [("Lp", "Lm"), ("Sp", "Sm"), ("Jp", "Jm")]:
        residual = np.linalg.norm(out[minus_name] - out[plus_name].conj().T, "fro")
        if residual > 1e-12:
            logger.warning("%s != %s^dag, residual=%.3e", minus_name, plus_name, residual)

    return out
