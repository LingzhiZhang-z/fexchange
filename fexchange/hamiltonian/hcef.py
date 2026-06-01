"""
Crystal electric field term H_cef via Stevens operators.

Spec reference: 02-03-HCEF_FORM, Hutchings 1964.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fexchange.core.fermion import one_body_operator_matrix
from fexchange.core.space_j import normalize_J
from fexchange.core.stevens import build_cef_stevens_operators
from fexchange.utils.constants import N_ORB
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError
from fexchange.utils.numerics import DTYPE_COMPLEX

logger = logging.getLogger("fexchange")


def build_hcef_matrix_J(
    J: float,
    B_params: dict[str, float],
    symmetry: Literal["Oh", "C3v"] = "Oh",
    mode_q3: Literal["cos", "sin"] = "cos",
) -> NDArray[np.complexfloating]:
    """
    Build H_cef in the |J,M> basis (02-03 §1-3).

    Parameters
    ----------
    J : total angular momentum quantum number.
    B_params : CEF parameters. Keys depend on symmetry:
        Oh:  {"B4": ..., "B6": ...}
        C3v: {"B20": ..., "B40": ..., "B60": ..., "B66": ..., "B43": ..., "B63": ...}
    """
    J = normalize_J(J, module="hcef")
    ops = build_cef_stevens_operators(J, symmetry=symmetry, mode_q3=mode_q3)
    dim = int(round(2 * J + 1))
    H = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)

    if symmetry == "Oh":
        B4 = B_params.get("B4", 0.0)
        B6 = B_params.get("B6", 0.0)
        H += B4 * (ops["O40"] + 5 * ops["O44c"])
        H += B6 * (ops["O60"] - 21 * ops["O64c"])
    elif symmetry == "C3v":
        H += B_params.get("B20", 0.0) * ops["O20"]
        H += B_params.get("B40", 0.0) * ops["O40"]
        H += B_params.get("B60", 0.0) * ops["O60"]
        H += B_params.get("B66", 0.0) * ops["O66"]
        H += B_params.get("B43", 0.0) * ops["O43_eta"]
        H += B_params.get("B63", 0.0) * ops["O63_eta"]
    else:
        raise PhysError(
            "FXE-PHYS-001",
            f"Unknown symmetry branch: {symmetry!r}",
            actual={"symmetry": symmetry},
        )

    check_hermitian(H, label="H_cef", module="hcef")
    return H


def build_hcef_matrix_fock(
    h_cef_1b: NDArray[np.complexfloating],
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """Lift a one-body CEF matrix to a fixed-occupation Fock sector."""
    h = np.asarray(h_cef_1b, dtype=DTYPE_COMPLEX)
    if h.shape != (n_orb, n_orb):
        raise PhysError(
            "FXE-PHYS-001",
            f"h_cef_1b shape {h.shape} != ({n_orb},{n_orb})",
            module="hcef",
            actual={"shape": list(h.shape), "n_orb": n_orb},
        )
    check_hermitian(h, label="h_cef_1b", module="hcef")
    H = one_body_operator_matrix(h, n_ele, n_orb)
    check_hermitian(H, label="H_cef_fock", module="hcef")
    return H
