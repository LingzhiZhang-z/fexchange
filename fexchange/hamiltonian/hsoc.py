"""
Spin-orbit coupling term H_soc.

Spec reference: 02-02-HSOC_FORM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fexchange.core.space_ls import build_space_ls_operator
from fexchange.core.fermion import one_body_operator_matrix
from fexchange.utils.constants import N_ORB
from fexchange.utils.checks import check_hermitian

__all__ = [
    "build_hsoc_unit_matrix",
    "build_hsoc_matrix",
]


def build_hsoc_unit_operator() -> NDArray[np.complexfloating]:
    """
    Build the unit operator for the SOC matrix.
    """
    one = build_space_ls_operator()
    return one["Lx"] @ one["Sx"] + one["Ly"] @ one["Sy"] + one["Lz"] @ one["Sz"]

def build_hsoc_unit_matrix(
    n_ele: int,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build O_soc such that H_soc = zeta * O_soc (02-02 §5).

    This is the zeta-independent SOC operator.
    """
    h_soc_1b = build_hsoc_unit_operator()
    O_soc = one_body_operator_matrix(h_soc_1b, n_ele, n_orb)
    check_hermitian(O_soc, label="O_soc", module="hsoc")
    return O_soc


def build_hsoc_matrix(
    n_ele: int,
    zeta: float = 1.0,
    n_orb: int = N_ORB,
) -> NDArray[np.complexfloating]:
    """
    Build many-body H_soc = zeta * O_soc.

    Returns the matrix in sector n_ele.
    """
    O_soc = build_hsoc_unit_matrix(n_ele, n_orb)
    H = zeta * O_soc
    check_hermitian(H, label="H_soc", module="hsoc")
    return H
