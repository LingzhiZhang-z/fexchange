from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp

from .fock_basis import FockBasis, popcount


def build_creation_matrices(basis_n: FockBasis, basis_np1: FockBasis, n_orb: int = 14) -> List[sp.csr_matrix]:
    """Build creation operator matrices D_p^(n) for p in [0, n_orb-1].

    D_p^(n) has shape (dim_{n+1}, dim_n) with
    (D_p)_{i,j} = sign if det_i = det_j with bit p set.
    """
    dim_n = len(basis_n.dets)
    dim_np1 = len(basis_np1.dets)
    matrices: List[sp.csr_matrix] = []

    for p in range(n_orb):
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        bit = 1 << p
        below_mask = bit - 1
        for col, det in enumerate(basis_n.dets):
            if det & bit:
                continue
            det_new = int(det | bit)
            row = basis_np1.index.get(det_new)
            if row is None:
                continue
            n_below = popcount(int(det) & below_mask)
            sign = -1.0 if (n_below % 2 == 1) else 1.0
            rows.append(row)
            cols.append(col)
            data.append(sign)
        mat = sp.csr_matrix((data, (rows, cols)), shape=(dim_np1, dim_n), dtype=np.float64)
        matrices.append(mat)
    return matrices


def build_annihilation_matrices(basis_nm1: FockBasis, basis_n: FockBasis, n_orb: int = 14) -> List[sp.csr_matrix]:
    """Build annihilation matrices C_p^(n) as hermitian of D_p^(n-1)."""
    d_create = build_creation_matrices(basis_nm1, basis_n, n_orb=n_orb)
    return [mat.conjugate().transpose().tocsr() for mat in d_create]
