"""
Global tolerance table and dtype policy.

Spec reference: 00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Dtype policy (00-02 §2)
# ---------------------------------------------------------------------------
DTYPE_REAL: type = np.float64
DTYPE_COMPLEX: type = np.complex128

# ---------------------------------------------------------------------------
# Global tolerance table (00-02 §3)
# ---------------------------------------------------------------------------
EPS_ZERO: float = 1e-12
EPS_NORM: float = 1e-10
EPS_ORTH: float = 1e-10
EPS_DIAG: float = 1e-9
EPS_EIG_CLUSTER: float = 1e-10
EPS_SVD_REL: float = 1e-12
EPS_MAP: float = 1e-8
EPS_HERM: float = 1e-10
EPS_UNITARY: float = 1e-10
EPS_NK_SPLIT: float = 1e-6
EPS_MAG_AB: float = 1e-1

# Runtime path defaults (00-02 §7)
DENSITY_SPARSE_SWITCH: float = 0.15
PEAK_MEMORY_BUDGET_GB: float = 8.0
CHUNK_POLICY: str = "auto-by-memory"


def numerics_meta() -> dict:
    """Return the active numerical profile as a metadata dict (00-02 §11)."""
    return {
        "dtype_real": str(DTYPE_REAL),
        "dtype_complex": str(DTYPE_COMPLEX),
        "eps_zero": EPS_ZERO,
        "eps_norm": EPS_NORM,
        "eps_orth": EPS_ORTH,
        "eps_diag": EPS_DIAG,
        "eps_eig_cluster": EPS_EIG_CLUSTER,
        "eps_svd_rel": EPS_SVD_REL,
        "eps_map": EPS_MAP,
        "eps_herm": EPS_HERM,
        "eps_unitary": EPS_UNITARY,
        "eps_nk_split": EPS_NK_SPLIT,
        "eps_mag_ab": EPS_MAG_AB,
        "density_sparse_switch": DENSITY_SPARSE_SWITCH,
        "peak_memory_budget_gb": PEAK_MEMORY_BUDGET_GB,
    }
