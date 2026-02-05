from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

Array = np.ndarray


@dataclass(frozen=True)
class CoulombShape:
    r4: float
    r6: float
    norm_rule: str = "F2=1"


@dataclass(frozen=True)
class CoulombScale:
    F2_abs: float
    F0_abs: float = 0.0


@dataclass(frozen=True)
class SOCParam:
    lambda_abs: float


@dataclass
class StateSet:
    n: int
    basis_id: str
    V_fock: Array
    labels: Optional[Dict] = None
    meta: Dict = field(default_factory=dict)


@dataclass
class EnergySet:
    n: int
    shape_key: str
    energy_key: str
    E_total: Array
    E_parts: Dict[str, Array] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


@dataclass
class GroundManifold:
    n: int
    basis_id: str
    G_fock: Array
    E0: float
    degeneracy: int
    tol: float
    meta: Dict = field(default_factory=dict)


@dataclass
class FermionOpsCache:
    n: int
    basis_id: str
    orbital_order_id: str
    D_create: List[sp.spmatrix]
    meta: Dict = field(default_factory=dict)


@dataclass
class ProjectedTransitionsCache:
    n: int
    basis_id: str
    orbital_order_id: str
    which: str  # "np1" or "nm1"
    src_lsjm_path: str
    truncation_id: str
    T: List[Array]
    E_mid: Array
    meta: Dict = field(default_factory=dict)
