"""
State-vector objects: BasisDet, StateVec, StateSet.

Spec reference: 01-01-STATE_VECTOR_CONVENTION.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from fexchange.core.fock import make_basis_id, check_basis_id, popcount
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_ZERO, EPS_NORM
from fexchange.utils.errors import PhysError, BindError

logger = logging.getLogger("fexchange")


# ---------------------------------------------------------------------------
# Phase gauge (01-01 §5, 00-02 §5)
# ---------------------------------------------------------------------------

def fix_phase(v: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Apply pivot-rule phase gauge to a single state vector.

    1. Find pivot = argmax |v|.
    2. Tie-break: smallest index.
    3. Multiply so v[pivot] is real >= 0.
    """
    v = v.copy()
    absv = np.abs(v)
    pivot = int(np.argmax(absv))
    if absv[pivot] < EPS_ZERO:
        return v
    phase = v[pivot] / absv[pivot]
    v = v / phase
    # ensure numerically real and non-negative at pivot
    v[pivot] = abs(v[pivot])
    return v


def fix_phase_columns(V: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Apply pivot-rule phase gauge to each column of V."""
    V = V.copy()
    for j in range(V.shape[1]):
        V[:, j] = fix_phase(V[:, j])
    return V


# ---------------------------------------------------------------------------
# BasisDet (01-01 §2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasisDet:
    """One Slater determinant in Fock basis."""

    det: int

    @property
    def n_ele(self) -> int:
        return popcount(self.det)


# ---------------------------------------------------------------------------
# StateVec (01-01 §2)
# ---------------------------------------------------------------------------

@dataclass
class StateVec:
    """
    One physical state as a linear combination of BasisDet.

    Attributes:
        basis_id: Fock-basis identifier.
        n_ele: electron number.
        coeffs: complex coefficient array of length dim_fock.
    """

    basis_id: str
    n_ele: int
    coeffs: NDArray[np.complexfloating]

    def normalize(self) -> StateVec:
        norm = np.linalg.norm(self.coeffs)
        if norm < EPS_ZERO:
            raise PhysError(
                "FXE-PHYS-001",
                "Cannot normalise zero vector",
                module="states",
            )
        self.coeffs = self.coeffs / norm
        return self

    def apply_phase_gauge(self) -> StateVec:
        self.coeffs = fix_phase(self.coeffs)
        return self


# ---------------------------------------------------------------------------
# StateSet (01-01 §2, §4)
# ---------------------------------------------------------------------------

@dataclass
class StateSet:
    """
    A set of StateVec stored as a matrix V_fock.

    Attributes:
        basis_id: Fock-basis identifier.
        n_ele: electron number.
        state_order_id: ordering identifier string.
        V_fock: complex matrix (dim_fock, n_states), columns are states.
        labels: per-column label dicts.
        meta: additional metadata.
    """

    basis_id: str
    n_ele: int
    state_order_id: str
    V_fock: NDArray[np.complexfloating]
    labels: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.V_fock.shape[1]

    @property
    def dim_fock(self) -> int:
        return self.V_fock.shape[0]

    def apply_phase_gauge(self) -> StateSet:
        """Apply pivot-rule phase gauge to every column."""
        self.V_fock = fix_phase_columns(self.V_fock)
        return self

    def check_binding(self, expected_basis_id: str) -> None:
        check_basis_id(expected_basis_id, self.basis_id)

    def validate(self) -> None:
        """Run basic consistency checks."""
        if self.V_fock.shape[1] != len(self.labels):
            raise PhysError(
                "FXE-PHYS-001",
                f"Column count {self.V_fock.shape[1]} != label count {len(self.labels)}",
                module="states",
            )
