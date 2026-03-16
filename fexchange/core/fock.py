"""
Fock-space foundations: bitstring encoding, determinant enumeration, basis_id.

Spec reference: 01-00-FOUNDATIONS_FOCK_SLATER.
"""

from __future__ import annotations

import logging
from math import comb
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.errors import PhysError, BindError
from fexchange.utils.constants import N_ORB

logger = logging.getLogger("fexchange")


# ---------------------------------------------------------------------------
# Bitstring helpers (01-00 §5)
# ---------------------------------------------------------------------------

def occ(det: int, p: int) -> int:
    """Occupation of orbital *p* in determinant *det*."""
    return (det >> p) & 1


def popcount(det: int) -> int:
    """Number of set bits (electron count) in *det*."""
    return bin(det).count("1")


def set_bit(det: int, p: int) -> int:
    return det | (1 << p)


def clear_bit(det: int, p: int) -> int:
    return det & ~(1 << p)


def count_below(det: int, p: int) -> int:
    """Number of occupied orbitals with index < *p* (01-00 §7)."""
    mask = (1 << p) - 1
    return bin(det & mask).count("1")


# ---------------------------------------------------------------------------
# Sector dimension (01-00 §1)
# ---------------------------------------------------------------------------

def dim_sector(n_ele: int, n_orb: int = N_ORB) -> int:
    """Hilbert-space dimension for *n_ele* electrons in *n_orb* spin-orbitals."""
    _validate_sector(n_ele, n_orb)
    return comb(n_orb, n_ele)


# ---------------------------------------------------------------------------
# Determinant enumeration (01-00 §6)
# ---------------------------------------------------------------------------

def enumerate_dets(n_ele: int, n_orb: int = N_ORB) -> NDArray[np.int64]:
    """
    Return sorted array of all Slater determinants (ascending integer, lex_v1).

    Each determinant is a non-negative integer whose bit-*p* encodes orbital *p*.
    """
    _validate_sector(n_ele, n_orb)
    dets: list[int] = []
    upper = 1 << n_orb
    for d in range(upper):
        if bin(d).count("1") == n_ele:
            dets.append(d)
    return np.array(dets, dtype=np.int64)


def det_index(det: int, dets: NDArray[np.int64]) -> int:
    """Return the position of *det* in sorted determinant list."""
    idx = np.searchsorted(dets, det)
    if idx >= len(dets) or dets[idx] != det:
        raise PhysError(
            "FXE-PHYS-001",
            f"Determinant {det} not found in basis",
            actual={"det": det},
        )
    return int(idx)


# ---------------------------------------------------------------------------
# basis_id (01-00 §6)
# ---------------------------------------------------------------------------

def make_basis_id(n_ele: int, n_orb: int = N_ORB) -> str:
    """
    Canonical basis_id string: ``fock{n_orb}_n{n_ele}_lex_v1``.
    """
    _validate_sector(n_ele, n_orb)
    return f"fock{n_orb}_n{n_ele}_lex_v1"


def check_basis_id(expected: str, actual: str) -> None:
    """Fail hard on basis_id mismatch (01-00 §6, 01-01 §3)."""
    if expected != actual:
        raise BindError(
            "FXE-BIND-001",
            f"basis_id mismatch: expected {expected!r}, got {actual!r}",
            actual={"lhs_id": expected, "rhs_id": actual},
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_sector(n_ele: int, n_orb: int = N_ORB) -> None:
    if not (0 <= n_ele <= n_orb):
        raise PhysError(
            "FXE-PHYS-001",
            f"Invalid sector: n_ele={n_ele} not in [0, {n_orb}]",
            actual={"n_ele": n_ele, "n_orb": n_orb},
        )
