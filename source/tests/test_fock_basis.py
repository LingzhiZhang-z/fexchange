"""Tests for core/fock_basis.py - Fock basis construction.

Expected behavior:
- For n electrons in 14 orbitals: C(14, n) determinants
- Determinants are integers with n bits set
- Sorted in ascending integer order (lexicographic)
- basis_id is deterministic

Expected outputs (dim = C(14, n)):
- n=0: dim=1 (vacuum)
- n=1: dim=14
- n=2: dim=91
- n=7: dim=3432 (half-filled)
- n=14: dim=1 (fully filled)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fexchange.core.fock_basis import FockBasis, build_fock_basis, iter_dets, popcount


def binomial(n: int, k: int) -> int:
    """Compute C(n, k)."""
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


class TestPopcount:
    """Test bit counting function."""

    def test_popcount_zero(self) -> None:
        assert popcount(0) == 0

    def test_popcount_single_bit(self) -> None:
        for i in range(14):
            assert popcount(1 << i) == 1

    def test_popcount_all_bits(self) -> None:
        # 14 bits all set: 2^14 - 1 = 16383
        assert popcount(0b11111111111111) == 14

    def test_popcount_various(self) -> None:
        assert popcount(0b101) == 2
        assert popcount(0b1111) == 4
        assert popcount(0b10101010) == 4


class TestIterDets:
    """Test determinant iteration."""

    def test_iter_dets_count(self) -> None:
        """iter_dets should yield C(14, n) determinants."""
        for n in range(15):
            dets = list(iter_dets(n, n_orb=14))
            assert len(dets) == binomial(14, n)

    def test_iter_dets_popcount(self) -> None:
        """Each determinant should have exactly n bits set."""
        for n in [0, 1, 2, 7, 13, 14]:
            for det in iter_dets(n, n_orb=14):
                assert popcount(det) == n


class TestBuildFockBasis:
    """Test Fock basis construction."""

    def test_dimension_formula(self) -> None:
        """Dimension should match C(14, n).

        Expected:
        - n=0: 1
        - n=1: 14
        - n=2: 91
        - n=7: 3432
        - n=14: 1
        """
        expected = {0: 1, 1: 14, 2: 91, 7: 3432, 14: 1}
        for n, dim in expected.items():
            basis = build_fock_basis(n)
            assert len(basis.dets) == dim, f"n={n}: expected {dim}, got {len(basis.dets)}"

    def test_dets_sorted(self) -> None:
        """Determinants should be sorted in ascending order."""
        for n in [1, 2, 3]:
            basis = build_fock_basis(n)
            assert np.all(basis.dets[:-1] < basis.dets[1:])

    def test_index_correct(self) -> None:
        """Index dict should map det -> position."""
        basis = build_fock_basis(2)
        for i, det in enumerate(basis.dets):
            assert basis.index[int(det)] == i

    def test_basis_id_deterministic(self) -> None:
        """Same n should produce same basis_id."""
        b1 = build_fock_basis(3)
        b2 = build_fock_basis(3)
        assert b1.basis_id == b2.basis_id

    def test_basis_id_format(self) -> None:
        """basis_id should follow expected format."""
        basis = build_fock_basis(2)
        assert "fock14" in basis.basis_id
        assert "n2" in basis.basis_id

    def test_n_stored(self) -> None:
        """FockBasis should store the electron count."""
        for n in [0, 1, 7, 14]:
            basis = build_fock_basis(n)
            assert basis.n == n

    def test_n0_vacuum(self) -> None:
        """n=0 should give single vacuum state (det=0)."""
        basis = build_fock_basis(0)
        assert len(basis.dets) == 1
        assert basis.dets[0] == 0

    def test_n14_fully_filled(self) -> None:
        """n=14 should give single fully-filled state."""
        basis = build_fock_basis(14)
        assert len(basis.dets) == 1
        assert basis.dets[0] == (1 << 14) - 1  # 16383
