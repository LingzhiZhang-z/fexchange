"""Tests for core/orbitals.py - Spin-orbital indexing.

Expected behavior:
- 14 spin-orbitals for f-shell: m in [-3, 3], sigma in {-0.5, +0.5}
- Ordering: m=-3 first, then for each m: sigma=-0.5, then sigma=+0.5
- p_to_ms and ms_to_p are inverses of each other

Expected outputs:
- p=0  -> (m=-3, sigma=-0.5)
- p=1  -> (m=-3, sigma=+0.5)
- p=6  -> (m=0,  sigma=-0.5)
- p=13 -> (m=3,  sigma=+0.5)
"""
from __future__ import annotations

import pytest

from fexchange.core.orbitals import ORBITAL_ORDER_ID, ms_to_p, p_to_ms


class TestOrbitalIndexing:
    """Test spin-orbital index conversions."""

    def test_orbital_order_id_defined(self) -> None:
        """ORBITAL_ORDER_ID should be a non-empty string."""
        assert isinstance(ORBITAL_ORDER_ID, str)
        assert len(ORBITAL_ORDER_ID) > 0

    def test_p_to_ms_range(self) -> None:
        """p_to_ms should accept p in [0, 13]."""
        for p in range(14):
            m, sigma = p_to_ms(p)
            assert -3 <= m <= 3
            assert sigma in (-0.5, 0.5)

    def test_p_to_ms_out_of_range(self) -> None:
        """p_to_ms should raise for invalid p."""
        with pytest.raises(IndexError):
            p_to_ms(14)
        with pytest.raises(IndexError):
            p_to_ms(-1)

    def test_ms_to_p_inverse(self) -> None:
        """ms_to_p should be the inverse of p_to_ms."""
        for p in range(14):
            m, sigma = p_to_ms(p)
            assert ms_to_p(m, sigma) == p

    def test_all_orbitals_unique(self) -> None:
        """All 14 (m, sigma) pairs should be unique."""
        pairs = [p_to_ms(p) for p in range(14)]
        assert len(pairs) == len(set(pairs))

    def test_expected_mappings(self) -> None:
        """Check specific expected mappings.

        Expected:
        - p=0: m=-3, sigma=-0.5
        - p=1: m=-3, sigma=+0.5
        - p=6: m=0,  sigma=-0.5
        - p=7: m=0,  sigma=+0.5
        - p=13: m=3, sigma=+0.5
        """
        assert p_to_ms(0) == (-3, -0.5)
        assert p_to_ms(1) == (-3, 0.5)
        assert p_to_ms(6) == (0, -0.5)
        assert p_to_ms(7) == (0, 0.5)
        assert p_to_ms(13) == (3, 0.5)

    def test_ms_to_p_all_m_values(self) -> None:
        """ms_to_p should work for all valid (m, sigma) pairs."""
        for m in range(-3, 4):
            for sigma in (-0.5, 0.5):
                p = ms_to_p(m, sigma)
                assert 0 <= p <= 13
                assert p_to_ms(p) == (m, sigma)
