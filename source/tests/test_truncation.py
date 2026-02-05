"""Tests for sopt/truncation.py - State truncation utilities.

Expected behavior:
- truncate_by_nstates: keep lowest n_keep energy states
- truncate_by_energy_window: keep states with E <= emax
- Returns (indices, truncation_id) tuple
- truncation_id is deterministic for caching

Expected outputs:
- truncate_by_nstates([0,1,2,3,4], 3) -> indices for 3 lowest
- truncate_by_energy_window([0,5,10,15], 7) -> indices where E <= 7
"""
from __future__ import annotations

import numpy as np
import pytest

from fexchange.sopt.truncation import truncate_by_energy_window, truncate_by_nstates


class TestTruncateByNstates:
    """Test truncation by number of states."""

    def test_keep_all(self) -> None:
        """If n_keep >= len(E), keep all."""
        E = np.array([0.0, 1.0, 2.0, 3.0])
        idx, trunc_id = truncate_by_nstates(E, n_keep=10)
        assert len(idx) == 4
        assert trunc_id == "all"

    def test_keep_none_is_none(self) -> None:
        """If n_keep is None, keep all."""
        E = np.array([0.0, 1.0, 2.0])
        idx, trunc_id = truncate_by_nstates(E, n_keep=None)
        assert len(idx) == 3
        assert trunc_id == "all"

    def test_keep_subset(self) -> None:
        """Keep n_keep lowest energy states."""
        E = np.array([2.0, 0.0, 1.0, 3.0])  # unsorted
        idx, trunc_id = truncate_by_nstates(E, n_keep=2)
        assert len(idx) == 2
        # Should keep indices of two lowest: E[1]=0, E[2]=1
        assert set(idx) == {1, 2}
        assert "nstates_2" in trunc_id

    def test_deterministic_id(self) -> None:
        """Same n_keep should give same truncation_id."""
        E = np.array([0.0, 1.0, 2.0])
        _, id1 = truncate_by_nstates(E, n_keep=2)
        _, id2 = truncate_by_nstates(E, n_keep=2)
        assert id1 == id2

    def test_n_keep_zero_raises(self) -> None:
        """n_keep=0 should raise ValueError."""
        E = np.array([0.0, 1.0])
        with pytest.raises(ValueError):
            truncate_by_nstates(E, n_keep=0)

    def test_n_keep_negative_raises(self) -> None:
        """Negative n_keep should raise ValueError."""
        E = np.array([0.0, 1.0])
        with pytest.raises(ValueError):
            truncate_by_nstates(E, n_keep=-1)


class TestTruncateByEnergyWindow:
    """Test truncation by energy window."""

    def test_keep_all(self) -> None:
        """If emax is None, keep all."""
        E = np.array([0.0, 1.0, 2.0])
        idx, trunc_id = truncate_by_energy_window(E, emax=None)
        assert len(idx) == 3
        assert trunc_id == "all"

    def test_energy_cutoff(self) -> None:
        """Keep states with E <= emax."""
        E = np.array([0.0, 5.0, 10.0, 15.0])
        idx, trunc_id = truncate_by_energy_window(E, emax=7.0)
        assert len(idx) == 2
        assert set(idx) == {0, 1}  # E=0 and E=5
        assert "emax" in trunc_id

    def test_no_states_pass(self) -> None:
        """If no states pass, return empty indices."""
        E = np.array([10.0, 20.0, 30.0])
        idx, trunc_id = truncate_by_energy_window(E, emax=5.0)
        assert len(idx) == 0

    def test_boundary_included(self) -> None:
        """States with E exactly at emax should be included."""
        E = np.array([0.0, 5.0, 10.0])
        idx, trunc_id = truncate_by_energy_window(E, emax=5.0)
        assert len(idx) == 2
        assert set(idx) == {0, 1}

    def test_deterministic_id(self) -> None:
        """Same emax should give same truncation_id."""
        E = np.array([0.0, 1.0, 2.0])
        _, id1 = truncate_by_energy_window(E, emax=1.5)
        _, id2 = truncate_by_energy_window(E, emax=1.5)
        assert id1 == id2
