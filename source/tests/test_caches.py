"""Tests for sopt/fermion_ops_cache.py and sopt/projected_transitions_cache.py.

Expected behavior:
- Caches are created on first call
- Caches are loaded on subsequent calls with same parameters
- Cache invalidation when parameters change
- Files are written to correct locations

Cache structure:
- fermion_ops_cache: D_p0.npz ... D_p13.npz, meta.json
- projected_transitions_cache: T_p0.npy ... T_p13.npy, E_mid.npy, params.json, source_paths.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fexchange.api.io import save_state_set
from fexchange.api.types import StateSet
from fexchange.core.fock_basis import build_fock_basis
from fexchange.core.orbitals import ORBITAL_ORDER_ID
from fexchange.sopt.fermion_ops_cache import build_or_load_ops_cache
from fexchange.sopt.projected_transitions_cache import build_or_load_projected_transitions_cache


class TestFermionOpsCache:
    """Test fermion operators caching."""

    def test_creates_cache(self, tmp_path: Path) -> None:
        """First call should create cache files."""
        basis_n = build_fock_basis(2)
        basis_np1 = build_fock_basis(3)

        cache = build_or_load_ops_cache(
            n=2,
            cache_dir=tmp_path,
            basis_n=basis_n,
            basis_np1=basis_np1,
            orbital_order_id=ORBITAL_ORDER_ID,
        )

        assert cache.n == 2
        assert len(cache.D_create) == 14

        # Check files exist
        cache_subdir = list(tmp_path.iterdir())[0]
        assert (cache_subdir / "meta.json").exists()
        for p in range(14):
            assert (cache_subdir / f"D_p{p}.npz").exists()

    def test_loads_existing_cache(self, tmp_path: Path) -> None:
        """Second call should load from cache."""
        basis_n = build_fock_basis(2)
        basis_np1 = build_fock_basis(3)

        # First call creates
        cache1 = build_or_load_ops_cache(
            n=2,
            cache_dir=tmp_path,
            basis_n=basis_n,
            basis_np1=basis_np1,
            orbital_order_id=ORBITAL_ORDER_ID,
        )

        # Second call loads
        cache2 = build_or_load_ops_cache(
            n=2,
            cache_dir=tmp_path,
            basis_n=basis_n,
            basis_np1=basis_np1,
            orbital_order_id=ORBITAL_ORDER_ID,
        )

        # Should get same matrices
        for p in range(14):
            np.testing.assert_array_almost_equal(
                cache1.D_create[p].toarray(), cache2.D_create[p].toarray()
            )

    def test_different_n_creates_new_cache(self, tmp_path: Path) -> None:
        """Different n should create separate cache."""
        basis_2 = build_fock_basis(2)
        basis_3 = build_fock_basis(3)
        basis_4 = build_fock_basis(4)

        cache_n2 = build_or_load_ops_cache(2, tmp_path, basis_2, basis_3, ORBITAL_ORDER_ID)
        cache_n3 = build_or_load_ops_cache(3, tmp_path, basis_3, basis_4, ORBITAL_ORDER_ID)

        assert cache_n2.n == 2
        assert cache_n3.n == 3

        # Should be two cache directories
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) == 2

    def test_meta_contains_required_fields(self, tmp_path: Path) -> None:
        """meta.json should contain required fields."""
        basis_n = build_fock_basis(2)
        basis_np1 = build_fock_basis(3)

        cache = build_or_load_ops_cache(2, tmp_path, basis_n, basis_np1, ORBITAL_ORDER_ID)

        assert "n" in cache.meta
        assert "basis_id" in cache.meta
        assert "orbital_order_id" in cache.meta


class TestProjectedTransitionsCache:
    """Test projected transitions caching."""

    @pytest.fixture
    def setup_lsjm(self, tmp_path: Path):
        """Create fake LSJM state files."""
        dim_fock = len(build_fock_basis(2).dets)
        n_states = 10

        V_fock = np.random.randn(dim_fock, n_states)
        state = StateSet(n=2, basis_id="test", V_fock=V_fock, labels=None, meta={})

        lsjm_path = tmp_path / "lsjm.npz"
        save_state_set(lsjm_path, state)

        return lsjm_path, V_fock, n_states

    def test_creates_cache_np1(self, tmp_path: Path, setup_lsjm) -> None:
        """Test np1 cache creation."""
        lsjm_path, V_fock, n_states = setup_lsjm

        basis_n = build_fock_basis(2)
        basis_np1 = build_fock_basis(3)
        ops = build_or_load_ops_cache(2, tmp_path / "ops", basis_n, basis_np1, ORBITAL_ORDER_ID)

        E_mid = np.linspace(0, 10, n_states)

        cache = build_or_load_projected_transitions_cache(
            n=2,
            cache_dir=tmp_path / "proj",
            basis_id=basis_n.basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="np1",
            lsjm_path=lsjm_path,
            E_mid=E_mid,
            truncation_indices=None,
            truncation_id="all",
            D_create=ops.D_create,
        )

        assert cache.which == "np1"
        assert len(cache.T) == 14
        assert len(cache.E_mid) == n_states

    def test_creates_cache_nm1(self, tmp_path: Path, setup_lsjm) -> None:
        """Test nm1 cache creation."""
        lsjm_path, V_fock, n_states = setup_lsjm

        basis_nm1 = build_fock_basis(1)
        basis_n = build_fock_basis(2)
        ops_nm1 = build_or_load_ops_cache(1, tmp_path / "ops", basis_nm1, basis_n, ORBITAL_ORDER_ID)

        E_mid = np.linspace(0, 10, n_states)

        cache = build_or_load_projected_transitions_cache(
            n=2,
            cache_dir=tmp_path / "proj",
            basis_id=basis_n.basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="nm1",
            lsjm_path=lsjm_path,
            E_mid=E_mid,
            truncation_indices=None,
            truncation_id="all",
            D_create_nm1=ops_nm1.D_create,
        )

        assert cache.which == "nm1"
        assert len(cache.T) == 14

    def test_truncation_applied(self, tmp_path: Path, setup_lsjm) -> None:
        """Truncation indices should reduce state count."""
        lsjm_path, V_fock, n_states = setup_lsjm

        basis_n = build_fock_basis(2)
        basis_np1 = build_fock_basis(3)
        ops = build_or_load_ops_cache(2, tmp_path / "ops", basis_n, basis_np1, ORBITAL_ORDER_ID)

        E_mid = np.linspace(0, 10, n_states)
        truncation_indices = np.array([0, 1, 2])  # Keep only 3 states

        cache = build_or_load_projected_transitions_cache(
            n=2,
            cache_dir=tmp_path / "proj",
            basis_id=basis_n.basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="np1",
            lsjm_path=lsjm_path,
            E_mid=E_mid,
            truncation_indices=truncation_indices,
            truncation_id="nstates_3",
            D_create=ops.D_create,
        )

        assert len(cache.E_mid) == 3
        for T_p in cache.T:
            assert T_p.shape[0] == 3

    def test_wrong_which_raises(self, tmp_path: Path, setup_lsjm) -> None:
        """Invalid 'which' should raise."""
        lsjm_path, V_fock, n_states = setup_lsjm

        with pytest.raises(ValueError, match="np1.*nm1"):
            build_or_load_projected_transitions_cache(
                n=2,
                cache_dir=tmp_path,
                basis_id="test",
                orbital_order_id=ORBITAL_ORDER_ID,
                which="invalid",
                lsjm_path=lsjm_path,
                E_mid=np.zeros(n_states),
                truncation_indices=None,
                truncation_id="all",
            )
