"""Integration tests for the full SOPT pipeline.

These tests verify that the components work together correctly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fexchange.api.io import save_energy_set, save_ground_manifold, save_state_set
from fexchange.api.types import EnergySet, GroundManifold, StateSet
from fexchange.core.fock_basis import build_fock_basis
from fexchange.core.orbitals import ORBITAL_ORDER_ID
from fexchange.sopt.fermion_ops_cache import build_or_load_ops_cache
from fexchange.sopt.heff import compute_heff
from fexchange.sopt.projected_transitions_cache import build_or_load_projected_transitions_cache


class TestFullPipeline:
    """Test the complete SOPT pipeline."""

    @pytest.fixture
    def setup_test_data(self, tmp_path: Path):
        """Create minimal test data for pipeline."""
        n = 2  # 2 electrons

        # Build bases
        basis_nm1 = build_fock_basis(n - 1)
        basis_n = build_fock_basis(n)
        basis_np1 = build_fock_basis(n + 1)

        dim_nm1 = len(basis_nm1.dets)
        dim_n = len(basis_n.dets)
        dim_np1 = len(basis_np1.dets)

        # Ground manifold: 2-fold degenerate
        d = 2
        G_fock = np.random.randn(dim_n, d)
        G_fock, _ = np.linalg.qr(G_fock)  # Orthonormalize
        E0 = -100.0

        ground = GroundManifold(
            n=n, basis_id=basis_n.basis_id, G_fock=G_fock, E0=E0, degeneracy=d, tol=1e-10, meta={}
        )
        ground_path = tmp_path / "ground.npz"
        save_ground_manifold(ground_path, ground)

        # LSJM states for n+1 sector
        n_np1_states = 20
        V_np1 = np.random.randn(dim_np1, n_np1_states)
        V_np1, _ = np.linalg.qr(V_np1)
        lsjm_np1 = StateSet(n=n + 1, basis_id=basis_np1.basis_id, V_fock=V_np1, labels=None, meta={})
        lsjm_np1_path = tmp_path / "lsjm_np1.npz"
        save_state_set(lsjm_np1_path, lsjm_np1)

        # Energies for n+1
        E_np1 = np.sort(np.random.uniform(10, 50, n_np1_states))
        e_np1 = EnergySet(n=n + 1, shape_key="", energy_key="", E_total=E_np1, E_parts={}, meta={})
        e_np1_path = tmp_path / "e_np1.npz"
        save_energy_set(e_np1_path, e_np1)

        # LSJM states for n-1 sector
        n_nm1_states = 10
        V_nm1 = np.random.randn(dim_nm1, n_nm1_states)
        V_nm1, _ = np.linalg.qr(V_nm1)
        lsjm_nm1 = StateSet(n=n - 1, basis_id=basis_nm1.basis_id, V_fock=V_nm1, labels=None, meta={})
        lsjm_nm1_path = tmp_path / "lsjm_nm1.npz"
        save_state_set(lsjm_nm1_path, lsjm_nm1)

        # Energies for n-1
        E_nm1 = np.sort(np.random.uniform(5, 30, n_nm1_states))
        e_nm1 = EnergySet(n=n - 1, shape_key="", energy_key="", E_total=E_nm1, E_parts={}, meta={})
        e_nm1_path = tmp_path / "e_nm1.npz"
        save_energy_set(e_nm1_path, e_nm1)

        # Hopping matrix
        t = np.random.randn(14, 14) + 1j * np.random.randn(14, 14)
        t = t + t.conj().T  # Make Hermitian
        t_path = tmp_path / "t.npy"
        np.save(t_path, t)

        return {
            "n": n,
            "d": d,
            "tmp_path": tmp_path,
            "ground_path": ground_path,
            "lsjm_np1_path": lsjm_np1_path,
            "e_np1_path": e_np1_path,
            "lsjm_nm1_path": lsjm_nm1_path,
            "e_nm1_path": e_nm1_path,
            "t_path": t_path,
            "G_fock": G_fock,
            "E0": E0,
            "E_np1": E_np1,
            "E_nm1": E_nm1,
            "basis_n": basis_n,
            "basis_np1": basis_np1,
            "basis_nm1": basis_nm1,
        }

    def test_full_pipeline(self, setup_test_data) -> None:
        """Run the full SOPT pipeline and check output."""
        data = setup_test_data
        n = data["n"]
        d = data["d"]
        tmp_path = data["tmp_path"]

        # Load hopping
        t = np.load(data["t_path"])

        # Build operator caches
        ops_n = build_or_load_ops_cache(
            n, tmp_path / "cache", data["basis_n"], data["basis_np1"], ORBITAL_ORDER_ID
        )
        ops_nm1 = build_or_load_ops_cache(
            n - 1, tmp_path / "cache", data["basis_nm1"], data["basis_n"], ORBITAL_ORDER_ID
        )

        # Build projected transitions caches
        cache_np1 = build_or_load_projected_transitions_cache(
            n=n,
            cache_dir=tmp_path / "proj_cache",
            basis_id=data["basis_n"].basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="np1",
            lsjm_path=data["lsjm_np1_path"],
            E_mid=data["E_np1"],
            truncation_indices=None,
            truncation_id="all",
            D_create=ops_n.D_create,
        )

        cache_nm1 = build_or_load_projected_transitions_cache(
            n=n,
            cache_dir=tmp_path / "proj_cache",
            basis_id=data["basis_n"].basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="nm1",
            lsjm_path=data["lsjm_nm1_path"],
            E_mid=data["E_nm1"],
            truncation_indices=None,
            truncation_id="all",
            D_create_nm1=ops_nm1.D_create,
        )

        # Compute Heff
        Heff = compute_heff(t, data["G_fock"], cache_np1, cache_nm1, data["E0"])

        # Verify output
        assert Heff.shape == (d * d, d * d)
        # Heff should be Hermitian
        np.testing.assert_array_almost_equal(Heff, Heff.conj().T, decimal=10)

    def test_pipeline_with_truncation(self, setup_test_data) -> None:
        """Test pipeline with state truncation."""
        data = setup_test_data
        n = data["n"]
        tmp_path = data["tmp_path"]

        t = np.load(data["t_path"])

        ops_n = build_or_load_ops_cache(
            n, tmp_path / "cache2", data["basis_n"], data["basis_np1"], ORBITAL_ORDER_ID
        )
        ops_nm1 = build_or_load_ops_cache(
            n - 1, tmp_path / "cache2", data["basis_nm1"], data["basis_n"], ORBITAL_ORDER_ID
        )

        # Truncate to 5 states each
        idx_np1 = np.arange(5)
        idx_nm1 = np.arange(5)

        cache_np1 = build_or_load_projected_transitions_cache(
            n=n,
            cache_dir=tmp_path / "proj_cache2",
            basis_id=data["basis_n"].basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="np1",
            lsjm_path=data["lsjm_np1_path"],
            E_mid=data["E_np1"],
            truncation_indices=idx_np1,
            truncation_id="nstates_5",
            D_create=ops_n.D_create,
        )

        cache_nm1 = build_or_load_projected_transitions_cache(
            n=n,
            cache_dir=tmp_path / "proj_cache2",
            basis_id=data["basis_n"].basis_id,
            orbital_order_id=ORBITAL_ORDER_ID,
            which="nm1",
            lsjm_path=data["lsjm_nm1_path"],
            E_mid=data["E_nm1"],
            truncation_indices=idx_nm1,
            truncation_id="nstates_5",
            D_create_nm1=ops_nm1.D_create,
        )

        # Should have 5 intermediate states each
        assert len(cache_np1.E_mid) == 5
        assert len(cache_nm1.E_mid) == 5

        # Heff should still work
        Heff = compute_heff(t, data["G_fock"], cache_np1, cache_nm1, data["E0"])
        assert Heff.shape == (data["d"] * data["d"], data["d"] * data["d"])
