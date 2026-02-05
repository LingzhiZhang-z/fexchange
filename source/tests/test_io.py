"""Tests for api/io.py - File I/O for data structures.

Expected behavior:
- StateSet, EnergySet, GroundManifold can be saved and loaded as NPZ
- JSON metadata is preserved
- Arrays are preserved exactly
- Path list I/O works correctly

File formats:
- StateSet: V_fock, labels_json, meta_json
- EnergySet: E_total, E_parts_json, meta_json, part__*
- GroundManifold: G_fock, E0, degeneracy, tol, meta_json
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fexchange.api.io import (
    load_energy_set,
    load_ground_manifold,
    load_state_set,
    read_source_paths_txt,
    save_energy_set,
    save_ground_manifold,
    save_state_set,
    write_source_paths_txt,
)
from fexchange.api.types import EnergySet, GroundManifold, StateSet


class TestStateSetIO:
    """Test StateSet save/load."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """StateSet should survive save/load roundtrip."""
        V_fock = np.random.randn(10, 5)
        labels = {"L": [0, 1, 2, 3, 4], "S": [0.5, 0.5, 0.5, 0.5, 0.5]}
        meta = {"source": "test", "version": 1}
        original = StateSet(n=2, basis_id="test_basis", V_fock=V_fock, labels=labels, meta=meta)

        path = tmp_path / "state.npz"
        save_state_set(path, original)
        loaded = load_state_set(path, n=2, basis_id="test_basis")

        np.testing.assert_array_almost_equal(loaded.V_fock, original.V_fock)
        assert loaded.labels == original.labels
        assert loaded.n == original.n

    def test_empty_labels(self, tmp_path: Path) -> None:
        """StateSet with no labels should work."""
        V_fock = np.eye(3)
        original = StateSet(n=1, basis_id="b", V_fock=V_fock, labels=None, meta={})

        path = tmp_path / "state.npz"
        save_state_set(path, original)
        loaded = load_state_set(path)

        np.testing.assert_array_almost_equal(loaded.V_fock, V_fock)
        assert loaded.labels is None

    def test_complex_array(self, tmp_path: Path) -> None:
        """Complex V_fock should be preserved."""
        V_fock = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
        original = StateSet(n=1, basis_id="b", V_fock=V_fock, labels=None, meta={})

        path = tmp_path / "state.npz"
        save_state_set(path, original)
        loaded = load_state_set(path)

        np.testing.assert_array_almost_equal(loaded.V_fock, V_fock)


class TestEnergySetIO:
    """Test EnergySet save/load."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """EnergySet should survive save/load roundtrip."""
        E_total = np.array([0.0, 1.0, 2.0, 3.0])
        E_parts = {"Coulomb": np.array([0.0, 0.5, 1.0, 1.5]), "SOC": np.array([0.0, 0.5, 1.0, 1.5])}
        meta = {"units": "meV"}
        original = EnergySet(
            n=2, shape_key="sk", energy_key="ek", E_total=E_total, E_parts=E_parts, meta=meta
        )

        path = tmp_path / "energy.npz"
        save_energy_set(path, original)
        loaded = load_energy_set(path)

        np.testing.assert_array_almost_equal(loaded.E_total, E_total)
        assert set(loaded.E_parts.keys()) == set(E_parts.keys())
        for name in E_parts:
            np.testing.assert_array_almost_equal(loaded.E_parts[name], E_parts[name])

    def test_empty_parts(self, tmp_path: Path) -> None:
        """EnergySet with no E_parts should work."""
        E_total = np.array([1.0, 2.0])
        original = EnergySet(n=1, shape_key="s", energy_key="e", E_total=E_total, E_parts={}, meta={})

        path = tmp_path / "energy.npz"
        save_energy_set(path, original)
        loaded = load_energy_set(path)

        np.testing.assert_array_almost_equal(loaded.E_total, E_total)
        assert loaded.E_parts == {}


class TestGroundManifoldIO:
    """Test GroundManifold save/load."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """GroundManifold should survive save/load roundtrip."""
        G_fock = np.random.randn(10, 3)  # 3-fold degenerate ground state
        original = GroundManifold(
            n=2, basis_id="b", G_fock=G_fock, E0=-100.5, degeneracy=3, tol=1e-8, meta={"source": "test"}
        )

        path = tmp_path / "ground.npz"
        save_ground_manifold(path, original)
        loaded = load_ground_manifold(path)

        np.testing.assert_array_almost_equal(loaded.G_fock, G_fock)
        assert loaded.E0 == pytest.approx(-100.5)
        assert loaded.degeneracy == 3
        assert loaded.tol == pytest.approx(1e-8)

    def test_complex_ground_state(self, tmp_path: Path) -> None:
        """Complex G_fock should be preserved."""
        G_fock = np.random.randn(5, 2) + 1j * np.random.randn(5, 2)
        original = GroundManifold(n=1, basis_id="b", G_fock=G_fock, E0=0.0, degeneracy=2, tol=0.0, meta={})

        path = tmp_path / "ground.npz"
        save_ground_manifold(path, original)
        loaded = load_ground_manifold(path)

        np.testing.assert_array_almost_equal(loaded.G_fock, G_fock)


class TestSourcePathsIO:
    """Test source paths file I/O."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Paths should survive write/read roundtrip."""
        paths = ["/path/to/file1.npz", "/path/to/file2.npz"]
        path = tmp_path / "source_paths.txt"
        write_source_paths_txt(path, paths)
        loaded = read_source_paths_txt(path)

        # Paths are resolved to absolute
        assert len(loaded) == 2

    def test_empty_list(self, tmp_path: Path) -> None:
        """Empty path list should work."""
        path = tmp_path / "empty.txt"
        write_source_paths_txt(path, [])
        loaded = read_source_paths_txt(path)
        assert loaded == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Nonexistent file should return empty list."""
        path = tmp_path / "nonexistent.txt"
        loaded = read_source_paths_txt(path)
        assert loaded == []
