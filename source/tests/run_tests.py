#!/usr/bin/env python
"""Simple test runner without pytest dependency.

Run with: python -m tests.run_tests
from the fexchange directory.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


def test_orbitals():
    """Test orbital indexing."""
    from fexchange.core.orbitals import ORBITAL_ORDER_ID, ms_to_p, p_to_ms

    print("Testing orbitals...")

    # Test ORBITAL_ORDER_ID
    assert isinstance(ORBITAL_ORDER_ID, str) and len(ORBITAL_ORDER_ID) > 0

    # Test p_to_ms range
    for p in range(14):
        m, sigma = p_to_ms(p)
        assert -3 <= m <= 3
        assert sigma in (-0.5, 0.5)

    # Test inverse
    for p in range(14):
        m, sigma = p_to_ms(p)
        assert ms_to_p(m, sigma) == p

    # Test specific mappings
    assert p_to_ms(0) == (-3, -0.5)
    assert p_to_ms(1) == (-3, 0.5)
    assert p_to_ms(6) == (0, -0.5)
    assert p_to_ms(13) == (3, 0.5)

    print("  ✓ ORBITAL_ORDER_ID defined")
    print("  ✓ p_to_ms returns valid (m, sigma)")
    print("  ✓ ms_to_p is inverse of p_to_ms")
    print("  ✓ Specific mappings correct")
    return True


def test_fock_basis():
    """Test Fock basis construction."""
    import math

    from fexchange.core.fock_basis import build_fock_basis, iter_dets, popcount

    print("Testing fock_basis...")

    def binomial(n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    # Test popcount
    assert popcount(0) == 0
    assert popcount(0b101) == 2
    assert popcount(0b11111111111111) == 14

    # Test dimensions
    expected_dims = {0: 1, 1: 14, 2: 91, 7: 3432, 14: 1}
    for n, expected in expected_dims.items():
        basis = build_fock_basis(n)
        assert len(basis.dets) == expected, f"n={n}: got {len(basis.dets)}, expected {expected}"

    # Test sorted
    basis = build_fock_basis(2)
    assert np.all(basis.dets[:-1] < basis.dets[1:])

    # Test index
    for i, det in enumerate(basis.dets):
        assert basis.index[int(det)] == i

    print(f"  ✓ popcount works correctly")
    print(f"  ✓ Dimensions: n=0→1, n=1→14, n=2→91, n=7→3432, n=14→1")
    print(f"  ✓ Determinants sorted in ascending order")
    print(f"  ✓ Index dict correctly maps det→position")
    return True


def test_fermion_ops():
    """Test fermion creation operators."""
    from fexchange.core.fermion_ops import build_creation_matrices
    from fexchange.core.fock_basis import build_fock_basis

    print("Testing fermion_ops...")

    basis_n = build_fock_basis(2)
    basis_np1 = build_fock_basis(3)
    D = build_creation_matrices(basis_n, basis_np1)

    # Check shape
    assert len(D) == 14
    for D_p in D:
        assert D_p.shape == (len(basis_np1.dets), len(basis_n.dets))

    # Check values are ±1
    for D_p in D:
        assert np.all(np.abs(D_p.data) == 1.0)

    # Test sign convention: D_0|vacuum> = +|0b1>
    basis_0 = build_fock_basis(0)
    basis_1 = build_fock_basis(1)
    D_vac = build_creation_matrices(basis_0, basis_1)
    for p in range(14):
        result = D_vac[p].toarray()
        assert result[p, 0] == 1.0, f"D_{p}|vac> should give +1"

    # Test D_1|0b1> = -|0b11> (one orbital below)
    D_1e = build_creation_matrices(basis_1, basis_n)
    det_0 = 0b1
    col_0 = basis_1.index[det_0]
    det_new = 0b11
    row = basis_n.index[det_new]
    assert D_1e[1].toarray()[row, col_0] == -1.0

    print(f"  ✓ D_p has shape (dim_{{n+1}}, dim_n)")
    print(f"  ✓ Nonzero entries are ±1")
    print(f"  ✓ D_p|vacuum> = +|p> for all p")
    print(f"  ✓ Sign convention correct (D_1|0b1> = -|0b11>)")
    return True


def test_keys():
    """Test cache key generation."""
    import math

    from fexchange.api.keys import energy_key, normalize_shape, quantize_float, shape_key

    print("Testing keys...")

    # Test quantize_float
    assert float(quantize_float(1.0)) == 1.0
    assert float(quantize_float(0.0)) == 0.0
    assert float(quantize_float(1e-15, precision=1e-12)) == 0.0  # zeroed

    # Test normalize_shape
    r4, r6, F2_abs = normalize_shape(100.0, 50.0, 25.0)
    assert abs(r4 - 0.5) < 1e-10
    assert abs(r6 - 0.25) < 1e-10
    assert abs(F2_abs - 100.0) < 1e-10

    # Test F2=0 raises
    try:
        normalize_shape(0.0, 50.0, 25.0)
        assert False, "Should have raised"
    except ValueError:
        pass

    # Test shape_key format
    key = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
    assert "shape|" in key
    assert "n=2" in key

    # Test determinism
    k1 = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
    k2 = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
    assert k1 == k2

    print(f"  ✓ quantize_float works (1.0, 0.0, tiny→0)")
    print(f"  ✓ normalize_shape: (100,50,25)→(0.5,0.25,100)")
    print(f"  ✓ normalize_shape raises on F2=0")
    print(f"  ✓ shape_key contains expected components")
    print(f"  ✓ Keys are deterministic")
    return True


def test_io():
    """Test file I/O."""
    from fexchange.api.io import (
        load_ground_manifold,
        load_state_set,
        read_source_paths_txt,
        save_ground_manifold,
        save_state_set,
        write_source_paths_txt,
    )
    from fexchange.api.types import GroundManifold, StateSet

    print("Testing io...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Test StateSet roundtrip
        V_fock = np.random.randn(10, 5)
        original = StateSet(n=2, basis_id="test", V_fock=V_fock, labels={"L": [1, 2]}, meta={})
        path = tmp_path / "state.npz"
        save_state_set(path, original)
        loaded = load_state_set(path)
        np.testing.assert_array_almost_equal(loaded.V_fock, original.V_fock)

        # Test GroundManifold roundtrip
        G_fock = np.random.randn(10, 3)
        ground = GroundManifold(n=2, basis_id="b", G_fock=G_fock, E0=-100.5, degeneracy=3, tol=1e-8, meta={})
        path = tmp_path / "ground.npz"
        save_ground_manifold(path, ground)
        loaded_g = load_ground_manifold(path)
        np.testing.assert_array_almost_equal(loaded_g.G_fock, G_fock)
        assert abs(loaded_g.E0 - (-100.5)) < 1e-10

        # Test source paths
        paths = ["/path/to/file1.npz"]
        path = tmp_path / "source.txt"
        write_source_paths_txt(path, paths)
        loaded_paths = read_source_paths_txt(path)
        assert len(loaded_paths) == 1

    print(f"  ✓ StateSet save/load roundtrip")
    print(f"  ✓ GroundManifold save/load roundtrip")
    print(f"  ✓ Source paths write/read")
    return True


def test_truncation():
    """Test truncation utilities."""
    from fexchange.sopt.truncation import truncate_by_energy_window, truncate_by_nstates

    print("Testing truncation...")

    E = np.array([2.0, 0.0, 1.0, 3.0])  # unsorted

    # Test truncate_by_nstates
    idx, trunc_id = truncate_by_nstates(E, n_keep=2)
    assert len(idx) == 2
    assert set(idx) == {1, 2}  # indices of E=0 and E=1
    assert "nstates" in trunc_id

    # Test keep all
    idx, trunc_id = truncate_by_nstates(E, n_keep=None)
    assert len(idx) == 4
    assert trunc_id == "all"

    # Test truncate_by_energy_window
    E2 = np.array([0.0, 5.0, 10.0, 15.0])
    idx, trunc_id = truncate_by_energy_window(E2, emax=7.0)
    assert len(idx) == 2
    assert set(idx) == {0, 1}
    assert "emax" in trunc_id

    print(f"  ✓ truncate_by_nstates keeps lowest n")
    print(f"  ✓ truncate_by_nstates with None keeps all")
    print(f"  ✓ truncate_by_energy_window filters by E <= emax")
    return True


def test_heff():
    """Test Heff computation."""
    from fexchange.api.types import ProjectedTransitionsCache
    from fexchange.sopt.heff import compute_heff

    print("Testing heff...")

    d = 2  # ground manifold dimension
    dim_n = 5
    n_np1 = 3
    n_nm1 = 3

    t = np.random.randn(14, 14)
    G_fock = np.random.randn(dim_n, d)

    # Create mock caches
    T_np1 = [np.random.randn(n_np1, dim_n) for _ in range(14)]
    T_nm1 = [np.random.randn(n_nm1, dim_n) for _ in range(14)]
    E_np1 = np.linspace(10, 50, n_np1)
    E_nm1 = np.linspace(5, 30, n_nm1)

    cache_np1 = ProjectedTransitionsCache(
        n=2, basis_id="t", orbital_order_id="t", which="np1",
        src_lsjm_path="", truncation_id="all", T=T_np1, E_mid=E_np1, meta={}
    )
    cache_nm1 = ProjectedTransitionsCache(
        n=2, basis_id="t", orbital_order_id="t", which="nm1",
        src_lsjm_path="", truncation_id="all", T=T_nm1, E_mid=E_nm1, meta={}
    )

    E0 = -100.0
    Heff = compute_heff(t, G_fock, cache_np1, cache_nm1, E0)

    # Check shape
    assert Heff.shape == (d * d, d * d)

    # Check Hermiticity
    np.testing.assert_array_almost_equal(Heff, Heff.conj().T, decimal=10)

    # Test t=0 gives Heff=0
    t_zero = np.zeros((14, 14))
    Heff_zero = compute_heff(t_zero, G_fock, cache_np1, cache_nm1, E0)
    np.testing.assert_array_almost_equal(Heff_zero, np.zeros((d * d, d * d)))

    print(f"  ✓ Heff has shape (d², d²) = ({d*d}, {d*d})")
    print(f"  ✓ Heff is Hermitian")
    print(f"  ✓ Heff=0 when t=0")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("fexchange Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("core/orbitals", test_orbitals),
        ("core/fock_basis", test_fock_basis),
        ("core/fermion_ops", test_fermion_ops),
        ("api/keys", test_keys),
        ("api/io", test_io),
        ("sopt/truncation", test_truncation),
        ("sopt/heff", test_heff),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            if test_func():
                passed += 1
                print(f"  → PASSED")
        except Exception as e:
            failed += 1
            print(f"  → FAILED: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
