"""Tests for sopt/heff.py - Second-order perturbation theory Heff.

Expected behavior:
- compute_heff takes t, G_fock, cache_np1, cache_nm1, E0
- Returns Heff with shape (d*d, d*d) where d is ground manifold dimension
- Heff should be Hermitian (up to numerical precision)
- Energy denominators must be nonzero

Physics formula:
- Aplus_p = T_p^{np1} @ G_fock
- Aminus_q = T_q^{nm1} @ G_fock
- X_{nu,rho}^{cd} = sum_{p,q} t[p,q] * Aplus_p[nu,c] * Aminus_q[rho,d]
- Heff_{cd,ab} = sum_{nu,rho} X^{cd} * conj(X^{ab}) / (E0 - E_{np1}[nu] - E_{nm1}[rho])
"""
from __future__ import annotations

import numpy as np
import pytest

from fexchange.api.types import ProjectedTransitionsCache
from fexchange.sopt.heff import compute_heff


def make_mock_cache(which: str, n_mid: int, dim_n: int) -> ProjectedTransitionsCache:
    """Create a mock ProjectedTransitionsCache for testing."""
    T = [np.random.randn(n_mid, dim_n) for _ in range(14)]
    E_mid = np.linspace(1.0, 10.0, n_mid)  # Positive energies
    return ProjectedTransitionsCache(
        n=2,
        basis_id="test",
        orbital_order_id="test",
        which=which,
        src_lsjm_path="/fake/path",
        truncation_id="all",
        T=T,
        E_mid=E_mid,
        meta={},
    )


class TestComputeHeff:
    """Test Heff computation."""

    def test_output_shape(self) -> None:
        """Heff should have shape (d*d, d*d)."""
        d = 3  # ground manifold dimension
        dim_n = 10  # Fock basis dimension
        n_np1 = 5  # intermediate states for n+1
        n_nm1 = 4  # intermediate states for n-1

        t = np.random.randn(14, 14) + 1j * np.random.randn(14, 14)
        G_fock = np.random.randn(dim_n, d) + 1j * np.random.randn(dim_n, d)

        cache_np1 = make_mock_cache("np1", n_np1, dim_n)
        cache_nm1 = make_mock_cache("nm1", n_nm1, dim_n)
        E0 = -50.0  # Ground state energy (should be below intermediate states)

        Heff = compute_heff(t, G_fock, cache_np1, cache_nm1, E0)

        assert Heff.shape == (d * d, d * d)

    def test_hermiticity(self) -> None:
        """Heff should be Hermitian."""
        d = 2
        dim_n = 5
        n_np1 = 3
        n_nm1 = 3

        # Use Hermitian t for cleaner test
        t_real = np.random.randn(14, 14)
        t = t_real + t_real.T

        G_fock = np.random.randn(dim_n, d)
        cache_np1 = make_mock_cache("np1", n_np1, dim_n)
        cache_nm1 = make_mock_cache("nm1", n_nm1, dim_n)
        E0 = -100.0

        Heff = compute_heff(t, G_fock, cache_np1, cache_nm1, E0)

        # Check Hermiticity
        np.testing.assert_array_almost_equal(Heff, Heff.conj().T, decimal=10)

    def test_wrong_t_shape_raises(self) -> None:
        """t must have shape (14, 14)."""
        d = 2
        dim_n = 5

        t_wrong = np.random.randn(10, 10)  # Wrong shape
        G_fock = np.random.randn(dim_n, d)
        cache_np1 = make_mock_cache("np1", 3, dim_n)
        cache_nm1 = make_mock_cache("nm1", 3, dim_n)

        with pytest.raises(ValueError, match="14"):
            compute_heff(t_wrong, G_fock, cache_np1, cache_nm1, E0=-100.0)

    def test_wrong_which_raises(self) -> None:
        """Cache which must be 'np1' and 'nm1' respectively."""
        d = 2
        dim_n = 5

        t = np.random.randn(14, 14)
        G_fock = np.random.randn(dim_n, d)

        # Both caches have wrong 'which'
        cache_wrong = make_mock_cache("wrong", 3, dim_n)
        cache_nm1 = make_mock_cache("nm1", 3, dim_n)

        with pytest.raises(ValueError):
            compute_heff(t, G_fock, cache_wrong, cache_nm1, E0=-100.0)

    def test_zero_denominator_raises(self) -> None:
        """Zero energy denominators should raise."""
        d = 2
        dim_n = 5

        t = np.random.randn(14, 14)
        G_fock = np.random.randn(dim_n, d)

        cache_np1 = make_mock_cache("np1", 3, dim_n)
        cache_nm1 = make_mock_cache("nm1", 3, dim_n)

        # Set E0 such that E0 - E_np1 - E_nm1 = 0 for some combination
        E0 = cache_np1.E_mid[0] + cache_nm1.E_mid[0]  # This makes denom = 0

        with pytest.raises(ValueError, match="denominator"):
            compute_heff(t, G_fock, cache_np1, cache_nm1, E0)

    def test_complex_input(self) -> None:
        """Complex t and G_fock should work."""
        d = 2
        dim_n = 5

        t = np.random.randn(14, 14) + 1j * np.random.randn(14, 14)
        G_fock = np.random.randn(dim_n, d) + 1j * np.random.randn(dim_n, d)
        cache_np1 = make_mock_cache("np1", 3, dim_n)
        cache_nm1 = make_mock_cache("nm1", 3, dim_n)

        Heff = compute_heff(t, G_fock, cache_np1, cache_nm1, E0=-100.0)

        assert Heff.dtype == np.complex128 or Heff.dtype == np.float64
        assert Heff.shape == (d * d, d * d)


class TestHeffPhysics:
    """Test physical properties of Heff."""

    def test_trivial_case(self) -> None:
        """With t=0, Heff should be zero."""
        d = 2
        dim_n = 5

        t = np.zeros((14, 14))
        G_fock = np.random.randn(dim_n, d)
        cache_np1 = make_mock_cache("np1", 3, dim_n)
        cache_nm1 = make_mock_cache("nm1", 3, dim_n)

        Heff = compute_heff(t, G_fock, cache_np1, cache_nm1, E0=-100.0)

        np.testing.assert_array_almost_equal(Heff, np.zeros((d * d, d * d)))

    def test_energy_scale(self) -> None:
        """Heff magnitude should scale with t^2 / Delta E."""
        d = 2
        dim_n = 5

        # Simple t with one nonzero element
        t = np.zeros((14, 14))
        t[0, 0] = 1.0

        G_fock = np.eye(dim_n)[:, :d]
        cache_np1 = make_mock_cache("np1", 3, dim_n)
        cache_nm1 = make_mock_cache("nm1", 3, dim_n)

        E0 = -100.0
        Heff1 = compute_heff(t, G_fock, cache_np1, cache_nm1, E0)

        # With 2*t, Heff should scale by 4
        t2 = 2.0 * t
        Heff2 = compute_heff(t2, G_fock, cache_np1, cache_nm1, E0)

        # Heff ~ t^2, so Heff2 ~ 4 * Heff1
        np.testing.assert_array_almost_equal(Heff2, 4.0 * Heff1)
