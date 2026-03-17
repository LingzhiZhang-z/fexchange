"""Tests for sopt/spin12.py (04-03)."""

import pytest
import numpy as np

from fexchange.sopt.spin12 import spin12_map


class TestSpin12:
    def test_identity_input(self):
        """Identity Heff should give zero exchange."""
        Heff = np.eye(4, dtype=np.complex128)
        result = spin12_map(Heff)
        assert set(result.keys()) == {"J_mu", "mapping_residual"}
        np.testing.assert_allclose(result["J_mu"], 0, atol=1e-10)

    def test_heisenberg(self):
        """Pure Heisenberg: H = J S_i · S_j = (J/4)(sigma_x x sigma_x + y + z)."""
        J_val = 2.0
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        H = (J_val / 4) * (np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
        result = spin12_map(H)
        np.testing.assert_allclose(result["J_mu"], np.eye(3) * J_val, atol=1e-10)
        assert result["mapping_residual"] < 1e-10

    def test_reconstruction_residual(self):
        """Random Hermitian 4x4 should reconstruct with small residual."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        H = (A + A.conj().T) / 2
        result = spin12_map(H)
        assert result["mapping_residual"] < 1e-10
