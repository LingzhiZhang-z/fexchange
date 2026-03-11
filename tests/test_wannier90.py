"""Tests for io/wannier90.py (05-03) — smoke tests."""

import pytest
import numpy as np

from fexchange.io.wannier90 import build_U_r2c, apply_basis_transform
from fexchange.utils.checks import check_unitary


class TestU_r2c:
    def test_non_spinor_unitary(self):
        U = build_U_r2c(spinor=False)
        assert U.shape == (7, 7)
        check_unitary(U, label="U_r2c")

    def test_spinor_unitary(self):
        U = build_U_r2c(spinor=True)
        assert U.shape == (14, 14)
        check_unitary(U, label="U_r2c_spinor")

    def test_basis_transform_preserves_trace(self):
        """Basis transform should preserve matrix trace."""
        rng = np.random.default_rng(42)
        h = rng.random((7, 7)) + 1j * rng.random((7, 7))
        h = (h + h.conj().T) / 2
        U = build_U_r2c(spinor=False)
        h_new = apply_basis_transform(h, U)
        np.testing.assert_allclose(np.trace(h_new), np.trace(h), atol=1e-10)
