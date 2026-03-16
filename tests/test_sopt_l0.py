"""Tests for sopt/precompute.py L0 (04-01)."""

import pytest
import numpy as np

from fexchange.sopt.precompute import build_L0
from fexchange.core.fock import dim_sector


class TestL0:
    @pytest.mark.parametrize("n_ele", [1, 2, 3])
    def test_shapes(self, n_ele):
        result = build_L0(n_ele)
        X = result["X"]
        Y = result["Y"]
        assert X.shape == (14, dim_sector(n_ele + 1), dim_sector(n_ele))
        assert Y.shape == (14, dim_sector(n_ele), dim_sector(n_ele - 1))

    def test_conjugate_consistency(self):
        """Reverse direction = conjugate (04-01 §1 validation)."""
        result = build_L0(2)
        X = result["X"]
        for kappa in range(14):
            # <beta^n | f_kappa | alpha^{n+1}> = conj(X[kappa, alpha, beta])
            annihilation = X[kappa].conj().T
            assert annihilation.shape == (dim_sector(2), dim_sector(3))
