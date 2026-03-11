"""Tests for models/hsoc.py (02-02)."""

import pytest
import numpy as np

from fexchange.models.hsoc import build_hsoc_matrix, build_hsoc_unit_matrix
from fexchange.utils.checks import check_hermitian


class TestHsocHermiticity:
    @pytest.mark.parametrize("n_ele", [1, 2, 3])
    def test_hermitian(self, n_ele):
        H = build_hsoc_matrix(n_ele, zeta=1.0)
        check_hermitian(H, label="H_soc")

    def test_zeta_scaling(self):
        """H_soc(zeta) = zeta * O_soc (02-02 §5)."""
        O = build_hsoc_unit_matrix(2)
        H = build_hsoc_matrix(2, zeta=2.5)
        np.testing.assert_allclose(H, 2.5 * O, atol=1e-14)
