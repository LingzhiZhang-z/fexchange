"""Tests for models/hcef.py (02-03)."""

import pytest
import numpy as np

from fexchange.core.stevens import build_cef_stevens_operators
from fexchange.models.hcef import build_hcef_matrix_J
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError


class TestStevensOperators:
    """All Stevens operators must be Hermitian (02-03 §6.4 validation)."""

    @pytest.mark.parametrize("J", [3.5, 4.0, 6.0])
    def test_oh_hermitian(self, J):
        ops = build_cef_stevens_operators(J, "Oh")
        for name, op in ops.items():
            check_hermitian(op, label=f"Stevens_{name}_J{J}")

    @pytest.mark.parametrize("J", [3.5, 4.0])
    def test_c3v_hermitian(self, J):
        ops = build_cef_stevens_operators(J, "C3v", "cos")
        for name, op in ops.items():
            check_hermitian(op, label=f"Stevens_{name}_J{J}_C3v")

    def test_o20_diagonal(self):
        """O_2^0 must be diagonal in |J,M> basis (02-03 §6.2)."""
        ops = build_cef_stevens_operators(4.0, "Oh")
        O20 = ops["O20"]
        off_diag = O20 - np.diag(np.diag(O20))
        assert np.linalg.norm(off_diag) < 1e-12

    def test_dimension(self):
        ops = build_cef_stevens_operators(3.5, "Oh")
        assert ops["O20"].shape == (8, 8)  # 2*3.5+1 = 8

    def test_j_small_float_deviation_auto_fix(self):
        ops = build_cef_stevens_operators(3.5 + 1e-10, "Oh")
        assert ops["O20"].shape == (8, 8)

    def test_j_large_float_deviation_fail(self):
        with pytest.raises(PhysError):
            build_cef_stevens_operators(3.5001, "Oh")


class TestHcefMatrix:
    def test_oh_hermitian(self):
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
        check_hermitian(H, label="H_cef_Oh")

    def test_c3v_hermitian(self):
        params = {"B20": 0.1, "B40": 0.01, "B60": 0.001, "B66": 0.0001, "B43": 0.01, "B63": 0.001}
        H = build_hcef_matrix_J(4.0, params, "C3v", "cos")
        check_hermitian(H, label="H_cef_C3v")
