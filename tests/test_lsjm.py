"""Tests for representations/lsjm.py (03-01)."""

import pytest
import numpy as np

from fexchange.representations.lsms import build_lsms
from fexchange.representations.lsjm import build_lsjm, select_soc_lowest_subspace
from fexchange.utils.checks import check_orthonormal

_F2 = 1.0
_F4 = 0.5
_F6 = 0.25


class TestLsjmF1:
    """f^1: LSJM should have 14 states (same space, J-coupled)."""

    @pytest.fixture(autouse=True)
    def build(self):
        lsms = build_lsms(1)
        self.result = build_lsjm(lsms)

    def test_total_states(self):
        assert self.result["V_fock"].shape[1] == 14

    def test_orthonormality(self):
        check_orthonormal(self.result["V_fock"], label="V_lsjm_f1")

    def test_coef_zeta(self):
        """For f^1, coef_zeta should follow J(J+1)-L(L+1)-S(S+1) / 2."""
        labels = self.result["labels"]
        coef_z = self.result["coef_zeta"]
        for idx, lab in enumerate(labels):
            L, S, J = lab["L"], lab["twoS"] / 2.0, lab["J"]
            expected = 0.5 * (J * (J + 1) - L * (L + 1) - S * (S + 1))
            np.testing.assert_allclose(
                coef_z[idx], expected, atol=1e-8,
                err_msg=f"coef_zeta mismatch at state {idx}"
            )

    def test_soc_lowest_subspace(self):
        """For f^1 (n<=2l), J0 = |L-S| = |3 - 0.5| = 2.5, n_j = 6."""
        lsms = build_lsms(1)
        lsjm = build_lsjm(lsms)
        soc0 = select_soc_lowest_subspace(lsjm, F2=_F2, F4=_F4, F6=_F6)
        assert soc0["J0"] == 2.5
        assert soc0["n_j"] == 6


class TestLsjmF2:
    """f^2: LSJM should have 91 states."""

    def test_total_states(self):
        lsms = build_lsms(2)
        lsjm = build_lsjm(lsms)
        assert lsjm["V_fock"].shape[1] == 91
        check_orthonormal(lsjm["V_fock"], label="V_lsjm_f2")


class TestSocLowestSelection:
    def test_uses_full_coulomb_energy(self):
        """Ground LS term selection must use F2/F4/F6, not coef_F2 alone."""
        labels = []
        # Term A: (alpha=0, L=4, twoS=0, J=4), M=-4..4 (9 states)
        for m in range(-4, 5):
            labels.append({"alpha": 0, "L": 4, "twoS": 0, "J": 4.0, "M": float(m)})
        # Term B: (alpha=0, L=5, twoS=2, J=4), M=-4..4 (9 states)
        for m in range(-4, 5):
            labels.append({"alpha": 0, "L": 5, "twoS": 2, "J": 4.0, "M": float(m)})

        # F2-only would choose term A (1 < 2); full energy with F4=1 chooses term B.
        coef_F2 = np.array([1.0] * 9 + [2.0] * 9)
        coef_F4 = np.array([0.0] * 9 + [-10.0] * 9)
        coef_F6 = np.zeros(18)

        lsjm_result = {
            "labels": labels,
            "n_ele": 2,
            "V_fock": np.eye(18, dtype=np.complex128),
            "coef_F2": coef_F2,
            "coef_F4": coef_F4,
            "coef_F6": coef_F6,
        }

        soc0 = select_soc_lowest_subspace(lsjm_result, F2=1.0, F4=1.0, F6=0.0)
        assert soc0["L0"] == 5
        assert soc0["S0"] == 1.0
        assert soc0["n_j"] == 9
