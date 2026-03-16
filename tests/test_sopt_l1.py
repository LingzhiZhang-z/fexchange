"""Tests for sopt/precompute.py L1 (04-01)."""

import pytest
import numpy as np

from fexchange.sopt.precompute import build_L0, build_L1
from fexchange.spectrum.lsms import build_lsms
from fexchange.spectrum.lsjm import build_lsjm, select_soc_lowest_subspace

_F2 = 1.0
_F4 = 0.5
_F6 = 0.25


class TestL1F1:
    """L1 vertex dimensions for f^1."""

    @pytest.fixture(autouse=True)
    def setup(self):
        n = 1
        l0 = build_L0(n)

        lsms_n = build_lsms(n)
        lsjm_n = build_lsjm(lsms_n)
        soc0 = select_soc_lowest_subspace(lsjm_n, F2=_F2, F4=_F4, F6=_F6)

        lsms_np1 = build_lsms(n + 1)
        lsjm_np1 = build_lsjm(lsms_np1)

        # n-1 = 0 sector has dim 1
        lsms_nm1 = build_lsms(n - 1) if n > 0 else None
        if lsms_nm1:
            lsjm_nm1 = build_lsjm(lsms_nm1)
            U_nm1 = lsjm_nm1["V_fock"]
        else:
            U_nm1 = np.eye(1, dtype=np.complex128)

        self.result = build_L1(l0, lsjm_np1["V_fock"], soc0["U_n_soc0"], U_nm1)
        self.n_j = soc0["n_j"]

    def test_A_shape(self):
        A = self.result["A"]
        assert A.shape[0] == 14  # n_orb
        assert A.shape[2] == self.n_j  # j dimension

    def test_B_shape(self):
        B = self.result["B"]
        assert B.shape[0] == 14
        assert B.shape[1] == self.n_j
