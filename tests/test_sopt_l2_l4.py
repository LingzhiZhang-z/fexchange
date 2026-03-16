"""Tests for sopt/contraction.py L2-L4 (04-02)."""

import pytest
import numpy as np

from fexchange.sopt.contraction import build_L2, build_L3, build_L4
from fexchange.sopt.precompute import build_L0, build_L1
from fexchange.spectrum.lsms import build_lsms
from fexchange.spectrum.lsjm import build_lsjm, select_soc_lowest_subspace
from fexchange.utils.checks import check_hermitian

_F2 = 1.0
_F4 = 0.5
_F6 = 0.25


class TestZeroHop:
    """Zero-hopping sanity: t=0 => Heff=0 (04-00 §6 validation)."""

    def test_zero_hop_f1(self):
        n = 1
        l0 = build_L0(n)

        lsms_n = build_lsms(n)
        lsjm_n = build_lsjm(lsms_n)
        soc0 = select_soc_lowest_subspace(lsjm_n, F2=_F2, F4=_F4, F6=_F6)

        lsms_np1 = build_lsms(n + 1)
        lsjm_np1 = build_lsjm(lsms_np1)

        # f^0 sector: dim=1
        U_nm1 = np.ones((1, 1), dtype=np.complex128)

        l1 = build_L1(l0, lsjm_np1["V_fock"], soc0["U_n_soc0"], U_nm1)

        # Zero hopping
        t_mu = np.zeros((14, 14), dtype=np.complex128)
        l2 = build_L2(l1, t_mu)

        assert np.allclose(l2["M_A"], 0)
        assert np.allclose(l2["M_B"], 0)

    def test_nonzero_hop_does_not_crash(self):
        """Regression: non-zero hopping must not trigger einsum ValueError."""
        n = 1
        l0 = build_L0(n)

        lsms_n = build_lsms(n)
        lsjm_n = build_lsjm(lsms_n)
        soc0 = select_soc_lowest_subspace(lsjm_n, F2=_F2, F4=_F4, F6=_F6)

        lsms_np1 = build_lsms(n + 1)
        lsjm_np1 = build_lsjm(lsms_np1)
        U_nm1 = np.ones((1, 1), dtype=np.complex128)

        l1 = build_L1(l0, lsjm_np1["V_fock"], soc0["U_n_soc0"], U_nm1)

        t_mu = np.zeros((14, 14), dtype=np.complex128)
        t_mu[0, 0] = 1.0
        l2 = build_L2(l1, t_mu)

        assert l2["M_A"].shape == (l1["n_u"], l1["n_v"], l1["n_j"], l1["n_j"])
        assert l2["M_B"].shape == (l1["n_v"], l1["n_u"], l1["n_j"], l1["n_j"])
        assert np.isfinite(l2["M_A"]).all()
        assert np.isfinite(l2["M_B"]).all()
