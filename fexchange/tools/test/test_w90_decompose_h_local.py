"""Tests for the standalone h_local decomposition tool."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from w90_decompose_h_local import decompose_h_local
from w90_extract import _build_U_r2c, _expand_to_spinor


def build_test_soc_matrix(zeta):
    """Build zeta * L.S in the internal (dn, up) per-m ordering."""
    ell = 3
    m_vals = np.arange(-ell, ell + 1, dtype=float)
    lz = np.diag(m_vals).astype(complex)
    lp = np.zeros((7, 7), dtype=complex)
    for col, m in enumerate(range(-ell, ell + 1)):
        if m == ell:
            continue
        row = col + 1
        lp[row, col] = np.sqrt(ell * (ell + 1) - m * (m + 1))
    lm = lp.conj().T

    dim = 14
    soc = np.zeros((dim, dim), dtype=complex)
    dn_idx = np.arange(0, dim, 2)
    up_idx = np.arange(1, dim, 2)

    soc[np.ix_(dn_idx, dn_idx)] = -0.5 * zeta * lz
    soc[np.ix_(up_idx, up_idx)] = 0.5 * zeta * lz
    soc[np.ix_(dn_idx, up_idx)] = 0.5 * zeta * lp
    soc[np.ix_(up_idx, dn_idx)] = 0.5 * zeta * lm
    return soc


class TestDecomposeHLocal:
    def test_pure_cef_extracts_zero_soc(self):
        h_cef_real = np.diag(np.linspace(-0.3, 0.3, 7)).astype(complex)
        h_cef_real[1, 4] = 0.17
        h_cef_real[4, 1] = 0.17
        h_cef_real[2, 5] = -0.09
        h_cef_real[5, 2] = -0.09

        U = _build_U_r2c(spinor=False)
        h_cef_complex = U @ h_cef_real @ U.T.conj()
        h_local = np.kron(h_cef_complex, np.eye(2, dtype=complex))

        result = decompose_h_local(h_local)

        assert result["h_cef"].shape == (7, 7)
        np.testing.assert_allclose(result["h_cef"], h_cef_complex, atol=1e-12)
        np.testing.assert_allclose(result["h_cef_real"], h_cef_real.real, atol=1e-12)
        assert abs(result["zeta_fit"]) < 1e-12
        assert result["soc_diag_residual"] == 0.0
        assert result["total_residual"] < 1e-12
        assert result["cef_reality_check"] < 1e-12

    def test_cef_plus_known_soc_recovers_parameters(self):
        zeta = 0.237
        h_cef_real = np.diag(np.array([-0.4, -0.15, 0.05, 0.2, 0.31, 0.48, 0.73]))
        h_cef_real[0, 3] = 0.08
        h_cef_real[3, 0] = 0.08
        h_cef_real[2, 6] = -0.06
        h_cef_real[6, 2] = -0.06

        U = _build_U_r2c(spinor=False)
        h_cef_complex = U @ h_cef_real @ U.T.conj()
        h_soc = build_test_soc_matrix(zeta)
        h_local = np.kron(h_cef_complex, np.eye(2, dtype=complex)) + h_soc

        result = decompose_h_local(h_local)

        np.testing.assert_allclose(result["h_cef"], h_cef_complex, atol=1e-12)
        np.testing.assert_allclose(result["h_cef_real"], h_cef_real.real, atol=1e-12)
        assert result["zeta_fit"] == pytest.approx(zeta, abs=1e-12)
        assert result["soc_diag_residual"] < 1e-12
        assert result["total_residual"] < 1e-12
        assert result["cef_reality_check"] < 1e-12

    def test_without_soc_pipeline_matrix_has_zero_soc(self):
        h_cef_real = np.diag(np.linspace(1.0, 1.6, 7)).astype(complex)
        h_cef_real[1, 4] = 0.12
        h_cef_real[4, 1] = 0.12

        U = _build_U_r2c(spinor=True)
        h_local = U @ _expand_to_spinor(h_cef_real) @ U.T.conj()

        result = decompose_h_local(h_local)

        assert abs(result["zeta_fit"]) < 1e-12
        assert result["total_residual"] < 1e-12
        assert result["cef_reality_check"] < 1e-12
