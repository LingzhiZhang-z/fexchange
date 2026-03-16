"""Tests for non-Kramers analytical basis builder."""

import numpy as np
import pytest

from fexchange.spectrum.doublet import (
    allowed_m_values,
    build_d3d_eg_doublet,
    build_oh_eg_doublet,
    cos_basis_vector,
    project_to_eg_doublet,
)
from fexchange.core.space_j import build_time_reversal_operator
from tests.conftest import random_u2


class TestAllowedMValues:
    """Verify allowed m-sector listings."""

    def test_d3d_mod3_res1_j4(self):
        assert allowed_m_values(4, 3, 1) == [4, 1, -2]

    def test_d3d_mod3_res1_j6(self):
        assert allowed_m_values(6, 3, 1) == [4, 1, -2, -5]

    def test_d3d_mod3_res1_j8(self):
        assert allowed_m_values(8, 3, 1) == [7, 4, 1, -2, -5, -8]

    def test_oh_mod4_res0_j4(self):
        assert allowed_m_values(4, 4, 0) == [4, 0, -4]

    def test_oh_mod4_res0_j6(self):
        assert allowed_m_values(6, 4, 0) == [4, 0, -4]

    def test_oh_mod4_res0_j8(self):
        assert allowed_m_values(8, 4, 0) == [8, 4, 0, -4, -8]

    def test_oh_mod4_res2_j4(self):
        assert allowed_m_values(4, 4, 2) == [2, -2]

    def test_oh_mod4_res2_j6(self):
        assert allowed_m_values(6, 4, 2) == [6, 2, -2, -6]

    def test_oh_mod4_res2_j8(self):
        assert allowed_m_values(8, 4, 2) == [6, 2, -2, -6]


class TestCosBasisVector:
    """Verify C_m = (|m> + |-m>) / sqrt(2) construction."""

    def test_m0_is_ket_zero(self):
        v = cos_basis_vector(4, 0)
        assert v.shape == (9,)
        expected = np.zeros(9)
        expected[4] = 1.0
        np.testing.assert_allclose(v, expected)

    def test_m4_j4_real_normalized(self):
        v = cos_basis_vector(4, 4)
        assert v.shape == (9,)
        np.testing.assert_allclose(v.imag, 0.0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0)
        scale = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(v[0], scale)
        np.testing.assert_allclose(v[8], scale)

    def test_m2_j6_real_normalized(self):
        v = cos_basis_vector(6, 2)
        assert v.shape == (13,)
        np.testing.assert_allclose(v.imag, 0.0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0)
        scale = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(v[4], scale)
        np.testing.assert_allclose(v[8], scale)

    def test_negative_m_raises(self):
        with pytest.raises(ValueError):
            cos_basis_vector(4, -2)


class TestBuildOhEgDoublet:
    @pytest.mark.parametrize(
        ("J", "coeffs_u", "coeffs_v"),
        [
            (4, [1.0, 1.0], [1.0]),
            (6, [1.0, 1.0], [1.0, 1.0]),
            (8, [1.0, 1.0, 1.0], [1.0, 1.0]),
        ],
    )
    def test_columns_are_real(self, J, coeffs_u, coeffs_v):
        psi = build_oh_eg_doublet(J, coeffs_u, coeffs_v)
        assert psi.shape == (int(2 * J + 1), 2)
        np.testing.assert_allclose(psi.imag, 0.0, atol=1e-15)

    @pytest.mark.parametrize(
        ("J", "coeffs_u", "coeffs_v"),
        [
            (4, [1.0, 1.0], [1.0]),
            (6, [1.0, 1.0], [1.0, 1.0]),
            (8, [1.0, 1.0, 1.0], [1.0, 1.0]),
        ],
    )
    def test_orthonormal(self, J, coeffs_u, coeffs_v):
        psi = build_oh_eg_doublet(J, coeffs_u, coeffs_v)
        np.testing.assert_allclose(psi.conj().T @ psi, np.eye(2), atol=1e-14)

    def test_sector_purity_j4(self):
        psi = build_oh_eg_doublet(4, [1.0, 1.0], [1.0])
        mask_0mod4 = np.array([True, False, False, False, True, False, False, False, True])
        mask_2mod4 = np.array([False, False, True, False, False, False, True, False, False])
        np.testing.assert_allclose(psi[~mask_0mod4, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(psi[~mask_2mod4, 1], 0.0, atol=1e-15)

    def test_j4_matches_known_fixture(self):
        psi = build_oh_eg_doublet(4, [0.0, 1.0], [1.0])
        scale = 1.0 / np.sqrt(2.0)
        np.testing.assert_allclose(psi[0, 0], scale, atol=1e-15)
        np.testing.assert_allclose(psi[8, 0], scale, atol=1e-15)
        np.testing.assert_allclose(psi[2, 1], scale, atol=1e-15)
        np.testing.assert_allclose(psi[6, 1], scale, atol=1e-15)

    def test_unsupported_j_raises(self):
        with pytest.raises(ValueError):
            build_oh_eg_doublet(10, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    def test_wrong_u_length_raises(self):
        with pytest.raises(ValueError):
            build_oh_eg_doublet(4, [1.0], [1.0])

    def test_zero_norm_column_raises(self):
        with pytest.raises(ValueError):
            build_oh_eg_doublet(4, [0.0, 0.0], [1.0])


class TestBuildD3dEgDoublet:
    @pytest.mark.parametrize(("J", "n_coeffs"), [(4, 3), (6, 4), (8, 6)])
    def test_orthonormal(self, J, n_coeffs):
        coeffs = [1.0] * n_coeffs
        psi = build_d3d_eg_doublet(J, coeffs)
        np.testing.assert_allclose(psi.conj().T @ psi, np.eye(2), atol=1e-14)

    @pytest.mark.parametrize(("J", "n_coeffs"), [(4, 3), (6, 4), (8, 6)])
    def test_second_column_is_time_reversal_partner(self, J, n_coeffs):
        coeffs = [1.0] * n_coeffs
        psi = build_d3d_eg_doublet(J, coeffs)
        U_T = build_time_reversal_operator(J)
        expected = U_T @ psi[:, 0].conj()
        np.testing.assert_allclose(psi[:, 1], expected, atol=1e-14)

    def test_sector_purity_j4(self):
        psi = build_d3d_eg_doublet(4, [1.0, 1.0, 1.0])
        mask_p1 = np.zeros(9, dtype=bool)
        mask_p1[[2, 5, 8]] = True
        np.testing.assert_allclose(np.abs(psi[~mask_p1, 0]), 0.0, atol=1e-15)

        mask_m1 = np.zeros(9, dtype=bool)
        mask_m1[[0, 3, 6]] = True
        np.testing.assert_allclose(np.abs(psi[~mask_m1, 1]), 0.0, atol=1e-15)

    def test_j4_matches_known_fixture(self):
        psi = build_d3d_eg_doublet(4, [1.0, 1.0, 1.0])
        scale = 1.0 / np.sqrt(3.0)
        np.testing.assert_allclose(np.abs(psi[2, 0]), scale, atol=1e-14)
        np.testing.assert_allclose(np.abs(psi[5, 0]), scale, atol=1e-14)
        np.testing.assert_allclose(np.abs(psi[8, 0]), scale, atol=1e-14)

    def test_unsupported_j_raises(self):
        with pytest.raises(ValueError):
            build_d3d_eg_doublet(10, [1.0] * 7)

    def test_wrong_coeff_length_raises(self):
        with pytest.raises(ValueError):
            build_d3d_eg_doublet(4, [1.0, 1.0])

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError):
            build_d3d_eg_doublet(4, [0.0, 0.0, 0.0])

class TestProjectToEgDoublet:
    @pytest.mark.parametrize("J", [4, 6, 8])
    def test_oh_roundtrip(self, J):
        n_u = len(set(abs(m) for m in allowed_m_values(J, 4, 0)))
        n_v = len(set(abs(m) for m in allowed_m_values(J, 4, 2)))
        psi_ref = build_oh_eg_doublet(J, [1.0] * n_u, [1.0] * n_v)
        psi_scrambled = psi_ref @ random_u2(seed=42)
        psi_out = project_to_eg_doublet(J, psi_scrambled, "Oh")
        np.testing.assert_allclose(psi_out.imag, 0.0, atol=1e-10)
        np.testing.assert_allclose(psi_out.conj().T @ psi_out, np.eye(2), atol=1e-10)

    @pytest.mark.parametrize(("J", "n_coeffs"), [(4, 3), (6, 4), (8, 6)])
    def test_d3d_roundtrip(self, J, n_coeffs):
        psi_ref = build_d3d_eg_doublet(J, [1.0] * n_coeffs)
        psi_scrambled = psi_ref @ random_u2(seed=77)
        psi_out = project_to_eg_doublet(J, psi_scrambled, "D3d")
        U_T = build_time_reversal_operator(J)
        np.testing.assert_allclose(psi_out[:, 1], U_T @ psi_out[:, 0].conj(), atol=1e-10)
        np.testing.assert_allclose(psi_out.conj().T @ psi_out, np.eye(2), atol=1e-10)

    @pytest.mark.parametrize(("J", "n_coeffs"), [(4, 3), (6, 4), (8, 6)])
    def test_c3v_routes_to_d3d_roundtrip(self, J, n_coeffs):
        psi_ref = build_d3d_eg_doublet(J, [1.0] * n_coeffs)
        psi_scrambled = psi_ref @ random_u2(seed=91)
        psi_out = project_to_eg_doublet(J, psi_scrambled, "C3v")
        U_T = build_time_reversal_operator(J)
        np.testing.assert_allclose(psi_out[:, 1], U_T @ psi_out[:, 0].conj(), atol=1e-10)
        np.testing.assert_allclose(psi_out.conj().T @ psi_out, np.eye(2), atol=1e-10)
