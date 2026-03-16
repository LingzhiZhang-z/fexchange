"""Tests for the independent w90_extract module."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from w90_extract import (
    read_hr,
    _build_U_r2c,
    _expand_to_spinor,
    _permute_ud_to_du,
    w90_extract,
    w90_extract_from_toml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_hr(path, matrices, num_wann):
    """Write a synthetic wannier90_hr.dat."""
    r_vecs = list(matrices.keys())
    lines = [
        "synthetic hr",
        str(num_wann),
        str(len(r_vecs)),
        " ".join(["1"] * len(r_vecs)),
    ]
    for R in r_vecs:
        mat = matrices[R]
        for j in range(num_wann):
            for i in range(num_wann):
                v = mat[i, j]
                lines.append(
                    f"{R[0]:4d}{R[1]:4d}{R[2]:4d}{i+1:4d}{j+1:4d}"
                    f" {v.real: .12f} {v.imag: .12f}"
                )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReadHr:
    def test_roundtrip(self, tmp_path):
        num_wann = 4
        mat0 = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
        write_hr(tmp_path / "hr.dat", {(0, 0, 0): mat0}, num_wann)
        H_R, nw = read_hr(tmp_path / "hr.dat")
        assert nw == 4
        np.testing.assert_allclose(H_R[(0, 0, 0)], mat0, atol=1e-10)


class TestBuildU:
    def test_unitary_7x7(self):
        U = _build_U_r2c(spinor=False)
        assert U.shape == (7, 7)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(7), atol=1e-12)

    def test_unitary_14x14(self):
        U = _build_U_r2c(spinor=True)
        assert U.shape == (14, 14)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(14), atol=1e-12)


class TestExpandToSpinor:
    def test_shape_and_rule(self):
        mat = np.array([[1+2j, 3+0j], [0+1j, 4+0j]])
        out = _expand_to_spinor(mat)
        assert out.shape == (4, 4)
        np.testing.assert_equal(out[1, 1], mat[0, 0])        # spin-up = raw
        np.testing.assert_equal(out[3, 3], mat[1, 1])
        np.testing.assert_equal(out[0, 0], mat[0, 0].conj()) # spin-down = conj
        np.testing.assert_equal(out[2, 2], mat[1, 1].conj())
        np.testing.assert_equal(out[0, 1], 0)                # spin-flip = 0


class TestSpinPermutation:
    def test_ud_to_du_interleaved_swap(self):
        mat_ud = np.diag([10, 20, 30, 40]).astype(complex)
        out = _permute_ud_to_du(mat_ud)
        expected = np.diag([20, 10, 40, 30]).astype(complex)
        np.testing.assert_allclose(out, expected, atol=1e-12)


class TestW90Extract:
    def _make_system(self, tmp_path):
        num_wann = 14
        H0 = np.zeros((num_wann, num_wann), dtype=complex)
        h_ii = np.diag(np.linspace(1.0, 1.6, 7)).astype(complex)
        h_ii[1, 4] = 0.2 - 0.05j; h_ii[4, 1] = 0.2 + 0.05j
        h_jj = np.diag(np.linspace(2.0, 2.6, 7)).astype(complex)
        h_jj[0, 6] = -0.1 + 0.07j; h_jj[6, 0] = -0.1 - 0.07j
        H0[0:7, 0:7] = h_ii
        H0[7:14, 7:14] = h_jj
        H1 = np.zeros((num_wann, num_wann), dtype=complex)
        H1[0:7, 7:14] = np.diag(np.linspace(0.1, 0.7, 7)) + 0.01j
        hr_path = tmp_path / "hr.dat"
        write_hr(hr_path, {(0, 0, 0): H0, (1, 0, 0): H1}, num_wann)
        return hr_path, h_ii, h_jj

    def test_output_shapes(self, tmp_path):
        hr_path, _, _ = self._make_system(tmp_path)
        result = w90_extract(
            hr_path=str(hr_path),
            n_orb_per_atom=[7, 7],
            f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
        )
        assert result["t_mu"].shape == (14, 14)
        assert result["h_local"].shape == (14, 14)

    def test_h_local_hermitian(self, tmp_path):
        hr_path, _, _ = self._make_system(tmp_path)
        result = w90_extract(
            hr_path=str(hr_path),
            n_orb_per_atom=[7, 7],
            f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
        )
        np.testing.assert_allclose(
            result["h_local"], result["h_local"].conj().T, atol=1e-10
        )

    def test_h_local_value(self, tmp_path):
        hr_path, h_ii, h_jj = self._make_system(tmp_path)
        result = w90_extract(
            hr_path=str(hr_path),
            n_orb_per_atom=[7, 7],
            f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
        )
        expected = _expand_to_spinor(0.5 * (h_ii + h_jj))
        U = _build_U_r2c(spinor=True)
        expected = U @ expected @ U.T.conj()
        np.testing.assert_allclose(result["h_local"], expected, atol=1e-10)

    def test_with_soc_reorders_ud_input_to_internal_du(self, tmp_path):
        num_wann = 28
        H0 = np.zeros((num_wann, num_wann), dtype=complex)
        H1 = np.zeros((num_wann, num_wann), dtype=complex)

        h_ii_ud = np.diag(np.arange(1, 15, dtype=float)).astype(complex)
        h_jj_ud = np.diag(np.arange(101, 115, dtype=float)).astype(complex)
        H0[0:14, 0:14] = h_ii_ud
        H0[14:28, 14:28] = h_jj_ud
        H1[0:14, 14:28] = np.diag(np.arange(201, 215, dtype=float)).astype(complex)

        hr_path = tmp_path / "hr_soc.dat"
        write_hr(hr_path, {(0, 0, 0): H0, (1, 0, 0): H1}, num_wann)

        result = w90_extract(
            hr_path=str(hr_path),
            n_orb_per_atom=[14, 14],
            f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
            spinor=True,
        )

        h_local_du = 0.5 * (_permute_ud_to_du(h_ii_ud) + _permute_ud_to_du(h_jj_ud))
        U = _build_U_r2c(spinor=True)
        expected = U @ h_local_du @ U.T.conj()
        np.testing.assert_allclose(result["h_local"], expected, atol=1e-10)

    def test_ligand_correction_without_soc(self, tmp_path):
        """Verify f-p-f second-order correction against hand calculation."""
        num_wann = 17  # 7 f_i + 7 f_j + 3 p
        H0 = np.zeros((num_wann, num_wann), dtype=complex)

        # Onsite energies
        eps_fi = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        eps_fj = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
        eps_p = np.array([5.0, 5.5, 6.0])
        H0[0:7, 0:7] = np.diag(eps_fi)
        H0[7:14, 7:14] = np.diag(eps_fj)
        H0[14:17, 14:17] = np.diag(eps_p)

        # f_i -> p hopping (same cell, in H0)
        t_ip = np.zeros((7, 3), dtype=complex)
        t_ip[0, 0] = 0.3
        t_ip[2, 1] = 0.2 + 0.1j
        H0[0:7, 14:17] = t_ip
        H0[14:17, 0:7] = t_ip.conj().T

        # f_j -> p hopping (f_j at cell [1,0,0], ligand at cell [0,0,0])
        # R_jo = [0,0,0]-[1,0,0] = (-1,0,0), stored via Hermitian pair in R=(1,0,0)
        t_jp = np.zeros((7, 3), dtype=complex)
        t_jp[0, 0] = 0.4
        t_jp[2, 1] = 0.15 - 0.05j

        H1 = np.zeros((num_wann, num_wann), dtype=complex)
        # Direct f-f hopping
        t_ff = np.zeros((7, 7), dtype=complex)
        t_ff[0, 0] = 0.05
        H1[0:7, 7:14] = t_ff
        # p -> f_j block in R=(1,0,0): H1[p, fj] = t_jp^H
        H1[14:17, 7:14] = t_jp.conj().T

        hr_path = tmp_path / "hr_lig.dat"
        write_hr(hr_path, {(0, 0, 0): H0, (1, 0, 0): H1}, num_wann)

        result = w90_extract(
            hr_path=str(hr_path),
            n_orb_per_atom=[7, 7, 3],
            f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
            ligands=[(2, [0, 0, 0])],
        )

        # Hand-compute t_lig
        t_lig = np.zeros((7, 7), dtype=complex)
        for u in range(7):
            for v in range(7):
                for p in range(3):
                    if t_ip[u, p] == 0 or t_jp[v, p] == 0:
                        continue
                    d_u = eps_fi[u] - eps_p[p]
                    d_v = eps_fj[v] - eps_p[p]
                    denom = d_u + d_v
                    if abs(denom) < 1e-12:
                        continue
                    delta = 2.0 * d_u * d_v / denom
                    t_lig[u, v] += t_ip[u, p] * np.conj(t_jp[v, p]) / delta

        t_mu_expected = _expand_to_spinor(t_ff + t_lig)
        U = _build_U_r2c(spinor=True)
        t_mu_expected = U @ t_mu_expected @ U.T.conj()

        np.testing.assert_allclose(result["t_mu"], t_mu_expected, atol=1e-10)

    def test_ligand_correction_with_soc(self, tmp_path):
        """Verify f-p-f correction with SOC uses consistent ordering."""
        num_wann = 34  # 14 f_i + 14 f_j + 6 p (spinor)
        H0 = np.zeros((num_wann, num_wann), dtype=complex)

        # Spinor onsite: interleaved (up, dn) per m
        # Make up/dn energies DIFFERENT to expose ordering bugs
        eps_fi = np.zeros(14)
        eps_fj = np.zeros(14)
        eps_p = np.zeros(6)
        for m in range(7):
            eps_fi[2 * m] = 1.0 + 0.1 * m          # up
            eps_fi[2 * m + 1] = 1.05 + 0.1 * m      # dn (different!)
            eps_fj[2 * m] = 2.0 + 0.1 * m
            eps_fj[2 * m + 1] = 2.05 + 0.1 * m
        for p in range(3):
            eps_p[2 * p] = 5.0 + 0.5 * p
            eps_p[2 * p + 1] = 5.03 + 0.5 * p

        H0[0:14, 0:14] = np.diag(eps_fi)
        H0[14:28, 14:28] = np.diag(eps_fj)
        H0[28:34, 28:34] = np.diag(eps_p)

        # f_i -> p hopping (same cell)
        t_ip = np.zeros((14, 6), dtype=complex)
        t_ip[0, 0] = 0.3     # m0_up -> p0_up
        t_ip[1, 1] = 0.25    # m0_dn -> p0_dn
        H0[0:14, 28:34] = t_ip
        H0[28:34, 0:14] = t_ip.conj().T

        # f_j -> p hopping (via Hermitian pair in R=(1,0,0))
        t_jp = np.zeros((14, 6), dtype=complex)
        t_jp[0, 0] = 0.4     # m0_up -> p0_up
        t_jp[1, 1] = 0.35    # m0_dn -> p0_dn

        H1 = np.zeros((num_wann, num_wann), dtype=complex)
        H1[28:34, 14:28] = t_jp.conj().T  # p -> f_j in R=(1,0,0)

        hr_path = tmp_path / "hr_soc_lig.dat"
        write_hr(hr_path, {(0, 0, 0): H0, (1, 0, 0): H1}, num_wann)

        result = w90_extract(
            hr_path=str(hr_path),
            n_orb_per_atom=[14, 14, 6],
            f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
            ligands=[(2, [0, 0, 0])],
            spinor=True,
        )

        # Hand-compute: all in original ud ordering, then permute at end
        t_lig_ud = np.zeros((14, 14), dtype=complex)
        for u in range(14):
            for v in range(14):
                for p in range(6):
                    if t_ip[u, p] == 0 or t_jp[v, p] == 0:
                        continue
                    d_u = eps_fi[u] - eps_p[p]
                    d_v = eps_fj[v] - eps_p[p]
                    denom = d_u + d_v
                    if abs(denom) < 1e-12:
                        continue
                    delta = 2.0 * d_u * d_v / denom
                    t_lig_ud[u, v] += t_ip[u, p] * np.conj(t_jp[v, p]) / delta

        t_mu_du = _permute_ud_to_du(t_lig_ud)
        U = _build_U_r2c(spinor=True)
        t_mu_expected = U @ t_mu_du @ U.T.conj()

        np.testing.assert_allclose(result["t_mu"], t_mu_expected, atol=1e-10)

    def test_bad_energy_unit(self, tmp_path):
        hr_path, _, _ = self._make_system(tmp_path)
        with pytest.raises(ValueError, match="energy_unit"):
            w90_extract(
                hr_path=str(hr_path),
                n_orb_per_atom=[7, 7],
                f_sites=[(0, [0, 0, 0]), (1, [1, 0, 0])],
                energy_unit="kcal",
            )


class TestTomlEntry:
    def test_from_toml(self, tmp_path):
        num_wann = 14
        H0 = np.diag(np.arange(1, 15, dtype=float)).astype(complex)
        H1 = np.zeros((num_wann, num_wann), dtype=complex)
        H1[0:7, 7:14] = 0.1 * np.eye(7, dtype=complex)
        write_hr(tmp_path / "hr.dat", {(0, 0, 0): H0, (1, 0, 0): H1}, num_wann)

        toml_text = f"""
hr_path = "{tmp_path / 'hr.dat'}"
spinor = false
energy_unit = "eV"
output = "{tmp_path / 'out'}"
n_orb_per_atom = [7, 7]

[f_site_i]
atom = 0
cell = [0, 0, 0]

[f_site_j]
atom = 1
cell = [1, 0, 0]
"""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(toml_text)

        result = w90_extract_from_toml(str(toml_path))
        assert result["t_mu"].shape == (14, 14)
        assert (tmp_path / "out_t_mu_real.txt").exists()
        assert (tmp_path / "out_t_mu_imag.txt").exists()
        assert (tmp_path / "out_h_local_real.txt").exists()
        assert (tmp_path / "out_h_local_imag.txt").exists()

        t_mu_real = np.loadtxt(tmp_path / "out_t_mu_real.txt")
        np.testing.assert_allclose(t_mu_real, result["t_mu"].real, atol=1e-10)
