"""Tests for io/wannier90.py (05-03) — smoke tests."""

from pathlib import Path

import numpy as np

from fexchange.io.wannier90 import (
    _expand_without_soc_to_spinor,
    apply_basis_transform,
    build_U_r2c,
    w90_extract,
)
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


def _write_win(path: Path, *, num_wann: int) -> None:
    path.write_text(
        "\n".join(
            [
                f"num_wann = {num_wann}",
                "begin projections",
                "A:l=3",
                "B:l=3",
                "O:l=1",
                "end projections",
                "begin atoms_frac",
                "A 0.0 0.0 0.0",
                "B 0.5 0.0 0.0",
                "O 0.25 0.0 0.0",
                "end atoms_frac",
                "begin unit_cell_cart",
                "ang",
                "1.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "end unit_cell_cart",
            ]
        ),
        encoding="utf-8",
    )


def _write_hr(path: Path, matrices: dict[tuple[int, int, int], np.ndarray]) -> None:
    r_vectors = list(matrices.keys())
    num_wann = next(iter(matrices.values())).shape[0]
    lines = ["synthetic hr", str(num_wann), str(len(r_vectors)), " ".join(["1"] * len(r_vectors))]
    for R in r_vectors:
        mat = matrices[R]
        for jj in range(num_wann):
            for ii in range(num_wann):
                value = mat[ii, jj]
                lines.append(
                    f"{R[0]:4d}{R[1]:4d}{R[2]:4d}{ii + 1:4d}{jj + 1:4d}"
                    f" {value.real: .12f} {value.imag: .12f}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_w90_extract_returns_t_mu_h_local_and_meta(tmp_path: Path):
    num_wann = 21
    h0 = np.zeros((num_wann, num_wann), dtype=np.complex128)
    hij = np.zeros((num_wann, num_wann), dtype=np.complex128)

    h_ii = np.diag(np.linspace(1.0, 1.6, 7)).astype(np.complex128)
    h_ii[1, 4] = 0.2 - 0.05j
    h_ii[4, 1] = 0.2 + 0.05j
    h_jj = np.diag(np.linspace(2.0, 2.6, 7)).astype(np.complex128)
    h_jj[0, 6] = -0.1 + 0.07j
    h_jj[6, 0] = -0.1 - 0.07j
    h0[0:7, 0:7] = h_ii
    h0[7:14, 7:14] = h_jj
    h0[14:21, 14:21] = np.diag(np.linspace(-3.0, -2.4, 7))

    hij[0:7, 7:14] = np.diag(np.linspace(0.1, 0.7, 7)) + 0.01j

    hr_path = tmp_path / "wannier90_hr.dat"
    win_path = tmp_path / "wannier.win"
    _write_hr(hr_path, {(0, 0, 0): h0, (1, 0, 0): hij})
    _write_win(win_path, num_wann=num_wann)

    cfg = {
        "soc_mode": "without_soc",
        "spin_completion_rule": "up_raw_down_conj_zero_flip_v1",
        "hr_path": str(hr_path),
        "win_path": str(win_path),
        "orbital_basis": "real_harmonic_default_w90",
        "orbital_order_id": "w90_f_default_v1",
        "energy_unit": "eV",
        "f_site_i": 0,
        "f_site_j": 1,
        "f_site_i_cell": [0, 0, 0],
        "f_site_j_cell": [1, 0, 0],
        "ligand_indices": [],
        "ligand_cells": [],
        "all_wannier_atom_indices": [0, 1, 2],
        "delta_mode": "from_onsite",
        "delta_reduction": "channelwise",
    }

    result = w90_extract(cfg)

    assert set(result) == {"t_mu", "h_local", "meta"}
    assert result["t_mu"].shape == (14, 14)
    assert result["h_local"].shape == (14, 14)
    np.testing.assert_allclose(result["h_local"], result["h_local"].conj().T, atol=1e-10)

    h_local_raw = 0.5 * (h_ii + h_jj)
    expected = apply_basis_transform(
        _expand_without_soc_to_spinor(h_local_raw),
        build_U_r2c("w90_f_default_v1", spinor=True),
    )
    np.testing.assert_allclose(result["h_local"], expected, atol=1e-10)
