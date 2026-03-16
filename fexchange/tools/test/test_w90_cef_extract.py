"""Tests for CEF extraction helpers in w90_decompose_h_local."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j


TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS_DIR))

from w90_decompose_h_local import (  # noqa: E402
    decompose_orbital_tensors,
    extract_cef_stevens_fit,
)


def _reference_ckq_matrix(ell: int, k: int, q: int) -> np.ndarray:
    dim = 2 * ell + 1
    matrix = np.zeros((dim, dim), dtype=complex)
    red = ((-1) ** ell) * (2 * ell + 1) * float(wigner_3j(ell, k, ell, 0, 0, 0))
    m_vals = list(range(-ell, ell + 1))
    for row, mp in enumerate(m_vals):
        for col, m in enumerate(m_vals):
            three_j = float(wigner_3j(ell, k, ell, -mp, q, m))
            matrix[row, col] = ((-1) ** (ell - mp)) * three_j * red
    return matrix


def test_decompose_orbital_tensors_recovers_known_coefficients():
    ell = 3
    trace_avg = 1.75
    coeffs = {
        (2, 0): 0.12,
        (4, 2): -0.05 + 0.03j,
        (4, -2): -0.05 - 0.03j,
        (6, 4): 0.08,
        (6, -4): 0.08,
    }
    h_cef = trace_avg * np.eye(2 * ell + 1, dtype=complex)
    for key, value in coeffs.items():
        h_cef += value * _reference_ckq_matrix(ell, *key)

    result = decompose_orbital_tensors(h_cef, ell=ell)

    assert result["trace_avg"] == pytest.approx(trace_avg, abs=1e-12)
    assert result["residual"] < 1e-12
    for key, value in coeffs.items():
        assert result["a_kq"][key] == pytest.approx(value, abs=1e-11)


def test_extract_cef_stevens_fit_trace_only_matrix_returns_zero_parameters():
    h_cef = 2.5 * np.eye(7, dtype=complex)

    result = extract_cef_stevens_fit(
        h_cef,
        n_ele=13,
        zeta=0.24,
        symmetry="Oh",
    )

    assert result["J0"] == pytest.approx(3.5)
    assert result["alpha0"] == 0
    assert result["L0"] == 3
    assert result["S0"] == pytest.approx(0.5)
    assert 0.0 <= result["j0_weight"] <= 1.0
    assert result["stevens_fit_residual"] < 1e-12
    assert result["B_params"]["B4"] == pytest.approx(0.0, abs=1e-12)
    assert result["B_params"]["B6"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(result["cef_splitting_meV"], np.zeros(8), atol=1e-12)


def test_cli_reports_scheme_a_only(tmp_path: Path):
    h_cef = np.diag(np.linspace(-0.3, 0.3, 7)).astype(complex)
    h_local = np.kron(h_cef, np.eye(2, dtype=complex))
    h_local_path = tmp_path / "h_local.dat"
    np.savetxt(h_local_path, h_local, fmt="%.12f")

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "w90_decompose_h_local.py"),
            "--h_local_dat",
            str(h_local_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "=== SOC: zeta =" in proc.stdout
    assert "=== Scheme A:" in proc.stdout
    assert "=== Scheme B:" not in proc.stdout


def test_cli_reports_scheme_b_when_many_body_inputs_are_provided(tmp_path: Path):
    h_cef = np.diag(np.linspace(-0.2, 0.4, 7)).astype(complex)
    h_local = np.kron(h_cef, np.eye(2, dtype=complex))
    h_local_path = tmp_path / "h_local.dat"
    np.savetxt(h_local_path, h_local, fmt="%.12f")

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "w90_decompose_h_local.py"),
            "--h_local_dat",
            str(h_local_path),
            "--n_ele",
            "13",
            "--zeta_soc",
            "0.24",
            "--symmetry",
            "Oh",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "=== Scheme B: Stevens fit (Oh) ===" in proc.stdout
    assert "ground: alpha=0, L=3, S=0.5, J0=3.5" in proc.stdout
