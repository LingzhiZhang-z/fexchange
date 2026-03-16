"""Tests for non-Kramers doublet gauge-fixing functions."""
import numpy as np

from fexchange.spectrum.multipole import analyze_multipole_carrying
from fexchange.spectrum.doublet import (
    build_d3d_eg_doublet,
    build_oh_eg_doublet,
    gauge_fix_non_kramers_d3d,
    gauge_fix_non_kramers_oh,
)
from fexchange.core.space_j import build_time_reversal_operator, pauli_decompose
from fexchange.core.stevens import build_multipole_operators
from tests.conftest import random_u2


def _make_oh_j4_doublet():
    """Canonical J=4 Oh Gamma_3 doublet."""
    J = 4.0
    Psi = build_oh_eg_doublet(J, [0.0, 1.0], [1.0])
    return Psi.astype(np.complex128), J


# ---------------------------------------------------------------------------
# Oh gauge tests
# ---------------------------------------------------------------------------


def test_gauge_fix_oh_output_columns_are_real():
    """Both output columns must be real (TR-eigenstate property for cubic CEF)."""
    Psi, _ = _make_oh_j4_doublet()
    basis = gauge_fix_non_kramers_oh(Psi, tol=1e-10)
    np.testing.assert_allclose(basis.imag, 0.0, atol=1e-12)


def test_gauge_fix_oh_even_operator_has_zero_oy():
    """TR-even operator (Q_v = O22c) must project to sigma_x/z only: oy=0."""
    Psi, J = _make_oh_j4_doublet()
    basis = gauge_fix_non_kramers_oh(Psi, tol=1e-10)
    Q_v = build_multipole_operators(J, "electric_quadrupole")["O22c"]
    M = basis.conj().T @ Q_v @ basis
    coeffs = pauli_decompose(M)
    assert abs(complex(coeffs["oy"])) < 1e-10


def test_gauge_fix_oh_odd_operator_has_zero_ox_oz():
    """TR-odd operator (O33s octupole) must project to sigma_y only: ox=oz=0."""
    Psi, J = _make_oh_j4_doublet()
    basis = gauge_fix_non_kramers_oh(Psi, tol=1e-10)
    T_odd = build_multipole_operators(J, "magnetic_octupole")["O33s"]
    M = basis.conj().T @ T_odd @ basis
    coeffs = pauli_decompose(M)
    assert abs(complex(coeffs["ox"])) < 1e-10
    assert abs(complex(coeffs["oz"])) < 1e-10


def test_gauge_fix_oh_is_gauge_invariant():
    """Output must be identical regardless of input U(2) rotation."""
    Psi, _ = _make_oh_j4_doublet()
    Psi_rot = Psi @ random_u2(seed=7)
    basis1 = gauge_fix_non_kramers_oh(Psi, tol=1e-10)
    basis2 = gauge_fix_non_kramers_oh(Psi_rot, tol=1e-10)
    np.testing.assert_allclose(basis1, basis2, atol=1e-10)


def test_gauge_fix_oh_each_column_is_tr_eigenstate():
    """Each output column must satisfy U_T @ col.conj() = +/- col (TR eigenstate)."""
    Psi, J = _make_oh_j4_doublet()
    U_T = build_time_reversal_operator(J)
    basis = gauge_fix_non_kramers_oh(Psi, tol=1e-10)
    for i in range(2):
        col = basis[:, i]
        partner = U_T @ col.conj()
        for sign in (+1, -1):
            if np.linalg.norm(partner - sign * col) < 1e-10:
                break
        else:
            raise AssertionError(
                f"Column {i} is not a TR eigenstate: ||U_T col* - col|| = "
                f"{min(np.linalg.norm(partner - col), np.linalg.norm(partner + col)):.2e}"
            )


# ---------------------------------------------------------------------------
# D3d gauge helpers and tests
# ---------------------------------------------------------------------------


def _make_d3d_j4_doublet():
    """Canonical J=4 D3d E_g doublet."""
    J = 4.0
    Psi = build_d3d_eg_doublet(J, [1.0, 1.0, 1.0])
    U_T = build_time_reversal_operator(J)
    Jz = build_multipole_operators(J, "magnetic_dipole")["Jz"]
    return Psi, U_T, Jz, J


def test_gauge_fix_d3d_tr_pairing_satisfied():
    """Output must satisfy Theta|psi_1> = |psi_2> exactly."""
    Psi, U_T, Jz, _ = _make_d3d_j4_doublet()
    basis = gauge_fix_non_kramers_d3d(Psi @ random_u2(seed=13), tol=1e-10)
    w2_expected = U_T @ basis[:, 0].conj()
    np.testing.assert_allclose(basis[:, 1], w2_expected, atol=1e-10)


def test_gauge_fix_d3d_odd_anchor_is_diagonal_descending():
    """Odd anchor (Jz) must project to diagonal matrix with descending eigenvalues."""
    Psi, _, Jz, _ = _make_d3d_j4_doublet()
    basis = gauge_fix_non_kramers_d3d(Psi @ random_u2(seed=13), tol=1e-10)
    M_Jz = basis.conj().T @ Jz @ basis
    np.testing.assert_allclose(M_Jz[0, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(M_Jz[1, 0], 0.0, atol=1e-10)
    assert M_Jz[0, 0].real > M_Jz[1, 1].real


def test_gauge_fix_d3d_is_gauge_invariant():
    """Output must be identical for two different U(2) rotations of the same subspace."""
    Psi, _, _, _ = _make_d3d_j4_doublet()
    basis1 = gauge_fix_non_kramers_d3d(Psi @ random_u2(seed=3), tol=1e-10)
    basis2 = gauge_fix_non_kramers_d3d(Psi @ random_u2(seed=17), tol=1e-10)
    np.testing.assert_allclose(basis1, basis2, atol=1e-10)


def test_gauge_fix_d3d_first_column_largest_component_real_positive():
    """Largest-magnitude entry of first output column must be real positive."""
    Psi, _, _, _ = _make_d3d_j4_doublet()
    basis = gauge_fix_non_kramers_d3d(Psi @ random_u2(seed=99), tol=1e-10)
    w1 = basis[:, 0]
    pivot = int(np.argmax(np.abs(w1)))
    assert abs(w1[pivot].imag) < 1e-10
    assert w1[pivot].real > 0


# ---------------------------------------------------------------------------
# analyze_multipole_carrying integration
# ---------------------------------------------------------------------------


def test_analyze_multipole_uses_oh_gauge_when_requested():
    """analyze_multipole_carrying must set used_gauge='non_kramers_oh' for Oh."""
    Psi, J = _make_oh_j4_doublet()
    result = analyze_multipole_carrying(
        Psi,
        J,
        point_group="Oh",
    )
    assert result["used_gauge"] == "non_kramers_oh"


def test_analyze_multipole_uses_d3d_gauge_when_requested():
    """analyze_multipole_carrying must set used_gauge='non_kramers_d3d' for D3d."""
    Psi, _, _, J = _make_d3d_j4_doublet()
    result = analyze_multipole_carrying(
        Psi,
        J,
        point_group="D3d",
    )
    assert result["used_gauge"] == "non_kramers_d3d"


def test_analyze_multipole_falls_back_to_kramers_without_point_group():
    """Without point_group, Kramers check must still run as before."""
    Psi = np.eye(2, dtype=np.complex128)
    result = analyze_multipole_carrying(Psi, 0.5)
    assert result["used_gauge"] == "kramers"


def test_analyze_multipole_gauge_none_for_triplet():
    """For 3-column input, no gauge should be applied."""
    Psi = np.eye(3, dtype=np.complex128)
    result = analyze_multipole_carrying(Psi, 1.0)
    assert result["used_gauge"] == "none"
