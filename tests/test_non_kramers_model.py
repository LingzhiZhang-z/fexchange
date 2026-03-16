"""Tests for select_non_kramers_doublet gauge integration."""
import numpy as np

from fexchange.spectrum.doublet import build_d3d_eg_doublet, build_oh_eg_doublet
from fexchange.spectrum.ground import select_non_kramers_doublet


def _make_oh_j4_input():
    """J=4 Oh Gamma_3 doublet as CEF eigenvectors."""
    J = 4.0
    evecs = build_oh_eg_doublet(J, [0.0, 1.0], [1.0]).astype(np.complex128)
    evals = np.array([0.0, 0.0])
    return evals, evecs


def _make_c3v_j4_input():
    """J=4 C3v input reuses the D3d-compatible non-Kramers analytical basis."""
    J = 4.0
    evecs = build_d3d_eg_doublet(J, [1.0, 1.0, 1.0]).astype(np.complex128)
    evals = np.array([0.0, 0.0])
    return evals, evecs


def test_select_nonkramers_oh_outputs_gauge_fixed_basis():
    """Oh doublet_vectors must be real after canonical projection/gauge fixing."""
    evals, evecs = _make_oh_j4_input()
    out = select_non_kramers_doublet(
        4.0,
        evals,
        evecs,
        point_group="Oh",
        eps_mag_ab=10.0,
    )
    Psi = out["doublet_vectors"]
    np.testing.assert_allclose(Psi.imag, 0.0, atol=1e-10)


def test_select_nonkramers_oh_outputs_electric_channels():
    """Output must contain M_Q1, M_Q2, Q_labels, map_electric."""
    evals, evecs = _make_oh_j4_input()
    out = select_non_kramers_doublet(
        4.0,
        evals,
        evecs,
        point_group="Oh",
        eps_mag_ab=10.0,
    )
    assert "M_Q1" in out
    assert "M_Q2" in out
    assert "Q_labels" in out
    assert len(out["Q_labels"]) == 2
    assert "map_electric" in out
    assert len(out["map_electric"]) == 2


def test_select_nonkramers_oh_gauge_meta_records_gauge_used():
    """gauge_meta must record which gauge was applied."""
    evals, evecs = _make_oh_j4_input()
    out = select_non_kramers_doublet(
        4.0,
        evals,
        evecs,
        point_group="Oh",
        eps_mag_ab=10.0,
    )
    assert out["gauge_meta"].get("gauge_used") == "non_kramers_oh"


def test_select_nonkramers_oh_projects_without_external_anchor():
    """Canonical Oh projection should work without caller-supplied gauge data."""
    evals, evecs = _make_oh_j4_input()
    out = select_non_kramers_doublet(
        4.0,
        evals,
        evecs,
        point_group="Oh",
        eps_mag_ab=10.0,
    )
    assert "doublet_vectors" in out
    assert "M_Jx" in out
    assert "M_Jy" in out
    assert "M_Jz" in out


def test_select_nonkramers_c3v_routes_through_d3d_interface():
    """C3v non-Kramers path should reuse the D3d analytical projection."""
    evals, evecs = _make_c3v_j4_input()
    out = select_non_kramers_doublet(
        4.0,
        evals,
        evecs,
        point_group="C3v",
        eps_mag_ab=10.0,
    )
    Psi = out["doublet_vectors"]
    assert out["gauge_meta"].get("gauge_used") == "non_kramers_c3v"
    np.testing.assert_allclose(Psi.conj().T @ Psi, np.eye(2), atol=1e-10)
