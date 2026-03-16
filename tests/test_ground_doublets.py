import numpy as np

from fexchange.spectrum.doublet import (
    build_d3d_eg_doublet,
    build_oh_eg_doublet,
    gauge_fix_kramers_pair,
    project_to_eg_doublet,
)
from fexchange.core.space_j import build_time_reversal_operator
from fexchange.spectrum.ground import (
    build_W,
    select_kramers_doublet,
    select_non_kramers_doublet,
)


def test_target_module_imports_resolve():
    assert callable(gauge_fix_kramers_pair)
    assert callable(build_oh_eg_doublet)
    assert callable(build_d3d_eg_doublet)
    assert callable(project_to_eg_doublet)
    assert callable(select_kramers_doublet)
    assert callable(select_non_kramers_doublet)
    assert callable(build_W)


def test_build_oh_eg_doublet_still_returns_real_j4_basis():
    psi = build_oh_eg_doublet(4, [0.0, 1.0], [1.0])
    assert psi.shape == (9, 2)
    np.testing.assert_allclose(psi.imag, 0.0, atol=1e-15)


def test_build_d3d_eg_doublet_still_returns_tr_paired_basis():
    J = 4.0
    psi = build_d3d_eg_doublet(J, [1.0, 1.0, 1.0])
    u_t = build_time_reversal_operator(J)
    np.testing.assert_allclose(psi[:, 1], u_t @ psi[:, 0].conj(), atol=1e-14)


def test_project_to_eg_doublet_routes_c3v_through_trigonal_backend():
    J = 4.0
    psi = build_d3d_eg_doublet(J, [1.0, 1.0, 1.0]).astype(np.complex128)
    out = project_to_eg_doublet(J, psi, "C3v")
    u_t = build_time_reversal_operator(J)
    np.testing.assert_allclose(out[:, 1], u_t @ out[:, 0].conj(), atol=1e-10)


def test_select_kramers_doublet_preserves_output_keys():
    evals = np.array([0.0, 0.0], dtype=float)
    evecs = np.eye(2, dtype=complex)

    out = select_kramers_doublet(0.5, evals, evecs, point_group="Oh")

    assert {
        "kramer_vectors",
        "Lambda",
        "g_tensor",
        "gauge_meta",
        "symmetry_meta",
    }.issubset(out)


def test_select_non_kramers_doublet_preserves_output_keys():
    J = 4.0
    evals = np.array([0.0, 0.0], dtype=float)
    evecs = build_oh_eg_doublet(J, [0.0, 1.0], [1.0]).astype(complex)

    out = select_non_kramers_doublet(
        J,
        evals,
        evecs,
        point_group="Oh",
        eps_mag_ab=10.0,
    )

    assert {
        "doublet_vectors",
        "M_Q1",
        "M_Q2",
        "Q_labels",
        "map_electric",
        "gauge_meta",
        "symmetry_meta",
    }.issubset(out)
