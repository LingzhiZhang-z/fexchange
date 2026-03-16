import numpy as np
from fexchange.spectrum.doublet import (
    build_d3d_eg_doublet,
    gauge_fix_kramers_pair,
    project_to_eg_doublet,
)
from fexchange.core.space_ls import ELL, build_space_ls_operator, build_space_ls_matrix, orbital_index, orbital_map
from fexchange.core.space_j import (
    build_space_j_operator,
    build_time_reversal_operator,
    normalize_J,
    pauli_decompose,
    project_J_to_subspace,
    project_operators_to_subspace,
)
from fexchange.spectrum.multipole import analyze_multipole_carrying
from fexchange.core.stevens import build_multipole_operators

def test_space_ls_module_builds_expected_keys():
    mats = build_space_ls_operator()
    for key in ("Lz", "Sz", "Lp", "Lm", "Sp", "Sm", "Lx", "Ly", "Sx", "Sy", "Jz", "Jp", "Jm", "Jx", "Jy"):
        assert key in mats
        assert mats[key].shape == (14, 14)


def test_space_ls_module_exposes_orbital_mapping():
    assert ELL == 3
    for p in range(14):
        m, sigma = orbital_map(p)
        assert orbital_index(m, sigma) == p


def test_space_j_module_builds_cartesian_matrices():
    assert normalize_J(3.50000000001) == 3.5
    Jz, Jp, Jm, Jx, Jy = build_space_j_operator(0.5)
    np.testing.assert_allclose(Jx, 0.5 * (Jp + Jm))
    np.testing.assert_allclose(Jy, -0.5j * (Jp - Jm))
    assert Jz.shape == (2, 2)


def test_space_j_module_exposes_time_reversal_matrix():
    U = build_time_reversal_operator(0.5)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(2))
    np.testing.assert_allclose(U.T, -U)


def test_space_j_module_exposes_pauli_decompose():
    M = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    dec = pauli_decompose(M)
    assert abs(dec["ox"] - 1.0) < 1e-12
    assert abs(dec["o0"]) < 1e-12
    assert abs(dec["oy"]) < 1e-12
    assert abs(dec["oz"]) < 1e-12


def test_space_ls_module_exposes_fock_matrix_builder_symbol():
    assert callable(build_space_ls_matrix)


def test_space_j_module_projects_generic_operators_to_subspace():
    Jz, _, _, Jx, _ = build_space_j_operator(0.5)
    Psi = np.eye(2, dtype=np.complex128)
    projected = project_operators_to_subspace(
        Psi,
        {"Jx": Jx, "Jz": Jz},
        module="test_space_j",
    )
    assert set(projected.keys()) == {"Jx", "Jz"}
    np.testing.assert_allclose(projected["Jx"], Jx)
    np.testing.assert_allclose(projected["Jz"], Jz)


def test_space_j_module_projects_to_three_dim_subspace():
    # J=3/2 has full dimension 4; project to a 3D subspace.
    Jz, _, _, Jx, _ = build_space_j_operator(1.5)
    Psi = np.eye(4, dtype=np.complex128)[:, :3]
    projected = project_operators_to_subspace(
        Psi,
        {"Jx": Jx, "Jz": Jz},
        module="test_space_j",
    )
    assert projected["Jx"].shape == (3, 3)
    assert projected["Jz"].shape == (3, 3)
    np.testing.assert_allclose(projected["Jx"], projected["Jx"].conj().T)
    np.testing.assert_allclose(projected["Jz"], projected["Jz"].conj().T)


def test_space_j_module_projects_near_zero_operators_without_false_hermiticity_failure():
    J = 4.0
    Psi = project_to_eg_doublet(
        J,
        build_d3d_eg_doublet(J, [1.0, 1.0, 1.0]).astype(np.complex128),
        "C3v",
    )
    ops = build_multipole_operators(J, "magnetic_dipole")
    M_Jx, M_Jy, M_Jz = project_J_to_subspace(
        Psi,
        ops["Jx"],
        ops["Jy"],
        ops["Jz"],
        module="test_space_j",
    )
    np.testing.assert_allclose(M_Jx, M_Jx.conj().T, atol=1e-16)
    np.testing.assert_allclose(M_Jy, M_Jy.conj().T, atol=1e-16)
    np.testing.assert_allclose(M_Jz, np.diag([1.0, -1.0]), atol=1e-10)


def test_space_j_module_reports_carried_multipoles_for_doublet_with_pauli():
    Psi = np.eye(2, dtype=np.complex128)
    out = analyze_multipole_carrying(Psi, 0.5)

    assert out["subspace_dim"] == 2
    dipole = out["multipoles"]["magnetic_dipole"]
    assert dipole["Jx"]["carried"] is True
    assert dipole["Jy"]["carried"] is True
    assert dipole["Jz"]["carried"] is True

    for op_name in ("Jx", "Jy", "Jz"):
        assert "pauli" in dipole[op_name]
        coeffs = dipole[op_name]["pauli"]
        assert set(coeffs.keys()) == {"o0", "ox", "oy", "oz"}


def test_space_j_module_reports_carried_multipoles_for_triplet():
    Psi = np.eye(3, dtype=np.complex128)
    out = analyze_multipole_carrying(Psi, 1.0)

    assert out["subspace_dim"] == 3
    dipole = out["multipoles"]["magnetic_dipole"]
    quad = out["multipoles"]["electric_quadrupole"]
    octu = out["multipoles"]["magnetic_octupole"]
    assert any(entry["carried"] for entry in dipole.values())
    assert any(entry["carried"] for entry in quad.values())
    assert not any(entry["carried"] for entry in octu.values())
    assert "pauli" not in next(iter(dipole.values()))


def test_space_j_module_reports_norms_for_each_operator():
    Psi = np.eye(2, dtype=np.complex128)
    out = analyze_multipole_carrying(Psi, 0.5)
    for family in ("magnetic_dipole", "electric_quadrupole", "magnetic_octupole"):
        for payload in out["multipoles"][family].values():
            assert isinstance(payload["norm"], float)
            assert payload["norm"] >= 0.0


def _kramers_g_proxies(Psi: np.ndarray, J: float = 0.5) -> tuple[float, float, float]:
    ops = build_multipole_operators(J, "magnetic_dipole")
    basis = gauge_fix_kramers_pair(Psi, tol=1e-10)
    M_Jx = basis.conj().T @ ops["Jx"] @ basis
    M_Jy = basis.conj().T @ ops["Jy"] @ basis
    M_Jz = basis.conj().T @ ops["Jz"] @ basis
    gx = float(np.real(M_Jx[0, 1]))
    gy = float(np.real(1j * M_Jy[0, 1]))
    gz = float(np.real(M_Jz[0, 0]))
    return gx, gy, gz


def _all_nonzero_components_share_sign(*values: float, tol: float = 1e-10) -> bool:
    signs = {int(np.sign(value)) for value in values if abs(value) > tol}
    return len(signs) <= 1


def test_kramers_gauge_auto_flip_harmonizes_two_plus_one_sign_pattern():
    # Construct a basis that yields (-,-,+) before discrete harmonization.
    Psi = np.diag([1.0, -1.0]).astype(np.complex128)
    gx, gy, gz = _kramers_g_proxies(Psi, J=0.5)

    assert _all_nonzero_components_share_sign(gx, gy, gz)


def test_kramers_gauge_auto_flip_keeps_none_when_signs_already_consistent():
    Psi = np.eye(2, dtype=np.complex128)
    gx, gy, gz = _kramers_g_proxies(Psi, J=0.5)

    assert _all_nonzero_components_share_sign(gx, gy, gz)


def test_kramers_gauge_allows_two_zero_g_proxies_and_keeps_remaining_positive():
    # This basis drives gx and gy close to zero after phase alignment.
    Psi = np.eye(2, dtype=np.complex128) @ np.diag([1.0, 1.0j])
    gx, gy, gz = _kramers_g_proxies(Psi, J=0.5)

    assert abs(gx) <= 1e-10
    assert abs(gy) <= 1e-10
    assert gz > 0


def test_kramers_gauge_allows_one_zero_g_proxy_and_keeps_other_two_positive():
    Psi = np.array(
        [
            [0.014101 - 0.246172j, 0.398675 + 0.79671j],
            [-0.220797 + 0.008539j, 0.282353 - 0.130248j],
            [-0.282353 - 0.130248j, -0.220797 - 0.008539j],
            [0.398675 - 0.79671j, -0.014101 - 0.246172j],
        ],
        dtype=np.complex128,
    )
    ops = build_multipole_operators(1.5, "magnetic_dipole")
    basis = gauge_fix_kramers_pair(Psi, tol=1e-4)
    M_Jx = basis.conj().T @ ops["Jx"] @ basis
    M_Jy = basis.conj().T @ ops["Jy"] @ basis
    M_Jz = basis.conj().T @ ops["Jz"] @ basis
    gx = float(np.real(M_Jx[0, 1]))
    gy = float(np.real(1j * M_Jy[0, 1]))
    gz = float(np.real(M_Jz[0, 0]))

    assert abs(gy) <= 1e-4
    assert gx > 0
    assert gz > 0


def test_kramers_gauge_auto_flip_preserves_tr_pair_convention():
    Psi = np.diag([1.0, -1.0]).astype(np.complex128)
    U_T = build_time_reversal_operator(0.5)
    basis = gauge_fix_kramers_pair(Psi, tol=1e-10)

    w2_target = U_T @ basis[:, 0].conj()
    overlap = complex(w2_target.conj() @ basis[:, 1])
    phase = overlap / abs(overlap)
    residual = np.linalg.norm(w2_target - phase * basis[:, 1])
    assert residual < 1e-10
