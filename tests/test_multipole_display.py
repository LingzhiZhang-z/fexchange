import numpy as np
import pytest

from fexchange.spectrum.multipole import analyze_multipole_carrying, format_multipole_summary


def _analyze_doublet_j_half() -> dict:
    psi = np.eye(2, dtype=complex)
    return analyze_multipole_carrying(psi, 0.5)


def test_doublet_analysis_structure():
    analysis = _analyze_doublet_j_half()

    assert analysis["subspace_dim"] == 2
    assert "multipoles" in analysis
    assert set(analysis["multipoles"].keys()) == {
        "magnetic_dipole",
        "electric_quadrupole",
        "magnetic_octupole",
    }


def test_doublet_pauli_component_magnitudes():
    analysis = _analyze_doublet_j_half()
    dipole = analysis["multipoles"]["magnetic_dipole"]

    assert abs(dipole["Jx"]["pauli"]["ox"]) == pytest.approx(0.5)
    assert abs(dipole["Jy"]["pauli"]["oy"]) == pytest.approx(0.5)
    assert abs(dipole["Jz"]["pauli"]["oz"]) == pytest.approx(0.5)


def test_j_half_non_dipole_channels_not_carried():
    analysis = _analyze_doublet_j_half()

    for family in ("electric_quadrupole", "magnetic_octupole"):
        assert not any(row["carried"] for row in analysis["multipoles"][family].values())


def test_summary_contains_key_sections_for_doublet():
    analysis = _analyze_doublet_j_half()
    text = format_multipole_summary(analysis)

    assert "magnetic_dipole:" in text
    assert "electric_quadrupole:" in text
    assert "magnetic_octupole:" in text
    assert "Jx" in text
    assert "Yes" in text
    assert "ox=" in text


def test_triplet_has_no_pauli_payload_and_summary_has_no_pauli_terms():
    psi = np.eye(3, dtype=complex)
    analysis = analyze_multipole_carrying(psi, 1.0)

    assert analysis["subspace_dim"] == 3
    for family in analysis["multipoles"].values():
        for payload in family.values():
            assert "pauli" not in payload

    text = format_multipole_summary(analysis)
    assert "o0=" not in text


def test_larger_j_doublet_has_pauli_for_carried_dipole_ops():
    # Use |M=+1/2>, |M=-1/2> in J=7/2 manifold so g proxies stay nonzero.
    psi = np.eye(8, dtype=complex)[:, [4, 3]]
    analysis = analyze_multipole_carrying(psi, 3.5)
    dipole = analysis["multipoles"]["magnetic_dipole"]

    carried = [name for name, payload in dipole.items() if payload["carried"]]
    assert carried
    for name in carried:
        assert "pauli" in dipole[name]
