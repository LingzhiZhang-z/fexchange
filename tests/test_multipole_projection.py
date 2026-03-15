"""Integration tests for multipole projection pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from fexchange.core.space_j import _fix_column_phases
from fexchange.hamiltonian.hcef import build_hcef_matrix_J
from fexchange.spectrum.multipole import analyze_multipole_carrying, format_multipole_summary


def _diag_and_project(J, B_params, symmetry="Oh", mode_q3="cos"):
    H = build_hcef_matrix_J(J, B_params, symmetry=symmetry, mode_q3=mode_q3)
    evals, evecs = np.linalg.eigh(H)
    evecs = _fix_column_phases(evecs)
    return evals, evecs


def _load_cef_states_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "cef_states.py"
    spec = importlib.util.spec_from_file_location("cef_states_tool", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestKramersDoublet:
    """Half-integer J gives Kramers doublet ground state."""

    def test_oh_j72_ground_doublet_has_pauli(self):
        _, evecs = _diag_and_project(3.5, {"B4": 0.01, "B6": 0.0001})
        Psi = evecs[:, :2]
        result = analyze_multipole_carrying(Psi, 3.5, point_group="Oh")
        assert result["subspace_dim"] == 2
        assert result["used_gauge"] == "kramers"
        dipole = result["multipoles"]["magnetic_dipole"]
        assert any(dipole[op]["carried"] for op in dipole)
        for _, payload in dipole.items():
            if payload["carried"]:
                assert "pauli" in payload

    def test_c3v_j72_ground_doublet_has_pauli(self):
        _, evecs = _diag_and_project(
            3.5,
            {"B40": 0.01, "B60": 0.0001, "B43": 0.005},
            symmetry="C3v",
        )
        Psi = evecs[:, :2]
        result = analyze_multipole_carrying(Psi, 3.5, point_group="C3v")
        assert result["subspace_dim"] == 2
        assert result["used_gauge"] == "kramers"


class TestIntegerTriplet:
    """Integer J in Oh can give triplet ground state."""

    def test_oh_j4_triplet_has_spin1(self):
        evals, evecs = _diag_and_project(4.0, {"B4": 0.01, "B6": 0.0001})
        assert abs(evals[0] - evals[1]) < 1e-6
        assert abs(evals[1] - evals[2]) < 1e-6
        Psi = evecs[:, :3]
        result = analyze_multipole_carrying(Psi, 4.0, point_group="Oh")
        assert result["subspace_dim"] == 3
        for family in result["multipoles"].values():
            for payload in family.values():
                if payload["carried"]:
                    assert "spin1" in payload

    def test_triplet_summary_includes_spin1_components(self):
        _, evecs = _diag_and_project(4.0, {"B4": 0.01, "B6": 0.0001})
        analysis = analyze_multipole_carrying(evecs[:, :3], 4.0, point_group="Oh")
        text = format_multipole_summary(analysis)
        assert "sx=" in text
        assert "q3z2r2=" in text


class TestIntegerSinglet:
    """One-dimensional subspaces should report scalar expectations."""

    def test_j4_basis_singlet_has_expectation(self):
        Psi = np.eye(9, dtype=complex)[:, -1:]
        result = analyze_multipole_carrying(Psi, 4.0, point_group="Oh")
        assert result["subspace_dim"] == 1
        jz = result["multipoles"]["magnetic_dipole"]["Jz"]
        assert jz["carried"] is True
        assert jz["expectation"] == 4.0

    def test_singlet_summary_includes_expectation(self):
        analysis = analyze_multipole_carrying(np.eye(9, dtype=complex)[:, -1:], 4.0, point_group="Oh")
        text = format_multipole_summary(analysis)
        assert "<O>=" in text


def test_cef_states_computes_level_multipoles_for_supported_dimensions():
    module = _load_cef_states_module()
    result = module.compute_cef_states(
        point_group="Oh",
        J=4.0,
        B_params={"B4": 0.01, "B6": 0.0001},
        verbose=False,
    )
    assert len(result["level_multipoles"]) == len(result["level_info"])
    assert result["level_multipoles"][0]["subspace_dim"] == 3


def test_cef_states_skips_fourfold_levels():
    module = _load_cef_states_module()
    result = module.compute_cef_states(
        point_group="Oh",
        J=1.5,
        B_params={},
        verbose=False,
    )
    assert [level["degeneracy"] for level in result["level_info"]] == [4]
    assert result["level_multipoles"] == [None]
