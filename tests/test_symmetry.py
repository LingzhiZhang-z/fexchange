"""Tests for models/symmetry.py - irrep classification."""

import pytest
import numpy as np

from fexchange.models.symmetry import (
    CHARACTER_TABLES,
    MULTIPOLE_RULES,
    ROTATIONAL_CORE_TABLES,
    parity_from_J,
    build_active_irrep_table,
    build_projectors,
    irrep_metadata,
    build_representation_matrices,
    classify_irreps,
    allowed_multipoles,
    classify_with_multipoles,
    analyze_cef_symmetry,
)
from fexchange.models.hcef import build_hcef_matrix_J


def test_o_star_class_sizes_match_standard():
    assert ROTATIONAL_CORE_TABLES["O_star"]["class_sizes"] == {
        "E": 1,
        "R": 1,
        "C2_mix": 6,
        "C4": 6,
        "RC4": 6,
        "C2p_mix": 12,
        "C3": 8,
        "RC3": 8,
    }


def test_c3v_star_class_sizes_match_standard():
    assert ROTATIONAL_CORE_TABLES["C3v_star"]["class_sizes"] == {
        "E": 1,
        "R": 1,
        "2C3": 2,
        "2RC3": 2,
        "3sigma_v": 3,
        "3Rsigma_v": 3,
    }


def test_parity_is_determined_by_j_only():
    assert parity_from_J(4.0) == +1
    assert parity_from_J(3.5) == -1


def test_active_table_is_single_branch_for_oh_and_d3d():
    oh = build_active_irrep_table(J=4.0, point_group="Oh")
    assert oh["branch_mode"] == "single"
    d3d = build_active_irrep_table(J=3.5, point_group="D3d")
    assert d3d["branch_mode"] == "single"
    assert all("+" not in k for k in d3d["rows"].keys()) is False


def test_o_star_row_norms_are_48():
    tbl = ROTATIONAL_CORE_TABLES["O_star"]
    sizes = tbl["class_sizes"]
    rows = tbl["rows"]
    for required in ("Gamma6", "Gamma7", "Gamma8"):
        assert required in rows
    for row in rows.values():
        norm2 = sum(sizes[c] * abs(row[c]) ** 2 for c in sizes)
        assert abs(norm2 - 48.0) < 1e-10


def test_c3v_star_row_norms_are_12():
    tbl = ROTATIONAL_CORE_TABLES["C3v_star"]
    sizes = tbl["class_sizes"]
    rows = tbl["rows"]
    for required in ("Gamma4", "Gamma5", "Gamma6"):
        assert required in rows
    for row in rows.values():
        norm2 = sum(sizes[c] * abs(row[c]) ** 2 for c in sizes)
        assert abs(norm2 - 12.0) < 1e-10


def test_o_star_spinor_r_sign():
    rows = ROTATIONAL_CORE_TABLES["O_star"]["rows"]
    for name in ("Gamma6", "Gamma7", "Gamma8"):
        assert rows[name]["R"] == -rows[name]["E"]


def test_d3d_labels_follow_j_parity():
    even_tbl = build_active_irrep_table(J=4.0, point_group="D3d")
    odd_tbl = build_active_irrep_table(J=3.5, point_group="D3d")
    assert set(even_tbl["rows"].keys()) == {
        "Gamma1+",
        "Gamma2+",
        "Gamma3+",
        "Gamma4+",
        "Gamma5+",
        "Gamma6+",
    }
    assert set(odd_tbl["rows"].keys()) == {
        "Gamma1-",
        "Gamma2-",
        "Gamma3-",
        "Gamma4-",
        "Gamma5-",
        "Gamma6-",
    }


def test_n_f_inconsistency_raises():
    with pytest.raises(ValueError):
        build_active_irrep_table(J=3.5, point_group="Oh", n_f=4)


def test_projector_set_covers_all_active_irreps():
    _h = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
    projectors = build_projectors(4.0, point_group="Oh")
    assert set(projectors.keys()) == {
        "Gamma1",
        "Gamma2",
        "Gamma3",
        "Gamma4",
        "Gamma5",
        "Gamma6",
        "Gamma7",
        "Gamma8",
    }


def test_projector_completeness_on_j4_oh():
    projectors = build_projectors(4.0, point_group="Oh")
    total = np.zeros((9, 9), dtype=complex)
    for p in projectors.values():
        total += p
    np.testing.assert_allclose(total, np.eye(9, dtype=complex), atol=1e-6)


def test_irrep_metadata_contains_display_primary_aliases():
    meta = irrep_metadata("Gamma1", point_group="Oh", J=4.0)
    assert meta["irrep_display"] == "Γ1"
    assert meta["irrep_primary"] == "Gamma1"
    assert "A1g" in meta["irrep_aliases"]
    assert meta["mapping_unverified"] is False


def test_spinor_aliases_default_empty():
    meta = irrep_metadata("Gamma6", point_group="Oh", J=3.5)
    assert meta["irrep_aliases"] == []
    assert meta["mapping_unverified"] is True


def test_oh_spinor_multipole_rules_exist():
    assert "dipole" in allowed_multipoles("Gamma6", "Oh")
    assert "quadrupole" in allowed_multipoles("Gamma8", "Oh")


def test_d3d_family_multipole_rules_exist():
    assert "dipole" in allowed_multipoles("Gamma5+", "D3d")
    assert "octupole" in allowed_multipoles("Gamma2-", "D3d")


def test_classify_with_multipoles_new_schema_single_branch():
    h = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
    _, evecs = np.linalg.eigh(h)
    out = classify_with_multipoles(4.0, evecs, point_group="Oh", n_f=2)
    assert "irrep_display" in out
    assert "irrep_primary" in out
    assert "irrep_aliases" in out
    assert "allowed_multipoles" in out
    assert "excited_irreps" in out
    assert "parity_unknown" not in out


def test_analyze_cef_symmetry_requires_exactly_one_input():
    with pytest.raises(ValueError):
        analyze_cef_symmetry(4.0, "Oh")
    h = build_hcef_matrix_J(4.0, {"B4": 0.01}, "Oh")
    _, evecs = np.linalg.eigh(h)
    with pytest.raises(ValueError):
        analyze_cef_symmetry(4.0, "Oh", B_params={"B4": 0.01}, evecs=evecs)


class TestCharacterTablesOh:
    def test_oh_exists(self):
        assert "Oh" in CHARACTER_TABLES

    def test_oh_irrep_names(self):
        expected = {"Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5"}
        assert set(CHARACTER_TABLES["Oh"].keys()) - {"_class_sizes"} == expected

    def test_oh_dimensions(self):
        """Gamma1=1, Gamma2=1, Gamma3=2, Gamma4=3, Gamma5=3 (Bethe notation)."""
        dims = {
            name: row["E"]
            for name, row in CHARACTER_TABLES["Oh"].items()
            if name != "_class_sizes"
        }
        assert dims == {"Gamma1": 1, "Gamma2": 1, "Gamma3": 2, "Gamma4": 3, "Gamma5": 3}

    def test_oh_orthogonality(self):
        """Character orthogonality: sum_g chi_i(g)* chi_j(g) = |G| delta_ij."""
        table = CHARACTER_TABLES["Oh"]
        for name, row in table.items():
            if name == "_class_sizes":
                continue
            norm_sq = sum(
                size * abs(row[op]) ** 2
                for op, size in table["_class_sizes"].items()
            )
            assert abs(norm_sq - 48) < 1e-10, f"{name} norm failed"


class TestMultipoleRulesOh:
    def test_oh_rules_exist(self):
        assert "Oh" in MULTIPOLE_RULES

    def test_gamma4_carries_dipole(self):
        """Gamma4 (T1) carries magnetic dipole in Oh."""
        assert "dipole" in MULTIPOLE_RULES["Oh"]["Gamma4"]

    def test_gamma3_carries_quadrupole(self):
        """Gamma3 (E) carries electric quadrupole in Oh."""
        assert "quadrupole" in MULTIPOLE_RULES["Oh"]["Gamma3"]

    def test_gamma1_no_dipole(self):
        """Gamma1 (A1) singlet cannot carry dipole."""
        assert "dipole" not in MULTIPOLE_RULES["Oh"]["Gamma1"]


class TestCharacterTablesD3d:
    def test_d3d_exists(self):
        assert "D3d" in CHARACTER_TABLES

    def test_d3d_irrep_names(self):
        expected = {"A1g", "A2g", "Eg", "A1u", "A2u", "Eu"}
        assert set(CHARACTER_TABLES["D3d"].keys()) - {"_class_sizes"} == expected

    def test_d3d_dimensions(self):
        dims = {
            name: row["E"]
            for name, row in CHARACTER_TABLES["D3d"].items()
            if name != "_class_sizes"
        }
        assert dims == {"A1g": 1, "A2g": 1, "Eg": 2, "A1u": 1, "A2u": 1, "Eu": 2}

    def test_d3d_orthogonality(self):
        table = CHARACTER_TABLES["D3d"]
        order = sum(table["_class_sizes"].values())
        assert order == 12
        for name, row in table.items():
            if name == "_class_sizes":
                continue
            norm_sq = sum(
                size * abs(row[op]) ** 2
                for op, size in table["_class_sizes"].items()
            )
            assert abs(norm_sq - 12) < 1e-10, f"{name} norm failed"


class TestMultipoleRulesD3d:
    def test_d3d_rules_exist(self):
        assert "D3d" in MULTIPOLE_RULES

    def test_a2g_carries_dipole(self):
        """A2g carries z-component of magnetic dipole in D3d."""
        assert "dipole" in MULTIPOLE_RULES["D3d"]["A2g"]

    def test_eg_carries_dipole(self):
        """Eg carries xy-component of magnetic dipole in D3d."""
        assert "dipole" in MULTIPOLE_RULES["D3d"]["Eg"]


class TestRepresentationMatrices:
    def test_identity_is_identity(self):
        """D(E) = I for any J."""
        reps = build_representation_matrices(J=4.0, point_group="Oh")
        I = np.eye(9, dtype=complex)
        np.testing.assert_allclose(reps["E"], I, atol=1e-12)

    def test_c4_fourth_power_is_identity(self):
        """D(C4)^4 = I."""
        reps = build_representation_matrices(J=4.0, point_group="Oh")
        D_c4 = reps["C4"]
        np.testing.assert_allclose(
            np.linalg.matrix_power(D_c4, 4),
            np.eye(9, dtype=complex),
            atol=1e-10,
        )

    def test_c3_third_power_is_identity(self):
        """D(C3)^3 = I."""
        reps = build_representation_matrices(J=4.0, point_group="Oh")
        D_c3 = reps["C3"]
        np.testing.assert_allclose(
            np.linalg.matrix_power(D_c3, 3),
            np.eye(9, dtype=complex),
            atol=1e-10,
        )

    def test_inversion(self):
        """D(i) = (-1)^J * I for integer J (parity)."""
        reps = build_representation_matrices(J=4.0, point_group="Oh")
        evals = np.linalg.eigvalsh(reps["i"].real)
        np.testing.assert_allclose(evals, np.ones(9), atol=1e-10)

    def test_unitarity(self):
        """All D(g) must be unitary."""
        reps = build_representation_matrices(J=4.0, point_group="Oh")
        I = np.eye(9, dtype=complex)
        for name, D in reps.items():
            np.testing.assert_allclose(
                D @ D.conj().T,
                I,
                atol=1e-10,
                err_msg=f"D({name}) not unitary",
            )


class TestClassifyIrreps:
    def test_oh_j4_known_degeneracies(self):
        """J=4 in Oh splits as Gamma1 + Gamma3 + Gamma4 + Gamma5."""
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.0}, "Oh")
        _, evecs = np.linalg.eigh(H)
        labels = classify_irreps(4.0, evecs, point_group="Oh")
        from collections import Counter

        counts = Counter(labels)
        assert counts["Gamma1"] == 1
        assert counts["Gamma3"] == 2
        assert counts["Gamma4"] == 3
        assert counts["Gamma5"] == 3

    def test_oh_j4_all_states_labeled(self):
        """Every eigenstate gets a label."""
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
        _, evecs = np.linalg.eigh(H)
        labels = classify_irreps(4.0, evecs, point_group="Oh")
        assert len(labels) == 9
        assert all(label.startswith("Gamma") for label in labels)

    def test_oh_j3_5_half_integer(self):
        """Half-integer J needs the Oh double group; deferred."""
        pass


class TestAllowedMultipoles:
    def test_gamma4_oh(self):
        result = allowed_multipoles("Gamma4", "Oh")
        assert "dipole" in result

    def test_gamma1_oh(self):
        result = allowed_multipoles("Gamma1", "Oh")
        assert "dipole" not in result

    def test_unknown_irrep_raises(self):
        with pytest.raises(KeyError):
            allowed_multipoles("GammaX", "Oh")


class TestClassifyWithMultipoles:
    def test_returns_ground_and_excited(self):
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
        _, evecs = np.linalg.eigh(H)
        result = classify_with_multipoles(4.0, evecs, "Oh")
        assert "ground" in result
        assert "excited" in result
        assert "irrep" in result["ground"]
        assert "multipoles" in result["ground"]
        assert len(result["excited"]) == 8


class TestAnalyzeCefSymmetry:
    def test_from_b_params(self):
        """Standalone: J + B_params -> classification."""
        result = analyze_cef_symmetry(4.0, "Oh", B_params={"B4": 0.01, "B6": 0.001})
        assert "ground" in result
        assert "eigenvalues" in result
        assert len(result["all_irreps"]) == 9

    def test_from_evecs(self):
        """Standalone: J + evecs -> classification."""
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.0}, "Oh")
        _, evecs = np.linalg.eigh(H)
        result = analyze_cef_symmetry(4.0, "Oh", evecs=evecs)
        assert "ground" in result

    def test_must_provide_one_input(self):
        with pytest.raises(ValueError):
            analyze_cef_symmetry(4.0, "Oh")

    def test_must_not_provide_both(self):
        H = build_hcef_matrix_J(4.0, {"B4": 0.01}, "Oh")
        _, evecs = np.linalg.eigh(H)
        with pytest.raises(ValueError):
            analyze_cef_symmetry(4.0, "Oh", B_params={"B4": 0.01}, evecs=evecs)
