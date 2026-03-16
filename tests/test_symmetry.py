"""Tests for core/symmetry.py - irrep classification."""

import pytest
import numpy as np

from fexchange.spectrum.classify import (
    MULTIPOLE_RULES,
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
from fexchange.spectrum.tables import (
    C3V_STAR_CHI,
    C3V_STAR_CLASSES,
    C3V_STAR_IRREPS,
    C3V_STAR_SIZES,
    O_STAR_CHI,
    O_STAR_CLASSES,
    O_STAR_IRREPS,
    O_STAR_SIZES,
)
from fexchange.hamiltonian.hcef import build_hcef_matrix_J


def test_o_star_class_sizes_match_standard():
    assert O_STAR_SIZES == (1, 1, 3, 3, 6, 6, 6, 6, 8, 8)


def test_c3v_star_class_sizes_match_standard():
    assert C3V_STAR_SIZES == (1, 1, 2, 2, 3, 3)


def test_parity_is_determined_by_j_only():
    assert parity_from_J(4.0) == +1
    assert parity_from_J(3.5) == -1


def test_active_table_is_single_branch_for_oh_and_d3d():
    oh = build_active_irrep_table(J=4.0, point_group="Oh")
    assert oh["branch_mode"] == "single"
    d3d = build_active_irrep_table(J=3.5, point_group="D3d")
    assert d3d["branch_mode"] == "single"
    assert all("+" not in k for k in d3d["rows"].keys())
    assert all("-" in k for k in d3d["rows"].keys())


def test_o_star_row_norms_are_48():
    for required in ("Gamma6", "Gamma7", "Gamma8"):
        assert required in O_STAR_IRREPS
    for row in O_STAR_CHI:
        norm2 = sum(size * abs(char) ** 2 for size, char in zip(O_STAR_SIZES, row, strict=True))
        assert abs(norm2 - 48.0) < 1e-10


def test_c3v_star_row_norms_are_12():
    for required in ("Gamma4", "Gamma5", "Gamma6"):
        assert required in C3V_STAR_IRREPS
    for row in C3V_STAR_CHI:
        norm2 = sum(size * abs(char) ** 2 for size, char in zip(C3V_STAR_SIZES, row, strict=True))
        assert abs(norm2 - 12.0) < 1e-10


def test_o_star_column_orthogonality():
    """Column orthogonality holds for O* at parent-class level after sub-class split."""
    idx = {name: i for i, name in enumerate(O_STAR_CLASSES)}
    np.testing.assert_allclose(O_STAR_CHI[:, idx["3C2"]], O_STAR_CHI[:, idx["3RC2"]], atol=1e-12)
    np.testing.assert_allclose(O_STAR_CHI[:, idx["6C2p"]], O_STAR_CHI[:, idx["6RC2p"]], atol=1e-12)

    grouped = (
        ("E", ("E",)),
        ("R", ("R",)),
        ("C2_mix", ("3C2", "3RC2")),
        ("C4", ("C4",)),
        ("RC4", ("RC4",)),
        ("C2p_mix", ("6C2p", "6RC2p")),
        ("C3", ("C3",)),
        ("RC3", ("RC3",)),
    )
    for i, (c, sub_c) in enumerate(grouped):
        col_c = O_STAR_CHI[:, idx[sub_c[0]]]
        n_c = sum(O_STAR_SIZES[idx[name]] for name in sub_c)
        for j, (cp, sub_cp) in enumerate(grouped):
            col_cp = O_STAR_CHI[:, idx[sub_cp[0]]]
            total = np.vdot(col_c, col_cp)
            if i == j:
                expected = 48.0 / n_c
                assert abs(total - expected) < 1e-10, f"O* col orth ({c},{cp}): {total} != {expected}"
            else:
                assert abs(total) < 1e-10, f"O* col orth ({c},{cp}): {total} != 0"


def test_c3v_star_column_orthogonality():
    """Column orthogonality for C3v*."""
    classes = list(C3V_STAR_CLASSES)
    for i, c in enumerate(classes):
        for j, cp in enumerate(classes):
            total = np.vdot(C3V_STAR_CHI[:, i], C3V_STAR_CHI[:, j])
            if i == j:
                expected = 12.0 / C3V_STAR_SIZES[i]
                assert abs(total - expected) < 1e-10, f"C3v* col orth ({c},{cp}): {total} != {expected}"
            else:
                assert abs(total) < 1e-10, f"C3v* col orth ({c},{cp}): {total} != 0"


def test_o_star_spinor_r_sign():
    idx = {name: i for i, name in enumerate(O_STAR_CLASSES)}
    irrep_idx = {name: i for i, name in enumerate(O_STAR_IRREPS)}
    for name in ("Gamma6", "Gamma7", "Gamma8"):
        row = O_STAR_CHI[irrep_idx[name], :]
        assert row[idx["R"]] == -row[idx["E"]]


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


def test_projector_idempotency_oh_j4():
    """P_Gamma^2 = P_Gamma for all active irreps."""
    projectors = build_projectors(4.0, point_group="Oh")
    for name, P in projectors.items():
        np.testing.assert_allclose(
            P @ P,
            P,
            atol=1e-10,
            err_msg=f"P_{name} not idempotent",
        )


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
    assert "magnetic_dipole" in allowed_multipoles("Gamma6", "Oh")
    assert "electric_quadrupole" in allowed_multipoles("Gamma8", "Oh")


def test_d3d_family_multipole_rules_exist():
    assert "magnetic_dipole" in allowed_multipoles("Gamma5+", "D3d")
    assert "magnetic_octupole" in allowed_multipoles("Gamma2-", "D3d")


def test_oh_multipole_rules_match_tensor_product_theory():
    """
    Verify Oh single-valued multipole rules against the 02-07 contract table.
    """
    expected = {
        "Gamma1": {"magnetic_octupole"},
        "Gamma2": {"magnetic_octupole"},
        "Gamma3": {"electric_quadrupole", "magnetic_octupole"},
        "Gamma4": {"magnetic_dipole", "magnetic_octupole"},
        "Gamma5": {"electric_quadrupole", "magnetic_octupole"},
    }

    for irrep, allowed in expected.items():
        assert set(MULTIPOLE_RULES["Oh"][irrep]) == allowed


def test_oh_spinor_multipole_rules_match_tensor_product_theory():
    """
    Verify Oh spinor multipole rules against the 02-07 contract table.
    """
    expected = {
        "Gamma6": {"magnetic_dipole", "magnetic_octupole"},
        "Gamma7": {"magnetic_dipole", "magnetic_octupole"},
        "Gamma8": {"magnetic_dipole", "electric_quadrupole", "magnetic_octupole"},
    }

    for irrep, allowed in expected.items():
        assert set(MULTIPOLE_RULES["Oh"][irrep]) == allowed


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


def test_removed_dual_parity_keys_are_absent():
    h = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
    _, evecs = np.linalg.eigh(h)
    out = analyze_cef_symmetry(4.0, "Oh", evecs=evecs, n_f=2)
    assert "parity_unknown" not in out
    assert "even_branch" not in out
    assert "odd_branch" not in out


class TestMultipoleRulesOh:
    def test_oh_rules_exist(self):
        assert "Oh" in MULTIPOLE_RULES

    def test_gamma4_carries_dipole(self):
        """Gamma4 (T1) carries magnetic dipole in Oh."""
        assert "magnetic_dipole" in MULTIPOLE_RULES["Oh"]["Gamma4"]

    def test_gamma3_carries_quadrupole(self):
        """Gamma3 (E) carries electric quadrupole in Oh."""
        assert "electric_quadrupole" in MULTIPOLE_RULES["Oh"]["Gamma3"]

    def test_gamma1_no_dipole(self):
        """Gamma1 (A1) singlet cannot carry dipole."""
        assert "magnetic_dipole" not in MULTIPOLE_RULES["Oh"]["Gamma1"]


class TestMultipoleRulesD3d:
    def test_d3d_rules_exist(self):
        assert "D3d" in MULTIPOLE_RULES

    def test_gamma2_plus_carries_dipole(self):
        """Gamma2+ carries z-component of magnetic dipole in D3d."""
        assert "magnetic_dipole" in MULTIPOLE_RULES["D3d"]["Gamma2+"]

    def test_gamma3_plus_carries_dipole(self):
        """Gamma3+ carries xy-component of magnetic dipole in D3d."""
        assert "magnetic_dipole" in MULTIPOLE_RULES["D3d"]["Gamma3+"]

    def test_d3d_removed_alias_keys_are_absent(self):
        for key in ("A1g", "A2g", "Eg", "A1u", "A2u", "Eu"):
            assert key not in MULTIPOLE_RULES["D3d"]


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


def test_oh_j3_5_half_integer_decomposition():
    """J=7/2 in Oh decomposes as Gamma6 + Gamma7 + Gamma8 (dimensions 2+2+4=8)."""
    from collections import Counter

    H = build_hcef_matrix_J(3.5, {"B4": 0.01, "B6": 0.001}, "Oh")
    _, evecs = np.linalg.eigh(H)
    labels = classify_irreps(3.5, evecs, point_group="Oh")
    counts = Counter(labels)
    assert counts["Gamma6"] == 2
    assert counts["Gamma7"] == 2
    assert counts["Gamma8"] == 4
    assert sum(counts.values()) == 8


class TestAllowedMultipoles:
    def test_gamma4_oh(self):
        result = allowed_multipoles("Gamma4", "Oh")
        assert "magnetic_dipole" in result

    def test_gamma1_oh(self):
        result = allowed_multipoles("Gamma1", "Oh")
        assert "magnetic_dipole" not in result

    def test_unknown_irrep_raises(self):
        with pytest.raises(KeyError):
            allowed_multipoles("GammaX", "Oh")


class TestClassifyWithMultipoles:
    def test_returns_ground_and_excited(self):
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.001}, "Oh")
        _, evecs = np.linalg.eigh(H)
        result = classify_with_multipoles(4.0, evecs, "Oh")
        assert "irrep_display" in result
        assert "irrep_primary" in result
        assert "irrep_aliases" in result
        assert "allowed_multipoles" in result
        assert "excited_irreps" in result
        assert len(result["excited_irreps"]) == 8


class TestAnalyzeCefSymmetry:
    def test_from_b_params(self):
        """Standalone: J + B_params -> classification."""
        result = analyze_cef_symmetry(4.0, "Oh", B_params={"B4": 0.01, "B6": 0.001})
        assert "irrep_primary" in result
        assert "eigenvalues" in result
        assert "excited_irreps" in result
        assert len(result["all_irreps"]) == 9

    def test_from_evecs(self):
        """Standalone: J + evecs -> classification."""
        H = build_hcef_matrix_J(4.0, {"B4": 0.01, "B6": 0.0}, "Oh")
        _, evecs = np.linalg.eigh(H)
        result = analyze_cef_symmetry(4.0, "Oh", evecs=evecs)
        assert "irrep_primary" in result
        assert "excited_irreps" in result

    def test_must_provide_one_input(self):
        with pytest.raises(ValueError):
            analyze_cef_symmetry(4.0, "Oh")

    def test_must_not_provide_both(self):
        H = build_hcef_matrix_J(4.0, {"B4": 0.01}, "Oh")
        _, evecs = np.linalg.eigh(H)
        with pytest.raises(ValueError):
            analyze_cef_symmetry(4.0, "Oh", B_params={"B4": 0.01}, evecs=evecs)
