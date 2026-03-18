"""Tests for io/run_input.py validation."""

from __future__ import annotations

import textwrap

import pytest

from fexchange.io.run_input import load_run_input
from fexchange.utils.errors import InputError


def _write(tmp_path, content: str):
    path = tmp_path / "run_input.toml"
    path.write_text(textwrap.dedent(content))
    return path


def _slater_from_jh_ratios(*, r42: float, r62: float, jh: float) -> tuple[float, float, float]:
    denom = 286.0 + 195.0 * r42 + 250.0 * r62
    f2 = 0.0 if jh == 0.0 else 6435.0 * jh / denom
    return f2, r42 * f2, r62 * f2


class TestRunInputValidation:
    def test_unknown_top_level_section_is_rejected(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "extra"
            title = "unsupported section"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L1"

            [checks]
            strict_mode = true
            eps_profile = "default"

            [extra]
            value = true
            """,
        )

        with pytest.raises(InputError, match="Unknown sections"):
            load_run_input(path)

    def test_l1_window_allows_missing_physics_section(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "r1"
            title = "missing physics"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L1"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )
        cfg = load_run_input(path)
        assert "physics" not in cfg
        assert "_derived" not in cfg

    def test_valid_minimal_l1_input(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "r2"
            title = "valid l1"

            [physics]
            n_ele = 2
            F2_ratio = 1.0
            F4_ratio = 0.5
            F6_ratio = 0.25

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L1"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        assert cfg["physics"]["n_ele"] == 2
        assert "_derived" in cfg

    def test_l2_window_requires_inputs_section(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "l2_missing_inputs"
            title = "missing inputs"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L2"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        with pytest.raises(InputError, match=r"\[inputs\] required"):
            load_run_input(path)

    def test_valid_l2_with_inputs(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "l2_valid_inputs"
            title = "valid l2 with inputs"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L2"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        assert cfg["inputs"]["hopping_file"] == str(hopping)

    def test_sources_labels_are_normalized_to_canonical_names(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        projector = tmp_path / "proj.npy"
        hopping.write_bytes(b"placeholder")
        projector.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "sources_normalized"
            title = "sources section normalized"

            [physics]
            n_ele = 13
            F2_ratio = 1.0
            F4_ratio = 0.6
            F6_ratio = 0.4

            [sources]
            hopping_label = "YbOCl_bond1"
            projection_label = "cef_ground"

            [inputs]
            hopping_file = "{hopping}"
            projector_file = "{projector}"

            [sopt]
            U = 6.0
            Jh = 0.3
            zeta = 0.4

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L4"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        assert cfg["sources"]["hopping_label"] == "YbOCl_bond1"
        assert cfg["sources"]["projection_label"] == "cef_ground"
        assert cfg["sources"]["hopping_name"] == "YbOCl_bond1"
        assert cfg["sources"]["kramer_name"] == "cef_ground"

    def test_re_defaults_fill_missing_physics_and_zeta(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "re_defaults"
            title = "re defaults"

            [physics]
            n_ele = 13
            RE = "Yb"

            [sopt]
            U = 6.0
            Jh = 0.3

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L1"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        assert cfg["physics"]["RE"] == "Yb"
        assert cfg["_branches"]["n"]["physics"]["F2"] == pytest.approx(12.32692411 * 0.3)
        assert cfg["_branches"]["n"]["physics"]["F4"] == pytest.approx(7.75712981 * 0.3)
        assert cfg["_branches"]["n"]["physics"]["F6"] == pytest.approx(5.58743756 * 0.3)
        assert cfg["sopt"]["zeta"] == pytest.approx(408.0)

    def test_f_ratio_takes_priority_over_re(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "re_override"
            title = "re override"

            [physics]
            n_ele = 13
            RE = "Yb"
            F2_ratio = 9.0
            F4_ratio = 5.0
            F6_ratio = 7.0

            [sopt]
            U = 6.0
            Jh = 0.3
            zeta = 0.5

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L1"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        expected_r42 = 5.0 / 9.0
        expected_r62 = 7.0 / 9.0
        expected_f2, expected_f4, expected_f6 = _slater_from_jh_ratios(
            r42=expected_r42,
            r62=expected_r62,
            jh=0.3,
        )
        assert cfg["_derived"]["r42"] == pytest.approx(expected_r42)
        assert cfg["_derived"]["r62"] == pytest.approx(expected_r62)
        assert cfg["_branches"]["n"]["physics"]["F2"] == pytest.approx(expected_f2)
        assert cfg["_branches"]["n"]["physics"]["F4"] == pytest.approx(expected_f4)
        assert cfg["_branches"]["n"]["physics"]["F6"] == pytest.approx(expected_f6)
        assert cfg["sopt"]["zeta"] == pytest.approx(0.5)

    def test_main_physics_values_are_resolved_from_jh_and_input_ratios(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "main_physics_from_jh"
            title = "main physics from jh"

            [physics]
            n_ele = 11
            F2_ratio = 1200.0
            F4_ratio = 750.0
            F6_ratio = 540.0

            [sopt]
            U = 5000.0
            Jh = 400.0
            zeta = 330.0

            [sources]
            hopping_label = "main_physics_from_jh_hop"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L3"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)

        expected_f2, expected_f4, expected_f6 = _slater_from_jh_ratios(
            r42=750.0 / 1200.0,
            r62=540.0 / 1200.0,
            jh=400.0,
        )
        assert cfg["_derived"]["r42"] == pytest.approx(750.0 / 1200.0)
        assert cfg["_derived"]["r62"] == pytest.approx(540.0 / 1200.0)
        assert cfg["_branches"]["n"]["physics"]["F2"] == pytest.approx(expected_f2)
        assert cfg["_branches"]["n"]["physics"]["F4"] == pytest.approx(expected_f4)
        assert cfg["_branches"]["n"]["physics"]["F6"] == pytest.approx(expected_f6)

    def test_branch_sections_are_normalized_with_mev_units(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "branch_mev"
            title = "branch mev"

            [units]
            energy = "meV"

            [physics]
            n_ele = 13
            RE = "Yb"

            [physics_nm1]
            RE = "Tm"

            [sopt]
            U = 6000.0
            Jh = 300.0

            [sopt_nm1]
            U = 0.0
            Jh = 500.0
            offset = 24203.2925903531

            [sopt_np1]
            U = 0.0
            offset = 6400.0

            [sources]
            hopping_label = "branch_mev_hop"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L3"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)

        assert cfg["units"]["energy"] == "meV"
        assert cfg["_branches"]["n"]["physics"]["n_ele"] == 13
        assert cfg["_branches"]["nm1"]["physics"]["n_ele"] == 12
        assert cfg["_branches"]["np1"]["physics"]["n_ele"] == 14
        assert cfg["_branches"]["n"]["physics"]["F2"] == pytest.approx(12.32692411 * 300.0)
        assert cfg["_branches"]["nm1"]["physics"]["F2"] == pytest.approx(12.32881596 * 500.0)
        assert cfg["_branches"]["n"]["sopt"]["zeta"] == pytest.approx(408.0)
        assert cfg["_branches"]["nm1"]["sopt"]["zeta"] == pytest.approx(372.7348)
        assert cfg["_branches"]["nm1"]["sopt"]["offset"] == pytest.approx(24203.2925903531)
        assert cfg["_branches"]["np1"]["sopt"]["offset"] == pytest.approx(6400.0)

    def test_branch_sections_fall_back_to_main_values_when_omitted(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "branch_fallback"
            title = "branch fallback"

            [physics]
            n_ele = 11
            F2_ratio = 1200.0
            F4_ratio = 750.0
            F6_ratio = 540.0

            [sopt]
            U = 5000.0
            Jh = 400.0
            zeta = 330.0

            [sopt_nm1]
            offset = 12000.0

            [sources]
            hopping_label = "branch_fallback_hop"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L3"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)

        expected_f2, expected_f4, expected_f6 = _slater_from_jh_ratios(
            r42=750.0 / 1200.0,
            r62=540.0 / 1200.0,
            jh=400.0,
        )
        assert cfg["_branches"]["n"]["physics"]["F2"] == pytest.approx(expected_f2)
        assert cfg["_branches"]["n"]["physics"]["F4"] == pytest.approx(expected_f4)
        assert cfg["_branches"]["n"]["physics"]["F6"] == pytest.approx(expected_f6)
        assert cfg["_branches"]["nm1"]["physics"]["F2"] == pytest.approx(expected_f2)
        assert cfg["_branches"]["nm1"]["physics"]["F4"] == pytest.approx(expected_f4)
        assert cfg["_branches"]["nm1"]["physics"]["F6"] == pytest.approx(expected_f6)
        assert cfg["_branches"]["np1"]["physics"]["F2"] == pytest.approx(expected_f2)
        assert cfg["_branches"]["np1"]["sopt"]["U"] == pytest.approx(5000.0)
        assert cfg["_branches"]["np1"]["sopt"]["zeta"] == pytest.approx(330.0)
        assert cfg["_branches"]["nm1"]["sopt"]["offset"] == pytest.approx(12000.0)
        assert cfg["_branches"]["np1"]["sopt"]["offset"] == pytest.approx(0.0)

    def test_branch_sections_share_resolved_main_physics_when_jh_is_zero(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "branch_share_main_zero_jh"
            title = "branch share main zero jh"

            [physics]
            n_ele = 13
            RE = "Yb"
            F2_ratio = 7396.154466
            F4_ratio = 4654.277886
            F6_ratio = 3352.462536

            [sopt]
            U = 6000.0
            Jh = 0.0
            zeta = 391.738

            [sources]
            hopping_label = "branch_share_main_zero_jh_hop"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L3"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)

        assert cfg["_derived"]["r42"] == pytest.approx(4654.277886 / 7396.154466)
        assert cfg["_derived"]["r62"] == pytest.approx(3352.462536 / 7396.154466)
        for branch_name in ("nm1", "n", "np1"):
            assert cfg["_branches"][branch_name]["physics"]["F2"] == pytest.approx(0.0)
            assert cfg["_branches"][branch_name]["physics"]["F4"] == pytest.approx(0.0)
            assert cfg["_branches"][branch_name]["physics"]["F6"] == pytest.approx(0.0)

    def test_sopt_defaults_include_energy_reference_and_offset(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "sopt_defaults"
            title = "sopt defaults"

            [physics]
            n_ele = 13
            F2_ratio = 1200.0
            F4_ratio = 750.0
            F6_ratio = 540.0

            [sopt]
            U = 5000.0
            Jh = 400.0
            zeta = 330.0

            [sources]
            hopping_label = "sopt_defaults_hop"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L3"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)

        assert cfg["sopt"]["energy_reference"] == "lsjm_ground"
        assert cfg["sopt"]["offset"] == pytest.approx(0.0)
        assert cfg["_branches"]["n"]["sopt"]["offset"] == pytest.approx(0.0)

    def test_invalid_energy_reference_is_rejected(self, tmp_path):
        hopping = tmp_path / "hop.npy"
        hopping.write_bytes(b"placeholder")
        path = _write(
            tmp_path,
            f"""
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "bad_energy_reference"
            title = "bad energy reference"

            [physics]
            n_ele = 13
            F2_ratio = 1200.0
            F4_ratio = 750.0
            F6_ratio = 540.0

            [sopt]
            U = 5000.0
            Jh = 400.0
            zeta = 330.0
            energy_reference = "bad_mode"

            [sources]
            hopping_label = "bad_energy_reference_hop"

            [inputs]
            hopping_file = "{hopping}"

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L3"

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        with pytest.raises(InputError, match="energy_reference"):
            load_run_input(path)

    def test_f_ratio_partial_is_rejected(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "partial"
            title = "partial ratio"

            [physics]
            n_ele = 13
            F2_ratio = 12.0

            [model]
            scheme = "RS"

            [sopt]
            U = 6000.0
            Jh = 600.0
            zeta = 400.0

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L4"

            [checks]
            strict_mode = true
            eps_profile = "default"

            [sources]
            hopping_label = "test"
            projection_label = "test"

            [inputs]
            hopping_file = "t.dat"
            projector_file = "w.dat"
            """,
        )

        with pytest.raises(InputError, match="F_ratios must appear together"):
            load_run_input(path)

    def test_side_branch_fallback_to_n(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "fall"
            title = "fallback"

            [physics]
            n_ele = 13
            F2_ratio = 12.0
            F4_ratio = 7.0
            F6_ratio = 5.0

            [model]
            scheme = "RS"

            [sopt]
            U = 6000.0
            Jh = 600.0
            zeta = 400.0

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L4"

            [checks]
            strict_mode = true
            eps_profile = "default"

            [sources]
            hopping_label = "test"
            projection_label = "test"

            [inputs]
            hopping_file = "t.dat"
            projector_file = "w.dat"
            """,
        )

        cfg = load_run_input(path)

        r42_n = cfg["_branches"]["n"]["derived"]["r42"]
        r42_nm1 = cfg["_branches"]["nm1"]["derived"]["r42"]
        r42_np1 = cfg["_branches"]["np1"]["derived"]["r42"]
        assert r42_nm1 == pytest.approx(r42_n)
        assert r42_np1 == pytest.approx(r42_n)
        assert cfg["_branches"]["nm1"]["physics"]["F2"] == pytest.approx(
            cfg["_branches"]["n"]["physics"]["F2"]
        )

    def test_side_branch_offset_defaults_to_zero(self, tmp_path):
        path = _write(
            tmp_path,
            """
            schema_version = "fxe.run_input.v1"
            standard_version = "2026-02"
            run_id = "off"
            title = "offset"

            [physics]
            n_ele = 13
            F2_ratio = 12.0
            F4_ratio = 7.0
            F6_ratio = 5.0

            [model]
            scheme = "RS"

            [sopt]
            U = 6000.0
            Jh = 600.0
            zeta = 400.0
            offset = 99.0

            [sopt_nm1]
            zeta = 350.0

            [paths]
            output_root = "./outputs"

            [runtime]
            end_level = "L4"

            [checks]
            strict_mode = true
            eps_profile = "default"

            [sources]
            hopping_label = "test"
            projection_label = "test"

            [inputs]
            hopping_file = "t.dat"
            projector_file = "w.dat"
            """,
        )

        cfg = load_run_input(path)

        assert cfg["_branches"]["n"]["sopt"]["offset"] == pytest.approx(99.0)
        assert cfg["_branches"]["nm1"]["sopt"]["offset"] == pytest.approx(0.0)
        assert cfg["_branches"]["nm1"]["sopt"]["zeta"] == pytest.approx(350.0)
        assert cfg["_branches"]["nm1"]["sopt"]["U"] == pytest.approx(6000.0)
        assert cfg["_branches"]["nm1"]["sopt"]["Jh"] == pytest.approx(600.0)
