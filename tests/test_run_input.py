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
            start_level = "L1"
            end_level = "L1"
            on_missing_upstream = "fail"
            read_first = true

            [checks]
            strict_mode = true
            eps_profile = "default"

            [extra]
            value = true
            """,
        )

        with pytest.raises(InputError, match="Unsupported top-level section"):
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
            start_level = "L1"
            end_level = "L1"
            on_missing_upstream = "fail"
            read_first = true

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
            F2 = 1.0
            F4 = 0.5
            F6 = 0.25

            [paths]
            output_root = "./outputs"

            [runtime]
            start_level = "L1"
            end_level = "L1"
            on_missing_upstream = "fail"
            read_first = true

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
            start_level = "L2"
            end_level = "L2"
            on_missing_upstream = "fail"
            read_first = true

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        with pytest.raises(InputError, match=r"Missing required section for selected window: \[inputs\]"):
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
            start_level = "L2"
            end_level = "L2"
            on_missing_upstream = "fail"
            read_first = true

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
            F2 = 1.0
            F4 = 0.6
            F6 = 0.4

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
            start_level = "L4"
            end_level = "L4"
            on_missing_upstream = "fail"
            read_first = true

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
            start_level = "L1"
            end_level = "L1"
            on_missing_upstream = "fail"
            read_first = true

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        assert cfg["physics"]["RE"] == "Yb"
        assert cfg["physics"]["F2"] == pytest.approx(12.32692411 * 0.3)
        assert cfg["physics"]["F4"] == pytest.approx(7.75712981 * 0.3)
        assert cfg["physics"]["F6"] == pytest.approx(5.58743756 * 0.3)
        assert cfg["sopt"]["zeta"] == pytest.approx(0.408)

    def test_explicit_values_override_re_defaults(self, tmp_path):
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
            F2 = 9.0
            F6 = 7.0

            [sopt]
            U = 6.0
            Jh = 0.3
            zeta = 0.5

            [paths]
            output_root = "./outputs"

            [runtime]
            start_level = "L1"
            end_level = "L1"
            on_missing_upstream = "fail"
            read_first = true

            [checks]
            strict_mode = true
            eps_profile = "default"
            """,
        )

        cfg = load_run_input(path)
        assert cfg["physics"]["F2"] == pytest.approx(9.0)
        assert cfg["physics"]["F4"] == pytest.approx(7.75712981 * 0.3)
        assert cfg["physics"]["F6"] == pytest.approx(7.0)
        assert cfg["sopt"]["zeta"] == pytest.approx(0.5)
