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
