"""Tests for io/disk.py (05-00)."""

import json
import pytest
import numpy as np
import tempfile
from pathlib import Path

from fexchange.io.disk import (
    fmt12, core_dir_token, ujhz_dir_token,
    build_stage_path, atomic_write_npz, atomic_write_json,
    build_meta, point_result_filename, write_point_result_txt,
)


class TestPathTokens:
    def test_fmt12(self):
        assert fmt12(1.0) == "1.000000000000"
        assert len(fmt12(0.123456789012).split(".")[-1]) == 12

    def test_core_dir_token(self):
        token = core_dir_token(6, 0.5, 0.25, "RS")
        assert "n-6" in token
        assert "r42-" in token
        assert "scheme-RS" in token

    def test_build_stage_path(self):
        p = build_stage_path("./outputs", "LMSM", n=2, r42=0.5, r62=0.25)
        assert "LMSM" in str(p)

    def test_l2_and_l4_paths_include_canonical_labels(self):
        p_l2 = build_stage_path(
            "./outputs",
            "L2",
            n=2,
            r42=0.5,
            r62=0.25,
            hopping_name="bond_a",
        )
        p_l4 = build_stage_path(
            "./outputs",
            "L4",
            n=2,
            r42=0.5,
            r62=0.25,
            hopping_name="bond_a",
            U=1.0,
            Jh=2.0,
            zeta=3.0,
            kramer_name="proj_a",
        )
        assert "hopping/bond_a/L2" in str(p_l2)
        assert "hopping/bond_a" in str(p_l4)
        assert "kramer/proj_a/L4" in str(p_l4)

    def test_l4_path_can_include_branch_and_denominator_signatures(self):
        p_l4 = build_stage_path(
            "./outputs",
            "L4",
            n=2,
            r42=0.5,
            r62=0.25,
            hopping_name="bond_a",
            U=1.0,
            Jh=2.0,
            zeta=3.0,
            kramer_name="proj_a",
            branch_signature="branchsig-deadbeef",
            denom_signature="denomsig-cafebabe",
        )
        assert "branchsig-deadbeef" in str(p_l4)
        assert "denomsig-cafebabe" in str(p_l4)
        assert "kramer/proj_a/L4" in str(p_l4)

    def test_point_result_filename_includes_cfg_signature_and_fixed_6_decimals(self):
        name = point_result_filename(
            RE="Yb",
            n_ele=13,
            hopping_label="bond_a",
            projection_label="proj_a",
            U=8000.0,
            Jh=300.0,
            zeta=391.1234567,
            cfg_signature="cfg-deadbeefcafe",
        )
        assert name == "Yb_13_bond_a_proj_a_8000.000000_300.000000_391.123457__cfg-deadbeefcafe.txt"

    def test_write_point_result_txt_writes_sidecar_meta(self, tmp_path):
        path = write_point_result_txt(
            str(tmp_path),
            RE="Yb",
            n_ele=13,
            hopping_label="bond_a",
            projection_label="proj_a",
            U=8000.0,
            Jh=300.0,
            zeta=391.1234567,
            J_mu=np.eye(3),
            mapping_residual=1.0e-8,
            cfg_signature="cfg-deadbeefcafe",
            cfg_meta={
                "cfg_signature": "cfg-deadbeefcafe",
                "resolved_branches": {
                    "n": {"physics": {"F2": 1.0, "F4": 0.6, "F6": 0.4}},
                    "nm1": {"physics": {"F2": 2.0, "F4": 1.2, "F6": 0.8}},
                    "np1": {"physics": {"F2": 3.0, "F4": 1.8, "F6": 1.2}},
                },
            },
        )

        assert path.name == "Yb_13_bond_a_proj_a_8000.000000_300.000000_391.123457__cfg-deadbeefcafe.txt"
        meta_path = path.with_suffix(".meta.json")
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["cfg_signature"] == "cfg-deadbeefcafe"
        assert data["resolved_branches"]["nm1"]["physics"]["F2"] == pytest.approx(2.0)


class TestAtomicWrite:
    def test_round_trip_npz(self, tmp_path):
        path = tmp_path / "test.npz"
        arr = np.random.rand(3, 3)
        atomic_write_npz(path, data=arr)
        loaded = dict(np.load(str(path)))
        np.testing.assert_allclose(loaded["data"], arr)

    def test_round_trip_json(self, tmp_path):
        import json
        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        atomic_write_json(path, data)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["key"] == "value"
        assert loaded["num"] == 42
