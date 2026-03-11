"""Tests for io/disk.py (05-00)."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from fexchange.io.disk import (
    fmt12, core_dir_token, ujhz_dir_token,
    build_stage_path, atomic_write_npz, atomic_write_json,
    build_meta,
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
