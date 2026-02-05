"""Tests for utils/params.py - Parameter file loading.

Expected behavior:
- load_params_file loads JSON config
- Relative paths are resolved relative to the config file location
- Only .json files are supported
- Returns a dict with parameter values

Expected outputs:
- Relative path "data/file.npy" in /home/user/config.json
  -> resolves to /home/user/data/file.npy
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fexchange.utils.params import PATH_KEYS, load_params_file


class TestLoadParamsFile:
    """Test parameter file loading."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Basic JSON loading should work."""
        config = {"n": 2, "some_value": 100}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = load_params_file(config_path)

        assert result["n"] == 2
        assert result["some_value"] == 100

    def test_relative_path_resolution(self, tmp_path: Path) -> None:
        """Relative paths should be resolved relative to config file."""
        # Create subdirectory
        (tmp_path / "data").mkdir()

        config = {"t_npy": "data/t.npy", "n": 2}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = load_params_file(config_path)

        expected_path = (tmp_path / "data" / "t.npy").resolve()
        assert result["t_npy"] == str(expected_path)

    def test_absolute_path_preserved(self, tmp_path: Path) -> None:
        """Absolute paths should be preserved."""
        abs_path = "/absolute/path/to/file.npy"
        config = {"t_npy": abs_path, "n": 2}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = load_params_file(config_path)

        assert result["t_npy"] == abs_path

    def test_non_json_raises(self, tmp_path: Path) -> None:
        """Non-JSON files should raise ValueError."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("n: 2")

        with pytest.raises(ValueError, match="JSON"):
            load_params_file(config_path)

    def test_non_dict_raises(self, tmp_path: Path) -> None:
        """JSON that's not an object should raise."""
        config_path = tmp_path / "config.json"
        config_path.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="object"):
            load_params_file(config_path)

    def test_path_keys_defined(self) -> None:
        """PATH_KEYS should contain expected keys."""
        expected = {"t_npy", "ground_npz", "cache_dir", "out_dir"}
        assert expected.issubset(PATH_KEYS)

    def test_none_path_ignored(self, tmp_path: Path) -> None:
        """null/None path values should be ignored."""
        config = {"t_npy": None, "n": 2}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = load_params_file(config_path)

        assert result["t_npy"] is None

    def test_all_path_keys_resolved(self, tmp_path: Path) -> None:
        """All PATH_KEYS should be resolved if present."""
        config = {key: f"relative/{key}" for key in PATH_KEYS}
        config["n"] = 2
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        result = load_params_file(config_path)

        for key in PATH_KEYS:
            # Should be absolute path now
            assert Path(result[key]).is_absolute()
