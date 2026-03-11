"""API surface tests for CEF-only Stevens module."""

from pathlib import Path

import fexchange.core.stevens as stevens


def test_only_cef_entrypoint_is_exposed():
    assert hasattr(stevens, "build_cef_stevens_operators")
    assert not hasattr(stevens, "build_stevens_operator")
    assert not hasattr(stevens, "build_stevens_set")
    assert not hasattr(stevens, "convert_stevens_to_tensors")
    assert not hasattr(stevens, "convert_tensor_to_stevens")


def test_spherical_tensors_module_removed():
    root = Path(__file__).resolve().parents[1] / "fexchange" / "core"
    assert not (root / "spherical_tensors.py").exists()
