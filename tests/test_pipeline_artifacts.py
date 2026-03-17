"""Tests for optional J_mu payload in persisted L4 artifacts."""

from __future__ import annotations

import numpy as np

from fexchange.pipeline.artifacts import persist_l4, try_load_l4


def _cfg(tmp_path):
    return {
        "standard_version": "2026-02",
        "paths": {"output_root": str(tmp_path / "outputs")},
        "sopt": {"U": 1.0, "Jh": 2.0, "zeta": 3.0},
        "sources": {"hopping_name": "bond_a", "kramer_name": "proj_a"},
    }


def test_persist_l4_round_trips_optional_jmu_payload(tmp_path):
    cfg = _cfg(tmp_path)
    result = {
        "h_mu_abcd": np.zeros((2, 2, 2, 2), dtype=np.complex128),
        "Heff_mu_abcd": np.zeros((2, 2, 2, 2), dtype=np.complex128),
        "J_mu": np.eye(3, dtype=float),
        "mapping_residual": 1.0e-12,
    }
    labels = np.arange(16, dtype=np.int64).reshape(4, 4)
    W = np.eye(2, dtype=np.complex128)

    persist_l4(
        cfg,
        result,
        n_ele=13,
        r42=0.6,
        r62=0.4,
        scheme="RS",
        labels=labels,
        W=W,
    )

    stage_dir = (
        tmp_path
        / "outputs"
        / "core"
        / "n-13_r42-0.600000000000_r62-0.400000000000_scheme-RS"
        / "hopping"
        / "bond_a"
        / "U-1.000000000000_Jh-2.000000000000_z-3.000000000000"
        / "kramer"
        / "proj_a"
        / "L4"
    )
    loaded = try_load_l4(stage_dir)
    assert loaded is not None
    np.testing.assert_allclose(loaded["J_mu"], np.eye(3))
    assert loaded["mapping_residual"] == 1.0e-12


def test_try_load_l4_accepts_artifact_without_jmu(tmp_path):
    cfg = _cfg(tmp_path)
    result = {
        "h_mu_abcd": np.zeros((3, 3, 3, 3), dtype=np.complex128),
        "Heff_mu_abcd": np.zeros((3, 3, 3, 3), dtype=np.complex128),
    }
    labels = np.arange(324, dtype=np.int64).reshape(81, 4)
    W = np.eye(3, dtype=np.complex128)

    persist_l4(
        cfg,
        result,
        n_ele=13,
        r42=0.6,
        r62=0.4,
        scheme="RS",
        labels=labels,
        W=W,
    )

    stage_dir = (
        tmp_path
        / "outputs"
        / "core"
        / "n-13_r42-0.600000000000_r62-0.400000000000_scheme-RS"
        / "hopping"
        / "bond_a"
        / "U-1.000000000000_Jh-2.000000000000_z-3.000000000000"
        / "kramer"
        / "proj_a"
        / "L4"
    )
    loaded = try_load_l4(stage_dir)
    assert loaded is not None
    assert "J_mu" not in loaded
    assert "mapping_residual" not in loaded
