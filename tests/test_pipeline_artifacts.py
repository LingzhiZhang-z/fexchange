"""Tests for optional J_mu payload in persisted L4 artifacts."""

from __future__ import annotations

import numpy as np

from fexchange.io.disk import build_stage_path
from fexchange.pipeline.artifacts import persist_l4, try_load_l4
from fexchange.pipeline.keys import level_key


def _cfg(tmp_path):
    return {
        "standard_version": "2026-02",
        "paths": {"output_root": str(tmp_path / "outputs")},
        "sopt": {"U": 1.0, "Jh": 2.0, "zeta": 3.0, "offset": 0.0, "energy_reference": "lsjm_ground"},
        "sources": {"hopping_name": "bond_a", "kramer_name": "proj_a"},
    }


def _branch_payload(*, n_ele, F2, F4, F6, U, Jh, zeta, offset=0.0):
    return {
        "physics": {"n_ele": n_ele, "F2": F2, "F4": F4, "F6": F6, "RE": "auto"},
        "sopt": {"U": U, "Jh": Jh, "zeta": zeta, "offset": offset},
        "derived": {"r42": F4 / F2, "r62": F6 / F2},
    }


def _cfg_with_branches(tmp_path, *, nm1_offset=0.0, nm1_f4=6.0, n_offset=0.0, energy_reference="lsjm_ground"):
    cfg = _cfg(tmp_path)
    cfg["physics"] = {"n_ele": 13, "F2": 10.0, "F4": 6.0, "F6": 4.0}
    cfg["sopt"]["offset"] = n_offset
    cfg["sopt"]["energy_reference"] = energy_reference
    cfg["_branches"] = {
        "n": _branch_payload(n_ele=13, F2=10.0, F4=6.0, F6=4.0, U=1.0, Jh=2.0, zeta=3.0, offset=n_offset),
        "nm1": _branch_payload(n_ele=12, F2=10.0, F4=nm1_f4, F6=4.0, U=1.0, Jh=2.0, zeta=3.0, offset=nm1_offset),
        "np1": _branch_payload(n_ele=14, F2=10.0, F4=6.0, F6=4.0, U=1.0, Jh=2.0, zeta=3.0),
    }
    return cfg


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
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L4",
        n=13,
        r42=0.6,
        r62=0.4,
        scheme="RS",
        hopping_name="bond_a",
        U=1.0,
        Jh=2.0,
        zeta=3.0,
        kramer_name="proj_a",
    )

    persist_l4(
        stage_dir,
        cfg,
        result,
        n_ele=13,
        r42=0.6,
        r62=0.4,
        labels=labels,
        W=W,
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
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L4",
        n=13,
        r42=0.6,
        r62=0.4,
        scheme="RS",
        hopping_name="bond_a",
        U=1.0,
        Jh=2.0,
        zeta=3.0,
        kramer_name="proj_a",
    )

    persist_l4(
        stage_dir,
        cfg,
        result,
        n_ele=13,
        r42=0.6,
        r62=0.4,
        labels=labels,
        W=W,
    )
    loaded = try_load_l4(stage_dir)
    assert loaded is not None
    assert "J_mu" not in loaded
    assert "mapping_residual" not in loaded


def test_l1_key_changes_when_branch_ratios_change(tmp_path):
    cfg_a = _cfg_with_branches(tmp_path, nm1_f4=6.0)
    cfg_b = _cfg_with_branches(tmp_path, nm1_f4=5.0)

    key_a = level_key("L1", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_a)
    key_b = level_key("L1", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_b)

    assert key_a != key_b


def test_l3_key_changes_when_branch_denominator_changes(tmp_path):
    cfg_a = _cfg_with_branches(tmp_path, nm1_offset=0.0)
    cfg_b = _cfg_with_branches(tmp_path, nm1_offset=10.0)

    key_a = level_key("L3", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_a)
    key_b = level_key("L3", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_b)

    assert key_a != key_b


def test_l3_key_changes_when_energy_reference_changes(tmp_path):
    cfg_a = _cfg_with_branches(tmp_path, energy_reference="lsjm_ground")
    cfg_b = _cfg_with_branches(tmp_path, energy_reference="zero")

    key_a = level_key("L3", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_a)
    key_b = level_key("L3", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_b)

    assert key_a != key_b


def test_l3_key_changes_when_main_sector_offset_changes(tmp_path):
    cfg_a = _cfg_with_branches(tmp_path, n_offset=0.0)
    cfg_b = _cfg_with_branches(tmp_path, n_offset=10.0)

    key_a = level_key("L3", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_a)
    key_b = level_key("L3", n_ele=13, r42=0.6, r62=0.4, cfg=cfg_b)

    assert key_a != key_b
