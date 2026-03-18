"""Tests for pipeline refactoring helpers."""
from __future__ import annotations

from fexchange.pipeline.artifacts import ARTIFACT_FILE_SPEC
from fexchange.pipeline.keys import extract_source_names


def test_extract_source_names_from_full_cfg():
    cfg = {"sources": {"hopping_name": "bond_a", "kramer_name": "proj_a"}}
    sn = extract_source_names(cfg)
    assert sn.hopping_name == "bond_a"
    assert sn.kramer_name == "proj_a"


def test_extract_source_names_missing_sources():
    cfg = {}
    sn = extract_source_names(cfg)
    assert sn.hopping_name == ""
    assert sn.kramer_name == ""


def test_extract_source_names_non_dict_sources():
    cfg = {"sources": "invalid"}
    sn = extract_source_names(cfg)
    assert sn.hopping_name == ""
    assert sn.kramer_name == ""


def test_extract_source_names_partial():
    cfg = {"sources": {"hopping_name": "bond_a"}}
    sn = extract_source_names(cfg)
    assert sn.hopping_name == "bond_a"
    assert sn.kramer_name == ""


def test_artifact_file_spec_has_all_levels():
    expected = {"LMSM", "LSJM", "L0", "L1", "L2", "L3", "L4"}
    assert set(ARTIFACT_FILE_SPEC.keys()) == expected


def test_artifact_file_spec_lmsm_structure():
    spec = ARTIFACT_FILE_SPEC["LMSM"]
    assert ("V.npz", ["V_fock"]) in spec
    assert ("E_terms.npz", ["coef_F0", "coef_F2", "coef_F4", "coef_F6"]) in spec
    assert ("meta.json", None) in spec


def test_artifact_file_spec_lsjm_has_coef_zeta():
    spec = ARTIFACT_FILE_SPEC["LSJM"]
    e_terms_entry = [s for s in spec if s[0] == "E_terms.npz"][0]
    assert "coef_zeta" in e_terms_entry[1]
