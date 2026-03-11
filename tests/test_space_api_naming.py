from __future__ import annotations

from pathlib import Path

import fexchange.core.space_j as space_j
import fexchange.core.space_ls as space_ls


def test_space_ls_exports_singular_api_only():
    assert hasattr(space_ls, "build_space_ls_operator")
    assert hasattr(space_ls, "build_space_ls_matrix")
    assert not hasattr(space_ls, "build_space_ls_matrices")


def test_space_j_exports_singular_api_only():
    assert hasattr(space_j, "build_space_j_operator")
    assert hasattr(space_j, "build_time_reversal_operator")
    assert not hasattr(space_j, "build_J_matrices")
    assert not hasattr(space_j, "build_time_reversal_J_basis")
    assert not hasattr(space_j, "build_space_j_matrix")


def test_runtime_code_has_no_legacy_space_identifiers():
    root = Path(__file__).resolve().parents[1] / "fexchange"
    forbidden = (
        "build_space_ls_matrices",
        "build_J_matrices",
        "build_time_reversal_J_basis",
        "build_angular_operators",
    )

    runtime_files = [
        root / "core" / "space_ls.py",
        root / "core" / "space_j.py",
        root / "models" / "projection.py",
        root / "models" / "hcef.py",
        root / "core" / "symmetry.py",
    ]

    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} found in {path}"


def test_angular_module_is_removed():
    root = Path(__file__).resolve().parents[1] / "fexchange"
    assert not (root / "models" / "angular.py").exists()


def test_kramers_uses_space_j_pauli_decompose():
    root = Path(__file__).resolve().parents[1] / "fexchange"
    text = (root / "models" / "kramers.py").read_text(encoding="utf-8")
    assert "from fexchange.core.space_j import pauli_decompose" in text
