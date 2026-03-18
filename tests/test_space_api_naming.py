from __future__ import annotations

from pathlib import Path
import re

import fexchange.core.space_j as space_j
import fexchange.core.space_ls as space_ls
import fexchange.utils.constants as constants


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


def test_runtime_code_has_no_removed_space_identifiers():
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
        root / "hamiltonian" / "hcef.py",
    ]

    for path in runtime_files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token} found in {path}"


def test_angular_module_is_removed():
    root = Path(__file__).resolve().parents[1] / "fexchange"
    assert not (root / "hamiltonian" / "angular.py").exists()


def test_shared_constants_live_in_single_utils_entrypoint():
    expected_names = (
        "N_ORB",
        "ELL",
        "SIGMA_0",
        "SIGMA_X",
        "SIGMA_Y",
        "SIGMA_Z",
        "RUN_SCHEMA_VERSION",
        "STANDARD_VERSION",
        "LEVELS",
        "LEVEL_ORDER",
    )

    for name in expected_names:
        assert hasattr(constants, name), f"missing shared constant {name}"


def test_runtime_files_do_not_redefine_shared_constants():
    root = Path(__file__).resolve().parents[1] / "fexchange"
    forbidden = (
        r"^N_ORB\s*:",
        r"^ELL\s*:",
        r"^RUN_SCHEMA_VERSION\s*=",
        r"^STANDARD_VERSION\s*=",
        r"^LEVELS\s*=",
        r"^LEVEL_ORDER\s*=",
    )

    runtime_files = [
        root / "core" / "fock.py",
        root / "core" / "space_ls.py",
        root / "io" / "run_input.py",
        root / "cli.py",
    ]

    for path in runtime_files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert re.search(pattern, text, flags=re.MULTILINE) is None, f"{pattern} found in {path}"


def test_space_j_reuses_shared_pauli_constants():
    root = Path(__file__).resolve().parents[1] / "fexchange"
    text = (root / "core" / "space_j.py").read_text(encoding="utf-8")
    forbidden = (
        "sigma_x = np.array(",
        "sigma_y = np.array(",
        "sigma_z = np.array(",
    )

    for token in forbidden:
        assert token not in text, f"{token} found in core/space_j.py"
