from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "outputs" / "llw_diagram.py"


def _load_module():
    assert SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("outputs.llw_diagram", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("J", "expected_f4", "expected_f6"),
    [
        (4.0, 1680.0, 100800.0),
        (6.0, 7560.0, 1330560.0),
        (8.0, 23520.0, 5872799.799986662),
    ],
)
def test_compute_f_factors_matches_oh_reference_values(J, expected_f4, expected_f6):
    module = _load_module()

    f4, f6 = module.compute_F_factors(J)

    assert f4 == pytest.approx(expected_f4)
    assert f6 == pytest.approx(expected_f6)


def test_llw_scan_j4_matches_reference_sample_points():
    module = _load_module()

    scan = module.llw_scan(4.0, n_points=5)

    assert scan["x_grid"].tolist() == pytest.approx(np.linspace(-1.0, 1.0, 5).tolist())
    assert scan["energies"].shape == (5, 9)
    assert scan["irreps"][0] == [
        "Gamma1",
        "Gamma4",
        "Gamma4",
        "Gamma4",
        "Gamma3",
        "Gamma3",
        "Gamma5",
        "Gamma5",
        "Gamma5",
    ]
    assert scan["irreps"][2] == [
        "Gamma1",
        "Gamma5",
        "Gamma5",
        "Gamma5",
        "Gamma4",
        "Gamma4",
        "Gamma4",
        "Gamma3",
        "Gamma3",
    ]
    assert scan["irreps"][4] == [
        "Gamma5",
        "Gamma5",
        "Gamma5",
        "Gamma3",
        "Gamma3",
        "Gamma4",
        "Gamma4",
        "Gamma4",
        "Gamma1",
    ]
