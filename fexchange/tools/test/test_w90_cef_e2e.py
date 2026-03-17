"""End-to-end CEF extraction tests on real YbOX Wannier90 data.

Covers YbOF, YbOCl, YbOBr × {NSOC, SOC} = 6 cases.
All are Yb³⁺ (f¹³), D3d structure, C3v CEF branch with sin mode for q=3 terms.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TOOLS_DIR.parent.parent  # fexchange/tools -> fexchange -> project root
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from w90_extract import read_hr, _build_U_r2c, _permute_ud_to_du  # noqa: E402
from w90_decompose_h_local import (  # noqa: E402
    decompose_h_local,
    decompose_orbital_tensors,
    extract_cef_stevens_fit,
    extract_local_stevens_fit,
    convert_akq_to_bkq,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

def _resolve_data_dir() -> Path:
    local = PROJECT_ROOT / "data"
    if local.exists():
        return local
    fallback = Path("/Users/lingzhi/Documents/Code/fexchange/data")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not resolve repository data directory.")


DATA_DIR = _resolve_data_dir()
MATERIALS = ["YbOF", "YbOCl", "YbOBr"]

N_ELE = 13
ZETA_ATOMIC = 300.0  # used for NSOC cases (no extracted ζ)
MODE_Q3 = "sin"  # D3d structure uses sin tesseral for q=3 Stevens terms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_h_cef_nsoc(mat: str) -> np.ndarray:
    """Extract 7×7 h_cef (meV) from NSOC wannier_hr.dat."""
    path = DATA_DIR / mat / "NSOC" / "wannier_hr.dat"
    H_R, _ = read_hr(str(path))
    h_raw = H_R[(0, 0, 0)][0:7, 0:7].copy()
    U = _build_U_r2c(spinor=False)
    return (U @ h_raw @ U.T.conj()) * 1000.0


def _extract_h_local_soc(mat: str) -> np.ndarray:
    """Extract 14×14 h_local (meV) from SOC wannier_hr.dat."""
    path = DATA_DIR / mat / "SOC" / "wannier_hr.dat"
    H_R, _ = read_hr(str(path))
    h_raw_14 = H_R[(0, 0, 0)][0:14, 0:14].copy()
    h_raw_14 = _permute_ud_to_du(h_raw_14)
    U14 = _build_U_r2c(spinor=True)
    return (U14 @ h_raw_14 @ U14.T.conj()) * 1000.0


def _parse_ref_cef(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# NSOC tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mat", MATERIALS)
class TestNSOC:
    """CEF extraction from NSOC Wannier data (no SOC separation needed)."""

    def test_scheme_a_residual(self, mat: str):
        """Tensor expansion onto C_q^(k) basis should be exact for ℓ=3."""
        h_cef = _extract_h_cef_nsoc(mat)
        result = decompose_orbital_tensors(h_cef)
        assert result["residual"] < 1e-10

    def test_scheme_b_ground_state(self, mat: str):
        """Ground term must be ²F_{7/2} for f¹³."""
        h_cef = _extract_h_cef_nsoc(mat)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=ZETA_ATOMIC, symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert sb["J0"] == 3.5
        assert sb["L0"] == 3
        assert sb["S0"] == pytest.approx(0.5)

    def test_scheme_b_kramers_doublets(self, mat: str):
        """J0=7/2 with Kramers ion → 4 doublets (pairwise degenerate)."""
        h_cef = _extract_h_cef_nsoc(mat)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=ZETA_ATOMIC, symmetry="C3v", mode_q3=MODE_Q3,
        )
        eigvals = sb["cef_splitting_meV"]
        assert len(eigvals) == 8
        for i in range(0, 8, 2):
            assert eigvals[i] == pytest.approx(eigvals[i + 1], abs=1e-6)

    def test_scheme_b_stevens_fit_quality(self, mat: str):
        """Stevens fit residual must be small with correct tesseral mode."""
        h_cef = _extract_h_cef_nsoc(mat)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=ZETA_ATOMIC, symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert sb["stevens_fit_residual"] < 0.01

    def test_scheme_b_nonzero_splitting(self, mat: str):
        """CEF splitting must be nonzero (real crystal field present)."""
        h_cef = _extract_h_cef_nsoc(mat)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=ZETA_ATOMIC, symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert sb["cef_splitting_meV"][-1] > 1.0  # at least 1 meV total


# ---------------------------------------------------------------------------
# Scheme A → B conversion tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mat", MATERIALS)
class TestAtoB:
    """Scheme A a_kq → Stevens B_k^q analytical conversion must match Scheme B."""

    def test_conversion_matches_scheme_b(self, mat: str):
        h_cef = _extract_h_cef_nsoc(mat)
        sa = decompose_orbital_tensors(h_cef)
        result = convert_akq_to_bkq(
            sa["a_kq"], n_ele=N_ELE, J0=3.5, L0=3, S0=0.5,
            symmetry="C3v", mode_q3=MODE_Q3,
        )

        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=ZETA_ATOMIC, symmetry="C3v", mode_q3=MODE_Q3,
        )

        for name in result["B_params"]:
            assert result["B_params"][name] == pytest.approx(
                sb["B_params"][name], abs=1e-6
            ), f"{mat} {name}: A→B={result['B_params'][name]}, SchemeB={sb['B_params'][name]}"

    def test_conversion_fit_residual(self, mat: str):
        """Analytical A→B conversion Stevens fit residual must be small."""
        h_cef = _extract_h_cef_nsoc(mat)
        sa = decompose_orbital_tensors(h_cef)
        result = convert_akq_to_bkq(
            sa["a_kq"], n_ele=N_ELE, J0=3.5, L0=3, S0=0.5,
            symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert result["stevens_fit_residual"] < 0.01


# ---------------------------------------------------------------------------
# SOC tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mat", MATERIALS)
class TestSOC:
    """CEF extraction from SOC Wannier data (SOC separated first)."""

    def test_soc_decomposition(self, mat: str):
        """SOC decomposition must succeed with reasonable ζ."""
        h_local = _extract_h_local_soc(mat)
        zeta, h_cef = decompose_h_local(h_local, rtol=1e-2)
        assert 300.0 < zeta < 500.0  # Yb³⁺ ζ range
        assert h_cef.shape == (7, 7)
        assert np.allclose(h_cef, h_cef.conj().T, atol=1e-8)

    def test_scheme_a_residual(self, mat: str):
        """Tensor expansion must be exact after SOC separation."""
        h_local = _extract_h_local_soc(mat)
        _, h_cef = decompose_h_local(h_local, rtol=1e-2)
        result = decompose_orbital_tensors(h_cef)
        assert result["residual"] < 1e-10

    def test_scheme_b_ground_state(self, mat: str):
        """Ground term must be ²F_{7/2} for f¹³."""
        h_local = _extract_h_local_soc(mat)
        zeta, h_cef = decompose_h_local(h_local, rtol=1e-2)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=zeta, symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert sb["J0"] == 3.5

    def test_scheme_b_kramers_doublets(self, mat: str):
        """J0=7/2 with Kramers ion → 4 doublets."""
        h_local = _extract_h_local_soc(mat)
        zeta, h_cef = decompose_h_local(h_local, rtol=1e-2)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=zeta, symmetry="C3v", mode_q3=MODE_Q3,
        )
        eigvals = sb["cef_splitting_meV"]
        assert len(eigvals) == 8
        for i in range(0, 8, 2):
            assert eigvals[i] == pytest.approx(eigvals[i + 1], abs=1e-6)

    def test_scheme_b_stevens_fit_quality(self, mat: str):
        """Stevens fit residual must be small with correct tesseral mode."""
        h_local = _extract_h_local_soc(mat)
        zeta, h_cef = decompose_h_local(h_local, rtol=1e-2)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=zeta, symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert sb["stevens_fit_residual"] < 0.01

    def test_scheme_b_nonzero_splitting(self, mat: str):
        """CEF splitting must be nonzero."""
        h_local = _extract_h_local_soc(mat)
        zeta, h_cef = decompose_h_local(h_local, rtol=1e-2)
        sb = extract_cef_stevens_fit(
            h_cef, n_ele=N_ELE,
            zeta=zeta, symmetry="C3v", mode_q3=MODE_Q3,
        )
        assert sb["cef_splitting_meV"][-1] > 1.0


def test_soc_direct_local_fit_matches_ybocl_reference():
    """Direct J0 fit from SOC h_local must reproduce the YbOCl HOLE reference."""
    h_local = np.loadtxt(
        DATA_DIR / "YbOCl" / "test" / "w90" / "w90_soc_h_local.dat",
        dtype=complex,
    )
    h_local_hole = -h_local.conj()
    zeta_hole, _ = decompose_h_local(h_local_hole, rtol=5e-3)
    out = extract_local_stevens_fit(
        h_local_hole,
        n_ele=1,
        zeta=zeta_hole,
        symmetry="C3v",
        mode_q3=MODE_Q3,
    )
    ref = _parse_ref_cef(DATA_DIR / "YbOCl" / "test" / "ref_cef_soc.txt")

    assert abs(zeta_hole) == pytest.approx(ref["soc_lambda_meV"], abs=0.1)
    assert out["stevens_fit_residual"] == pytest.approx(ref["cef_fit_error"], abs=1e-5)
    assert out["B_params"]["B20"] == pytest.approx(ref["B20"], abs=1e-6)
    assert out["B_params"]["B40"] == pytest.approx(ref["B40"], abs=1e-6)
    assert out["B_params"]["B60"] == pytest.approx(ref["B60"], abs=1e-6)
    assert out["B_params"]["B43"] == pytest.approx(ref["B43s"], abs=1e-6)
    assert out["B_params"]["B63"] == pytest.approx(ref["B63s"], abs=1e-6)
    assert out["B_params"]["B66"] == pytest.approx(ref["B66"], abs=1e-6)

    expected = np.array(
        [
            ref["level_0"],
            ref["level_1"],
            ref["level_2"],
            ref["level_3"],
            ref["level_4"],
            ref["level_5"],
            ref["level_6"],
            ref["level_7"],
        ]
    )
    expected -= expected.min()
    np.testing.assert_allclose(out["cef_splitting_meV"], expected, atol=2e-4)


def test_soc_direct_local_cli_writes_ybocl_reference(tmp_path):
    """CLI direct-local mode must emit the YbOCl reference Stevens config."""
    h_local = np.loadtxt(
        DATA_DIR / "YbOCl" / "test" / "w90" / "w90_soc_h_local.dat",
        dtype=complex,
    )
    h_local_hole = -h_local.conj()
    h_local_path = tmp_path / "h_local_hole.dat"
    np.savetxt(h_local_path, h_local_hole, fmt="%.12f")

    cef_cfg = tmp_path / "cef_config.toml"
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "w90_decompose_h_local.py"),
        "--h_local_dat", str(h_local_path),
        "--rtol", "5e-3",
        "--n_ele", "1",
        "--symmetry", "C3v",
        "--mode_q3", "sin",
        "--direct_local_fit",
        "--write_cef_config", str(cef_cfg),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    cfg = _parse_ref_cef(cef_cfg)
    ref = _parse_ref_cef(DATA_DIR / "YbOCl" / "test" / "ref_cef_soc.txt")
    assert cfg["B20"] == pytest.approx(ref["B20"], abs=1e-6)
    assert cfg["B40"] == pytest.approx(ref["B40"], abs=1e-6)
    assert cfg["B60"] == pytest.approx(ref["B60"], abs=1e-6)
    assert cfg["B43"] == pytest.approx(ref["B43s"], abs=1e-6)
    assert cfg["B63"] == pytest.approx(ref["B63s"], abs=1e-6)
    assert cfg["B66"] == pytest.approx(ref["B66"], abs=1e-6)
