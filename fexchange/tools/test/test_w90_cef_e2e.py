"""End-to-end CEF extraction tests on real YbOX Wannier90 data.

Covers YbOF, YbOCl, YbOBr × {NSOC, SOC} = 6 cases.
All are Yb³⁺ (f¹³), D3d structure, C3v CEF branch with sin mode for q=3 terms.
"""

from __future__ import annotations

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
    convert_akq_to_bkq,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
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
