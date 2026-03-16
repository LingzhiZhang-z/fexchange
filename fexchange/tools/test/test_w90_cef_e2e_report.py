"""
Diagnostic report for CEF extraction on real YbOX Wannier90 data.

Not a pytest test — run directly to print a summary table:

    python tools/test/test_w90_cef_e2e_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from w90_extract import read_hr, _build_U_r2c, _permute_ud_to_du
from w90_decompose_h_local import (
    decompose_h_local,
    decompose_orbital_tensors,
    extract_cef_stevens_fit,
)

DATA_DIR = ROOT_DIR / "data"
MATERIALS = ["YbOF", "YbOCl", "YbOBr"]

# Yb³⁺ f¹³, rough free-ion parameters
N_ELE = 13
ZETA_ATOMIC = 300.0
MODE_Q3 = "sin"  # D3d structure


def _extract_h_cef_nsoc(mat: str) -> np.ndarray:
    path = DATA_DIR / mat / "NSOC" / "wannier_hr.dat"
    H_R, _ = read_hr(str(path))
    h_raw = H_R[(0, 0, 0)][0:7, 0:7].copy()
    U = _build_U_r2c(spinor=False)
    return (U @ h_raw @ U.T.conj()) * 1000.0


def _extract_h_cef_soc(mat: str) -> tuple[float, np.ndarray]:
    path = DATA_DIR / mat / "SOC" / "wannier_hr.dat"
    H_R, _ = read_hr(str(path))
    h_raw_14 = H_R[(0, 0, 0)][0:14, 0:14].copy()
    h_raw_14 = _permute_ud_to_du(h_raw_14)
    U14 = _build_U_r2c(spinor=True)
    h_local_meV = (U14 @ h_raw_14 @ U14.T.conj()) * 1000.0
    zeta, h_cef = decompose_h_local(h_local_meV, rtol=1e-2)
    return zeta, h_cef


def _run_one(mat: str, mode: str) -> None:
    if mode == "NSOC":
        h_cef = _extract_h_cef_nsoc(mat)
        zeta_for_fit = ZETA_ATOMIC
        zeta_str = f"{ZETA_ATOMIC:.1f} (atomic)"
    else:
        zeta_extracted, h_cef = _extract_h_cef_soc(mat)
        zeta_for_fit = zeta_extracted
        zeta_str = f"{zeta_extracted:.4f} (extracted)"

    sa = decompose_orbital_tensors(h_cef)
    sb = extract_cef_stevens_fit(
        h_cef, n_ele=N_ELE,
        zeta=zeta_for_fit, symmetry="C3v", mode_q3=MODE_Q3,
    )

    print(f"=== {mat}/{mode} ===")
    print(f"  zeta: {zeta_str}")
    print(f"  Scheme A residual: {sa['residual']:.3e}")
    print(f"  Ground: alpha={sb['alpha0']}, L={sb['L0']}, S={sb['S0']}, J0={sb['J0']}")
    print(f"  J0 weight: {sb['j0_weight']:.4f}")
    print(f"  Stevens fit residual: {sb['stevens_fit_residual']:.3e}")
    print(f"  Stevens parameters (C3v):")
    for name, val in sb["B_params"].items():
        print(f"    {name} = {val:+.6f} meV")
    print(f"  CEF splitting (meV): {np.round(sb['cef_splitting_meV'], 2).tolist()}")
    print(f"  Fit splitting (meV): {np.round(sb['cef_splitting_fit_meV'], 2).tolist()}")

    # Scheme A: significant a_kq
    print(f"  Non-zero a_kq:")
    for (k, q), v in sorted(sa["a_kq"].items()):
        if abs(v) > 1e-3:
            print(f"    a({k},{q:+d}) = {v.real:+.6f} {v.imag:+.6f}j")
    print()


if __name__ == "__main__":
    for mat in MATERIALS:
        for mode in ["NSOC", "SOC"]:
            _run_one(mat, mode)
