"""
Tests for w90_extract and w90_decompose_h_local.

Test 1: hopping matrices match reference data in data/*/test/
Test 2: SOC strength from decompose matches input SOC_LAMBDA (soc) or zero (nsoc)
"""

import sys
from pathlib import Path

import numpy as np

# allow imports from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from w90_extract import w90_extract, _build_U_r2c
from w90_decompose_h_local import decompose_h_local

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

NDIMS = [7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# All test cases parsed from data/*/test/input/*.dat
# (material, bond, soc_mode, hr_path, f_sites, ligands, soc_lambda_eV)
CASES = [
    # YbOCl
    ("YbOCl", "1st", "soc",  "YbOCl/SOC/wannier_hr.dat",
     [(0, [0,0,0]), (3, [0,0,0])], [(6, [0,0,0]), (7, [0,0,0])], None),
    ("YbOCl", "1st", "nsoc", "YbOCl/NSOC/wannier_hr.dat",
     [(0, [0,0,0]), (3, [0,0,0])], [(6, [0,0,0]), (7, [0,0,0])], 0.39326860966235533),
    ("YbOCl", "2nd", "soc",  "YbOCl/SOC/wannier_hr.dat",
     [(0, [0,0,0]), (0, [1,0,0])], [(6, [0,0,0]), (12, [0,-1,0])], None),
    ("YbOCl", "2nd", "nsoc", "YbOCl/NSOC/wannier_hr.dat",
     [(0, [0,0,0]), (0, [1,0,0])], [(6, [0,0,0]), (12, [0,-1,0])], 0.39326860966235533),
    # YbOBr
    ("YbOBr", "1st", "soc",  "YbOBr/SOC/wannier_hr.dat",
     [(0, [0,0,0]), (3, [0,0,0])], [(6, [0,0,0]), (7, [0,0,0])], None),
    ("YbOBr", "1st", "nsoc", "YbOBr/NSOC/wannier_hr.dat",
     [(0, [0,0,0]), (3, [0,0,0])], [(6, [0,0,0]), (7, [0,0,0])], 0.3917807293867309),
    ("YbOBr", "2nd", "soc",  "YbOBr/SOC/wannier_hr.dat",
     [(0, [0,0,0]), (0, [1,0,0])], [(6, [0,0,0]), (12, [0,-1,0])], None),
    ("YbOBr", "2nd", "nsoc", "YbOBr/NSOC/wannier_hr.dat",
     [(0, [0,0,0]), (0, [1,0,0])], [(6, [0,0,0]), (12, [0,-1,0])], 0.3917807293867309),
    # YbOF
    ("YbOF", "1st", "soc",  "YbOF/SOC/wannier_hr.dat",
     [(0, [0,0,0]), (3, [0,0,0])], [(6, [0,0,0]), (7, [0,0,0])], None),
    ("YbOF", "1st", "nsoc", "YbOF/NSOC/wannier_hr.dat",
     [(0, [0,0,0]), (3, [0,0,0])], [(6, [0,0,0]), (7, [0,0,0])], 0.393832481654664),
    ("YbOF", "2nd", "soc",  "YbOF/SOC/wannier_hr.dat",
     [(0, [0,0,0]), (0, [-1,0,0])], [(6, [0,0,0]), (12, [0,0,0])], None),
    ("YbOF", "2nd", "nsoc", "YbOF/NSOC/wannier_hr.dat",
     [(0, [0,0,0]), (0, [-1,0,0])], [(6, [0,0,0]), (12, [0,0,0])], 0.393832481654664),
]


def _to_ref_basis(h):
    """
    Convert 14x14 from complex-harmonic (dn,up) interleaved
    to real-harmonic (up,dn) interleaved (reference format).
    """
    U = _build_U_r2c(spinor=True)
    h_real = U.T.conj() @ h @ U
    # swap (dn,up) -> (up,dn)
    perm = [idx for m in range(7) for idx in (2 * m + 1, 2 * m)]
    return h_real[np.ix_(perm, perm)]


def _ref_path(material, bond, soc_mode):
    """Path to reference h_offdiag.dat."""
    tag = f"{material}_{soc_mode}_{bond}_bond1"
    return DATA_ROOT / material / "test" / tag / "h_offdiag.dat"


def _run_extract(case):
    """Run w90_extract for a given case."""
    material, bond, soc_mode, hr_rel, f_sites, ligands, _ = case
    is_soc = soc_mode == "soc"
    n_orb = [2 * d for d in NDIMS] if is_soc else NDIMS
    return w90_extract(
        hr_path=str(DATA_ROOT / hr_rel),
        n_orb_per_atom=n_orb,
        f_sites=f_sites,
        ligands=ligands,
        spinor=is_soc,
        energy_unit="eV",
    )


# ============================================================
# Test 1: hopping matches reference
# ============================================================
def test_hopping():
    print("=" * 60)
    print("TEST 1: hopping matrix vs reference")
    print("=" * 60)
    all_pass = True
    for case in CASES:
        material, bond, soc_mode, *_ = case
        label = f"{material} {bond} {soc_mode}"
        ref_file = _ref_path(material, bond, soc_mode)
        if not ref_file.exists():
            print(f"  SKIP {label}: reference not found")
            continue

        result = _run_extract(case)
        t_mu_ref_basis = _to_ref_basis(result["t_mu"])
        ref = np.loadtxt(str(ref_file), dtype=complex)

        # reference is in meV, our output is in eV
        t_mu_meV = t_mu_ref_basis * 1000.0

        diff = np.max(np.abs(t_mu_meV - ref))
        rel = diff / np.max(np.abs(ref)) if np.max(np.abs(ref)) > 0 else diff
        ok = rel < 1e-6
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {label}: max_rel_diff = {rel:.12e}")
        if not ok:
            all_pass = False
    return all_pass


# ============================================================
# Test 2: SOC strength
# ============================================================
def test_soc_strength():
    print("=" * 60)
    print("TEST 2: SOC strength extraction")
    print("=" * 60)
    all_pass = True
    for case in CASES:
        material, bond, soc_mode, *_, soc_lambda_eV = case
        label = f"{material} {bond} {soc_mode}"
        result = _run_extract(case)
        h_local_meV = result["h_local"] * 1000.0

        zeta, _ = decompose_h_local(h_local_meV, rtol=1e-2)

        if soc_mode == "nsoc":
            ok = zeta == 0.0
            status = "PASS" if ok else "FAIL"
            print(f"  {status} {label}: zeta = {zeta} (expect 0)")
        else:
            # find paired nsoc case for SOC_LAMBDA reference
            ref_lambda = None
            for c2 in CASES:
                if c2[0] == material and c2[1] == bond and c2[2] == "nsoc":
                    ref_lambda = c2[6]
                    break
            if ref_lambda is not None:
                ref_meV = ref_lambda * 1000.0
                rel = abs(zeta - ref_meV) / ref_meV
                ok = rel < 0.01
                status = "PASS" if ok else "FAIL"
                print(f"  {status} {label}: zeta = {zeta:.4f} meV, "
                      f"ref = {ref_meV:.4f} meV, rel_diff = {rel:.4e}")
            else:
                ok = True
                print(f"  INFO {label}: zeta = {zeta:.4f} meV (no nsoc ref)")
        if not ok:
            all_pass = False
    return all_pass


if __name__ == "__main__":
    p1 = test_hopping()
    print()
    p2 = test_soc_strength()
    print()
    if p1 and p2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
