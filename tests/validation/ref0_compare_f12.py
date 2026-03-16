"""Compare f12 wavefunctions derived from ref0 f2 via particle-hole conjugation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_REF0_DIR = str(Path(__file__).resolve().parents[2] / "docs" / "reference0")
if _REF0_DIR not in sys.path:
    sys.path.insert(0, _REF0_DIR)

from fexchange.core.fock import N_ORB, enumerate_dets
from fexchange.spectrum.lsjm import build_lsjm
from fexchange.spectrum.lsms import build_lsms
from tests.validation.checks import VALIDATION_R42, VALIDATION_R62
from tests.validation.ref0_compare_f2 import (
    ANALYTICAL_COEF,
    OVERLAP_TOL,
    VALUE_TOL,
    build_conversion_matrix,
    ket_to_vector,
)
from space_lz import LSJM_RS_f2, LSMM_RS_f2

FULL_SHELL = (1 << N_ORB) - 1
F12_ANALYTICAL_SHIFT = {
    "coef_F2": -4.0 / 3.0,
    "coef_F4": -10.0 / 11.0,
    "coef_F6": -500.0 / 429.0,
}


def _occupied_orbitals(det: int) -> list[int]:
    return [orb for orb in range(N_ORB) if (det >> orb) & 1]


def _particle_hole_phase(det_f2: int) -> int:
    """Return the fermionic phase for particle-hole conjugation of a 2-electron determinant."""
    p0, p1 = _occupied_orbitals(det_f2)
    return (-1) ** (p0 + p1 + 1)


def build_particle_hole_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the f2 -> f12 particle-hole transformation matrix in fexchange ordering."""
    dets_f2 = enumerate_dets(2)
    dets_f12 = enumerate_dets(12)
    det_to_idx_f12 = {int(det): idx for idx, det in enumerate(dets_f12)}
    dim = len(dets_f2)
    matrix = np.zeros((dim, dim), dtype=complex)
    for col, det_f2 in enumerate(dets_f2):
        det_f12 = FULL_SHELL ^ int(det_f2)
        row = det_to_idx_f12[det_f12]
        matrix[row, col] = _particle_hole_phase(int(det_f2))
    return matrix, dets_f2, dets_f12


def _coef_check(name: str, fex: float, ref: float, *, ref_label: str = "ref") -> dict[str, Any]:
    delta = abs(float(fex) - float(ref))
    return {
        "name": name,
        "fex": float(fex),
        "ref": float(ref),
        "ref_label": ref_label,
        "delta": delta,
        "passed": delta <= VALUE_TOL,
    }


def _lsms_lookup(labels: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], int]:
    lookup: dict[tuple[int, int, int, int], int] = {}
    for idx, label in enumerate(labels):
        key = (int(label["L"]), int(label["twoS"]), 2 * int(label["ML"]), int(label["twoMS"]))
        lookup[key] = idx
    return lookup


def _lsjm_lookup(labels: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], int]:
    lookup: dict[tuple[int, int, int, int], int] = {}
    for idx, label in enumerate(labels):
        key = (int(label["L"]), int(label["twoS"]), int(label["twoJ"]), int(label["twoM"]))
        lookup[key] = idx
    return lookup


def compare_f12_lsms() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare f12 LSMS states against particle-hole transformed ref0 f2 states."""
    ref_lsms_f2 = LSMM_RS_f2()
    fex_f12 = build_lsms(12, r42=VALIDATION_R42, r62=VALIDATION_R62)
    fex_lookup = _lsms_lookup(fex_f12["labels"])
    dets_f2, conversion, det_to_idx_f2 = build_conversion_matrix(2)
    particle_hole, _, _ = build_particle_hole_matrix()

    results: list[dict[str, Any]] = []
    for (L, S_ref), sector in ref_lsms_f2.items():
        two_s = 2 * S_ref
        for ml in sorted(sector["ML"], reverse=True):
            for ms in sorted(sector["MS"], reverse=True):
                ref_vec_f2 = ket_to_vector(sector["kets"][(ml, ms)], dets_f2, conversion, det_to_idx_f2)
                ref_vec_f12 = particle_hole @ ref_vec_f2
                f12_ml = -ml
                f12_ms = -ms
                key = (L, two_s, 2 * f12_ml, 2 * int(f12_ms))
                if key not in fex_lookup:
                    results.append(
                        {
                            "L": L,
                            "twoS": two_s,
                            "ML": f12_ml,
                            "MS": float(f12_ms),
                            "overlap": 0.0,
                            "passed": False,
                            "coef_checks": [],
                            "error": "no match",
                        }
                    )
                    continue
                idx = fex_lookup[key]
                overlap = float(abs(np.vdot(ref_vec_f12, fex_f12["V_fock"][:, idx])))
                coef_checks: list[dict[str, Any]] = []
                if f12_ml == L and f12_ms == S_ref:
                    coef_ref = ANALYTICAL_COEF[(two_s, L)]
                    coef_checks = [
                        _coef_check(
                            "coef_F2",
                            fex_f12["coef_F2"][idx],
                            coef_ref[1] + F12_ANALYTICAL_SHIFT["coef_F2"],
                            ref_label="analytic",
                        ),
                        _coef_check(
                            "coef_F4",
                            fex_f12["coef_F4"][idx],
                            coef_ref[2] + F12_ANALYTICAL_SHIFT["coef_F4"],
                            ref_label="analytic",
                        ),
                        _coef_check(
                            "coef_F6",
                            fex_f12["coef_F6"][idx],
                            coef_ref[3] + F12_ANALYTICAL_SHIFT["coef_F6"],
                            ref_label="analytic",
                        ),
                    ]
                results.append(
                    {
                        "L": L,
                        "twoS": two_s,
                        "ML": f12_ml,
                        "MS": float(f12_ms),
                        "overlap": overlap,
                        "passed": overlap > OVERLAP_TOL,
                        "coef_checks": coef_checks,
                    }
                )

    results.sort(key=lambda item: (-item["L"], -item["twoS"], -item["ML"], -item["MS"]))
    return results, fex_f12


def compare_f12_lsjm(fex_f12_lsms: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare f12 LSJM states against particle-hole transformed ref0 f2 states."""
    ref_lsjm_f2 = LSJM_RS_f2(LSMM_RS_f2())
    fex_f12_lsjm = build_lsjm(fex_f12_lsms)
    fex_lookup = _lsjm_lookup(fex_f12_lsjm["labels"])
    dets_f2, conversion, det_to_idx_f2 = build_conversion_matrix(2)
    particle_hole, _, _ = build_particle_hole_matrix()

    results: list[dict[str, Any]] = []
    for (L, S_ref), sector in ref_lsjm_f2.items():
        two_s = 2 * S_ref
        for J, M in sorted(sector["JM"], key=lambda item: (item[0], item[1]), reverse=True):
            ref_vec_f2 = ket_to_vector(sector["kets"][(J, M)], dets_f2, conversion, det_to_idx_f2)
            ref_vec_f12 = particle_hole @ ref_vec_f2
            f12_m = -M
            key = (L, two_s, 2 * J, 2 * int(f12_m))
            if key not in fex_lookup:
                results.append(
                    {
                        "L": L,
                        "twoS": two_s,
                        "J": float(J),
                        "M": float(f12_m),
                        "overlap": 0.0,
                        "passed": False,
                        "coef_checks": [],
                        "error": "no match",
                    }
                )
                continue
            idx = fex_lookup[key]
            overlap = float(abs(np.vdot(ref_vec_f12, fex_f12_lsjm["V_fock"][:, idx])))
            coef_checks: list[dict[str, Any]] = []
            if f12_m == J:
                S = two_s / 2.0
                zeta_fex = float(fex_f12_lsjm["coef_zeta"][idx])
                zeta_ana = float(-(J * (J + 1) - L * (L + 1) - S * (S + 1)) / 4.0)
                coef_checks = [_coef_check("coef_zeta", zeta_fex, zeta_ana)]
            results.append(
                {
                    "L": L,
                    "twoS": two_s,
                    "J": float(J),
                    "M": float(f12_m),
                    "overlap": overlap,
                    "passed": overlap > OVERLAP_TOL,
                    "coef_checks": coef_checks,
                }
            )

    results.sort(key=lambda item: (-item["L"], -item["twoS"], -item["J"], -item["M"]))
    return results, fex_f12_lsjm


def _print_results(title: str, results: list[dict[str, Any]]) -> int:
    fail_count = 0
    pass_count = 0
    for result in results:
        item_failed = (not result["passed"]) or any(not check["passed"] for check in result.get("coef_checks", []))
        status = "FAIL" if item_failed else "PASS"
        if item_failed:
            fail_count += 1
        if result["passed"]:
            pass_count += 1

        if "J" in result:
            label = f"L={result['L']} S={result['twoS']//2} J={int(result['J'])} M={result['M']:+.1f}"
        else:
            label = f"L={result['L']} S={result['twoS']//2} ML={result['ML']:+d} MS={result['MS']:+.1f}"
        if "error" in result:
            label = f"{label} ({result['error']})"
        print(f"  [{status}] ({label})  overlap={result['overlap']:.10f}")

        for check in result.get("coef_checks", []):
            check_status = "PASS" if check["passed"] else "FAIL"
            if not check["passed"]:
                fail_count += 1
            print(
                f"    [{check_status}] {check['name']}: "
                f"fex={check['fex']:+.10f} {check['ref_label']}={check['ref']:+.10f} "
                f"delta={check['delta']:.2e}"
            )

    print(f"  {title}: {pass_count}/{len(results)} overlap > {OVERLAP_TOL:.4f}")
    return fail_count


def main() -> int:
    particle_hole, _, _ = build_particle_hole_matrix()
    ident = np.eye(particle_hole.shape[0], dtype=complex)
    assert np.allclose(particle_hole @ particle_hole.conj().T, ident)
    assert np.allclose(particle_hole.conj().T @ particle_hole, ident)

    print("=== f12 Wavefunction Comparison: ref0 f2 particle-hole ===")
    print()
    print("--- f12 LSMS ---")
    lsms_results, fex_lsms = compare_f12_lsms()
    lsms_fail = _print_results("f12 LSMS", lsms_results)
    print()
    print("--- f12 LSJM ---")
    lsjm_results, _ = compare_f12_lsjm(fex_lsms)
    lsjm_fail = _print_results("f12 LSJM", lsjm_results)
    print()
    print(f"=== Summary: LSMS {lsms_fail} FAIL, LSJM {lsjm_fail} FAIL ===")
    return 1 if (lsms_fail + lsjm_fail) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
