"""Analytical wavefunction validation for f1 and f13."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from fexchange.core.fock import N_ORB, enumerate_dets
from fexchange.spectrum.lsjm import build_lsjm
from fexchange.spectrum.lsms import build_lsms
from tests.validation.checks import VALIDATION_R42, VALIDATION_R62

OVERLAP_TOL = 0.9999
VALUE_TOL = 1.0e-10
L_F = 3
F13_ANALYTICAL_COEF = {
    "coef_F2": -8.0 / 5.0,
    "coef_F4": -12.0 / 11.0,
    "coef_F6": -200.0 / 143.0,
}


def cg_l_half(l: int, ml: float, ms: float, j: float, m: float) -> float:
    """CG coefficient <l, ml, 1/2, ms | j, m> for l x s=1/2."""
    if abs(ml + ms - m) > VALUE_TOL or abs(ml) > l or abs(ms) != 0.5:
        return 0.0
    if abs(j - (l + 0.5)) <= VALUE_TOL:
        if ms > 0.0:
            return math.sqrt((l + m + 0.5) / (2 * l + 1))
        return math.sqrt((l - m + 0.5) / (2 * l + 1))
    if abs(j - (l - 0.5)) <= VALUE_TOL:
        if ms > 0.0:
            return -math.sqrt((l - m + 0.5) / (2 * l + 1))
        return math.sqrt((l + m + 0.5) / (2 * l + 1))
    return 0.0


def _fex_orbital_index(ml: int, two_ms: int) -> int:
    """Return the fexchange spin-orbital index for (ml, ms)."""
    return (ml + L_F) * 2 + (0 if two_ms == -1 else 1)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


def _slater_zero_check(name: str, values: np.ndarray) -> dict[str, Any]:
    max_abs = float(np.max(np.abs(values)))
    return {
        "name": name,
        "metric": max_abs,
        "metric_label": "max|delta|",
        "passed": max_abs <= VALUE_TOL,
    }


def _slater_constant_check(name: str, values: np.ndarray, expected: float) -> dict[str, Any]:
    max_abs = float(np.max(np.abs(np.asarray(values, dtype=float) - expected)))
    return {
        "name": name,
        "metric": max_abs,
        "metric_label": "max|delta|",
        "passed": max_abs <= VALUE_TOL,
    }


def _slater_match_check(name: str, values: np.ndarray, expected: float) -> dict[str, Any]:
    max_abs = float(np.max(np.abs(np.asarray(values, dtype=float) - expected)))
    return {
        "name": name,
        "metric": max_abs,
        "metric_label": "max|delta|",
        "passed": max_abs <= VALUE_TOL,
    }


def compare_f1_lsms() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Compare f1 LSMS states against one-determinant analytical states."""
    lsms = build_lsms(1, r42=VALIDATION_R42, r62=VALIDATION_R62)
    dets = enumerate_dets(1)
    det_to_idx = {int(det): idx for idx, det in enumerate(dets)}
    V_fock = lsms["V_fock"]

    results: list[dict[str, Any]] = []
    for idx, label in enumerate(lsms["labels"]):
        ml = int(label["ML"])
        two_ms = int(label["twoMS"])
        det = 1 << _fex_orbital_index(ml, two_ms)
        ref_vec = np.zeros(len(dets), dtype=complex)
        ref_vec[det_to_idx[det]] = 1.0
        overlap = float(abs(np.vdot(ref_vec, V_fock[:, idx])))
        results.append(
            {
                "ML": ml,
                "twoMS": two_ms,
                "overlap": overlap,
                "passed": overlap > OVERLAP_TOL,
            }
        )

    slater_checks = [
        _slater_zero_check("coef_F0", lsms["coef_F0"]),
        _slater_zero_check("coef_F2", lsms["coef_F2"]),
        _slater_zero_check("coef_F4", lsms["coef_F4"]),
        _slater_zero_check("coef_F6", lsms["coef_F6"]),
    ]
    return results, lsms, slater_checks


def compare_f1_lsjm(lsms: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare f1 LSJM states against analytical l x s=1/2 CG coefficients."""
    lsjm = build_lsjm(lsms)
    V_lsms = lsms["V_fock"]
    V_lsjm = lsjm["V_fock"]

    lsms_lookup: dict[tuple[int, int], int] = {}
    for idx, label in enumerate(lsms["labels"]):
        lsms_lookup[(int(label["ML"]), int(label["twoMS"]))] = idx

    overlap_results: list[dict[str, Any]] = []
    soc_checks: list[dict[str, Any]] = []
    S = 0.5
    for idx, label in enumerate(lsjm["labels"]):
        j = int(label["twoJ"]) / 2.0
        m = int(label["twoM"]) / 2.0
        ref_vec = np.zeros(V_lsjm.shape[0], dtype=complex)
        for two_ms in (-1, 1):
            ms = two_ms / 2.0
            ml = int(round(m - ms))
            if abs(ml) > L_F:
                continue
            cg = cg_l_half(L_F, float(ml), ms, j, m)
            if abs(cg) <= VALUE_TOL:
                continue
            ref_vec += cg * V_lsms[:, lsms_lookup[(ml, two_ms)]]
        ref_vec = _normalize(ref_vec)
        overlap = float(abs(np.vdot(ref_vec, V_lsjm[:, idx])))
        overlap_results.append({"J": j, "M": m, "overlap": overlap, "passed": overlap > OVERLAP_TOL})

        if int(label["twoM"]) == int(label["twoJ"]):
            zeta_fex = float(lsjm["coef_zeta"][idx])
            zeta_ana = float((j * (j + 1) - L_F * (L_F + 1) - S * (S + 1)) / 2.0)
            delta = abs(zeta_fex - zeta_ana)
            soc_checks.append(
                {
                    "J": j,
                    "zeta_fex": zeta_fex,
                    "zeta_ana": zeta_ana,
                    "delta": delta,
                    "passed": delta <= VALUE_TOL,
                }
            )

    return overlap_results, soc_checks


def compare_f13_lsms() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Compare f13 LSMS states against single-hole analytical determinants."""
    lsms = build_lsms(13, r42=VALIDATION_R42, r62=VALIDATION_R62)
    dets = enumerate_dets(13)
    det_to_idx = {int(det): idx for idx, det in enumerate(dets)}
    V_fock = lsms["V_fock"]
    full_shell = (1 << N_ORB) - 1

    results: list[dict[str, Any]] = []
    for idx, label in enumerate(lsms["labels"]):
        ml = int(label["ML"])
        two_ms = int(label["twoMS"])
        hole_orb = _fex_orbital_index(-ml, -two_ms)
        det = full_shell ^ (1 << hole_orb)
        ref_vec = np.zeros(len(dets), dtype=complex)
        ref_vec[det_to_idx[det]] = 1.0
        overlap = float(abs(np.vdot(ref_vec, V_fock[:, idx])))
        results.append(
            {
                "ML": ml,
                "twoMS": two_ms,
                "overlap": overlap,
                "passed": overlap > OVERLAP_TOL,
            }
        )

    slater_checks = [
        _slater_constant_check("coef_F0", lsms["coef_F0"], math.comb(13, 2)),
        _slater_match_check("coef_F2", lsms["coef_F2"], F13_ANALYTICAL_COEF["coef_F2"]),
        _slater_match_check("coef_F4", lsms["coef_F4"], F13_ANALYTICAL_COEF["coef_F4"]),
        _slater_match_check("coef_F6", lsms["coef_F6"], F13_ANALYTICAL_COEF["coef_F6"]),
    ]
    return results, lsms, slater_checks


def compare_f13_lsjm(lsms: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare f13 LSJM states against analytical CG coefficients."""
    lsjm = build_lsjm(lsms)
    V_lsms = lsms["V_fock"]
    V_lsjm = lsjm["V_fock"]

    lsms_lookup: dict[tuple[int, int], int] = {}
    for idx, label in enumerate(lsms["labels"]):
        lsms_lookup[(int(label["ML"]), int(label["twoMS"]))] = idx

    overlap_results: list[dict[str, Any]] = []
    soc_checks: list[dict[str, Any]] = []
    S = 0.5
    for idx, label in enumerate(lsjm["labels"]):
        j = int(label["twoJ"]) / 2.0
        m = int(label["twoM"]) / 2.0
        ref_vec = np.zeros(V_lsjm.shape[0], dtype=complex)
        for two_ms in (-1, 1):
            ms = two_ms / 2.0
            ml = int(round(m - ms))
            if abs(ml) > L_F:
                continue
            cg = cg_l_half(L_F, float(ml), ms, j, m)
            if abs(cg) <= VALUE_TOL:
                continue
            ref_vec += cg * V_lsms[:, lsms_lookup[(ml, two_ms)]]
        ref_vec = _normalize(ref_vec)
        overlap = float(abs(np.vdot(ref_vec, V_lsjm[:, idx])))
        overlap_results.append({"J": j, "M": m, "overlap": overlap, "passed": overlap > OVERLAP_TOL})

        if int(label["twoM"]) == int(label["twoJ"]):
            zeta_fex = float(lsjm["coef_zeta"][idx])
            zeta_ana = float(-(j * (j + 1) - L_F * (L_F + 1) - S * (S + 1)) / 2.0)
            delta = abs(zeta_fex - zeta_ana)
            soc_checks.append(
                {
                    "J": j,
                    "zeta_fex": zeta_fex,
                    "zeta_ana": zeta_ana,
                    "delta": delta,
                    "passed": delta <= VALUE_TOL,
                }
            )

    return overlap_results, soc_checks


def _print_lsms(tag: str, results: list[dict[str, Any]], slater_checks: list[dict[str, Any]]) -> int:
    fail_count = 0
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            fail_count += 1
        print(
            f"  [{status}] ML={result['ML']:+d} MS={result['twoMS']:+d}/2  "
            f"overlap={result['overlap']:.10f}"
        )
    for check in slater_checks:
        status = "PASS" if check["passed"] else "FAIL"
        if not check["passed"]:
            fail_count += 1
        print(f"  [{status}] {check['name']}: {check['metric_label']}={check['metric']:.2e}")
    passed = sum(1 for result in results if result["passed"])
    print(f"  {tag} LSMS overlap: {passed}/{len(results)}")
    return fail_count


def _print_lsjm(tag: str, overlaps: list[dict[str, Any]], soc_checks: list[dict[str, Any]]) -> int:
    fail_count = 0
    for result in overlaps:
        status = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            fail_count += 1
        print(
            f"  [{status}] J={result['J']:.1f} M={result['M']:+.1f}  "
            f"overlap={result['overlap']:.10f}"
        )
    for check in soc_checks:
        status = "PASS" if check["passed"] else "FAIL"
        if not check["passed"]:
            fail_count += 1
        print(
            f"  [{status}] J={check['J']:.1f} coef_zeta: "
            f"fex={check['zeta_fex']:+.10f} ana={check['zeta_ana']:+.10f} "
            f"delta={check['delta']:.2e}"
        )
    passed = sum(1 for result in overlaps if result["passed"])
    print(f"  {tag} LSJM overlap: {passed}/{len(overlaps)}")
    return fail_count


def run_f1() -> int:
    print("=== f1 Analytical Validation ===")
    print("--- f1 LSMS ---")
    lsms_results, lsms, slater_checks = compare_f1_lsms()
    fail_count = _print_lsms("f1", lsms_results, slater_checks)
    print("--- f1 LSJM ---")
    lsjm_overlaps, soc_checks = compare_f1_lsjm(lsms)
    fail_count += _print_lsjm("f1", lsjm_overlaps, soc_checks)
    print(f"=== f1 total FAIL: {fail_count} ===")
    print()
    return fail_count


def run_f13() -> int:
    print("=== f13 Analytical Validation ===")
    print("--- f13 LSMS ---")
    lsms_results, lsms, slater_checks = compare_f13_lsms()
    fail_count = _print_lsms("f13", lsms_results, slater_checks)
    print("--- f13 LSJM ---")
    lsjm_overlaps, soc_checks = compare_f13_lsjm(lsms)
    fail_count += _print_lsjm("f13", lsjm_overlaps, soc_checks)
    print(f"=== f13 total FAIL: {fail_count} ===")
    print()
    return fail_count


def main() -> int:
    import sys

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["f1", "f13"]
    total_fail = 0
    if "f1" in targets:
        total_fail += run_f1()
    if "f13" in targets:
        total_fail += run_f13()
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
