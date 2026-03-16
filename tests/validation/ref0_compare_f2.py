"""Compare f2 wavefunctions between Reference-0 and fexchange."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_REF0_DIR = str(Path(__file__).resolve().parents[2] / "docs" / "reference0")
if _REF0_DIR not in sys.path:
    sys.path.insert(0, _REF0_DIR)

from fexchange.core.fock import enumerate_dets, N_ORB
from fexchange.hamiltonian.hsoc import build_hsoc_unit_matrix
from fexchange.spectrum.lsjm import build_lsjm
from fexchange.spectrum.lsms import build_lsms
from tests.validation.checks import VALIDATION_R42, VALIDATION_R62
from space_lz import LSJM_RS_f2, LSMM_RS_f2

ANALYTICAL_COEF = {
    (2, 5): (1.0, -25 / 225, -51 / 1089, -325 / 184041),
    (2, 3): (1.0, -10 / 225, -33 / 1089, -7150 / 184041),
    (0, 4): (1.0, -30 / 225, 97 / 1089, 1950 / 184041),
    (0, 2): (1.0, 19 / 225, -99 / 1089, 17875 / 184041),
    (2, 1): (1.0, 45 / 225, 33 / 1089, -32175 / 184041),
    (0, 6): (1.0, 25 / 225, 9 / 1089, 25 / 184041),
    (0, 0): (1.0, 60 / 225, 198 / 1089, 42900 / 184041),
}

OVERLAP_TOL = 0.9999
VALUE_TOL = 1.0e-10


def _swap_spin_bits(det: int) -> int:
    swapped = 0
    for p in range(N_ORB):
        if (det >> p) & 1:
            swapped |= 1 << (p ^ 1)
    return swapped


def _fermion_sign(det: int) -> int:
    sign = 1
    for k in range(N_ORB // 2):
        if ((det >> (2 * k)) & 1) and ((det >> (2 * k + 1)) & 1):
            sign *= -1
    return sign


def build_conversion_matrix(n_ele: int = 2) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    dets = enumerate_dets(n_ele)
    det_to_idx = {int(det): idx for idx, det in enumerate(dets)}
    U = np.zeros((len(dets), len(dets)), dtype=complex)
    for col, det_ref in enumerate(dets):
        det_fex = _swap_spin_bits(int(det_ref))
        U[det_to_idx[det_fex], col] = _fermion_sign(int(det_ref))

    ident = np.eye(len(dets), dtype=complex)
    assert np.allclose(U.conj().T @ U, ident)
    assert np.allclose(U @ U.conj().T, ident)
    return dets, U, det_to_idx


def ket_to_vector(
    ket: Any,
    dets: np.ndarray,
    U: np.ndarray,
    det_to_idx: dict[int, int],
) -> np.ndarray:
    ref_vec = np.zeros(len(dets), dtype=complex)
    outer = ket.coeff()
    for ketline in ket.ketlines():
        ref_vec[det_to_idx[int(ketline.base())]] += outer * ketline.coeff()
    return U @ ref_vec


def _fmt_quantum(value: float) -> str:
    return f"{value:+.1f}"


def _fmt_value(value: float) -> str:
    return f"{value:+.10f}"


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
    for idx, lab in enumerate(labels):
        key = (int(lab["L"]), int(lab["twoS"]), 2 * int(lab["ML"]), int(lab["twoMS"]))
        lookup[key] = idx
    return lookup


def _lsjm_lookup(labels: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], int]:
    lookup: dict[tuple[int, int, int, int], int] = {}
    for idx, lab in enumerate(labels):
        key = (int(lab["L"]), int(lab["twoS"]), int(lab["twoJ"]), int(lab["twoM"]))
        lookup[key] = idx
    return lookup


def compare_lsms() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dets, U, det_to_idx = build_conversion_matrix(2)
    ref_lsms = LSMM_RS_f2()
    fex_lsms = build_lsms(2, r42=VALIDATION_R42, r62=VALIDATION_R62)
    fex_lookup = _lsms_lookup(fex_lsms["labels"])

    results: list[dict[str, Any]] = []
    for (L, S_ref), sector in ref_lsms.items():
        for ML in sorted(sector["ML"], reverse=True):
            for MS in sorted(sector["MS"], reverse=True):
                ref_ket = sector["kets"][(ML, MS)]
                ref_vec = ket_to_vector(ref_ket, dets, U, det_to_idx)
                idx = fex_lookup[(L, 2 * S_ref, 2 * ML, 2 * MS)]
                fex_vec = fex_lsms["V_fock"][:, idx]
                overlap = float(abs(np.vdot(ref_vec, fex_vec)))
                coef_checks: list[dict[str, Any]] = []
                if ML == L and MS == S_ref:
                    coef_ref = ANALYTICAL_COEF[(2 * S_ref, L)]
                    coef_checks = [
                        _coef_check("coef_F0", fex_lsms["coef_F0"][idx], coef_ref[0]),
                        _coef_check("coef_F2", fex_lsms["coef_F2"][idx], coef_ref[1]),
                        _coef_check("coef_F4", fex_lsms["coef_F4"][idx], coef_ref[2]),
                        _coef_check("coef_F6", fex_lsms["coef_F6"][idx], coef_ref[3]),
                    ]

                results.append(
                    {
                        "L": L,
                        "twoS": 2 * S_ref,
                        "ML": float(ML),
                        "MS": float(MS),
                        "overlap": overlap,
                        "passed": overlap > OVERLAP_TOL,
                        "coef_checks": coef_checks,
                    }
                )

    results.sort(key=lambda item: (-item["L"], -item["twoS"], -item["ML"], -item["MS"]))
    return results, fex_lsms


def compare_lsjm(fex_lsms: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dets, U, det_to_idx = build_conversion_matrix(2)
    ref_lsjm = LSJM_RS_f2(LSMM_RS_f2())
    fex_lsjm = build_lsjm(fex_lsms)
    fex_lookup = _lsjm_lookup(fex_lsjm["labels"])
    O_soc = build_hsoc_unit_matrix(2)

    results: list[dict[str, Any]] = []
    for (L, S_ref), sector in ref_lsjm.items():
        for J, M in sorted(sector["JM"], key=lambda item: (item[0], item[1]), reverse=True):
            ref_ket = sector["kets"][(J, M)]
            ref_vec = ket_to_vector(ref_ket, dets, U, det_to_idx)
            idx = fex_lookup[(L, 2 * S_ref, 2 * J, 2 * M)]
            fex_vec = fex_lsjm["V_fock"][:, idx]
            overlap = float(abs(np.vdot(ref_vec, fex_vec)))
            coef_checks: list[dict[str, Any]] = []
            if M == J:
                S = 0.5 * (2 * S_ref)
                coef_num = float((ref_vec.conj() @ O_soc @ ref_vec).real)
                coef_ana = float((J * (J + 1) - L * (L + 1) - S * (S + 1)) / 4.0)
                coef_checks = [
                    _coef_check(
                        "coef_zeta",
                        fex_lsjm["coef_zeta"][idx],
                        coef_num,
                        ref_label="ref_num",
                    ),
                    _coef_check(
                        "coef_zeta",
                        fex_lsjm["coef_zeta"][idx],
                        coef_ana,
                        ref_label="analytical",
                    ),
                ]

            results.append(
                {
                    "L": L,
                    "twoS": 2 * S_ref,
                    "J": float(J),
                    "M": float(M),
                    "overlap": overlap,
                    "passed": overlap > OVERLAP_TOL,
                    "coef_checks": coef_checks,
                }
            )

    results.sort(key=lambda item: (-item["L"], -item["twoS"], -item["J"], -item["M"]))
    return results, fex_lsjm


def print_lsms_results(results: list[dict[str, Any]]) -> int:
    fail_count = 0
    pass_count = 0
    for result in results:
        item_failed = not result["passed"] or any(not check["passed"] for check in result["coef_checks"])
        status = "FAIL" if item_failed else "PASS"
        if item_failed:
            fail_count += 1
        if result["passed"]:
            pass_count += 1
        print(
            "  "
            f"[{status}] (L={result['L']}, S={result['twoS'] // 2}, "
            f"ML={_fmt_quantum(result['ML'])}, MS={_fmt_quantum(result['MS'])})  "
            f"overlap={result['overlap']:.10f}"
        )
        for check in result["coef_checks"]:
            coef_status = "PASS" if check["passed"] else "FAIL"
            print(
                "    "
                f"[{coef_status}] {check['name']}: "
                f"fex={_fmt_value(check['fex'])}  "
                f"{check['ref_label']}={_fmt_value(check['ref'])}  "
                f"delta={check['delta']:.2e}"
            )
    print(f"  LSMS overlap: {pass_count}/{len(results)} states with |overlap| > {OVERLAP_TOL:.4f}")
    return fail_count


def print_lsjm_results(results: list[dict[str, Any]]) -> int:
    fail_count = 0
    pass_count = 0
    for result in results:
        item_failed = not result["passed"] or any(not check["passed"] for check in result["coef_checks"])
        status = "FAIL" if item_failed else "PASS"
        if item_failed:
            fail_count += 1
        if result["passed"]:
            pass_count += 1
        print(
            "  "
            f"[{status}] (L={result['L']}, S={result['twoS'] // 2}, "
            f"J={int(result['J'])}, M={_fmt_quantum(result['M'])})  "
            f"overlap={result['overlap']:.10f}"
        )
        for check in result["coef_checks"]:
            coef_status = "PASS" if check["passed"] else "FAIL"
            print(
                "    "
                f"[{coef_status}] {check['name']}: "
                f"fex={_fmt_value(check['fex'])}  "
                f"{check['ref_label']}={_fmt_value(check['ref'])}  "
                f"delta={check['delta']:.2e}"
            )
    print(f"  LSJM overlap: {pass_count}/{len(results)} states with |overlap| > {OVERLAP_TOL:.4f}")
    return fail_count


def compare_ref0_f2(lsjm_result: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if lsjm_result is None:
        _, fex_lsms = compare_lsms()
        results, _ = compare_lsjm(fex_lsms)
        return results

    dets, U, det_to_idx = build_conversion_matrix(2)
    ref_lsjm = LSJM_RS_f2(LSMM_RS_f2())
    fex_lookup = _lsjm_lookup(lsjm_result["labels"])
    O_soc = build_hsoc_unit_matrix(2)

    results: list[dict[str, Any]] = []
    for (L, S_ref), sector in ref_lsjm.items():
        for J, M in sorted(sector["JM"]):
            ref_vec = ket_to_vector(sector["kets"][(J, M)], dets, U, det_to_idx)
            idx = fex_lookup[(L, 2 * S_ref, 2 * J, 2 * M)]
            coef_num = float((ref_vec.conj() @ O_soc @ ref_vec).real)
            results.append(
                {
                    "L": L,
                    "S": S_ref,
                    "J": J,
                    "M": M,
                    "overlap": float(abs(np.vdot(ref_vec, lsjm_result["V_fock"][:, idx]))),
                    "coef_zeta_fex": float(lsjm_result["coef_zeta"][idx]),
                    "coef_zeta_ref_num": coef_num,
                }
            )
    return results


def main() -> int:
    print("=== f2 Wavefunction Comparison: ref0 vs fexchange ===")
    print()
    print("--- LSMS ---")
    lsms_results, fex_lsms = compare_lsms()
    lsms_fail = print_lsms_results(lsms_results)
    print()
    print("--- LSJM ---")
    lsjm_results, _ = compare_lsjm(fex_lsms)
    lsjm_fail = print_lsjm_results(lsjm_results)
    print()
    print(f"=== Summary: LSMS {lsms_fail} FAIL, LSJM {lsjm_fail} FAIL ===")
    return 0 if (lsms_fail == 0 and lsjm_fail == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
