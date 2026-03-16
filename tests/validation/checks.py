from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fexchange.core.fock import N_ORB, dim_sector
from fexchange.core.space_ls import build_space_ls_matrix
from fexchange.hamiltonian.hsoc import build_hsoc_unit_matrix
from fexchange.spectrum.lsjm import build_lsjm
from fexchange.spectrum.lsms import build_lsms
from fexchange.utils.checks import check_orthonormal
from fexchange.utils.numerics import EPS_DIAG, EPS_NORM

VALIDATION_F2 = 12.98
VALIDATION_F4 = 8.163
VALIDATION_F6 = 5.878
VALIDATION_R42 = VALIDATION_F4 / VALIDATION_F2
VALIDATION_R62 = VALIDATION_F6 / VALIDATION_F2


@dataclass
class ValidationContext:
    n_ele: int
    expected_dim: int
    build_elapsed_s: float
    lsms: dict[str, Any]
    lsjm: dict[str, Any]
    angular_ops: dict[str, np.ndarray]
    osoc_unit: np.ndarray


def eigenstate_diag_residual(
    V: np.ndarray,
    O: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    OV = O @ V
    diag_values = np.einsum("ij,ij->j", V.conj(), OV)
    D = np.diag(diag_values)
    residual = OV - V @ D
    max_diag_imag = float(np.max(np.abs(diag_values.imag))) if diag_values.size else 0.0
    max_residual = float(np.max(np.linalg.norm(residual, axis=0))) if residual.size else 0.0
    return diag_values, max_diag_imag, max_residual


def build_context(n_ele: int) -> ValidationContext:
    import time

    start = time.perf_counter()
    lsms = build_lsms(n_ele, r42=VALIDATION_R42, r62=VALIDATION_R62)
    lsjm = build_lsjm(lsms)
    angular_ops = build_space_ls_matrix(n_ele, N_ORB)
    osoc_unit = build_hsoc_unit_matrix(n_ele, N_ORB)
    elapsed = time.perf_counter() - start
    return ValidationContext(
        n_ele=n_ele,
        expected_dim=dim_sector(n_ele, N_ORB),
        build_elapsed_s=elapsed,
        lsms=lsms,
        lsjm=lsjm,
        angular_ops=angular_ops,
        osoc_unit=osoc_unit,
    )


def compute_numerical_section(ctx: ValidationContext) -> dict[str, Any]:
    lsms_dim = int(ctx.lsms["V_fock"].shape[1])
    lsjm_dim = int(ctx.lsjm["V_fock"].shape[1])
    lsms_orth = float(check_orthonormal(ctx.lsms["V_fock"], label="V_lsms_validation"))
    lsjm_orth = float(check_orthonormal(ctx.lsjm["V_fock"], label="V_lsjm_validation"))

    term_counts: dict[tuple[int, int, int], int] = {}
    for label in ctx.lsms["labels"]:
        key = (int(label["alpha"]), int(label["L"]), int(label["twoS"]))
        term_counts[key] = term_counts.get(key, 0) + 1

    multiplicity_errors = []
    for alpha, L, twoS in sorted(term_counts):
        count = term_counts[(alpha, L, twoS)]
        expected = (2 * L + 1) * (twoS + 1)
        multiplicity_errors.append(abs(count - expected))

    return {
        "checks": [
            _comparison_check(
                name="LSMS dimension",
                operator="dimension",
                scope="lsms",
                value=float(abs(lsms_dim - ctx.expected_dim)),
                threshold=0.0,
                detail=f"got {lsms_dim}, expected {ctx.expected_dim}",
            ),
            _comparison_check(
                name="LSJM dimension",
                operator="dimension",
                scope="lsjm",
                value=float(abs(lsjm_dim - ctx.expected_dim)),
                threshold=0.0,
                detail=f"got {lsjm_dim}, expected {ctx.expected_dim}",
            ),
            _comparison_check(
                name="LSMS orthonormality",
                operator="orthonormality",
                scope="lsms",
                value=lsms_orth,
                threshold=EPS_NORM,
                detail=f"residual = {lsms_orth:.2e}",
            ),
            _comparison_check(
                name="LSJM orthonormality",
                operator="orthonormality",
                scope="lsjm",
                value=lsjm_orth,
                threshold=EPS_NORM,
                detail=f"residual = {lsjm_orth:.2e}",
            ),
            _comparison_check(
                name="Term multiplicities",
                operator="multiplicity",
                scope="lsms",
                value=float(max(multiplicity_errors, default=0)),
                threshold=0.0,
                detail=f"max |count-expected| = {max(multiplicity_errors, default=0)}",
            ),
        ]
    }


def compute_lsms_section(ctx: ValidationContext) -> dict[str, Any]:
    V = ctx.lsms["V_fock"]
    labels = ctx.lsms["labels"]
    ops = {
        "L2": ctx.angular_ops["L2"],
        "S2": ctx.angular_ops["S2"],
        "Lz": ctx.angular_ops["Lz"],
        "Sz": ctx.angular_ops["Sz"],
        "H0": ctx.lsms["hint_rank_matrix_map"][0],
        "H2": ctx.lsms["hint_rank_matrix_map"][2],
        "H4": ctx.lsms["hint_rank_matrix_map"][4],
        "H6": ctx.lsms["hint_rank_matrix_map"][6],
    }
    expected = {
        "L2": np.array([lab["L"] * (lab["L"] + 1) for lab in labels], dtype=float),
        "S2": np.array(
            [
                (lab["twoS"] / 2.0) * (lab["twoS"] / 2.0 + 1.0)
                for lab in labels
            ],
            dtype=float,
        ),
        "Lz": np.array([lab["ML"] for lab in labels], dtype=float),
        "Sz": np.array([lab["MS"] for lab in labels], dtype=float),
        "H0": np.asarray(ctx.lsms["coef_F0"], dtype=float),
        "H2": np.asarray(ctx.lsms["coef_F2"], dtype=float),
        "H4": np.asarray(ctx.lsms["coef_F4"], dtype=float),
        "H6": np.asarray(ctx.lsms["coef_F6"], dtype=float),
    }

    # Detect repeated LS terms: same (L, twoS) with more than one alpha.
    # When present, individual H_k (k=2,4,6) residuals are not expected to
    # vanish because build_lsms diagonalizes the total H_int = F2*H2+F4*H4+F6*H6,
    # and the H_k do not commute within repeated-term subspaces.
    ls_alpha_counts: dict[tuple[int, int], set[int]] = {}
    for lab in labels:
        key = (int(lab["L"]), int(lab["twoS"]))
        ls_alpha_counts.setdefault(key, set()).add(int(lab["alpha"]))
    has_repeated_terms = any(len(v) > 1 for v in ls_alpha_counts.values())

    results: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for name, op in ops.items():
        diag, diag_imag, residual = eigenstate_diag_residual(V, op)
        results[name] = {"diag": diag.real, "diag_imag": diag_imag, "residual": residual}
        rank_residual_ref = has_repeated_terms and name in {"H2", "H4", "H6"}
        checks.extend(
            [
                _comparison_check(
                    name=f"LSMS {name} diagonal imaginary part",
                    operator=name,
                    scope="full",
                    value=diag_imag,
                    threshold=EPS_DIAG,
                    detail=f"max |Im diag| = {diag_imag:.2e}",
                ),
                _comparison_check(
                    name=f"LSMS {name} residual",
                    operator=name,
                    scope="full",
                    value=residual,
                    threshold=EPS_NORM,
                    detail=f"max residual = {residual:.2e}",
                    reference_only=rank_residual_ref,
                ),
                _comparison_check(
                    name=f"LSMS {name} label/stored consistency",
                    operator=name,
                    scope="full",
                    value=float(np.max(np.abs(diag.real - expected[name]))),
                    threshold=EPS_DIAG,
                    detail=f"max |diag-expected| = {np.max(np.abs(diag.real - expected[name])):.2e}",
                ),
            ]
        )

    if has_repeated_terms:
        physics = ctx.lsms["physics"]
        H_int = physics["F2"] * ops["H2"] + physics["F4"] * ops["H4"] + physics["F6"] * ops["H6"]
        hint_diag, hint_imag, hint_residual = eigenstate_diag_residual(V, H_int)
        results["H_int"] = {"diag": hint_diag.real, "diag_imag": hint_imag, "residual": hint_residual}
        expected_hint = physics["F2"] * expected["H2"] + physics["F4"] * expected["H4"] + physics["F6"] * expected["H6"]
        checks.extend(
            [
                _comparison_check(
                    name="LSMS H_int diagonal imaginary part",
                    operator="H_int",
                    scope="full",
                    value=hint_imag,
                    threshold=EPS_DIAG,
                    detail=f"max |Im diag| = {hint_imag:.2e}",
                ),
                _comparison_check(
                    name="LSMS H_int residual",
                    operator="H_int",
                    scope="full",
                    value=hint_residual,
                    threshold=EPS_NORM,
                    detail=f"max residual = {hint_residual:.2e}",
                ),
                _comparison_check(
                    name="LSMS H_int label/stored consistency",
                    operator="H_int",
                    scope="full",
                    value=float(np.max(np.abs(hint_diag.real - expected_hint))),
                    threshold=EPS_DIAG,
                    detail=f"max |diag-expected| = {np.max(np.abs(hint_diag.real - expected_hint)):.2e}",
                ),
            ]
        )

    term_groups: dict[tuple[int, int, int], list[int]] = {}
    for idx, label in enumerate(labels):
        key = (int(label["alpha"]), int(label["L"]), int(label["twoS"]))
        term_groups.setdefault(key, []).append(idx)

    term_rows = []
    for alpha, L, twoS in sorted(term_groups):
        idxs = term_groups[(alpha, L, twoS)]
        L2_mean, L2_spread = _term_value(results["L2"]["diag"], idxs)
        S2_mean, S2_spread = _term_value(results["S2"]["diag"], idxs)
        H0_mean, H0_spread = _term_value(results["H0"]["diag"], idxs)
        H2_mean, H2_spread = _term_value(results["H2"]["diag"], idxs)
        H4_mean, H4_spread = _term_value(results["H4"]["diag"], idxs)
        H6_mean, H6_spread = _term_value(results["H6"]["diag"], idxs)
        term_rows.append(
            {
                "alpha": alpha,
                "L": L,
                "S": twoS / 2.0,
                "n": len(idxs),
                "L2": L2_mean, "L2_spread": L2_spread,
                "S2": S2_mean, "S2_spread": S2_spread,
                "H0": H0_mean, "H0_spread": H0_spread,
                "H2": H2_mean, "H2_spread": H2_spread,
                "H4": H4_mean, "H4_spread": H4_spread,
                "H6": H6_mean, "H6_spread": H6_spread,
            }
        )

    state_rows = []
    for idx, label in enumerate(labels):
        state_rows.append(
            {
                "alpha": int(label["alpha"]),
                "L": int(label["L"]),
                "S": float(label["twoS"]) / 2.0,
                "ML": float(label["ML"]),
                "MS": float(label["MS"]),
                "Lz": float(results["Lz"]["diag"][idx]),
                "Sz": float(results["Sz"]["diag"][idx]),
            }
        )

    all_spreads = []
    for row in term_rows:
        for key in ("L2_spread", "S2_spread", "H0_spread", "H2_spread", "H4_spread", "H6_spread"):
            all_spreads.append(row[key])
    max_spread = max(all_spreads)
    checks.append(
        _comparison_check(
            name="LSMS term eigenvalue spread",
            operator="spread",
            scope="lsms",
            value=max_spread,
            threshold=EPS_DIAG,
            detail=f"max spread across terms = {max_spread:.2e}",
        )
    )

    return {
        "term_rows": term_rows,
        "state_rows": state_rows,
        "checks": checks,
    }


def compute_lsjm_section(ctx: ValidationContext) -> dict[str, Any]:
    V = ctx.lsjm["V_fock"]
    labels = ctx.lsjm["labels"]
    ops = {
        "L2": ctx.angular_ops["L2"],
        "S2": ctx.angular_ops["S2"],
        "J2": ctx.angular_ops["J2"],
        "Jz": ctx.angular_ops["Jz"],
        "O_soc": ctx.osoc_unit,
    }
    expected = {
        "L2": np.array([lab["L"] * (lab["L"] + 1) for lab in labels], dtype=float),
        "S2": np.array(
            [
                (lab["twoS"] / 2.0) * (lab["twoS"] / 2.0 + 1.0)
                for lab in labels
            ],
            dtype=float,
        ),
        "J2": np.array([lab["J"] * (lab["J"] + 1) for lab in labels], dtype=float),
        "Jz": np.array([lab["M"] for lab in labels], dtype=float),
        "O_soc": np.asarray(ctx.lsjm["coef_zeta"], dtype=float),
    }

    results: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for name in ("L2", "S2", "J2", "Jz"):
        diag, diag_imag, residual = eigenstate_diag_residual(V, ops[name])
        results[name] = {"diag": diag.real, "diag_imag": diag_imag, "residual": residual}
        checks.extend(
            [
                _comparison_check(
                    name=f"LSJM {name} diagonal imaginary part",
                    operator=name,
                    scope="full",
                    value=diag_imag,
                    threshold=EPS_DIAG,
                    detail=f"max |Im diag| = {diag_imag:.2e}",
                ),
                _comparison_check(
                    name=f"LSJM {name} residual",
                    operator=name,
                    scope="full",
                    value=residual,
                    threshold=EPS_NORM,
                    detail=f"max residual = {residual:.2e}",
                ),
                _comparison_check(
                    name=f"LSJM {name} label consistency",
                    operator=name,
                    scope="full",
                    value=float(np.max(np.abs(diag.real - expected[name]))),
                    threshold=EPS_DIAG,
                    detail=f"max |diag-expected| = {np.max(np.abs(diag.real - expected[name])):.2e}",
                ),
            ]
        )

    global_soc_diag, global_soc_imag, global_soc_residual = eigenstate_diag_residual(V, ops["O_soc"])
    results["O_soc"] = {
        "diag": global_soc_diag.real,
        "diag_imag": global_soc_imag,
        "residual": global_soc_residual,
    }
    checks.extend(
        [
            _comparison_check(
                name="LSJM global O_soc diagonal imaginary part",
                operator="O_soc",
                scope="global",
                value=global_soc_imag,
                threshold=EPS_DIAG,
                detail=f"max |Im diag| = {global_soc_imag:.2e}",
                reference_only=True,
            ),
            _comparison_check(
                name="LSJM global O_soc residual",
                operator="O_soc",
                scope="global",
                value=global_soc_residual,
                threshold=EPS_NORM,
                detail=f"max residual = {global_soc_residual:.2e}",
                reference_only=True,
            ),
        ]
    )

    term_groups: dict[tuple[int, int, int], list[int]] = {}
    multiplet_groups: dict[tuple[int, int, int, int], list[int]] = {}
    for idx, label in enumerate(labels):
        term_key = (int(label["alpha"]), int(label["L"]), int(label["twoS"]))
        multiplet_key = (int(label["alpha"]), int(label["L"]), int(label["twoS"]), int(label["twoJ"]))
        term_groups.setdefault(term_key, []).append(idx)
        multiplet_groups.setdefault(multiplet_key, []).append(idx)

    term_soc_summary: dict[tuple[int, int, int], dict[str, float]] = {}
    for alpha, L, twoS in sorted(term_groups):
        idxs = term_groups[(alpha, L, twoS)]
        V_block = V[:, idxs]
        projector = V_block @ V_block.conj().T
        diag, diag_imag, residual = eigenstate_diag_residual(
            V_block,
            projector @ ops["O_soc"] @ projector,
        )
        expected_block = expected["O_soc"][idxs]
        checks.extend(
            [
                _comparison_check(
                    name=f"LSJM term O_soc diagonal imaginary part (alpha={alpha}, L={L}, S={twoS/2.0:g})",
                    operator="O_soc",
                    scope="term",
                    value=diag_imag,
                    threshold=EPS_DIAG,
                    detail=f"max |Im diag| = {diag_imag:.2e}",
                ),
                _comparison_check(
                    name=f"LSJM term O_soc residual (alpha={alpha}, L={L}, S={twoS/2.0:g})",
                    operator="O_soc",
                    scope="term",
                    value=residual,
                    threshold=EPS_NORM,
                    detail=f"max residual = {residual:.2e}",
                ),
                _comparison_check(
                    name=f"LSJM term O_soc stored consistency (alpha={alpha}, L={L}, S={twoS/2.0:g})",
                    operator="O_soc",
                    scope="term",
                    value=float(np.max(np.abs(diag.real - expected_block))),
                    threshold=EPS_DIAG,
                    detail=f"max |diag-expected| = {np.max(np.abs(diag.real - expected_block)):.2e}",
                ),
            ]
        )
        term_soc_summary[(alpha, L, twoS)] = {
            "diag_imag_term": float(diag_imag),
            "residual_term": float(residual),
        }

    multiplet_rows = []
    for alpha, L, twoS, twoJ in sorted(multiplet_groups):
        idxs = multiplet_groups[(alpha, L, twoS, twoJ)]
        term_summary = term_soc_summary[(alpha, L, twoS)]
        L2_mean, L2_spread = _term_value(results["L2"]["diag"], idxs)
        S2_mean, S2_spread = _term_value(results["S2"]["diag"], idxs)
        J2_mean, J2_spread = _term_value(results["J2"]["diag"], idxs)
        soc_mean, soc_spread = _term_value(results["O_soc"]["diag"], idxs)
        multiplet_rows.append(
            {
                "alpha": alpha,
                "L": L,
                "S": twoS / 2.0,
                "J": twoJ / 2.0,
                "n": len(idxs),
                "L2": L2_mean, "L2_spread": L2_spread,
                "S2": S2_mean, "S2_spread": S2_spread,
                "J2": J2_mean, "J2_spread": J2_spread,
                "O_soc": soc_mean, "O_soc_spread": soc_spread,
                "diag_imag_term": term_summary["diag_imag_term"],
                "residual_term": term_summary["residual_term"],
            }
        )

    state_rows = []
    for idx, label in enumerate(labels):
        state_rows.append(
            {
                "alpha": int(label["alpha"]),
                "L": int(label["L"]),
                "S": float(label["twoS"]) / 2.0,
                "J": float(label["J"]),
                "M": float(label["M"]),
                "Jz": float(results["Jz"]["diag"][idx]),
            }
        )

    all_spreads = []
    for row in multiplet_rows:
        for key in ("L2_spread", "S2_spread", "J2_spread", "O_soc_spread"):
            all_spreads.append(row[key])
    max_spread = max(all_spreads)
    checks.append(
        _comparison_check(
            name="LSJM multiplet eigenvalue spread",
            operator="spread",
            scope="lsjm",
            value=max_spread,
            threshold=EPS_DIAG,
            detail=f"max spread across multiplets = {max_spread:.2e}",
        )
    )

    return {
        "multiplet_rows": multiplet_rows,
        "state_rows": state_rows,
        "global_soc": {
            "reference_only": True,
            "diag_imag": global_soc_imag,
            "residual": global_soc_residual,
            "diag_preview": [float(x) for x in global_soc_diag.real[: min(8, len(global_soc_diag))]],
        },
        "checks": checks,
    }


def _comparison_check(
    *,
    name: str,
    operator: str,
    scope: str,
    value: float,
    threshold: float,
    detail: str,
    reference_only: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "operator": operator,
        "scope": scope,
        "value": float(value),
        "threshold": float(threshold),
        "passed": float(value) <= float(threshold),
        "detail": detail,
        "reference_only": reference_only,
    }


def _term_value(values: np.ndarray, idxs: list[int]) -> tuple[float, float]:
    block = values[idxs]
    mean = float(np.mean(block))
    spread = float(np.max(np.abs(block - mean)))
    return mean, spread
