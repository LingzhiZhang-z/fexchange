"""
Numerical and analytic validation checks for LSMS/LSJM construction.

Each check function returns a list of result dicts:
    {"name": str, "passed": bool, "detail": str}
"""
from __future__ import annotations

from collections.abc import Callable
from math import comb
from typing import Any

import numpy as np

from fexchange.core.space_ls import build_space_ls_matrix
from fexchange.models.hsoc import build_hsoc_unit_matrix
from fexchange.representations.lsjm import build_lsjm, select_soc_lowest_subspace
from fexchange.representations.lsms import build_lsms
from fexchange.utils.numerics import EPS_DIAG, EPS_ORTH

# ---------------------------------------------------------------------------
# Slater-Condon ratio
# ---------------------------------------------------------------------------
F2_RATIO, F4_RATIO, F6_RATIO = 12.980, 8.163, 5.878
R42 = F4_RATIO / F2_RATIO
R62 = F6_RATIO / F2_RATIO

# ---------------------------------------------------------------------------
# f2 analytic coefficients from Reference-0 (RS_f2_En)
# ---------------------------------------------------------------------------
# Denominators that convert integer c_k to coef_Fk per unit F^k:
#   coef_F2 = c2 / 225,  coef_F4 = c4 / 1089,  coef_F6 = c6 / (184041/25)
_DENOM = {2: 225.0, 4: 1089.0, 6: 184041.0 / 25.0}

# (L, twoS) -> (c2, c4, c6)
F2_ANALYTIC = {
    (5, 2): (-25, -51, -13),    # 3H
    (3, 2): (-10, -33, -286),   # 3F
    (1, 2): (45, 33, -1287),    # 3P
    (6, 0): (25, 9, 1),         # 1I
    (4, 0): (-30, 97, 78),      # 1G
    (2, 0): (19, -99, 715),     # 1D
    (0, 0): (60, 198, 1716),    # 1S
}

TERM_NAMES = {
    (0, 0): "1S",
    (2, 0): "1D",
    (4, 0): "1G",
    (6, 0): "1I",
    (1, 2): "3P",
    (3, 2): "3F",
    (5, 2): "3H",
}

EPS_CHECK = 1e-8

NUMERICAL_SECTION = "Numerical Self-Consistency Checks"
LSMS_OPS_SECTION = "LSMS Operator Checks"
LSJM_OPS_SECTION = "LSJM Operator Checks"
REF0_SECTION = "Reference-0 Wavefunction Comparison (f2)"

SectionPayload = tuple[str, str, Any]


class ValidationContext:
    """Shared computation context for a single n_ele validation run."""

    def __init__(self, n_ele: int, *, r42: float = R42, r62: float = R62):
        self.n_ele = n_ele
        self.lsms = build_lsms(n_ele, r42=r42, r62=r62)
        self.lsjm = build_lsjm(self.lsms)
        self._angular_ops: dict[str, np.ndarray] | None = None
        self._hsoc_unit: np.ndarray | None = None

    @property
    def angular_ops(self) -> dict[str, np.ndarray]:
        if self._angular_ops is None:
            self._angular_ops = build_space_ls_matrix(self.n_ele)
        return self._angular_ops

    @property
    def hsoc_unit(self) -> np.ndarray:
        if self._hsoc_unit is None:
            self._hsoc_unit = build_hsoc_unit_matrix(self.n_ele)
        return self._hsoc_unit


# ===================================================================
# Numerical checks (valid for all n_ele)
# ===================================================================

def check_dimension(lsms: dict, lsjm: dict, n_ele: int) -> list[dict]:
    expected = comb(14, n_ele)
    n_lsms = lsms["V_fock"].shape[1]
    n_lsjm = lsjm["V_fock"].shape[1]
    return [
        _r("LSMS dimension", n_lsms == expected, f"got {n_lsms}, expected {expected}"),
        _r("LSJM dimension", n_lsjm == expected, f"got {n_lsjm}, expected {expected}"),
    ]


def check_orthonormality(lsms: dict, lsjm: dict) -> list[dict]:
    r_lsms = _orth_residual(lsms["V_fock"])
    r_lsjm = _orth_residual(lsjm["V_fock"])
    return [
        _r("LSMS orthonormality", r_lsms < EPS_ORTH, f"residual = {r_lsms:.2e}"),
        _r("LSJM orthonormality", r_lsjm < EPS_ORTH, f"residual = {r_lsjm:.2e}"),
    ]


def check_term_multiplicities(lsms: dict, n_ele: int) -> list[dict]:
    """Sum of (2L+1)(2S+1) over all terms must equal C(14, n)."""
    labels = lsms["labels"]
    terms: dict[tuple, int] = {}
    for lab in labels:
        key = (lab["alpha"], lab["L"], lab["twoS"])
        terms[key] = terms.get(key, 0) + 1
    total = sum(terms.values())
    expected = comb(14, n_ele)
    bad_terms = []
    for (alpha, L, twoS), count in terms.items():
        exp_count = (2 * L + 1) * (twoS + 1)
        if count != exp_count:
            bad_terms.append(f"alpha={alpha},L={L},2S={twoS}: {count}!={exp_count}")
    ok = total == expected and len(bad_terms) == 0
    detail = f"total={total}, expected={expected}"
    if bad_terms:
        detail += "; BAD: " + "; ".join(bad_terms)
    return [_r("Term multiplicities", ok, detail)]


def eigenstate_check(
    V: np.ndarray,
    O: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return diagonal expectations plus imaginary and residual diagnostics."""
    OV = O @ V
    diag = np.einsum("ij,ij->j", V.conj(), OV)
    residual = OV - V * diag[np.newaxis, :]
    max_imag = float(np.max(np.abs(diag.imag)))
    max_offdiag = float(np.max(np.linalg.norm(residual, axis=0)))
    return diag, max_imag, max_offdiag


def analytic_lsms_angular(labels: list[dict]) -> dict[str, np.ndarray]:
    return {
        "L2": np.array([lab["L"] * (lab["L"] + 1) for lab in labels], dtype=float),
        "S2": np.array([(lab["twoS"] / 2) * (lab["twoS"] / 2 + 1) for lab in labels], dtype=float),
        "Lz": np.array([float(lab["ML"]) for lab in labels], dtype=float),
        "Sz": np.array([float(lab["MS"]) for lab in labels], dtype=float),
    }


def analytic_lsms_fk(n_ele: int, labels: list[dict]) -> dict[int, np.ndarray] | None:
    if n_ele == 1:
        zeros = np.zeros(len(labels), dtype=float)
        return {0: zeros.copy(), 2: zeros.copy(), 4: zeros.copy(), 6: zeros.copy()}
    if n_ele != 2:
        return None

    analytic = {0: np.ones(len(labels), dtype=float)}
    for k in (2, 4, 6):
        arr = np.zeros(len(labels), dtype=float)
        analytic[k] = arr

    for idx, lab in enumerate(labels):
        coeffs = F2_ANALYTIC[(lab["L"], lab["twoS"])]
        analytic[2][idx] = coeffs[0] / _DENOM[2]
        analytic[4][idx] = coeffs[1] / _DENOM[4]
        analytic[6][idx] = coeffs[2] / _DENOM[6]
    return analytic


def analytic_lsjm_angular(labels: list[dict]) -> dict[str, np.ndarray]:
    return {
        "L2": np.array([lab["L"] * (lab["L"] + 1) for lab in labels], dtype=float),
        "S2": np.array([(lab["twoS"] / 2) * (lab["twoS"] / 2 + 1) for lab in labels], dtype=float),
        "J2": np.array([(lab["twoJ"] / 2) * (lab["twoJ"] / 2 + 1) for lab in labels], dtype=float),
        "Jz": np.array([lab["twoM"] / 2 for lab in labels], dtype=float),
    }


def analytic_lsjm_zeta(n_ele: int, labels: list[dict]) -> np.ndarray | None:
    prefactor = {1: 0.5, 2: 0.25, 12: -0.25, 13: -0.5}.get(n_ele)
    if prefactor is None:
        return None

    analytic = np.zeros(len(labels), dtype=float)
    for idx, lab in enumerate(labels):
        L = float(lab["L"])
        S = lab["twoS"] / 2
        J = lab["twoJ"] / 2
        analytic[idx] = prefactor * (J * (J + 1) - L * (L + 1) - S * (S + 1))
    return analytic


def analytic_lsjm_fk(n_ele: int, labels: list[dict]) -> dict[int, np.ndarray] | None:
    return analytic_lsms_fk(n_ele, labels)


# ===================================================================
# Analytic checks for specific n
# ===================================================================

def check_f1_analytic(lsms: dict, lsjm: dict) -> list[dict]:
    """f1: single 2F term, zero Coulomb, coef_zeta = (J(J+1)-12-3/4)/2."""
    results = []

    for k_name in ("coef_F2", "coef_F4", "coef_F6"):
        mx = float(np.max(np.abs(lsms[k_name])))
        results.append(_r(f"f1 {k_name} = 0", mx < EPS_CHECK, f"max |{k_name}| = {mx:.2e}"))

    terms = _unique_terms(lsms)
    results.append(_r("f1 single term 2F", terms == {(0, 3, 1)}, f"terms = {_fmt_terms(terms)}"))

    for twoJ, expected in [(5, -2.0), (7, 1.5)]:
        vals = _coef_zeta_for_twoJ(lsjm, twoJ)
        if vals:
            actual = vals[0]
            results.append(
                _r(
                    f"f1 coef_zeta J={twoJ}/2",
                    abs(actual - expected) < EPS_CHECK,
                    f"computed={actual:.6f}, analytic={expected:.6f}",
                )
            )
        else:
            results.append(_r(f"f1 coef_zeta J={twoJ}/2", False, "no states found"))

    soc0 = select_soc_lowest_subspace(lsjm)
    results.append(_r("f1 ground J = 5/2", abs(soc0["J0"] - 2.5) < 0.01, f"J0 = {soc0['J0']}"))
    results.append(_r("f1 ground multiplet size = 6", soc0["n_j"] == 6, f"n_j = {soc0['n_j']}"))
    return results


def check_f2_analytic(lsms: dict, lsjm: dict) -> list[dict]:
    """f2: 7 LS terms, analytic coef_Fk and coef_zeta."""
    results = []

    for (L, twoS), (c2, c4, c6) in sorted(F2_ANALYTIC.items()):
        name = TERM_NAMES.get((L, twoS), f"L={L},2S={twoS}")
        exp2, exp4, exp6 = c2 / _DENOM[2], c4 / _DENOM[4], c6 / _DENOM[6]
        got2, got4, got6 = _coef_fk_for_term(lsms, L, twoS)
        if got2 is None:
            results.append(_r(f"f2 coef_Fk {name}", False, "term not found"))
            continue
        err2, err4, err6 = abs(got2 - exp2), abs(got4 - exp4), abs(got6 - exp6)
        max_err = max(err2, err4, err6)
        results.append(
            _r(
                f"f2 coef_Fk {name}",
                max_err < EPS_CHECK,
                f"F2: {got2:+.6f} (exp {exp2:+.6f}, err {err2:.1e})  "
                f"F4: {got4:+.6f} (exp {exp4:+.6f}, err {err4:.1e})  "
                f"F6: {got6:+.6f} (exp {exp6:+.6f}, err {err6:.1e})",
            )
        )

    labels = lsjm["labels"]
    coef_z = lsjm["coef_zeta"]
    checked: set[tuple] = set()
    for idx, lab in enumerate(labels):
        key = (lab["L"], lab["twoS"], lab["twoJ"])
        if key in checked:
            continue
        checked.add(key)
        L, twoS, twoJ = key
        S, J = twoS / 2, twoJ / 2
        expected = 0.25 * (J * (J + 1) - L * (L + 1) - S * (S + 1))
        actual = float(coef_z[idx])
        name_ls = TERM_NAMES.get((L, twoS), f"L={L},2S={twoS}")
        results.append(
            _r(
                f"f2 coef_zeta {name_ls} J={J}",
                abs(actual - expected) < EPS_CHECK,
                f"computed={actual:.6f}, analytic={expected:.6f}",
            )
        )

    soc0 = select_soc_lowest_subspace(lsjm)
    results.append(
        _r(
            "f2 ground term = 3H (L=5, S=1)",
            soc0["L0"] == 5 and abs(soc0["S0"] - 1.0) < 0.01,
            f"L0={soc0['L0']}, S0={soc0['S0']}",
        )
    )
    results.append(_r("f2 ground J = 4", abs(soc0["J0"] - 4.0) < 0.01, f"J0 = {soc0['J0']}"))
    return results


def check_f13_analytic(lsms: dict, lsjm: dict) -> list[dict]:
    """f13: particle-hole conjugate of f1, sign-flipped coef_zeta."""
    results = []

    terms = _unique_terms(lsms)
    results.append(_r("f13 single term 2F", terms == {(0, 3, 1)}, f"terms = {_fmt_terms(terms)}"))

    for twoJ, expected in [(5, 2.0), (7, -1.5)]:
        vals = _coef_zeta_for_twoJ(lsjm, twoJ)
        if vals:
            actual = vals[0]
            results.append(
                _r(
                    f"f13 coef_zeta J={twoJ}/2",
                    abs(actual - expected) < EPS_CHECK,
                    f"computed={actual:.6f}, analytic={expected:.6f}",
                )
            )
        else:
            results.append(_r(f"f13 coef_zeta J={twoJ}/2", False, "no states found"))

    for k_name in ("coef_F2", "coef_F4", "coef_F6"):
        arr = lsms[k_name]
        spread = float(np.max(arr) - np.min(arr))
        results.append(
            _r(
                f"f13 {k_name} uniform",
                spread < EPS_CHECK,
                f"spread = {spread:.2e}, value = {float(arr[0]):.6f}",
            )
        )

    soc0 = select_soc_lowest_subspace(lsjm)
    results.append(_r("f13 ground J = 7/2", abs(soc0["J0"] - 3.5) < 0.01, f"J0 = {soc0['J0']}"))
    results.append(_r("f13 ground multiplet size = 8", soc0["n_j"] == 8, f"n_j = {soc0['n_j']}"))
    return results


# ===================================================================
# Operator-suite checks and tables
# ===================================================================

def compute_lsms_checks_and_table(
    lsms: dict,
    n_ele: int,
    angular_ops: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    del n_ele

    V = lsms["V_fock"]
    labels = lsms["labels"]
    stored_angular = analytic_lsms_angular(labels)
    analytic_fk = analytic_lsms_fk(lsms["n_ele"], labels)

    op_specs = [
        ("H0", "H0", lsms["hint_rank_matrix_map"][0], np.asarray(lsms["coef_F0"], dtype=float), analytic_fk[0] if analytic_fk else None),
        ("H2", "H2", lsms["hint_rank_matrix_map"][2], np.asarray(lsms["coef_F2"], dtype=float), analytic_fk[2] if analytic_fk else None),
        ("H4", "H4", lsms["hint_rank_matrix_map"][4], np.asarray(lsms["coef_F4"], dtype=float), analytic_fk[4] if analytic_fk else None),
        ("H6", "H6", lsms["hint_rank_matrix_map"][6], np.asarray(lsms["coef_F6"], dtype=float), analytic_fk[6] if analytic_fk else None),
        ("L2", "L2", angular_ops["L2"], stored_angular["L2"], None),
        ("S2", "S2", angular_ops["S2"], stored_angular["S2"], None),
        ("Lz", "Lz", angular_ops["Lz"], stored_angular["Lz"], None),
        ("Sz", "Sz", angular_ops["Sz"], stored_angular["Sz"], None),
    ]

    diag_real: dict[str, np.ndarray] = {}
    checks: list[dict] = []

    for display_name, key, operator, stored_expected, analytic_expected in op_specs:
        diag, max_imag, max_offdiag = eigenstate_check(V, operator)
        diag_real[key] = diag.real
        stored_err = float(np.max(np.abs(diag.real - stored_expected)))
        checks.extend(
            [
                _r(f"LSMS {display_name} diagonal imaginary part", max_imag < EPS_CHECK, f"max |Im diag| = {max_imag:.2e}"),
                _r(f"LSMS {display_name} off-diagonal residual", max_offdiag < EPS_DIAG, f"max residual = {max_offdiag:.2e}"),
                _r(f"LSMS {display_name} vs stored", stored_err < EPS_CHECK, f"max |Re diag - expected| = {stored_err:.2e}"),
            ]
        )
        if analytic_expected is not None:
            analytic_err = float(np.max(np.abs(diag.real - analytic_expected)))
            checks.append(
                _r(
                    f"LSMS {display_name} vs analytic",
                    analytic_err < EPS_CHECK,
                    f"max |Re diag - analytic| = {analytic_err:.2e}",
                )
            )

    table_rows: list[dict] = []
    m_rows: list[dict] = []
    term_blocks = _term_blocks(labels)
    for key in sorted(term_blocks):
        alpha, L, twoS = key
        idxs = term_blocks[key]
        row = {
            "alpha": alpha,
            "L": L,
            "twoS": twoS,
            "n_states": len(idxs),
        }
        for op_key, stored_key in (
            ("H0", "coef_F0"),
            ("H2", "coef_F2"),
            ("H4", "coef_F4"),
            ("H6", "coef_F6"),
            ("L2", "L2"),
            ("S2", "S2"),
        ):
            values = diag_real[op_key][idxs]
            row[f"{op_key}_num"] = float(np.mean(values))
            row[f"{op_key}_ana"] = float(lsms[stored_key][idxs[0]]) if stored_key.startswith("coef_") else float(stored_angular[stored_key][idxs[0]])
            row[f"{op_key}_spr"] = float(np.max(values) - np.min(values))
        table_rows.append(row)
        for idx in idxs:
            m_rows.append(
                {
                    "alpha": alpha,
                    "L": L,
                    "twoS": twoS,
                    "ML": int(labels[idx]["ML"]),
                    "MS": float(labels[idx]["MS"]),
                    "Lz_num": float(diag_real["Lz"][idx]),
                    "Sz_num": float(diag_real["Sz"][idx]),
                }
            )

    analytic_text = None
    if analytic_fk is not None:
        def _fmt_half_int(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        lines = [
            f"Analytic (f{lsms['n_ele']}):",
            "  Term  L  S  coef_F2_formula  coef_F4_formula  coef_F6_formula  err_F2  err_F4  err_F6",
        ]
        for key in sorted(term_blocks):
            _, L, twoS = key
            idx = term_blocks[key][0]
            name = TERM_NAMES.get((L, twoS), f"L={L},2S={twoS}")
            exp2 = float(analytic_fk[2][idx])
            exp4 = float(analytic_fk[4][idx])
            exp6 = float(analytic_fk[6][idx])
            err2 = abs(float(lsms["coef_F2"][idx]) - exp2)
            err4 = abs(float(lsms["coef_F4"][idx]) - exp4)
            err6 = abs(float(lsms["coef_F6"][idx]) - exp6)
            lines.append(
                f"  {name:<4} {L:>2} {_fmt_half_int(twoS):>2}  "
                f"{exp2:+15.6f}  {exp4:+15.6f}  {exp6:+15.6f}  "
                f"{err2:.1e}  {err4:.1e}  {err6:.1e}"
            )
        analytic_text = "\n".join(lines)

    m_rows.sort(key=lambda row: (row["alpha"], row["L"], row["twoS"], row["ML"], row["MS"]))
    return checks, table_rows, m_rows, analytic_text


def compute_lsjm_checks_and_table(
    lsjm: dict,
    n_ele: int,
    angular_ops: dict[str, np.ndarray],
    hsoc_unit: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict], list[dict], str | None, str | None]:
    del n_ele

    V = lsjm["V_fock"]
    labels = lsjm["labels"]
    stored_angular = analytic_lsjm_angular(labels)
    analytic_zeta = analytic_lsjm_zeta(lsjm["n_ele"], labels)
    analytic_fk = analytic_lsjm_fk(lsjm["n_ele"], labels)

    op_specs = [
        ("L2", "L2", angular_ops["L2"], stored_angular["L2"], None),
        ("S2", "S2", angular_ops["S2"], stored_angular["S2"], None),
        ("J2", "J2", angular_ops["J2"], stored_angular["J2"], None),
        ("Jz", "Jz", angular_ops["Jz"], stored_angular["Jz"], None),
        ("H0", "H0", lsjm["hint_rank_matrix_map"][0], np.asarray(lsjm["coef_F0"], dtype=float), analytic_fk[0] if analytic_fk else None),
        ("H2", "H2", lsjm["hint_rank_matrix_map"][2], np.asarray(lsjm["coef_F2"], dtype=float), analytic_fk[2] if analytic_fk else None),
        ("H4", "H4", lsjm["hint_rank_matrix_map"][4], np.asarray(lsjm["coef_F4"], dtype=float), analytic_fk[4] if analytic_fk else None),
        ("H6", "H6", lsjm["hint_rank_matrix_map"][6], np.asarray(lsjm["coef_F6"], dtype=float), analytic_fk[6] if analytic_fk else None),
    ]

    diag_real: dict[str, np.ndarray] = {}
    checks: list[dict] = []

    for display_name, key, operator, stored_expected, analytic_expected in op_specs:
        diag, max_imag, max_offdiag = eigenstate_check(V, operator)
        diag_real[key] = diag.real
        stored_err = float(np.max(np.abs(diag.real - stored_expected)))
        checks.extend(
            [
                _r(f"LSJM {display_name} diagonal imaginary part", max_imag < EPS_CHECK, f"max |Im diag| = {max_imag:.2e}"),
                _r(f"LSJM {display_name} off-diagonal residual", max_offdiag < EPS_DIAG, f"max residual = {max_offdiag:.2e}"),
                _r(f"LSJM {display_name} vs stored", stored_err < EPS_CHECK, f"max |Re diag - expected| = {stored_err:.2e}"),
            ]
        )
        if analytic_expected is not None:
            analytic_err = float(np.max(np.abs(diag.real - analytic_expected)))
            checks.append(
                _r(
                    f"LSJM {display_name} vs analytic",
                    analytic_err < EPS_CHECK,
                    f"max |Re diag - analytic| = {analytic_err:.2e}",
                )
            )

    zeta_diag_real = np.zeros(len(labels), dtype=float)
    soc_term_rows: list[dict] = []
    term_blocks = _term_blocks(labels)
    for key in sorted(term_blocks):
        alpha, L, twoS = key
        idxs = term_blocks[key]
        V_term = V[:, idxs]
        hsoc_term = V_term.conj().T @ hsoc_unit @ V_term
        diag, max_imag, max_offdiag = eigenstate_check(np.eye(len(idxs), dtype=complex), hsoc_term)
        diag_real_term = diag.real
        zeta_diag_real[idxs] = diag_real_term

        stored_expected = np.asarray(lsjm["coef_zeta"], dtype=float)[idxs]
        stored_err = float(np.max(np.abs(diag_real_term - stored_expected)))
        checks.extend(
            [
                _r(
                    f"LSJM O_soc term alpha={alpha} L={L} twoS={twoS} diagonal imaginary part",
                    max_imag < EPS_CHECK,
                    f"max |Im diag| = {max_imag:.2e}",
                ),
                _r(
                    f"LSJM O_soc term alpha={alpha} L={L} twoS={twoS} off-diagonal residual",
                    max_offdiag < EPS_DIAG,
                    f"max residual = {max_offdiag:.2e}",
                ),
                _r(
                    f"LSJM O_soc term alpha={alpha} L={L} twoS={twoS} vs stored",
                    stored_err < EPS_CHECK,
                    f"max |Re diag - expected| = {stored_err:.2e}",
                ),
            ]
        )
        if analytic_zeta is not None:
            analytic_expected = analytic_zeta[idxs]
            analytic_err = float(np.max(np.abs(diag_real_term - analytic_expected)))
            checks.append(
                _r(
                    f"LSJM O_soc term alpha={alpha} L={L} twoS={twoS} vs analytic",
                    analytic_err < EPS_CHECK,
                    f"max |Re diag - analytic| = {analytic_err:.2e}",
                )
            )

        j_blocks: dict[int, list[int]] = {}
        for pos, idx in enumerate(idxs):
            j_blocks.setdefault(labels[idx]["twoJ"], []).append(pos)

        zeta_blocks: list[dict] = []
        for twoJ in sorted(j_blocks):
            local_positions = j_blocks[twoJ]
            values = diag_real_term[local_positions]
            first_global_idx = idxs[local_positions[0]]
            zeta_blocks.append(
                {
                    "twoJ": twoJ,
                    "n_states": len(local_positions),
                    "zeta_num": float(np.mean(values)),
                    "zeta_ana": float(lsjm["coef_zeta"][first_global_idx]),
                    "zeta_spr": float(np.max(values) - np.min(values)),
                }
            )

        soc_term_rows.append(
            {
                "alpha": alpha,
                "L": L,
                "twoS": twoS,
                "n_states": len(idxs),
                "soc_max_imag": max_imag,
                "soc_max_offdiag": max_offdiag,
                "zeta_blocks": zeta_blocks,
            }
        )

    multiplets: dict[tuple[int, int, int, int], list[int]] = {}
    for idx, lab in enumerate(labels):
        key = (lab["alpha"], lab["L"], lab["twoS"], lab["twoJ"])
        multiplets.setdefault(key, []).append(idx)

    table_rows: list[dict] = []
    m_rows: list[dict] = []
    for key in sorted(multiplets):
        alpha, L, twoS, twoJ = key
        idxs = multiplets[key]
        row = {
            "alpha": alpha,
            "L": L,
            "twoS": twoS,
            "twoJ": twoJ,
            "n_states": len(idxs),
        }
        for op_key, stored_key in (
            ("L2", "L2"),
            ("S2", "S2"),
            ("J2", "J2"),
            ("H0", "coef_F0"),
            ("H2", "coef_F2"),
            ("H4", "coef_F4"),
            ("H6", "coef_F6"),
        ):
            values = diag_real[op_key][idxs]
            row[f"{op_key}_num"] = float(np.mean(values))
            row[f"{op_key}_ana"] = float(lsjm[stored_key][idxs[0]]) if stored_key.startswith("coef_") else float(stored_angular[stored_key][idxs[0]])
            row[f"{op_key}_spr"] = float(np.max(values) - np.min(values))

        zeta_values = zeta_diag_real[idxs]
        row["zeta_num"] = float(np.mean(zeta_values))
        row["zeta_ana"] = float(lsjm["coef_zeta"][idxs[0]])
        row["zeta_spr"] = float(np.max(zeta_values) - np.min(zeta_values))
        table_rows.append(row)
        for idx in idxs:
            m_rows.append(
                {
                    "alpha": alpha,
                    "L": L,
                    "twoS": twoS,
                    "twoJ": twoJ,
                    "twoM": int(labels[idx]["twoM"]),
                    "Jz_num": float(diag_real["Jz"][idx]),
                }
            )

    analytic_text = None
    if analytic_zeta is not None:
        def _fmt_half_int(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        lines = [
            f"Analytic SOC (f{lsjm['n_ele']}):",
            "  Term  L  S  J  zeta_formula  zeta_stored  err",
        ]
        for key in sorted(multiplets):
            _, L, twoS, twoJ = key
            idx = multiplets[key][0]
            name = TERM_NAMES.get((L, twoS), f"L={L},2S={twoS}")
            formula = float(analytic_zeta[idx])
            stored = float(lsjm["coef_zeta"][idx])
            err = abs(stored - formula)
            lines.append(
                f"  {name:<4} {L:>2} {_fmt_half_int(twoS):>2} {_fmt_half_int(twoJ):>3}  "
                f"{formula:+12.6f}  {stored:+11.6f}  {err:.1e}"
            )
        analytic_text = "\n".join(lines)

    global_diag, global_max_imag, global_max_offdiag = eigenstate_check(V, hsoc_unit)
    preview_items = "  ".join(
        f"(state_{idx + 1},{value.real:+.6f})"
        for idx, value in enumerate(global_diag[: min(len(global_diag), 8)])
    )
    soc_reference_text = "\n".join(
        [
            "Reference O_soc (global LSJM basis):",
            f"  max_imag     {global_max_imag:.1e}",
            f"  max_offdiag  {global_max_offdiag:.1e}",
            f"  diag preview: {preview_items}",
        ]
    )

    m_rows.sort(key=lambda row: (row["alpha"], row["L"], row["twoS"], row["twoJ"], row["twoM"]))
    return checks, table_rows, m_rows, soc_term_rows, analytic_text, soc_reference_text


# ===================================================================
# Validation modules
# ===================================================================

def mod_structural(ctx: ValidationContext) -> list[SectionPayload]:
    return [
        (
            NUMERICAL_SECTION,
            "checks",
            (
                check_dimension(ctx.lsms, ctx.lsjm, ctx.n_ele)
                + check_orthonormality(ctx.lsms, ctx.lsjm)
                + check_term_multiplicities(ctx.lsms, ctx.n_ele)
            ),
        )
    ]


def mod_lsms_operators(ctx: ValidationContext) -> list[SectionPayload]:
    checks, table_rows, m_rows, analytic_text = compute_lsms_checks_and_table(
        ctx.lsms,
        ctx.n_ele,
        angular_ops=ctx.angular_ops,
    )
    payloads: list[SectionPayload] = [
        (LSMS_OPS_SECTION, "lsms_operator_table", table_rows),
        (LSMS_OPS_SECTION, "lsms_m_table", m_rows),
    ]
    if analytic_text is not None:
        payloads.append((LSMS_OPS_SECTION, "analytic_fk_list", analytic_text))
    payloads.append((LSMS_OPS_SECTION, "checks", checks))
    return payloads


def mod_lsjm_operators(ctx: ValidationContext) -> list[SectionPayload]:
    lsjm = dict(ctx.lsjm)
    lsjm["hint_rank_matrix_map"] = ctx.lsms["hint_rank_matrix_map"]
    checks, table_rows, m_rows, soc_term_rows, analytic_text, soc_reference_text = compute_lsjm_checks_and_table(
        lsjm,
        ctx.n_ele,
        angular_ops=ctx.angular_ops,
        hsoc_unit=ctx.hsoc_unit,
    )
    payloads: list[SectionPayload] = [
        (LSJM_OPS_SECTION, "lsjm_soc_term_table", soc_term_rows),
        (LSJM_OPS_SECTION, "lsjm_operator_table", table_rows),
        (LSJM_OPS_SECTION, "lsjm_m_table", m_rows),
    ]
    if analytic_text is not None:
        payloads.append((LSJM_OPS_SECTION, "analytic_soc_list", analytic_text))
    if soc_reference_text is not None:
        payloads.append((LSJM_OPS_SECTION, "lsjm_soc_global_reference", soc_reference_text))
    payloads.append((LSJM_OPS_SECTION, "checks", checks))
    return payloads


def mod_analytic(ctx: ValidationContext) -> list[SectionPayload]:
    if ctx.n_ele == 1:
        return [("Analytic Checks (f1)", "checks", check_f1_analytic(ctx.lsms, ctx.lsjm))]
    if ctx.n_ele == 2:
        return [("Analytic Checks (f2)", "checks", check_f2_analytic(ctx.lsms, ctx.lsjm))]
    if ctx.n_ele == 13:
        return [("Analytic Checks (f13)", "checks", check_f13_analytic(ctx.lsms, ctx.lsjm))]
    return []


def mod_ref0(ctx: ValidationContext) -> list[SectionPayload]:
    if ctx.n_ele != 2:
        return []

    from tests.validation.ref0_compare import compare_ref0_f2

    ref0_results = compare_ref0_f2(ctx.lsjm)
    all_matched = all(r["matched"] for r in ref0_results)
    min_overlap = min(r["overlap_sq"] for r in ref0_results if r["matched"])
    max_dE = max(r["dE"] for r in ref0_results if r["matched"])

    return [
        (REF0_SECTION, "ref0_table", ref0_results),
        (REF0_SECTION, "text", ""),
        (
            REF0_SECTION,
            "checks",
            [
                {
                    "name": "Ref0 all states matched",
                    "passed": all_matched,
                    "detail": f"{len(ref0_results)} states",
                },
                {
                    "name": "Ref0 min |<ref|fex>|^2",
                    "passed": min_overlap > 1.0 - 1e-8,
                    "detail": f"{min_overlap:.10f}",
                },
                {
                    "name": "Ref0 max |dE|",
                    "passed": max_dE < 1e-8,
                    "detail": f"{max_dE:.2e}",
                },
            ],
        ),
    ]


MODULE_REGISTRY: dict[str, Callable[[ValidationContext], list[SectionPayload]]] = {
    "structural": mod_structural,
    "lsms_operators": mod_lsms_operators,
    "lsjm_operators": mod_lsjm_operators,
    "analytic": mod_analytic,
    "ref0": mod_ref0,
}

DEFAULT_MODULES = ["structural", "lsms_operators", "lsjm_operators", "analytic", "ref0"]


# ===================================================================
# Internal helpers
# ===================================================================

def _r(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": passed, "detail": detail}


def _orth_residual(V: np.ndarray) -> float:
    gram = V.conj().T @ V
    return float(np.linalg.norm(gram - np.eye(V.shape[1]), "fro"))


def _term_blocks(labels: list[dict]) -> dict[tuple, list[int]]:
    blocks: dict[tuple, list[int]] = {}
    for idx, lab in enumerate(labels):
        key = (lab["alpha"], lab["L"], lab["twoS"])
        blocks.setdefault(key, []).append(idx)
    return blocks


def _unique_terms(lsms: dict) -> set[tuple]:
    return {(lab["alpha"], lab["L"], lab["twoS"]) for lab in lsms["labels"]}


def _fmt_terms(terms: set[tuple]) -> str:
    return ", ".join(f"(a={a},L={L},2S={s})" for a, L, s in sorted(terms))


def _coef_zeta_for_twoJ(lsjm: dict, twoJ: int) -> list[float]:
    """Return coef_zeta values for all states with given twoJ."""
    labels = lsjm["labels"]
    coef_z = lsjm["coef_zeta"]
    return [float(coef_z[i]) for i, lab in enumerate(labels) if lab["twoJ"] == twoJ]


def _coef_fk_for_term(
    lsms: dict,
    L: int,
    twoS: int,
) -> tuple[float | None, float | None, float | None]:
    """Return (coef_F2, coef_F4, coef_F6) for the first state in the given LS term."""
    for idx, lab in enumerate(lsms["labels"]):
        if lab["L"] == L and lab["twoS"] == twoS:
            return (
                float(lsms["coef_F2"][idx]),
                float(lsms["coef_F4"][idx]),
                float(lsms["coef_F6"][idx]),
            )
    return None, None, None
