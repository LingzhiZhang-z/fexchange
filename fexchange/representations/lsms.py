"""
LSMS representation via raising-operator kernel null-space method.

Spec reference: 03-00-REPRESENTATION_LSMS.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

from fexchange.core.fock_basis import (
    N_ORB, dim_sector, make_basis_id,
)
from fexchange.core.space_ls import ELL
from fexchange.core.space_ls import build_space_ls_matrix
from fexchange.models.hint import build_hint_rank_matrix
from fexchange.utils.numerics import (
    DTYPE_COMPLEX, EPS_ZERO, EPS_NORM, EPS_DIAG, EPS_SVD_REL,
)
from fexchange.utils.checks import check_orthonormal
from fexchange.utils.errors import PhysError, NumError

logger = logging.getLogger("fexchange")


def _build_hint_reference_matrix(
    hint_rank_matrix_map: dict[int, NDArray[np.complexfloating]],
    *,
    r42: float,
    r62: float,
    Jh: float = 1.0,
) -> tuple[NDArray[np.complexfloating], float, float, float]:
    """
    Build one explicit H_int reference operator used for deterministic gauges.

    LSMS does not need absolute F0/U. For deterministic alpha-fixing we use
    U=F0=0 with ratios r42 = F4/F2, r62 = F6/F2, and Hund coupling Jh.

    From 02-01:
        Jh = (286*F2 + 195*F4 + 250*F6) / 6435.
    With F4=r42*F2, F6=r62*F2:
        F2 = 6435*Jh / (286 + 195*r42 + 250*r62).

    Returns (H_int_ref, F2, F4, F6).
    """
    denom = 286.0 + 195.0 * float(r42) + 250.0 * float(r62)
    if abs(denom) < EPS_ZERO:
        raise PhysError(
            "FXE-PHYS-001",
            "Invalid r42/r62 for Jh reference: denominator is zero",
            module="lsms",
            actual={"r42": float(r42), "r62": float(r62), "denom": denom},
        )
    F2 = 6435.0 * float(Jh) / denom
    F4 = float(r42) * F2
    F6 = float(r62) * F2
    H_int_ref = F2 * hint_rank_matrix_map[2] + F4 * hint_rank_matrix_map[4] + F6 * hint_rank_matrix_map[6]
    return H_int_ref, F2, F4, F6


def _expectation_error(
    op: NDArray[np.complexfloating],
    v: NDArray[np.complexfloating],
    expected: float,
) -> float:
    """Absolute expectation-value error |<v|op|v> - expected|."""
    value = float((v.conj() @ op @ v).real)
    return float(abs(value - expected))


def build_lsms(
    n_ele: int,
    n_orb: int = N_ORB,
    *,
    U: float = 0.0,
    Jh: float = 1.0,
    r42: float = 1.0,
    r62: float = 1.0,
) -> dict[str, Any]:
    """
    Build complete LSMS representation for sector n_ele (03-00 §3).

    Returns dict with:
        V_fock: (dim_fock, n_states) matrix of LSMS states.
        labels: list of per-column label dicts {alpha, L, twoS, ML, MS}.
        hint_rank_matrix_map: rank-resolved H_int matrix representations for
            energy decomposition.
        coef_Fk: dict of per-state expectation value arrays.
    """
    logger.info("Building LSMS for n_ele=%d", n_ele)
    ang = build_space_ls_matrix(n_ele, n_orb)
    L2 = ang["L2"]
    S2 = ang["S2"]
    Lz = ang["Lz"]
    Sz = ang["Sz"]
    Lp = ang["Lp"]
    Sp = ang["Sp"]
    Lm = ang["Lm"]
    Sm = ang["Sm"]

    hint_rank_matrix_map = {k: build_hint_rank_matrix(k, n_ele, n_orb) for k in (0, 2, 4, 6)}
    H_int_ref, F2, F4, F6 = _build_hint_reference_matrix(hint_rank_matrix_map, r42=r42, r62=r62, Jh=Jh)
    dim = dim_sector(n_ele, n_orb)

    # Build (ML, MS) subspace projectors (03-00 §3.4 step 1)
    Lz_diag = np.diag(Lz).real
    Sz_diag = np.diag(Sz).real
    sectors: dict[tuple[float, float], list[int]] = {}
    for i in range(dim):
        ml = round(Lz_diag[i] * 2) / 2
        ms = round(Sz_diag[i] * 2) / 2
        sectors.setdefault((ml, ms), []).append(i)

    def _get_embedding(ml: float, ms: float) -> NDArray[np.complexfloating] | None:
        """R_{ML,MS} embedding matrix: (dim_fock, dim_sector)."""
        idxs = sectors.get((ml, ms))
        if idxs is None or len(idxs) == 0:
            return None
        R = np.zeros((dim, len(idxs)), dtype=DTYPE_COMPLEX)
        for col, row in enumerate(idxs):
            R[row, col] = 1.0
        return R

    # Enumerate valid (L, S) pairs (prompt note 6)
    ell = ELL
    max_L = n_ele * ell
    max_twoS = n_ele  # 2*S_max = n_ele (if n_ele <= n_orb/2)
    if n_ele > n_orb // 2:
        max_twoS = n_orb - n_ele

    all_columns: list[NDArray[np.complexfloating]] = []
    all_labels: list[dict[str, Any]] = []

    term_alpha_counter: dict[tuple[int, int], int] = {}

    # Collect all (L, S) candidates; store R_LS to avoid recomputation below.
    candidates: list[tuple[int, int, NDArray[np.complexfloating]]] = []
    for L in range(0, max_L + 1):
        for twoS in range(0, max_twoS + 1):
            # S must match n_ele parity: n_ele even -> twoS even, n_ele odd -> twoS odd
            if twoS % 2 != n_ele % 2:
                continue
            S = twoS / 2.0
            R_hw = _get_embedding(float(L), S)
            if R_hw is None:
                continue
            candidates.append((L, twoS, R_hw))

    for L, twoS, R_LS in candidates:
        cols, labs = _process_LS_term(
            L, twoS,
            Lp, Sp, Lm, Sm,
            H_int_ref,
            R_LS, _get_embedding,
            term_alpha_counter,
        )
        all_columns.extend(cols)
        all_labels.extend(labs)

    if len(all_columns) == 0:
        raise PhysError(
            "FXE-PHYS-001",
            f"No LSMS states found for n_ele={n_ele}",
            actual={"n_ele": n_ele},
        )

    V_fock = np.column_stack(all_columns)

    # Orthonormality check
    check_orthonormal(V_fock, label="V_lsms", module="lsms")

    # Compute energy coefficients (03-00 §7.1)
    coef_Fk: dict[int, NDArray[np.floating]] = {}
    for k in (0, 2, 4, 6):
        coef = np.array([
            (V_fock[:, j].conj() @ hint_rank_matrix_map[k] @ V_fock[:, j]).real
            for j in range(V_fock.shape[1])
        ])
        coef_Fk[k] = coef

    # Diagonal-subspace check for H_int (03-00 §7.2)
    H_proj = V_fock.conj().T @ H_int_ref @ V_fock
    offdiag = H_proj - np.diag(np.diag(H_proj))
    max_offdiag = float(np.max(np.abs(offdiag))) if offdiag.size else 0.0
    if max_offdiag > EPS_DIAG:
        raise NumError(
            "FXE-NUM-001",
            f"LSMS diagonal-subspace check failed: max_offdiag={max_offdiag:.3e} > eps_diag={EPS_DIAG:.1e}",
            module="lsms",
            level="LMSM",
            actual={
                "residual_name": "r_diag_lsms",
                "residual_value": max_offdiag,
                "threshold": EPS_DIAG,
            },
        )

    # Required runtime residual checks (03-00 §3.3)
    max_r_L2 = 0.0
    max_r_S2 = 0.0
    max_r_Lz = 0.0
    max_r_Sz = 0.0
    for j, label in enumerate(all_labels):
        v = V_fock[:, j]
        L = int(label["L"])
        S = 0.5 * int(label["twoS"])
        ML = float(label["ML"])
        MS = float(label["MS"])

        max_r_L2 = max(max_r_L2, _expectation_error(L2, v, float(L * (L + 1))))
        max_r_S2 = max(max_r_S2, _expectation_error(S2, v, float(S * (S + 1))))
        max_r_Lz = max(max_r_Lz, _expectation_error(Lz, v, ML))
        max_r_Sz = max(max_r_Sz, _expectation_error(Sz, v, MS))

    max_residual = max(max_r_L2, max_r_S2, max_r_Lz, max_r_Sz)
    if max_residual > EPS_NORM:
        raise NumError(
            "FXE-NUM-001",
            f"LSMS residual checks failed: max_residual={max_residual:.3e} > eps_norm={EPS_NORM:.1e}",
            module="lsms",
            level="LMSM",
            actual={
                "residual_name": "r_lsms_ops",
                "residual_value": max_residual,
                "threshold": EPS_NORM,
                "breakdown": {
                    "L2": max_r_L2,
                    "S2": max_r_S2,
                    "Lz": max_r_Lz,
                    "Sz": max_r_Sz,
                },
            },
        )

    basis_id = make_basis_id(n_ele, n_orb)
    logger.info("LSMS complete: %d states, %d terms", V_fock.shape[1], len(term_alpha_counter))

    return {
        "V_fock": V_fock,
        "labels": all_labels,
        "basis_id": basis_id,
        "n_ele": n_ele,
        "n_orb": n_orb,
        "coef_F0": coef_Fk[0],
        "coef_F2": coef_Fk[2],
        "coef_F4": coef_Fk[4],
        "coef_F6": coef_Fk[6],
        "hint_rank_matrix_map": hint_rank_matrix_map,
        "physics": {"U": U, "Jh": Jh, "r42": r42, "r62": r62, "F2": F2, "F4": F4, "F6": F6},
        "state_order_id": "lsms_canonical_v1",
        "orbital_order_id": "f14_m-3..3_sigma(-1/2,+1/2)_interleaved_v1",
    }


def _process_LS_term(
    L: int,
    twoS: int,
    Lp: NDArray[np.complexfloating],
    Sp: NDArray[np.complexfloating],
    Lm: NDArray[np.complexfloating],
    Sm: NDArray[np.complexfloating],
    H_int_ref: NDArray[np.complexfloating],
    R_LS: NDArray[np.complexfloating],
    get_embedding: Callable[[float, float], NDArray[np.complexfloating] | None],
    term_alpha_counter: dict[tuple[int, int], int],
) -> tuple[list[NDArray[np.complexfloating]], list[dict[str, Any]]]:
    """Process one (L, S) term: find highest-weight states and generate multiplet."""
    S = twoS / 2.0
    if R_LS.shape[1] == 0:
        return [], []

    # Build raising maps A_L and A_S (03-00 §3.4 step 2)
    R_Lp1_S = get_embedding(float(L + 1), S)
    R_L_Sp1 = get_embedding(float(L), S + 1)

    blocks = []
    if R_Lp1_S is not None and R_Lp1_S.shape[1] > 0:
        AL = R_Lp1_S.conj().T @ Lp @ R_LS
        blocks.append(AL)
    if R_L_Sp1 is not None and R_L_Sp1.shape[1] > 0:
        AS = R_L_Sp1.conj().T @ Sp @ R_LS
        blocks.append(AS)

    if len(blocks) == 0:
        # Both target sectors empty: all vectors in sector are highest-weight
        B_hw = np.eye(R_LS.shape[1], dtype=DTYPE_COMPLEX)
    else:
        A_stacked = np.vstack(blocks)
        # Null space of stacked matrix
        B_hw = null_space(A_stacked, rcond=EPS_SVD_REL)

    if B_hw.shape[1] == 0:
        return [], []

    # Alpha fixing: diagonalise projected H_int (03-00 §3.2)
    H_int_sub = R_LS.conj().T @ H_int_ref @ R_LS
    H_hw = B_hw.conj().T @ H_int_sub @ B_hw

    if B_hw.shape[1] > 1:
        evals_hw, evecs_hw = np.linalg.eigh(H_hw)
        B_hw = B_hw @ evecs_hw
    else:
        evals_hw = np.array([H_hw[0, 0].real])

    # Sort by ascending eigenvalue for alpha assignment.
    # Sort columns by ascending eigenvalue to assign alpha indices.
    # NOTE: within a near-degenerate group, eigh may return an arbitrary
    # orthonormal basis of the degenerate subspace; alpha ordering is therefore
    # not guaranteed to be reproducible across platforms or LAPACK versions.
    sort_idx = np.argsort(evals_hw.real)
    B_hw = B_hw[:, sort_idx]

    n_alpha = B_hw.shape[1]
    all_columns: list[NDArray[np.complexfloating]] = []
    all_labels: list[dict[str, Any]] = []

    for a_idx in range(n_alpha):
        alpha = term_alpha_counter.get((L, twoS), 0)
        term_alpha_counter[(L, twoS)] = alpha + 1

        hw_full = R_LS @ B_hw[:, a_idx]

        # Generate full multiplet by lowering (03-00 §3.3)
        cols, labs = _generate_multiplet(hw_full, alpha, L, twoS, Lm, Sm)
        all_columns.extend(cols)
        all_labels.extend(labs)

    return all_columns, all_labels


def _generate_multiplet(
    hw: NDArray[np.complexfloating],
    alpha: int, L: int, twoS: int,
    Lm: NDArray, Sm: NDArray,
) -> tuple[list[NDArray[np.complexfloating]], list[dict[str, Any]]]:
    """Generate all (ML, MS) states from highest-weight state |L, L; S, S>."""
    S = twoS / 2.0
    dim = hw.shape[0]
    # ml_states[k] = |ML = L - k, MS = S>; build by applying L- repeatedly.
    ml_states = np.empty((2 * L + 1, dim), dtype=hw.dtype)
    ml_states[0] = hw
    for k, ML in enumerate(range(L, -L, -1)):
        denom = np.sqrt(L * (L + 1) - ML * (ML - 1))
        v_new = Lm @ ml_states[k] / denom
        norm = np.linalg.norm(v_new)
        if abs(norm - 1.0) > EPS_NORM:
            raise NumError(
                "FXE-NUM-001",
                f"L- lowering norm deviation at ML={ML}: |norm-1|={abs(norm-1.0):.3e}",
                module="lsms",
                level="LMSM",
                actual={"residual_name": "r_lm_norm", "residual_value": abs(norm - 1.0), "threshold": EPS_NORM},
            )
        ml_states[k + 1] = v_new

    # For each ML (ascending), build ms_states by applying S- repeatedly,
    # then collect in ascending MS order.
    # ms_states[k] = |ML, MS = S - k>
    ms_states = np.empty((twoS + 1, dim), dtype=hw.dtype)
    columns: list[NDArray[np.complexfloating]] = []
    labels: list[dict[str, Any]] = []
    for ML in range(-L, L + 1):
        ms_states[0] = ml_states[L - ML]
        for k, twoMS in enumerate(range(twoS, -twoS, -2)):
            MS = twoMS / 2
            denom = np.sqrt(S * (S + 1) - MS * (MS - 1))
            v_new = Sm @ ms_states[k] / denom
            norm = np.linalg.norm(v_new)
            if abs(norm - 1.0) > EPS_NORM:
                raise NumError(
                    "FXE-NUM-001",
                    f"S- lowering norm deviation at ML={ML}, MS={MS}: |norm-1|={abs(norm-1.0):.3e}",
                    module="lsms",
                    level="LMSM",
                    actual={"residual_name": "r_sm_norm", "residual_value": abs(norm - 1.0), "threshold": EPS_NORM},
                )
            ms_states[k + 1] = v_new

        for twoMS in range(-twoS, twoS + 2, 2):
            columns.append(ms_states[(twoS - twoMS) // 2].copy())
            labels.append({
                "alpha": alpha,
                "L": L,
                "S": twoS / 2,
                "ML": ML,
                "MS": twoMS / 2,
                "twoS": twoS,
                "twoMS": twoMS,
            })

    return columns, labels


# ---------------------------------------------------------------------------
# LSMS energy spectrum (per-term Coulomb energy)
# ---------------------------------------------------------------------------

def build_lsms_spectrum(lsms_result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Per-term Coulomb energy spectrum from LSMS output.

    Each entry corresponds to one (alpha, L, S) term with its Coulomb energy
    computed from coef_Fk and physics r42/r62 under Jh=1 convention.
    Sorted by ascending Coulomb energy.
    """
    labels = lsms_result["labels"]
    coef_F2 = lsms_result["coef_F2"]
    coef_F4 = lsms_result["coef_F4"]
    coef_F6 = lsms_result["coef_F6"]

    physics = lsms_result.get("physics", {})
    F2, F4, F6 = float(physics["F2"]), float(physics["F4"]), float(physics["F6"])

    terms: dict[tuple[int, int, int], dict[str, Any]] = {}
    for idx, lab in enumerate(labels):
        key = (lab["alpha"], lab["L"], lab["twoS"])
        if key not in terms:
            terms[key] = {
                "alpha": lab["alpha"],
                "L": lab["L"],
                "S": lab["twoS"] / 2.0,
                "twoS": lab["twoS"],
                "E_coulomb": (
                    F2 * float(coef_F2[idx])
                    + F4 * float(coef_F4[idx])
                    + F6 * float(coef_F6[idx])
                ),
                "col_indices": [],
            }
        terms[key]["col_indices"].append(idx)

    spectrum = list(terms.values())
    for entry in spectrum:
        entry["n_states"] = len(entry["col_indices"])
    spectrum.sort(key=lambda x: x["E_coulomb"])
    return spectrum
