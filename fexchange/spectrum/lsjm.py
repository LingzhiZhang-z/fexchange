"""
LSJM representation via J± highest-weight method.

Spec reference: 03-01-REPRESENTATION_LSJM.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from fexchange.utils.constants import ELL, N_ORB
from fexchange.core.fock import make_basis_id
from fexchange.core.space_ls import build_space_ls_matrix
from fexchange.core.states import fix_phase
from fexchange.hamiltonian.hsoc import build_hsoc_unit_matrix
from fexchange.utils.numerics import (
    EPS_ZERO, EPS_NORM, EPS_DIAG, EPS_SVD_REL,
)
from fexchange.utils.checks import check_orthonormal
from fexchange.utils.errors import PhysError, NumError

logger = logging.getLogger("fexchange")


def _robust_null_space(
    A: NDArray[np.complexfloating],
    abs_tol: float = 1e-10,
) -> NDArray[np.complexfloating]:
    """
    Find null space of A using SVD with absolute singular-value threshold.

    Unlike scipy.null_space which uses a relative threshold (fails for
    single-column matrices where all singular values are small), this uses
    an absolute cutoff on singular values.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    n_cols = A.shape[1]
    if len(S) == 0:
        return np.eye(n_cols, dtype=A.dtype)
    null_mask = S < abs_tol
    # Columns of Vh beyond the number of singular values are always null
    null_indices = list(np.where(null_mask)[0])
    null_indices.extend(range(len(S), n_cols))
    if not null_indices:
        return np.zeros((n_cols, 0), dtype=A.dtype)
    return Vh[null_indices, :].conj().T


def build_lsjm(
    lsms_result: dict[str, Any],
    n_orb: int = N_ORB,
) -> dict[str, Any]:
    """
    Build LSJM representation from LSMS output (03-01 §2).

    Parameters
    ----------
    lsms_result : output from build_lsms().

    Returns dict with V_fock, labels, coef_Fk, coef_zeta, etc.
    """
    n_ele = lsms_result["n_ele"]
    V_lsms = lsms_result["V_fock"]
    lsms_labels = lsms_result["labels"]

    logger.info("Building LSJM from LSMS for n_ele=%d", n_ele)

    ang = build_space_ls_matrix(n_ele, n_orb)
    Jp = ang["Jp"]
    Jm = ang["Jm"]

    # Group LSMS columns by (alpha, L, twoS) term
    term_groups: dict[tuple[int, int, int], list[int]] = {}
    for col_idx, lab in enumerate(lsms_labels):
        key = (lab["alpha"], lab["L"], lab["twoS"])
        term_groups.setdefault(key, []).append(col_idx)

    all_lsjm_columns: list[NDArray[np.complexfloating]] = []
    all_lsjm_labels: list[dict[str, Any]] = []
    # term_indices maps (alpha, L, twoS) -> LSJM output indices for block checks
    term_indices: dict[tuple[int, int, int], list[int]] = {}

    # Process each term block (03-01 §2.1)
    for (alpha, L, twoS), col_indices in sorted(term_groups.items()):
        V_block = V_lsms[:, col_indices]
        cols, labs = _process_term_block(
            alpha, L, twoS,
            V_block, col_indices, lsms_labels,
            Jp, Jm,
        )
        start = len(all_lsjm_columns)
        all_lsjm_columns.extend(cols)
        all_lsjm_labels.extend(labs)
        term_indices[(alpha, L, twoS)] = list(range(start, len(all_lsjm_columns)))

    if not all_lsjm_columns:
        raise PhysError(
            "FXE-PHYS-001",
            "No LSJM states generated",
            module="lsjm",
            stage="LSJM",
            op="build_lsjm",
        )

    V_fock = np.column_stack(all_lsjm_columns)
    check_orthonormal(V_fock, label="V_lsjm", module="lsjm")

    # Compute energy coefficients (03-01 §6.1).
    # These are matrix representations of Coulomb rank operators.
    hint_rank_matrix_map = lsms_result.get("hint_rank_matrix_map")
    coef_Fk: dict[int, NDArray] = {}
    if hint_rank_matrix_map is not None:
        for k in (0, 2, 4, 6):
            coef_Fk[k] = np.array([
                (V_fock[:, j].conj() @ hint_rank_matrix_map[k] @ V_fock[:, j]).real
                for j in range(V_fock.shape[1])
            ])

    # coef_zeta: <psi| O_soc |psi> where H_soc = zeta * O_soc
    O_soc = build_hsoc_unit_matrix(n_ele, n_orb)
    coef_zeta = np.array([
        (V_fock[:, j].conj() @ O_soc @ V_fock[:, j]).real
        for j in range(V_fock.shape[1])
    ])

    # Diagonal-subspace checks (03-01 §6.2) for fixed (alpha,L,S) blocks.
    for key, idxs in term_indices.items():
        V_blk = V_fock[:, idxs]
        H_soc_blk = V_blk.conj().T @ O_soc @ V_blk
        off_soc = H_soc_blk - np.diag(np.diag(H_soc_blk))
        max_soc = float(np.max(np.abs(off_soc))) if off_soc.size else 0.0
        if max_soc > EPS_DIAG:
            raise NumError(
                "FXE-NUM-001",
                f"LSJM SOC block leakage for term {key}: {max_soc:.3e} > eps_diag={EPS_DIAG:.1e}",
                module="lsjm",
                level="LSJM",
                stage="LSJM",
                op="diag_subspace_check",
                actual={
                    "residual_name": "r_diag_soc_lsjm",
                    "residual_value": max_soc,
                    "threshold": EPS_DIAG,
                    "term": {"alpha": key[0], "L": key[1], "twoS": key[2]},
                },
            )

    basis_id = make_basis_id(n_ele, n_orb)
    logger.info("LSJM complete: %d states", V_fock.shape[1])

    return {
        "V_fock": V_fock,
        "labels": all_lsjm_labels,
        "basis_id": basis_id,
        "n_ele": n_ele,
        "n_orb": n_orb,
        "coef_F0": coef_Fk.get(0, np.array([])),
        "coef_F2": coef_Fk.get(2, np.array([])),
        "coef_F4": coef_Fk.get(4, np.array([])),
        "coef_F6": coef_Fk.get(6, np.array([])),
        "coef_zeta": coef_zeta,
        "physics": lsms_result.get("physics", {}),
        "state_order_id": "lsjm_canonical_v1",
        "j_order_id": "Jasc_Masc_v1",
        "orbital_order_id": "f14_m-3..3_sigma(-1/2,+1/2)_interleaved_v1",
    }


def _process_term_block(
    alpha: int, L: int, twoS: int,
    V_block: NDArray, col_indices: list[int], lsms_labels: list,
    Jp: NDArray, Jm: NDArray,
) -> tuple[list[NDArray[np.complexfloating]], list[dict[str, Any]]]:
    """Process one (alpha, L, S) block to generate LSJM states (03-01 §2.1)."""
    twoJ_max = 2 * L + twoS
    twoJ_min = abs(2 * L - twoS)
    dim_fock = V_block.shape[0]

    # Map block columns by twoM = 2*ML + twoMS (integer key avoids float-dict issues)
    n_block = V_block.shape[1]
    m_groups: dict[int, list[int]] = {}
    for local_idx in range(n_block):
        lab = lsms_labels[col_indices[local_idx]]
        twoM = 2 * int(lab["ML"]) + int(lab["twoMS"])
        m_groups.setdefault(twoM, []).append(local_idx)

    # Already-generated columns at each twoM for orthogonal deflation
    generated_at_twoM: dict[int, list[NDArray[np.complexfloating]]] = {}
    term_columns: list[tuple[dict[str, Any], NDArray[np.complexfloating]]] = []

    # Process J from highest to lowest (03-01 §2.1 step 2)
    for twoJ in range(twoJ_max, twoJ_min - 2, -2):
        local_cols = m_groups.get(twoJ, [])
        if not local_cols:
            continue

        # Solve ker(J+) at M=J: find c such that Jp @ V_M @ c = 0
        V_M = V_block[:, local_cols]
        ns = _robust_null_space(Jp @ V_M)
        if ns.shape[1] == 0:
            continue

        # Deflation at fixed M=J:
        # ns gives numerical null-space candidates of J+ in this subspace, but in
        # finite precision they can still contain tiny components from previously
        # constructed higher-J multiplets at the same M.
        #
        # For each candidate |v>, remove the projection onto every stored
        # higher-J vector |u> in generated_at_twoM[twoJ]:
        #   |v> <- |v> - <u|v> |u|
        #
        # This is Gram-Schmidt orthogonal deflation in the complex inner-product
        # space (note <u|v> = u.conj().T @ v in code).
        hw_candidates = V_M @ ns
        for prev_vec in generated_at_twoM.get(twoJ, []):
            for col_j in range(hw_candidates.shape[1]):
                hw_candidates[:, col_j] -= (prev_vec.conj() @ hw_candidates[:, col_j]) * prev_vec

        # Re-orthonormalise and remove near-zero columns
        valid_cols = []
        for col_j in range(hw_candidates.shape[1]):
            norm = np.linalg.norm(hw_candidates[:, col_j])
            if norm > EPS_ZERO:
                hw_candidates[:, col_j] /= norm
                valid_cols.append(col_j)
        if not valid_cols:
            continue
        if len(valid_cols) > 1:
            raise NumError(
                "FXE-NUM-001",
                f"Deflation incomplete at twoJ={twoJ}, alpha={alpha}, L={L}, twoS={twoS}: "
                f"{len(valid_cols)} residual hw candidates after deflation",
                module="lsjm",
                level="LSJM",
                stage="LSJM",
                op="hw_deflation",
                actual={"residual_name": "r_deflation", "n_residual": len(valid_cols), "twoJ": twoJ},
            )

        hw_vec = fix_phase(hw_candidates[:, valid_cols[0]])

        # Generate multiplet |J, M> by J- recursion (03-01 §2.1 step 5).
        # jm_states[k] = |J, twoM = twoJ - 2*k>
        # denom^2 = J(J+1) - M(M-1) = (twoJ*(twoJ+2) - twoM*(twoM-2)) / 4
        jm_states = np.empty((twoJ + 1, dim_fock), dtype=hw_vec.dtype)
        jm_states[0] = hw_vec
        for k in range(twoJ):
            twoM = twoJ - 2 * k
            denom = np.sqrt(twoJ * (twoJ + 2) - twoM * (twoM - 2)) / 2
            v_new = Jm @ jm_states[k] / denom
            norm = np.linalg.norm(v_new)
            if abs(norm - 1.0) > EPS_NORM:
                raise NumError(
                    "FXE-NUM-001",
                    f"J- lowering norm deviation at twoJ={twoJ}, twoM={twoM}: |norm-1|={abs(norm-1.0):.3e}",
                    module="lsjm",
                    level="LSJM",
                    stage="LSJM",
                    op="jm_recursion",
                    actual={"residual_name": "r_jm_norm", "residual_value": abs(norm - 1.0), "threshold": EPS_NORM},
                )
            jm_states[k + 1] = v_new

        # Collect states and track for deflation
        for k in range(twoJ + 1):
            twoM = twoJ - 2 * k
            vec = jm_states[k].copy()
            generated_at_twoM.setdefault(twoM, []).append(vec)
            term_columns.append((
                {"alpha": alpha, "L": L, "S": twoS / 2.0, "J": twoJ / 2.0, "M": twoM / 2.0,
                 "twoS": twoS, "twoJ": twoJ, "twoM": twoM},
                vec,
            ))

    # Canonical ordering (03-01 §5): J ascending, then M ascending.
    term_columns.sort(key=lambda item: (item[0]["twoJ"], item[0]["twoM"]))
    return [vec for _, vec in term_columns], [lab for lab, _ in term_columns]


# ---------------------------------------------------------------------------
# SOC-lowest subspace selection (04-00 §0.2.1)
# ---------------------------------------------------------------------------

def select_soc_lowest_subspace(
    lsjm_result: dict[str, Any],
    *,
    F2: float | None = None,
    F4: float | None = None,
    F6: float | None = None,
    zeta: float = 1.0,
) -> dict[str, Any]:
    """
    Identify the SOC-lowest J0 multiplet from LSJM output (04-00 §0.2.1).

    J0 is determined numerically by minimising zeta * coef_zeta over all J in
    the ground LS term. Hund's third rule is checked as validation.

    Parameters
    ----------
    F2, F4, F6 : optional Coulomb coefficients. If omitted, they are read from
                 lsjm_result["physics"].
    zeta : SOC coupling constant sign convention. Default +1.0 (positive).
           Only the sign matters for ordering; pass a negative value if the
           convention yields zeta < 0.

    Returns dict with U_n_soc0, J0, alpha0, L0, S0, n_j, column_indices.
    """
    labels = lsjm_result["labels"]
    n_ele = lsjm_result["n_ele"]
    ell = ELL

    coef_F2 = lsjm_result.get("coef_F2")
    coef_F4 = lsjm_result.get("coef_F4")
    coef_F6 = lsjm_result.get("coef_F6")
    coef_zeta = lsjm_result.get("coef_zeta")
    if coef_F2 is None or coef_F4 is None or coef_F6 is None:
        raise PhysError(
            "FXE-PHYS-001",
            "Missing coef_F2/coef_F4/coef_F6 in LSJM result",
            module="lsjm",
            stage="LSJM",
            op="select_soc_lowest",
            actual={"keys": sorted(lsjm_result.keys())},
        )
    if F2 is None and F4 is None and F6 is None:
        physics = lsjm_result.get("physics", {})
        if not isinstance(physics, dict) or any(k not in physics for k in ("F2", "F4", "F6")):
            raise PhysError(
                "FXE-PHYS-001",
                "Missing F2/F4/F6 in arguments and lsjm_result['physics']",
                module="lsjm",
                stage="LSJM",
                op="select_soc_lowest",
                actual={"physics": physics},
            )
        F2 = float(physics["F2"])
        F4 = float(physics["F4"])
        F6 = float(physics["F6"])
    elif F2 is None or F4 is None or F6 is None:
        raise PhysError(
            "FXE-PHYS-001",
            "F2/F4/F6 must be provided together",
            module="lsjm",
            stage="LSJM",
            op="select_soc_lowest",
            actual={"F2": F2, "F4": F4, "F6": F6},
        )
    else:
        F2 = float(F2)
        F4 = float(F4)
        F6 = float(F6)

    # Identify ground LS term by Coulomb energy: E = F2*coef_F2 + F4*coef_F4 + F6*coef_F6
    term_energies: dict[tuple[int, int, int], float] = {}
    for idx, lab in enumerate(labels):
        key = (lab["alpha"], lab["L"], lab["twoS"])
        if key not in term_energies:
            term_energies[key] = (
                F2 * float(coef_F2[idx])
                + F4 * float(coef_F4[idx])
                + F6 * float(coef_F6[idx])
            )

    alpha0, L0, twoS0 = min(term_energies, key=term_energies.get)  # type: ignore[arg-type]
    S0 = twoS0 / 2.0

    # Validate against Hund's third rule (04-00 §0.2.1)
    J0_hund = abs(L0 - S0) if n_ele <= 2 * ell else L0 + S0

    # Find J0 from coef_zeta if available; otherwise fall back to deterministic label rule.
    if coef_zeta is not None:
        # coef_zeta is constant across all M in a J multiplet; take first occurrence per twoJ.
        j_soc_energies: dict[int, float] = {}
        for idx, lab in enumerate(labels):
            if lab["alpha"] == alpha0 and lab["L"] == L0 and lab["twoS"] == twoS0:
                twoJ = int(lab["twoJ"]) if "twoJ" in lab else int(round(2.0 * float(lab["J"])))
                if twoJ not in j_soc_energies:
                    j_soc_energies[twoJ] = float(zeta) * float(coef_zeta[idx])
        twoJ0 = min(j_soc_energies, key=j_soc_energies.get)  # type: ignore[arg-type]
    else:
        candidates = sorted(
            {
                int(lab["twoJ"]) if "twoJ" in lab else int(round(2.0 * float(lab["J"])))
                for lab in labels
                if lab["alpha"] == alpha0 and lab["L"] == L0 and lab["twoS"] == twoS0
            }
        )
        if not candidates:
            raise PhysError(
                "FXE-PHYS-001",
                "No J candidates found for selected ground LS term",
                module="lsjm",
                stage="LSJM",
                op="select_soc_lowest",
                actual={"alpha0": alpha0, "L0": L0, "twoS0": twoS0},
            )
        if len(candidates) == 1:
            twoJ0 = candidates[0]
        else:
            twoJ_hund = int(round(2.0 * J0_hund))
            if twoJ_hund in candidates:
                twoJ0 = twoJ_hund
            else:
                raise PhysError(
                    "FXE-PHYS-001",
                    "Missing coef_zeta and cannot disambiguate J0 from labels",
                    module="lsjm",
                    stage="LSJM",
                    op="select_soc_lowest",
                    actual={"J_candidates_twoJ": candidates, "twoJ_hund": twoJ_hund},
                )

    J0 = twoJ0 / 2.0
    if abs(J0 - J0_hund) > 1e-10:
        raise PhysError(
            "FXE-PHYS-001",
            f"SOC-lowest J0={J0:.1f} disagrees with Hund's third rule J0={J0_hund:.1f} "
            f"(alpha={alpha0}, L={L0}, twoS={twoS0}); check zeta sign / coef_zeta.",
            module="lsjm",
            stage="LSJM",
            op="select_soc_lowest",
            actual={
                "J0": float(J0),
                "J0_hund": float(J0_hund),
                "alpha": int(alpha0),
                "L": int(L0),
                "twoS": int(twoS0),
            },
        )

    # Select columns matching (alpha0, L0, twoS0, twoJ0)
    col_indices = [
        i for i, lab in enumerate(labels)
        if lab["alpha"] == alpha0
        and lab["L"] == L0
        and lab["twoS"] == twoS0
        and ((int(lab["twoJ"]) if "twoJ" in lab else int(round(2.0 * float(lab["J"])))) == twoJ0)
    ]

    n_j = twoJ0 + 1
    if len(col_indices) != n_j:
        raise PhysError(
            "FXE-PHYS-001",
            f"Expected {n_j} states for J0={J0}, found {len(col_indices)}",
            module="lsjm",
            stage="LSJM",
            op="select_soc_lowest",
            actual={"n_j": n_j, "found": len(col_indices)},
        )

    U_n_soc0 = lsjm_result["V_fock"][:, col_indices]

    return {
        "U_n_soc0": U_n_soc0,
        "J0": J0,
        "alpha0": alpha0,
        "L0": L0,
        "S0": S0,
        "n_j": n_j,
        "col_indices": col_indices,
    }


# ---------------------------------------------------------------------------
# LSJM energy spectrum (per-multiplet Coulomb + SOC)
# ---------------------------------------------------------------------------

def build_lsjm_spectrum(
    lsjm_result: dict[str, Any],
    *,
    zeta: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Per-multiplet energy spectrum from LSJM output.

    Each entry corresponds to one (alpha, L, S, J) multiplet with Coulomb
    and SOC energies. SOC does not mix different (alpha, L, S) terms.
    Sorted by ascending total energy.

    Parameters
    ----------
    zeta : SOC coupling constant (default +1.0). Only sign matters for ordering.
    """
    labels = lsjm_result["labels"]
    coef_F2 = lsjm_result["coef_F2"]
    coef_F4 = lsjm_result["coef_F4"]
    coef_F6 = lsjm_result["coef_F6"]
    coef_zeta = lsjm_result["coef_zeta"]

    physics = lsjm_result.get("physics", {})
    F2, F4, F6 = float(physics["F2"]), float(physics["F4"]), float(physics["F6"])

    multiplets: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for idx, lab in enumerate(labels):
        key = (lab["alpha"], lab["L"], lab["twoS"], lab["twoJ"])
        if key not in multiplets:
            E_coulomb = (
                F2 * float(coef_F2[idx])
                + F4 * float(coef_F4[idx])
                + F6 * float(coef_F6[idx])
            )
            E_soc = float(zeta) * float(coef_zeta[idx])
            multiplets[key] = {
                "alpha": lab["alpha"],
                "L": lab["L"],
                "S": lab["twoS"] / 2.0,
                "J": lab["twoJ"] / 2.0,
                "twoS": lab["twoS"],
                "twoJ": lab["twoJ"],
                "E_coulomb": E_coulomb,
                "E_soc": E_soc,
                "E_total": E_coulomb + E_soc,
                "col_indices": [],
            }
        multiplets[key]["col_indices"].append(idx)

    spectrum = list(multiplets.values())
    for entry in spectrum:
        entry["n_states"] = len(entry["col_indices"])
    spectrum.sort(key=lambda x: x["E_total"])
    return spectrum
