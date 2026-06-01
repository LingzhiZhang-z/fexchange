"""Full single-ion ED spectrum for H_int + H_soc with optional H_cef."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.core.fock import make_basis_id
from fexchange.core.space_ls import build_space_ls_matrix
from fexchange.core.states import fix_phase
from fexchange.hamiltonian.hcef import build_hcef_matrix_fock
from fexchange.hamiltonian.hint import build_hint_matrix
from fexchange.hamiltonian.hsoc import build_hsoc_matrix
from fexchange.utils.checks import check_hermitian, check_orthonormal
from fexchange.utils.constants import N_ORB
from fexchange.utils.errors import NumError, PhysError
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_EIG_CLUSTER, EPS_NORM, EPS_ZERO

__all__ = ["build_ion_ed"]


def build_ion_ed(
    n_ele: int,
    *,
    F0: float = 0.0,
    F2: float = 0.0,
    F4: float = 0.0,
    F6: float = 0.0,
    zeta: float = 0.0,
    offset: float = 0.0,
    n_orb: int = N_ORB,
    lsjm_reference: dict[str, Any] | None = None,
    hcef_1b: NDArray[np.complexfloating] | None = None,
) -> dict[str, Any]:
    """Diagonalize Hion = H_int + H_soc (+ H_cef) in one fixed f-electron sector.

    The returned ``V_fock_ed`` columns are ED eigenvectors in the canonical Fock
    basis. Without CEF, degenerate clusters are fixed by J^2 and then Jz. With
    CEF, J is not generally a good quantum number, so degenerate clusters are
    fixed by projected Jz and only expectation-value diagnostics are reported.
    """
    if not 0 <= n_ele <= n_orb:
        raise PhysError(
            "FXE-PHYS-001",
            f"IONED requires 0 <= n_ele <= {n_orb}, got {n_ele}",
            module="ion_ed",
            actual={"n_ele": n_ele, "n_orb": n_orb},
        )

    H = build_hint_matrix(n_ele, {0: F0, 2: F2, 4: F4, 6: F6}, n_orb=n_orb)
    if zeta != 0.0:
        H = H + build_hsoc_matrix(n_ele, zeta=zeta, n_orb=n_orb)
    hcef_norm = 0.0
    if hcef_1b is not None:
        hcef_arr = np.asarray(hcef_1b, dtype=DTYPE_COMPLEX)
        hcef_norm = float(np.linalg.norm(hcef_arr))
        if hcef_norm > EPS_ZERO:
            H = H + build_hcef_matrix_fock(hcef_arr, n_ele, n_orb=n_orb)
    check_hermitian(H, label="Hion", module="ion_ed")

    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals.real)
    evals = evals.real[order]
    evecs = evecs[:, order]

    ang = build_space_ls_matrix(n_ele, n_orb)
    if hcef_norm > EPS_ZERO:
        V_fock, energies, labels, j_values, m_values, group_ids = _canonicalize_general_spaces(
            evals,
            evecs,
            J2=ang["J2"],
            Jz=ang["Jz"],
            offset=float(offset),
        )
        state_order_id = "ioned_energy_Jz_projected_v1"
        j_order_id = "ioned_general_Jz_projected_v1"
    else:
        V_fock, energies, labels, j_values, m_values, group_ids = _canonicalize_degenerate_spaces(
            evals,
            evecs,
            J2=ang["J2"],
            Jz=ang["Jz"],
            offset=float(offset),
        )
        state_order_id = "ioned_energy_J_M_v1"
        j_order_id = "ioned_J_M_v1"
    check_orthonormal(V_fock, label="V_ioned", module="ion_ed")

    if lsjm_reference is not None:
        _attach_lsjm_composition(labels, V_fock, lsjm_reference)

    return {
        "V_fock": V_fock,
        "V_fock_ed": V_fock,
        "energies": energies,
        "labels": labels,
        "J": j_values,
        "M": m_values,
        "energy_group": group_ids,
        "basis_id": make_basis_id(n_ele, n_orb),
        "n_ele": n_ele,
        "n_orb": n_orb,
        "state_order_id": state_order_id,
        "j_order_id": j_order_id,
        "orbital_order_id": "f14_m-3..3_sigma(-1/2,+1/2)_interleaved_v1",
        "physics": {
            "F0": float(F0),
            "F2": float(F2),
            "F4": float(F4),
            "F6": float(F6),
            "zeta": float(zeta),
            "offset": float(offset),
            "hcef_1b_norm": hcef_norm,
            "includes_hcef": bool(hcef_norm > EPS_ZERO),
        },
    }


def _canonicalize_degenerate_spaces(
    evals: NDArray[np.floating],
    evecs: NDArray[np.complexfloating],
    *,
    J2: NDArray[np.complexfloating],
    Jz: NDArray[np.complexfloating],
    offset: float,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.floating],
    list[dict[str, Any]],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.integer],
]:
    columns: list[NDArray[np.complexfloating]] = []
    energy_out: list[float] = []
    labels: list[dict[str, Any]] = []
    j_out: list[float] = []
    m_out: list[float] = []
    group_out: list[int] = []

    for group_id, (lo, hi) in enumerate(_clusters(evals, EPS_EIG_CLUSTER)):
        V_energy = evecs[:, lo:hi]
        e_group = float(np.mean(evals[lo:hi])) + offset

        j2_blk = V_energy.conj().T @ J2 @ V_energy
        j2_vals, U_j = np.linalg.eigh(j2_blk)
        j_order = np.argsort(j2_vals.real)
        j2_vals = j2_vals.real[j_order]
        U_j = U_j[:, j_order]
        V_j_all = V_energy @ U_j

        for jlo, jhi in _clusters(j2_vals, EPS_EIG_CLUSTER):
            V_j = V_j_all[:, jlo:jhi]
            j2_value = float(np.mean(j2_vals[jlo:jhi]))
            J = _j_from_j2(j2_value)
            twoJ = int(round(2.0 * J))
            J = twoJ / 2.0

            jz_blk = V_j.conj().T @ Jz @ V_j
            m_vals, U_m = np.linalg.eigh(jz_blk)
            m_order = np.argsort(m_vals.real)
            m_vals = m_vals.real[m_order]
            U_m = U_m[:, m_order]
            V_m = V_j @ U_m

            for local_idx in range(V_m.shape[1]):
                vec = fix_phase(V_m[:, local_idx])
                M = round(float(m_vals[local_idx]) * 2.0) / 2.0
                columns.append(vec)
                energy_out.append(e_group)
                j_out.append(J)
                m_out.append(M)
                group_out.append(group_id)
                labels.append(
                    {
                        "energy_group": group_id,
                        "energy": e_group,
                        "J": J,
                        "M": M,
                        "twoJ": twoJ,
                        "twoM": int(round(2.0 * M)),
                        "j2": j2_value,
                    }
                )

    if not columns:
        raise NumError(
            "FXE-NUM-001",
            "IONED produced no eigenvectors",
            module="ion_ed",
            level="IONED",
            actual={"n_evals": int(evals.size)},
        )

    V = np.column_stack(columns).astype(DTYPE_COMPLEX, copy=False)
    return (
        V,
        np.asarray(energy_out, dtype=float),
        labels,
        np.asarray(j_out, dtype=float),
        np.asarray(m_out, dtype=float),
        np.asarray(group_out, dtype=np.int64),
    )


def _canonicalize_general_spaces(
    evals: NDArray[np.floating],
    evecs: NDArray[np.complexfloating],
    *,
    J2: NDArray[np.complexfloating],
    Jz: NDArray[np.complexfloating],
    offset: float,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.floating],
    list[dict[str, Any]],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.integer],
]:
    columns: list[NDArray[np.complexfloating]] = []
    energy_out: list[float] = []
    labels: list[dict[str, Any]] = []
    j_out: list[float] = []
    m_out: list[float] = []
    group_out: list[int] = []

    for group_id, (lo, hi) in enumerate(_clusters(evals, EPS_EIG_CLUSTER)):
        V_energy = evecs[:, lo:hi]
        e_group = float(np.mean(evals[lo:hi])) + offset

        jz_blk = V_energy.conj().T @ Jz @ V_energy
        m_vals, U_m = np.linalg.eigh(jz_blk)
        m_order = np.argsort(m_vals.real)
        m_vals = m_vals.real[m_order]
        U_m = U_m[:, m_order]
        V_m = V_energy @ U_m

        for local_idx in range(V_m.shape[1]):
            vec = fix_phase(V_m[:, local_idx])
            j2_exp = float(np.real(np.vdot(vec, J2 @ vec)))
            m_exp = float(np.real(np.vdot(vec, Jz @ vec)))
            columns.append(vec)
            energy_out.append(e_group)
            j_out.append(np.nan)
            m_out.append(m_exp)
            group_out.append(group_id)
            labels.append(
                {
                    "energy_group": group_id,
                    "energy": e_group,
                    "J": None,
                    "M": m_exp,
                    "twoJ": None,
                    "twoM": None,
                    "j2_expectation": j2_exp,
                }
            )

    if not columns:
        raise NumError(
            "FXE-NUM-001",
            "IONED produced no eigenvectors",
            module="ion_ed",
            level="IONED",
            actual={"n_evals": int(evals.size)},
        )

    V = np.column_stack(columns).astype(DTYPE_COMPLEX, copy=False)
    return (
        V,
        np.asarray(energy_out, dtype=float),
        labels,
        np.asarray(j_out, dtype=float),
        np.asarray(m_out, dtype=float),
        np.asarray(group_out, dtype=np.int64),
    )


def _clusters(values: NDArray[np.floating], tol: float) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, int(values.size) + 1):
        if idx == values.size or abs(float(values[idx]) - float(values[start])) > tol:
            clusters.append((start, idx))
            start = idx
    return clusters


def _j_from_j2(j2: float) -> float:
    value = max(0.0, 1.0 + 4.0 * float(j2))
    J = 0.5 * (-1.0 + np.sqrt(value))
    if abs(J - round(2.0 * J) / 2.0) > 10.0 * EPS_NORM:
        raise NumError(
            "FXE-NUM-001",
            "IONED J2 eigenvalue is not compatible with a half-integer J",
            module="ion_ed",
            level="IONED",
            actual={"J": J, "J2": j2},
        )
    return float(J)


def _attach_lsjm_composition(
    labels: list[dict[str, Any]],
    V_ed: NDArray[np.complexfloating],
    lsjm_reference: dict[str, Any],
) -> None:
    V_lsjm = np.asarray(lsjm_reference["V_fock"], dtype=np.complex128)
    if V_lsjm.shape[0] != V_ed.shape[0]:
        raise PhysError(
            "FXE-PHYS-001",
            "LSJM reference dimension does not match IONED Fock dimension",
            module="ion_ed",
            actual={"V_lsjm": list(V_lsjm.shape), "V_ed": list(V_ed.shape)},
        )
    C = V_lsjm.conj().T @ V_ed
    ref_labels = lsjm_reference.get("labels", [])
    for col, label in enumerate(labels):
        weights: dict[tuple[int, int, int, int], float] = defaultdict(float)
        for row, ref_label in enumerate(ref_labels):
            key = (
                int(ref_label.get("alpha", 0)),
                int(ref_label.get("L", 0)),
                int(ref_label.get("twoS", 0)),
                int(round(2.0 * float(ref_label.get("J", 0.0)))),
            )
            weights[key] += float(abs(C[row, col]) ** 2)
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            continue
        dominant_key, dominant_weight = ranked[0]
        label["dominant_lsj"] = {
            "alpha": dominant_key[0],
            "L": dominant_key[1],
            "twoS": dominant_key[2],
            "J": dominant_key[3] / 2.0,
            "weight": dominant_weight,
        }
        label["lsj_weights"] = [
            {
                "alpha": key[0],
                "L": key[1],
                "twoS": key[2],
                "J": key[3] / 2.0,
                "weight": value,
            }
            for key, value in ranked[:5]
            if value > 1.0e-10
        ]
