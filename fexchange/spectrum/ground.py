"""
Ground-doublet selection and projector construction.

Spec references: 02-05-KRAMERS_DOUBLET_G_TENSOR, 02-06-NON_KRAMERS_DOUBLET.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.spectrum.doublet import gauge_fix_kramers_pair, project_to_eg_doublet
from fexchange.core.space_j import (
    build_space_j_operator,
    build_time_reversal_operator,
    normalize_J,
    pauli_decompose,
    project_J_to_subspace,
    project_operators_to_subspace,
)
from fexchange.core.stevens import build_multipole_operators
from fexchange.spectrum.classify import analyze_cef_symmetry, build_symmetry_metadata, irrep_dimension
from fexchange.hamiltonian.hcef import build_hcef_matrix_J
from fexchange.spectrum.lsjm import select_soc_lowest_subspace
from fexchange.utils.checks import check_orthonormal
from fexchange.utils.errors import BindError, InputError, PhysError
from fexchange.utils.numerics import (
    DTYPE_COMPLEX,
    EPS_EIG_CLUSTER,
    EPS_MAG_AB,
    EPS_NK_SPLIT,
    EPS_NORM,
    EPS_ZERO,
)

logger = logging.getLogger("fexchange")


def _extract_symmetry_outputs(
    J: float,
    evecs: NDArray[np.complexfloating],
    *,
    point_group: str,
    symmetry_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    symmetry_out = build_symmetry_metadata(
        J,
        evecs,
        point_group=point_group,
        symmetry_meta=symmetry_meta,
    )
    return {
        "irrep": str(symmetry_out.get("irrep_primary", "unknown")),
        "allowed_multipoles": list(symmetry_out.get("allowed_multipoles", [])),
        "excited_irreps": [
            {"index": int(item["energy_index"]), "irrep": str(item["irrep_primary"])}
            for item in symmetry_out.get("excited_irreps", [])
        ],
        "symmetry_meta": symmetry_out,
    }


# ── Kramers ──
def select_kramers_doublet(
    J: float,
    evals: NDArray[np.floating],
    evecs: NDArray[np.complexfloating],
    *,
    c_g: float = 1.0,
    point_group: str = "Oh",
    symmetry_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select lowest Kramers doublet and apply canonical gauge-fixing."""
    J = normalize_J(J, module="kramers")
    assert evecs.shape[0] == int(round(2 * J + 1)), "J/evecs dimension mismatch"
    Jz, _, _, Jx, Jy = build_space_j_operator(J, module="kramers")
    U_T_n = build_time_reversal_operator(J, module="kramers")

    if abs(evals[0] - evals[1]) > EPS_EIG_CLUSTER:
        raise PhysError(
            "FXE-PHYS-001",
            f"Lowest pair not degenerate: E0={evals[0]:.6e}, E1={evals[1]:.6e}",
            actual={"E0": float(evals[0]), "E1": float(evals[1])},
        )

    k1 = evecs[:, 0].copy()
    k2 = evecs[:, 1].copy()
    K = np.column_stack([k1, k2])
    K = gauge_fix_kramers_pair(K, tol=EPS_ZERO)

    M_Jx, M_Jy, M_Jz = project_J_to_subspace(K, Jx, Jy, Jz, module="kramers")

    Lambda = np.zeros((3, 3), dtype=DTYPE_COMPLEX)
    dec_list = [pauli_decompose(M_Jx), pauli_decompose(M_Jy), pauli_decompose(M_Jz)]
    for alpha, dec in enumerate(dec_list):
        Lambda[alpha, 0] = 2.0 * dec["ox"]
        Lambda[alpha, 1] = 2.0 * dec["oy"]
        Lambda[alpha, 2] = 2.0 * dec["oz"]

    g_tensor = c_g * Lambda

    k1_final = K[:, 0]
    k2_final = K[:, 1]
    tr_residual = np.linalg.norm(U_T_n @ k1_final.conj() - k2_final)
    if tr_residual > EPS_NORM:
        raise PhysError(
            "FXE-PHYS-001",
            f"Kramers TR-pair residual too large: {tr_residual:.3e} > eps_norm={EPS_NORM:.1e}",
            module="kramers",
            actual={
                "residual_name": "r_tr_pair",
                "residual_value": float(tr_residual),
                "threshold": float(EPS_NORM),
            },
        )

    symmetry_out = _extract_symmetry_outputs(
        J,
        evecs,
        point_group=point_group,
        symmetry_meta=symmetry_meta,
    )

    return {
        "kramer_vectors": K,
        "M_Jx": M_Jx,
        "M_Jy": M_Jy,
        "M_Jz": M_Jz,
        "Lambda": Lambda,
        "g_tensor": g_tensor,
        "gauge_meta": {
            "c_g": c_g,
            "tr_residual": float(tr_residual),
        },
        **symmetry_out,
    }


# ── Non-Kramers ──
def select_non_kramers_doublet(
    J: float,
    evals: NDArray[np.floating],
    evecs: NDArray[np.complexfloating],
    *,
    eps_nk_split: float = EPS_NK_SPLIT,
    eps_mag_ab: float = EPS_MAG_AB,
    point_group: str = "Oh",
    symmetry_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select lowest non-Kramers quasi-doublet and project it canonically."""
    J = normalize_J(J, module="non_kramers")
    assert evecs.shape[0] == int(round(2 * J + 1)), "J/evecs dimension mismatch"
    Jz, _, _, Jx, Jy = build_space_j_operator(J, module="non_kramers")

    psi1 = evecs[:, 0].copy()
    psi2 = evecs[:, 1].copy()
    Delta_nk = float(evals[1] - evals[0])
    is_quasi_doublet = Delta_nk <= eps_nk_split

    Psi = np.column_stack([psi1, psi2])
    Psi = project_to_eg_doublet(J, Psi, point_group)
    gauge_name = f"non_kramers_{point_group.lower()}"

    M_Jx, M_Jy, M_Jz = project_J_to_subspace(Psi, Jx, Jy, Jz, module="non_kramers")

    norm_z = np.linalg.norm(M_Jz, "fro")
    norm_x = np.linalg.norm(M_Jx, "fro")
    norm_y = np.linalg.norm(M_Jy, "fro")

    mag_ratio_x = norm_x / max(norm_z, EPS_ZERO)
    mag_ratio_y = norm_y / max(norm_z, EPS_ZERO)
    if mag_ratio_x > eps_mag_ab or mag_ratio_y > eps_mag_ab:
        raise PhysError(
            "FXE-PHYS-001",
            "Non-Kramers magnetic channel leakage exceeds eps_mag_ab",
            module="non_kramers",
            actual={
                "mag_ratio_x": float(mag_ratio_x),
                "mag_ratio_y": float(mag_ratio_y),
                "eps_mag_ab": float(eps_mag_ab),
            },
        )

    quad_ops = build_multipole_operators(J, "electric_quadrupole")
    M_quad = project_operators_to_subspace(Psi, quad_ops, module="non_kramers")
    top_quad = sorted(
        M_quad.items(),
        key=lambda item: (-float(np.linalg.norm(item[1], ord="fro")), item[0]),
    )[:2]
    Q1_label, M_Q1 = top_quad[0]
    Q2_label, M_Q2 = top_quad[1]
    map_electric = {name: pauli_decompose(mat) for name, mat in top_quad}

    symmetry_out = _extract_symmetry_outputs(
        J,
        evecs,
        point_group=point_group,
        symmetry_meta=symmetry_meta,
    )

    return {
        "doublet_vectors": Psi,
        "Delta_nk": Delta_nk,
        "is_quasi_doublet": is_quasi_doublet,
        "M_Jx": M_Jx,
        "M_Jy": M_Jy,
        "M_Jz": M_Jz,
        "M_Q1": M_Q1,
        "M_Q2": M_Q2,
        "Q_labels": [Q1_label, Q2_label],
        "map_dipole": {
            "Jx": pauli_decompose(M_Jx),
            "Jy": pauli_decompose(M_Jy),
            "Jz": pauli_decompose(M_Jz),
        },
        "map_electric": map_electric,
        "gauge_meta": {
            "eps_nk_split": eps_nk_split,
            "mag_ratio_x": float(mag_ratio_x),
            "mag_ratio_y": float(mag_ratio_y),
            "gauge_used": gauge_name,
        },
        **symmetry_out,
    }


# ── Orchestrator ──
def _build_cef_inputs(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
) -> dict[str, Any]:
    """Build CEF diagonalization inputs. Mutates *state* in place (caches soc0)."""
    if "soc0" not in state:
        fsite = cfg.get("fsite")
        if not isinstance(fsite, dict):
            raise InputError(
                "FXE-INPUT-003",
                "fsite section is required when CEF projector must build SOC-lowest subspace",
                expected={"field_path": "fsite"},
                actual={"present": False},
            )
        state["soc0"] = select_soc_lowest_subspace(state[f"lsjm_{n_ele}"])

    soc0 = state["soc0"]
    J0 = float(soc0["J0"])
    n_j = int(soc0["n_j"])
    lsjm_n = state[f"lsjm_{n_ele}"]
    model = cfg.get("model", {})
    symmetry = model.get("symmetry", "Oh")
    mode_q3 = model.get("c3v_mode_q3", "cos")

    B_params = {
        key: float(model[key])
        for key in ("B4", "B6", "B20", "B40", "B60", "B66", "B43", "B63")
        if key in model
    }
    H_cef = build_hcef_matrix_J(J0, B_params, symmetry=symmetry, mode_q3=mode_q3)
    if H_cef.shape != (n_j, n_j):
        raise BindError(
            "FXE-BIND-003",
            "CEF matrix dimension mismatch with SOC-lowest manifold",
            expected={"shape": [n_j, n_j]},
            actual={"shape": list(H_cef.shape)},
        )

    evals, evecs = np.linalg.eigh(H_cef)
    symmetry_meta = analyze_cef_symmetry(
        J0,
        point_group=symmetry,
        evecs=evecs,
        n_f=n_ele,
        symmetry=symmetry,
        mode_q3=mode_q3,
    )
    return {
        "J0": J0,
        "lsjm_n": lsjm_n,
        "symmetry": symmetry,
        "mode_q3": mode_q3,
        "evals": evals,
        "evecs": evecs,
        "symmetry_meta": symmetry_meta,
    }


def build_W(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
) -> dict[str, Any]:
    """Build ground-state doublet projector W from CEF diagonalization."""
    sources = cfg["sources"]
    cef = _build_cef_inputs(cfg, state, n_ele=n_ele)
    J0 = cef["J0"]
    ground_irrep = cef["symmetry_meta"].get("irrep_primary", "unknown")
    if ground_irrep != "unknown" and irrep_dimension(ground_irrep, point_group=cef["symmetry"]) == 1:
        raise PhysError(
            "FXE-PHYS-001",
            f"Ground state is a singlet ({ground_irrep}); "
            "SOPT spin-1/2 mapping requires a doublet ground state",
            module="projection",
            actual={"ground_irrep": ground_irrep, "point_group": cef["symmetry"]},
        )

    if n_ele % 2 == 1:
        k_out = select_kramers_doublet(
            J0,
            cef["evals"],
            cef["evecs"],
            c_g=1.0,
            point_group=cef["symmetry"],
            symmetry_meta=cef["symmetry_meta"],
        )
        W = np.asarray(k_out["kramer_vectors"], dtype=np.complex128)
        gauge_meta = k_out.get("gauge_meta", {})
        kramer_basis_id = "cef_lowest_kramers_v1"
    else:
        nk_out = select_non_kramers_doublet(
            J0,
            cef["evals"],
            cef["evecs"],
            eps_nk_split=EPS_NK_SPLIT,
            point_group=cef["symmetry"],
            symmetry_meta=cef["symmetry_meta"],
        )
        W = np.asarray(nk_out["doublet_vectors"], dtype=np.complex128)
        gauge_meta = nk_out.get("gauge_meta", {})
        kramer_basis_id = "cef_lowest_nonkramers_v1"

    check_orthonormal(W, label="W_cef", module="projection")
    return {
        "W": W,
        "kramer_basis_id": kramer_basis_id,
        "kramer_name": sources["kramer_name"],
        "j_order_id": cef["lsjm_n"].get("j_order_id"),
        "gauge_meta": gauge_meta,
        "symmetry_meta": cef["symmetry_meta"],
    }
