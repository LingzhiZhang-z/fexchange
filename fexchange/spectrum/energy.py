"""
Energy reconstruction helpers for LSJM-based denominator construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.spectrum.lsjm import select_soc_lowest_subspace
from fexchange.utils.errors import InputError


def compute_intermediate_energies(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Construct E_u^{n+1}, E_u^{n-1} from LSJM term coefficients (04-02 §0.0.0).

    Caller must ensure LSJM payloads for n-1/n/n+1 are present in ``state``.
    """

    branches = cfg.get("_branches")
    if isinstance(branches, dict):
        branch_n = branches.get("n")
        branch_nm1 = branches.get("nm1")
        branch_np1 = branches.get("np1")
        if not isinstance(branch_n, dict) or not isinstance(branch_nm1, dict) or not isinstance(branch_np1, dict):
            raise InputError(
                "FXE-INPUT-003",
                "cfg._branches must contain n, nm1, and np1 branch payloads",
                actual={"present": sorted(branches)},
            )
    else:
        physics = cfg.get("physics")
        if not isinstance(physics, dict):
            raise InputError(
                "FXE-INPUT-003",
                "physics section is required when L3 denominators must be recomputed",
                expected={"field_path": "physics"},
                actual={"present": False},
            )
        sopt = cfg.get("sopt")
        if not isinstance(sopt, dict):
            raise InputError(
                "FXE-INPUT-003",
                "sopt section is required when L3 denominators must be recomputed",
                expected={"field_path": "sopt"},
                actual={"present": False},
            )
        branch_n = {"physics": physics, "sopt": {**sopt, "offset": float(sopt.get("offset", 0.0))}}
        branch_nm1 = {"physics": physics, "sopt": {**sopt, "offset": 0.0}}
        branch_np1 = {"physics": physics, "sopt": {**sopt, "offset": 0.0}}

    def _energy(lsjm: dict[str, Any], branch: dict[str, Any]) -> NDArray[np.floating]:
        physics = branch["physics"]
        sopt = branch["sopt"]
        return (
            float(sopt.get("offset", 0.0))
            + float(sopt["U"]) * np.asarray(lsjm["coef_F0"], dtype=float)
            + float(physics["F2"]) * np.asarray(lsjm["coef_F2"], dtype=float)
            + float(physics["F4"]) * np.asarray(lsjm["coef_F4"], dtype=float)
            + float(physics["F6"]) * np.asarray(lsjm["coef_F6"], dtype=float)
            + float(sopt["zeta"]) * np.asarray(lsjm["coef_zeta"], dtype=float)
        )

    ref_mode = str(cfg.get("sopt", {}).get("energy_reference", "lsjm_ground"))
    if ref_mode == "zero":
        E_ref = 0.0
    elif ref_mode == "lsjm_ground":
        lsjm_n = state[f"lsjm_{n_ele}"]
        physics_n = branch_n["physics"]
        sopt_n = branch_n["sopt"]
        soc0 = select_soc_lowest_subspace(
            lsjm_n,
            F2=float(physics_n["F2"]),
            F4=float(physics_n["F4"]),
            F6=float(physics_n["F6"]),
            zeta=float(sopt_n["zeta"]),
        )
        ref_idx = int(soc0["col_indices"][0])
        E_ref = (
            float(sopt_n.get("offset", 0.0))
            + float(sopt_n["U"]) * float(np.asarray(lsjm_n["coef_F0"], dtype=float)[ref_idx])
            + float(physics_n["F2"]) * float(np.asarray(lsjm_n["coef_F2"], dtype=float)[ref_idx])
            + float(physics_n["F4"]) * float(np.asarray(lsjm_n["coef_F4"], dtype=float)[ref_idx])
            + float(physics_n["F6"]) * float(np.asarray(lsjm_n["coef_F6"], dtype=float)[ref_idx])
        )
    else:
        raise InputError(
            "FXE-INPUT-003",
            "sopt.energy_reference must be 'lsjm_ground' or 'zero'",
            actual={"energy_reference": ref_mode},
        )

    E_np1 = _energy(state[f"lsjm_{n_ele + 1}"], branch_np1) - E_ref
    E_nm1 = _energy(state[f"lsjm_{n_ele - 1}"], branch_nm1) - E_ref
    return E_np1, E_nm1


def compute_ligand_energies(
    spectrum: dict[str, Any],
    *,
    Delta: float,
    U_p: float,
    lambda_p: float = 0.0,
) -> NDArray[np.floating]:
    """Assemble ligand p^N intermediate-state energies from spectrum coefficients.

    Mirrors the LSJM `_energy` helper above (Slater params * coef arrays), but
    for the ligand p-shell model:

        E_p^N[i] = Delta * coef_Delta[i] + U_p * coef_U_p[i] + lambda_p * coef_lambda_p[i]

    Parameters
    ----------
    spectrum : payload from `spectrum.ligand.build_ligand_spectrum`.
    Delta    : ligand-to-f charge-transfer cost (uniform per state, scaled by n_hole).
    U_p      : ligand p-shell onsite Hubbard U (uniform per state, scaled by hole-pair count).
    lambda_p : ligand atomic SOC strength (per-state, traceless contribution).

    Returns
    -------
    NDArray[float] of shape (spectrum["dim"],), in the SOC eigenbasis ordering
    (i.e., aligned with `spectrum["V_fock"]` columns).
    """
    if not all(np.isfinite([Delta, U_p, lambda_p])):
        raise InputError(
            "FXE-INPUT-003",
            "Delta / U_p / lambda_p must be finite",
            actual={"Delta": Delta, "U_p": U_p, "lambda_p": lambda_p},
        )
    return (
        float(Delta)    * np.asarray(spectrum["coef_Delta"],    dtype=float)
        + float(U_p)    * np.asarray(spectrum["coef_U_p"],      dtype=float)
        + float(lambda_p) * np.asarray(spectrum["coef_lambda_p"], dtype=float)
    )
