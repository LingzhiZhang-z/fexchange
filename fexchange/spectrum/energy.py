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

    physics = cfg.get("physics")
    if not isinstance(physics, dict):
        raise InputError(
            "FXE-INPUT-003",
            "physics section is required when L3 denominators must be recomputed",
            expected={"field_path": "physics"},
            actual={"present": False},
        )
    F0 = float(cfg["sopt"]["U"])
    F2 = float(physics["F2"])
    F4 = float(physics["F4"])
    F6 = float(physics["F6"])
    zeta = float(cfg["sopt"]["zeta"])

    def _energy(lsjm: dict[str, Any]) -> NDArray[np.floating]:
        return (
            F0 * np.asarray(lsjm["coef_F0"], dtype=float)
            + F2 * np.asarray(lsjm["coef_F2"], dtype=float)
            + F4 * np.asarray(lsjm["coef_F4"], dtype=float)
            + F6 * np.asarray(lsjm["coef_F6"], dtype=float)
            + zeta * np.asarray(lsjm["coef_zeta"], dtype=float)
        )

    lsjm_n = state[f"lsjm_{n_ele}"]
    soc0 = select_soc_lowest_subspace(lsjm_n, F2=F2, F4=F4, F6=F6, zeta=zeta)

    # The SOPT branch reference is set by the f^n Coulomb ground multiplet;
    # the fn-site SOC splitting stays in the intermediate-state energies.
    ref_idx = int(soc0["col_indices"][0])
    E_ref = float(
        F0 * np.asarray(lsjm_n["coef_F0"], dtype=float)[ref_idx]
        + F2 * np.asarray(lsjm_n["coef_F2"], dtype=float)[ref_idx]
        + F4 * np.asarray(lsjm_n["coef_F4"], dtype=float)[ref_idx]
        + F6 * np.asarray(lsjm_n["coef_F6"], dtype=float)[ref_idx]
    )

    E_np1 = _energy(state[f"lsjm_{n_ele + 1}"]) - E_ref
    E_nm1 = _energy(state[f"lsjm_{n_ele - 1}"]) - E_ref
    return E_np1, E_nm1
