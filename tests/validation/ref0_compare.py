"""
Reference-0 wavefunction comparison for f2.

Runs Reference-0's RS_f2() to get LSJM kets in Ref0 basis,
converts to fexchange Fock basis, and compares overlap and energy
with our build_lsjm output.
"""
from __future__ import annotations

import sys
import os

import numpy as np

from fexchange.core.fock_basis import enumerate_dets


# ===================================================================
# Orbital basis conversion: Reference-0 <-> fexchange
# ===================================================================
#
# Reference-0: index 2k = (m=k-3, sigma=+1/2), 2k+1 = (m=k-3, sigma=-1/2)
# fexchange:   index 2k = (m=k-3, sigma=-1/2), 2k+1 = (m=k-3, sigma=+1/2)
#
# Permutation: p_fex = p_ref0 XOR 1
# Fermionic sign: (-1)^(number of m-pairs with both spins occupied)

def _convert_det(ref0_det: int) -> tuple[int, int]:
    """Convert a Reference-0 determinant integer to fexchange convention.

    Returns (fex_det, sign) where sign is +1 or -1.
    """
    fex_det = 0
    n_double = 0
    for p in range(14):
        if (ref0_det >> p) & 1:
            fex_det |= (1 << (p ^ 1))
    for k in range(7):
        if ((ref0_det >> (2 * k)) & 1) and ((ref0_det >> (2 * k + 1)) & 1):
            n_double += 1
    return fex_det, (-1) ** n_double


def _ref0_ket_to_vector(ket, det_to_idx: dict[int, int], dim: int) -> np.ndarray:
    """Convert a Reference-0 Ket object to a numpy vector in fexchange Fock basis."""
    vec = np.zeros(dim, dtype=complex)
    outer = ket.coeff()
    for ketline in ket.ketlines():
        d_ref0 = ketline.base()
        d_fex, sign = _convert_det(d_ref0)
        if d_fex in det_to_idx:
            vec[det_to_idx[d_fex]] += outer * ketline.coeff() * sign
    return vec


# ===================================================================
# Main comparison
# ===================================================================

def compare_ref0_f2(lsjm_result: dict) -> list[dict]:
    """Compare our f2 LSJM with Reference-0 RS_f2().

    Returns a list of per-state comparison dicts:
        {"L", "S", "J", "M", "overlap_sq", "E_our", "E_ref", "dE", "matched"}
    """
    # Import Reference-0 code (adds its directory to sys.path temporarily)
    ref0_dir = os.path.join(os.path.dirname(__file__), "../../docs/reference0")
    ref0_dir = os.path.abspath(ref0_dir)
    sys.path.insert(0, ref0_dir)
    try:
        from space_lz import RS_f2, RS_f2_En
    finally:
        sys.path.remove(ref0_dir)

    # --- Run Reference-0 ---
    ref0_kets, ref0_info = RS_f2()
    # ref0_info[i] = (J, M, L, S) where J, M, L, S are integers for f2

    # --- Build fexchange Fock basis lookup ---
    fex_dets = enumerate_dets(2)
    det_to_idx = {int(d): i for i, d in enumerate(fex_dets)}
    dim = len(fex_dets)

    # --- Convert Ref0 kets to fexchange vectors ---
    ref0_vectors = [_ref0_ket_to_vector(ket, det_to_idx, dim) for ket in ref0_kets]

    # --- Compare with our LSJM ---
    V_fock = lsjm_result["V_fock"]
    labels = lsjm_result["labels"]
    physics = lsjm_result["physics"]
    F2 = float(physics["F2"])
    F4 = float(physics["F4"])
    F6 = float(physics["F6"])

    results = []
    for ref0_vec, (J, M, L, S) in zip(ref0_vectors, ref0_info):
        # Find matching state in our LSJM by exact quantum numbers
        matched_idx = None
        for j, lab in enumerate(labels):
            if (lab["L"] == L and lab["twoS"] == 2 * S
                    and lab["twoJ"] == 2 * J and lab["twoM"] == 2 * M):
                matched_idx = j
                break

        if matched_idx is None:
            results.append({
                "L": L, "S": S, "J": J, "M": M,
                "overlap_sq": 0.0, "E_our": 0.0, "E_ref": 0.0, "dE": 0.0,
                "matched": False,
            })
            continue

        our_vec = V_fock[:, matched_idx]
        overlap_sq = float(abs(np.dot(ref0_vec.conj(), our_vec)) ** 2)

        # Our Coulomb energy (no F0, no SOC)
        E_our = (F2 * float(lsjm_result["coef_F2"][matched_idx])
                 + F4 * float(lsjm_result["coef_F4"][matched_idx])
                 + F6 * float(lsjm_result["coef_F6"][matched_idx]))

        # Reference-0 analytic energy (Lambda=0 for Coulomb only, F0=0)
        E_ref = RS_f2_En([0.0, 0.0, F2, F4, F6], [L, S, J])

        results.append({
            "L": L, "S": S, "J": J, "M": M,
            "overlap_sq": overlap_sq,
            "E_our": E_our,
            "E_ref": E_ref,
            "dE": abs(E_our - E_ref),
            "matched": True,
        })

    return results
