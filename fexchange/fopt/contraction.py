"""fopt L2: V_+ blocks for cluster (4 (f_site, ligand) pairs x 3 sectors = 12 blocks)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.numerics import DTYPE_COMPLEX
from fexchange.utils.errors import BindError

logger = logging.getLogger("fexchange")


# (sector_label, f_creation_key in compute_L1_f output, p_creation_key in compute_L1_p output)
_SECTORS = (
    ("fn_p6",   "F_n_np1", "P_5_6"),   # V_+: (f^n,    p^6) -> (f^{n+1}, p^5)
    ("fn_p5",   "F_n_np1", "P_4_5"),   # V_+: (f^n,    p^5) -> (f^{n+1}, p^4)
    ("fnm1_p6", "F_nm1_n", "P_5_6"),   # V_+: (f^{n-1}, p^6) -> (f^n,    p^5)
)


def _compute_pair_V_plus(
    F_creation: NDArray[np.complexfloating],
    P_creation: NDArray[np.complexfloating],
    t_pf: NDArray[np.complexfloating],
) -> dict[str, Any]:
    """V_+ 4-leg tensor for one (f_site, ligand) pair x one sector."""
    n_orb_f, dim_high_f, dim_low_f = F_creation.shape
    n_orb_p, dim_high_p, dim_low_p = P_creation.shape

    if t_pf.shape != (n_orb_f, n_orb_p):
        raise BindError(
            "FXE-BIND-003",
            f"t_pf shape {tuple(t_pf.shape)} != ({n_orb_f}, {n_orb_p})",
            actual={"t_shape": list(t_pf.shape), "expected": [n_orb_f, n_orb_p]},
        )

    V_plus = np.einsum(
        "ab,aik,blj->ijkl",
        t_pf.astype(DTYPE_COMPLEX),
        F_creation,
        P_creation.conj(),
        optimize=True,
    )

    return {
        "V_plus": V_plus,
        "dim_high_f": dim_high_f,
        "dim_low_f": dim_low_f,
        "dim_high_p": dim_high_p,
        "dim_low_p": dim_low_p,
    }


def build_L2(
    l1: dict[str, Any],
    t: dict[tuple[int, int], NDArray[np.complexfloating]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    """V_+ blocks: 4 (f_site, ligand) pairs x 3 sectors = 12 blocks."""
    return {
        (f_site, ligand, sector): _compute_pair_V_plus(
            l1["f"][f_site][f_key], l1["p"][ligand][p_key], t_pf,
        )
        for (f_site, ligand), t_pf in t.items()
        for sector, f_key, p_key in _SECTORS
    }


# ---------------------------------------------------------------------------
# L3 skeleton: H_eff from 24 path amplitudes, with W projection applied here
# (no separate L4 since W is consumed inside L3).
# ---------------------------------------------------------------------------

def project_W(
    l2: dict[tuple[int, int, str], dict[str, Any]],
    W: NDArray[np.complexfloating],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    """Project the f^n LSJM end of each V_+ block to doublet (n_k).

    For sector "fn_p6", "fn_p5": project low_f end (input is f^n, ket side, use W).
    For sector "fnm1_p6"     : project high_f end (output is f^n, bra side, use W.conj()).
    W : (2J+1, n_k) projector, shared by all f-sites. Column a = doublet state a in LSJM basis.
    """
    n_k = W.shape[1]
    W_conj = W.conj()
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for (f_site, ligand, sector), block in l2.items():
        V_plus = block["V_plus"]
        new_block = dict(block)
        if sector == "fnm1_p6":
            new_block["V_plus"] = np.einsum("ijkl,ia->ajkl", V_plus, W_conj)
            new_block["dim_high_f"] = n_k
        else:
            new_block["V_plus"] = np.einsum("ijkl,ka->ijal", V_plus, W)
            new_block["dim_low_f"] = n_k
        result[(f_site, ligand, sector)] = new_block
    return result


# ---------------------------------------------------------------------------
# Path amplitude helpers.
#
# Native einsum output convention:
#     A[a, b, c, d]
#       a = low_f index of V_4    (final doublet of the f-site V_4 acts on)
#       b = low_f index of V_3    (final doublet of the f-site V_3 acts on)
#       c = low_f index of V_1    (initial doublet of the f-site V_1 acts on)
#       d = low_f index of V_2    (initial doublet of the f-site V_2 acts on)
#
# `_to_standard(A_native, r1, r2, r3, r4)` permutes axes so the result is
# indexed [f1_fin, f2_fin, f1_init, f2_init], matching sopt's [c, d, a, b]
# bra/ket convention used by spin12_map.
# ---------------------------------------------------------------------------


def _to_standard(
    H_native: NDArray[np.complexfloating],
    r1: int, r2: int, r3: int, r4: int,
) -> NDArray[np.complexfloating]:
    """Permute native (V_4_fin, V_3_fin, V_1_init, V_2_init) axes to the
    standard [f1_fin, f2_fin, f1_init, f2_init] ordering.
    """
    site_to_axis: dict[tuple[int, str], int] = {
        (r4, "fin"):  0,
        (r3, "fin"):  1,
        (r1, "init"): 2,
        (r2, "init"): 3,
    }
    perm = (
        site_to_axis[(1, "fin")],
        site_to_axis[(2, "fin")],
        site_to_axis[(1, "init")],
        site_to_axis[(2, "init")],
    )
    return H_native.transpose(perm)


def _path_amplitude_process1(
    V_1: NDArray[np.complexfloating],
    V_2: NDArray[np.complexfloating],
    V_3: NDArray[np.complexfloating],
    V_4: NDArray[np.complexfloating],
    E_np1: NDArray[np.floating],
    E_nm1: NDArray[np.floating],
    E_p5:  NDArray[np.floating],
    *,
    E_0: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Process-1 (alternating single-ligand): V+(r_X) V-(r_Y) V+(r_Y) V-(r_X).

    Timeline (f-site [═══] solid, ligand hole [┄┄┄] dashed):
                              V_1        V_2        V_3        V_4
        r_X (particle):       [═══════════════════════════════]
        r_Y (hole)    :                  [══════════]
        lig (hole)    :       [┄┄┄┄┄┄┄┄┄]            [┄┄┄┄┄┄┄┄┄]

    Denominators: s_1, s_3 subtract 1*E_0 (1 f-site excited); s_2 subtracts 2*E_0.
    """
    M1 = V_1[:, :, :, 0]   # (A=r_X^{n+1}, G=p^5, c=r_X_init)
    M2 = V_2[:, :, :, 0]   # (d=r_Y_init,  G=p^5, B=r_Y^{n-1})
    M3 = V_3[:, :, :, 0]   # (b=r_Y_fin,   H=p^5, B=r_Y^{n-1})
    M4 = V_4[:, :, :, 0]   # (A=r_X^{n+1}, H=p^5, a=r_X_fin)

    G_s1 = -1.0 / (E_np1[:, None] + E_p5 [None, :] -       E_0)  # 1 f-site
    G_s2 = -1.0 / (E_np1[:, None] + E_nm1[None, :] - 2.0 * E_0)  # 2 f-sites
    G_s3 = -1.0 / (E_np1[:, None] + E_p5 [None, :] -       E_0)  # 1 f-site

    # ─── Derivation: full cluster sum -> local-index einsum ─────────────────
    # (ref: standards/fopt/L3_four_type_full_cluster_expansion.md §2)
    #
    # Start: full 4-site cluster sum with 12 indices (S_1, S_2, S_3 each carry
    # 4 block indices on (f_1, f_2, p_1, p_2)):
    #
    #     H[a,b,c,d] = sum_{S_1,S_2,S_3} <F|V_4|S_3> G(S_3) <S_3|V_3|S_2> G(S_2) <S_2|V_2|S_1> G(S_1) <S_1|V_1|I>
    #     S_1 = (u_1, u_2, ρ_1, ρ_2)   S_2 = (v_1, v_2, σ_1, σ_2)   S_3 = (w_1, w_2, τ_1, τ_2)
    #
    # Each vertex acts on only 2 blocks; spectator blocks emit Kronecker δ:
    #     V_1 = A_{11} (p_1 -> f_1):  δ_{u_2,d}     δ_{ρ_2,Ω_2}
    #     V_2 = B_{21} (f_2 -> p_1):  δ_{v_1,u_1}   δ_{σ_2,ρ_2}
    #     V_3 = A_{21} (p_1 -> f_2):  δ_{w_1,v_1}   δ_{τ_2,σ_2}
    #     V_4 = B_{11} (f_1 -> p_1):  δ_{b,w_2}     δ_{Ω_2,τ_2}
    #
    # Collapsing all 8 δ's forces:
    #     u_2 = d,         ρ_2 = σ_2 = τ_2 = Ω_2,
    #     v_1 = w_1 = u_1, σ_1 = Ω_1,
    #     w_2 = b
    #
    # 4 free dummies remain (the only true sums):
    #     α  = u_1 ∈ f_1^{n+1}      (V_1 → V_4 active site, "long")
    #     β  = v_2 ∈ f_2^{n-1}      (V_2 → V_3 hole-active, "short")
    #     γ  = ρ_1 ∈ p_1^5          (s_1 ligand photon)
    #     γ' = τ_1 ∈ p_1^5          (s_3 ligand photon)
    #
    # Three intermediate states reduce to:
    #     S_1 = (α, d, γ , Ω_2)     S_2 = (α, β, Ω_1, Ω_2)     S_3 = (α, b, γ', Ω_2)
    # so the denominators only depend on (α, γ), (α, β), (α, γ') -> G_s1, G_s2, G_s3 above.
    #
    # Renaming for einsum: A=α, B=β, G=γ, H=γ'. The 4-vertex sum becomes:
    #
    #   H[a,b,c,d] = sum_{A,B,G,H}  M4*[A,H,a] G_s3[A,H] M3[b,H,B] G_s2[A,B] M2*[d,G,B] G_s1[A,G] M1[A,G,c]
    #                               └─ V_4† ─┘ └─/E_s3─┘ └─ V_3 ─┘ └─/E_s2─┘ └─ V_2† ─┘ └─/E_s1─┘ └─ V_1 ─┘
    # Summed dummies: A = α (f_X^{n+1})  B = β (f_Y^{n-1})  G = γ_1 (lig p^5)  H = γ_3 (lig p^5)
    # Free outputs  : a = r_X_fin        b = r_Y_fin        c = r_X_init       d = r_Y_init   (n_k doublet)
    return np.einsum(
        "AHa, AH, bHB, AB, dGB, AG, AGc -> abcd",
        M4.conj(), G_s3,
        M3,        G_s2,
        M2.conj(), G_s1,
        M1,
        optimize=True,
    )


def _path_amplitude_process2(
    V_1: NDArray[np.complexfloating],
    V_2: NDArray[np.complexfloating],
    V_3: NDArray[np.complexfloating],
    V_4: NDArray[np.complexfloating],
    E_np1: NDArray[np.floating],
    E_p5:  NDArray[np.floating],
    E_p4:  NDArray[np.floating],
    *,
    pattern: str,
    E_0: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Process-2 (onion single-ligand): V+(r_X) V+(r_Y) V-(?) V-(?), same lig.

    p_lig trajectory: p^6 -> p^5(γ_1) -> p^4(γ_2) -> p^5(γ_3) -> p^6.
    Pattern A (FIFO): V_3 lowers r_X, V_4 lowers r_Y  (V_1↔V_3 share A, V_2↔V_4 share B).
    Pattern B (LIFO): V_3 lowers r_Y, V_4 lowers r_X  (V_1↔V_4 share A, V_2↔V_3 share B).

    Timeline (f-site [═══] solid, ligand hole [┄┄┄] dashed):
                              V_1        V_2        V_3        V_4
        Pattern A    r_X    :  [════════════════════]                        (staircase)
        (FIFO)       r_Y    :             [════════════════════]
                  lig hole1 :  [┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄]
                  lig hole2 :             [┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄]

        Pattern B    r_X    :  [═══════════════════════════════]             (nested)
        (LIFO)       r_Y    :             [══════════]
                  lig hole1 :  [┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄]
                  lig hole2 :             [┄┄┄┄┄┄┄┄┄┄]

    Denominators: s_1, s_3 subtract 1*E_0; s_2 subtracts 2*E_0.
    """
    M1 = V_1[:, :, :, 0]   # (A=r_X^{n+1}, G=p^5, c=r_X_init)
    M2 = V_2                # (B=r_Y^{n+1}, H=p^4, d=r_Y_init, G=p^5)
    M3 = V_3                # (high_f, H=p^4, low_f, K=p^5)
    M4 = V_4[:, :, :, 0]   # (high_f, K=p^5, low_f)

    G_s1 = -1.0 / (E_np1[:, None] + E_p5[None, :] -       E_0)                                                  # 1 f-site
    G_s2 = -1.0 / (E_np1[:, None, None] + E_np1[None, :, None] + E_p4[None, None, :] - 2.0 * E_0)               # 2 f-sites

    # ─── Derivation: full cluster sum -> local-index einsum ─────────────────
    # (ref: standards/fopt/L3_four_type_full_cluster_expansion.md §3, Pattern A path)
    #
    # Example path A_11 A_21 B_11 B_21 (Pattern A FIFO on lig=1):
    #     V_1 = A_11 (p_1 -> f_1):  δ_{u_2,d}   δ_{ρ_2,Ω_2}
    #     V_2 = A_21 (p_1 -> f_2):  δ_{v_1,u_1} δ_{σ_2,ρ_2}
    #     V_3 = B_11 (f_1 -> p_1):  δ_{w_2,v_2} δ_{τ_2,σ_2}
    #     V_4 = B_21 (f_2 -> p_1):  δ_{a,w_1}   δ_{Ω_2,τ_2}
    #
    # Collapsing δ's forces u_2 = d, v_1 = u_1, w_1 = a, w_2 = v_2, and
    # ρ_2 = σ_2 = τ_2 = Ω_2. Five free dummies remain (one more than process 1
    # because the lig hops through p^4 in s_2, exposing γ_2):
    #     α  = u_1 ∈ f_1^{n+1}      β  = v_2 ∈ f_2^{n+1}
    #     γ_1 = ρ_1 ∈ p^5            γ_2 = σ_1 ∈ p^4            γ_3 = τ_1 ∈ p^5
    #
    # Reduced S:  S_1 = (α, d, γ_1, Ω_2)   S_2 = (α, β, γ_2, Ω_2)   S_3 = (a, β, γ_3, Ω_2)
    # so the s_2 denominator is 3-D (α, β, γ_2). Renaming A=α, B=β, G=γ_1, H=γ_2, K=γ_3.
    if pattern == "A":
        G_s3 = -1.0 / (E_np1[:, None] + E_p5[None, :] - E_0)  # (B, K)  1 f-site (only r_Y still excited in s_3)
        #   H[a,b,c,d] = sum_{A,B,G,H,K}  M4*[B,K,a] G_s3[B,K] M3*[A,H,b,K] G_s2[A,B,H] M2[B,H,d,G] G_s1[A,G] M1[A,G,c]
        #                                 └─ V_4† ─┘ └─/E_s3─┘ └── V_3† ──┘ └─/E_s2──┘ └── V_2 ──┘ └─/E_s1─┘ └─ V_1 ─┘
        # Summed dummies: A = α (f_X^{n+1})  B = β (f_Y^{n+1})  G = γ_1 (p^5)  H = γ_2 (p^4)  K = γ_3 (p^5)
        # Free outputs  : a = r_Y_fin       b = r_X_fin       c = r_X_init     d = r_Y_init      (FIFO: V_4 on r_Y, V_3 on r_X)
        return np.einsum(
            "BKa, BK, AHbK, ABH, BHdG, AG, AGc -> abcd",
            M4.conj(), G_s3,
            M3.conj(), G_s2,
            M2,        G_s1,
            M1,
            optimize=True,
        )
    if pattern == "B":
        # LIFO: V_3 lowers r_Y, V_4 lowers r_X.  V_3↔V_2 share B (r_Y^{n+1});  V_4↔V_1 share A (r_X^{n+1}).
        # Same δ-collapse structure as Pattern A but the V_3/V_4 site-pairings swap, so the
        # high_f indices on M_3* and M_4* swap A<->B compared with FIFO.  Same 5 free dummies.
        G_s3 = -1.0 / (E_np1[:, None] + E_p5[None, :] - E_0)  # (A, K)  1 f-site (only r_X still excited in s_3)
        #   H[a,b,c,d] = sum_{A,B,G,H,K}  M4*[A,K,a] G_s3[A,K] M3*[B,H,b,K] G_s2[A,B,H] M2[B,H,d,G] G_s1[A,G] M1[A,G,c]
        # Free outputs (LIFO): a = r_X_fin, b = r_Y_fin, c = r_X_init, d = r_Y_init.
        return np.einsum(
            "AKa, AK, BHbK, ABH, BHdG, AG, AGc -> abcd",
            M4.conj(), G_s3,
            M3.conj(), G_s2,
            M2,        G_s1,
            M1,
            optimize=True,
        )
    raise ValueError(f"Unknown process-2 pattern: {pattern!r}")


def _path_amplitude_process3(
    V_1: NDArray[np.complexfloating],
    V_2: NDArray[np.complexfloating],
    V_3: NDArray[np.complexfloating],
    V_4: NDArray[np.complexfloating],
    E_np1: NDArray[np.floating],
    E_nm1: NDArray[np.floating],
    E_p5:  NDArray[np.floating],
    *,
    E_0: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Process-3 (alternating cross-ligand): V_1/V_2 on lig_first, V_3/V_4 on lig_other.

    Same f-site dynamics as process 1 (r_X particle nested with r_Y hole),
    but the two p^5 hole excursions live on different ligands.

    Timeline (f-site [═══] solid, ligand hole [┄┄┄] dashed):
                              V_1        V_2        V_3        V_4
        r_X (particle)    :  [═══════════════════════════════]
        r_Y (hole)        :             [══════════]
        lig_first (hole)  :  [┄┄┄┄┄┄┄┄┄]
        lig_other (hole)  :                        [┄┄┄┄┄┄┄┄┄]

    Denominators: s_1, s_3 subtract 1*E_0; s_2 subtracts 2*E_0.
    """
    M1 = V_1[:, :, :, 0]
    M2 = V_2[:, :, :, 0]
    M3 = V_3[:, :, :, 0]
    M4 = V_4[:, :, :, 0]

    G_s1 = -1.0 / (E_np1[:, None] + E_p5[None, :] -       E_0)   # 1 f-site (lig_first p^5)
    G_s2 = -1.0 / (E_np1[:, None] + E_nm1[None, :] - 2.0 * E_0)   # 2 f-sites
    G_s3 = -1.0 / (E_np1[:, None] + E_p5[None, :] -       E_0)   # 1 f-site (lig_other p^5)

    # ─── Derivation: full cluster sum -> local-index einsum ─────────────────
    # (ref: standards/fopt/L3_four_type_full_cluster_expansion.md §4)
    #
    # Example path A_11 B_21 A_22 B_12 (V_1, V_2 on lig_1; V_3, V_4 on lig_2):
    #     V_1 = A_11 (p_1 -> f_1):  δ_{u_2,d}   δ_{ρ_2,Ω_2}
    #     V_2 = B_21 (f_2 -> p_1):  δ_{v_1,u_1} δ_{σ_2,ρ_2}
    #     V_3 = A_22 (p_2 -> f_2):  δ_{w_1,v_1} δ_{τ_1,σ_1}
    #     V_4 = B_12 (f_1 -> p_2):  δ_{b,w_2}   δ_{Ω_1,τ_1}
    #
    # Collapsing δ's forces u_2 = d, v_1 = w_1 = u_1, w_2 = b, ρ_2 = σ_2 = Ω_2,
    # σ_1 = τ_1 = Ω_1.  Four free dummies remain (the two ligand photons now
    # live on different ligands, distinguishing γ on lig_1 from δ on lig_2):
    #     α = u_1 ∈ f_1^{n+1}     β = v_2 ∈ f_2^{n-1}
    #     γ = ρ_1 ∈ lig_first p^5  δ = τ_2 ∈ lig_other p^5
    #
    # Reduced S:  S_1 = (α, d, γ, Ω_2)   S_2 = (α, β, Ω_1, Ω_2)   S_3 = (α, b, Ω_1, δ)
    # Same structural shape as process 1 (Σ over 4 dummies); only the two
    # p^5 indices are on different ligands. Renaming A=α, B=β, G=γ, H=δ.
    #
    #   H[a,b,c,d] = sum_{A,B,G,H}  M4*[A,H,a] G_s3[A,H] M3[b,H,B] G_s2[A,B] M2*[d,G,B] G_s1[A,G] M1[A,G,c]
    #                               └─ V_4† ─┘ └─/E_s3─┘ └─ V_3 ─┘ └─/E_s2─┘ └─ V_2† ─┘ └─/E_s1─┘ └─ V_1 ─┘
    # Summed dummies: A = α (f_X^{n+1})  B = β (f_Y^{n-1})  G = γ (lig_first p^5)  H = δ (lig_other p^5)
    # Free outputs  : a = r_X_fin        b = r_Y_fin        c = r_X_init           d = r_Y_init   (n_k doublet)
    return np.einsum(
        "AHa, AH, bHB, AB, dGB, AG, AGc -> abcd",
        M4.conj(), G_s3,
        M3,        G_s2,
        M2.conj(), G_s1,
        M1,
        optimize=True,
    )


def _path_amplitude_process4(
    V_1: NDArray[np.complexfloating],
    V_2: NDArray[np.complexfloating],
    V_3: NDArray[np.complexfloating],
    V_4: NDArray[np.complexfloating],
    E_np1: NDArray[np.floating],
    E_p5:  NDArray[np.floating],
    *,
    pattern: str,
    E_0: float = 0.0,
) -> NDArray[np.complexfloating]:
    """Process-4 (onion crossed double-ligand): all 4 vertices on "fn_p6", crossed lig.

    V_1 raises r_X via lig_a, V_2 raises r_Y via lig_b; V_3, V_4 lower on the OTHER lig.
    Pattern A (FIFO): V_3 lowers r_X via lig_b, V_4 lowers r_Y via lig_a.
    Pattern B (LIFO): V_3 lowers r_Y via lig_a, V_4 lowers r_X via lig_b.

    Timeline (f-site [═══] solid, ligand hole [┄┄┄] dashed):
                              V_1        V_2        V_3        V_4
        Pattern A    r_X    :  [════════════════════]                        (staircase)
        (FIFO)       r_Y    :             [════════════════════]
                   lig_a    :  [┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄]
                   lig_b    :             [┄┄┄┄┄┄┄┄┄┄]

        Pattern B    r_X    :  [═══════════════════════════════]             (nested)
        (LIFO)       r_Y    :             [══════════]
                   lig_a    :  [┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄]
                   lig_b    :             [┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄]

    Denominators: s_1, s_3 subtract 1*E_0; s_2 subtracts 2*E_0 (4-D denominator).
    """
    M1 = V_1[:, :, :, 0]   # (A=r_X^{n+1}, G=γ_a, c=r_X_init)
    M2 = V_2[:, :, :, 0]   # (B=r_Y^{n+1}, H=γ_b, d=r_Y_init)
    M3 = V_3[:, :, :, 0]
    M4 = V_4[:, :, :, 0]

    G_s1 = -1.0 / (E_np1[:, None] + E_p5[None, :] - E_0)                                            # 1 f-site
    G_s2 = -1.0 / (
        E_np1[:, None, None, None]
      + E_np1[None, :, None, None]
      + E_p5[None, None, :, None]
      + E_p5[None, None, None, :]
      - 2.0 * E_0
    )                                                                                                # 2 f-sites (4-D)

    # ─── Derivation: full cluster sum -> local-index einsum ─────────────────
    # (ref: standards/fopt/L3_four_type_full_cluster_expansion.md §5, Pattern A path)
    #
    # Example path A_11 A_22 B_12 B_21 (Pattern A FIFO: V_3 lowers V_1's site):
    #     V_1 = A_11 (p_1 -> f_1):  δ_{u_2,d}     δ_{ρ_2,Ω_2}
    #     V_2 = A_22 (p_2 -> f_2):  δ_{v_1,u_1}   δ_{σ_1,ρ_1}
    #     V_3 = B_12 (f_1 -> p_2):  δ_{w_2,v_2}   δ_{τ_1,σ_1}
    #     V_4 = B_21 (f_2 -> p_1):  δ_{a,w_1}     δ_{Ω_2,τ_2}
    #
    # Collapsing δ's forces u_2 = d, v_1 = u_1, w_2 = v_2, w_1 = a,
    # ρ_2 = Ω_2, σ_1 = τ_1 = ρ_1, τ_2 = Ω_2.  Four free dummies remain:
    #     α = u_1 ∈ f_1^{n+1}      β = v_2 ∈ f_2^{n+1}
    #     γ = ρ_1 ∈ lig_a p^5       δ = σ_2 ∈ lig_b p^5
    #
    # Reduced S:  S_1 = (α, d, γ, Ω_2)   S_2 = (α, β, γ, δ)   S_3 = (a, β, γ, Ω_2)
    # Note s_2 is genuinely 4-active (both f sites excited AND both ligs in p^5),
    # giving the only 4-D denominator G_s2[A,B,G,H] in the whole framework.
    # Renaming A=α, B=β, G=γ, H=δ.

    if pattern == "A":
        G_s3 = -1.0 / (E_np1[:, None] + E_p5[None, :] - E_0)  # (B, G)  1 f-site (r_Y still up on lig_a)
        #   H[a,b,c,d] = sum_{A,B,G,H}  M4*[B,G,a] G_s3[B,G] M3*[A,H,b] G_s2[A,B,G,H] M2[B,H,d] G_s1[A,G] M1[A,G,c]
        #                               └─ V_4† ─┘ └─/E_s3─┘ └─ V_3† ─┘ └── /E_s2 ──┘ └─ V_2 ─┘ └─/E_s1─┘ └─ V_1 ─┘
        # Summed dummies: A = α (f_X^{n+1})  B = β (f_Y^{n+1})  G = γ (lig_a p^5)  H = δ (lig_b p^5)
        # Free outputs  : a = r_Y_fin       b = r_X_fin       c = r_X_init        d = r_Y_init   (FIFO: V_4 on r_Y, V_3 on r_X)
        return np.einsum(
            "BGa, BG, AHb, ABGH, BHd, AG, AGc -> abcd",
            M4.conj(), G_s3,
            M3.conj(), G_s2,
            M2,        G_s1,
            M1,
            optimize=True,
        )
    if pattern == "B":
        # LIFO: V_3 lowers r_Y via lig_a, V_4 lowers r_X via lig_b.  Same δ-collapse
        # produces the same 4 free dummies; V_3/V_4 site-pairings swap so M_3*, M_4*
        # high_f labels swap A <-> B compared with FIFO.
        G_s3 = -1.0 / (E_np1[:, None] + E_p5[None, :] - E_0)  # (A, H)  1 f-site (r_X still up on lig_b)
        #   H[a,b,c,d] = sum_{A,B,G,H}  M4*[A,H,a] G_s3[A,H] M3*[B,G,b] G_s2[A,B,G,H] M2[B,H,d] G_s1[A,G] M1[A,G,c]
        # Free outputs (LIFO): a = r_X_fin, b = r_Y_fin, c = r_X_init, d = r_Y_init.
        return np.einsum(
            "AHa, AH, BGb, ABGH, BHd, AG, AGc -> abcd",
            M4.conj(), G_s3,
            M3.conj(), G_s2,
            M2,        G_s1,
            M1,
            optimize=True,
        )
    raise ValueError(f"Unknown process-4 pattern: {pattern!r}")


# ---------------------------------------------------------------------------
# 24 paths split into 4 process tables.
# Each entry encodes which (f_site, ligand, sector) provides V_1..V_4 plus the
# cluster-sign integer constant: sign = (-1)^(10*n_ele + sign_const).
# ---------------------------------------------------------------------------

# (f_first, ligand, sign_const)
# f_first = r1 = r4 ; f_other = r2 = r3.
_PROCESS1_PATHS: list[tuple[int, int, int]] = [
    (1, 1, 2),
    (1, 2, 26),
    (2, 1, 0),
    (2, 2, 24),
]

# (r1, r2, r3, r4, ligand, sign_const)
# pattern is "A" iff r3 == r1 (V_3 lowers V_1's site), else "B".
_PROCESS2_PATHS: list[tuple[int, int, int, int, int, int]] = [
    (1, 2, 1, 2, 1, 3),
    (1, 2, 2, 1, 1, 4),
    (2, 1, 1, 2, 1, 2),
    (2, 1, 2, 1, 1, 3),
    (1, 2, 1, 2, 2, 27),
    (1, 2, 2, 1, 2, 28),
    (2, 1, 1, 2, 2, 26),
    (2, 1, 2, 1, 2, 27),
]

# (f_first, lig_first, lig_other, sign_const)
_PROCESS3_PATHS: list[tuple[int, int, int, int]] = [
    (1, 1, 2, 14),
    (1, 2, 1, 14),
    (2, 1, 2, 12),
    (2, 2, 1, 12),
]

# (r_a, lig_a, r_b, lig_b, r_c, lig_c, r_d, lig_d, sign_const)
# pattern is "A" iff r_c == r_a (V_3 lowers V_1's site), else "B".
_PROCESS4_PATHS: list[tuple[int, int, int, int, int, int, int, int, int]] = [
    (1, 1, 2, 2, 1, 2, 2, 1, 13),
    (1, 1, 2, 2, 2, 1, 1, 2, 15),
    (2, 2, 1, 1, 1, 2, 2, 1, 13),
    (2, 2, 1, 1, 2, 1, 1, 2, 15),
    (1, 2, 2, 1, 1, 1, 2, 2, 15),
    (1, 2, 2, 1, 2, 2, 1, 1, 15),
    (2, 1, 1, 2, 1, 1, 2, 2, 13),
    (2, 1, 1, 2, 2, 2, 1, 1, 13),
]


def _cluster_sign(n_ele: int, sign_const: int) -> int:
    return -1 if (10 * n_ele + sign_const) % 2 == 1 else 1


def build_L3(
    l2: dict[tuple[int, int, str], dict[str, Any]],
    energies: dict[str, NDArray[np.floating]],
    W: NDArray[np.complexfloating],
    n_ele: int,
    *,
    return_per_path: bool = False,
) -> NDArray[np.complexfloating] | dict[str, Any]:
    """H_eff[f1_fin, f2_fin, f1_init, f2_init] (n_k,)*4 from 24 path amplitudes.

    l2       : build_L2 output {(f_site, ligand, sector): pair_dict}.
    energies : {"f_np1", "f_nm1", "p_5", "p_4"} -> 1D ndarray. Optional "E_0"
               key (default 0) is the single-site f^n GS energy; cluster
               GS = 2*E_0. Path denominators subtract n_exc * E_0 where
               n_exc = number of f-sites excited in the intermediate state
               (s_1 / s_3 -> 1*E_0,  s_2 -> 2*E_0).
    W        : (2J+1, n_k) doublet projector, shared by all f-sites.
    n_ele    : f-shell electron count (used for cluster sign).
    return_per_path : if False (default), return H_total only. If True, return
                      a dict {"H_total", "per_path", "per_process"} for debug:
                          per_path    -> dict {(process, path_idx, pattern): H_path}
                          per_process -> dict {process: sum of its paths}
                      Each stored amplitude is already sign-applied and
                      transposed to standard (f_1_fin, f_2_fin, f_1_init, f_2_init).
                      Identity: sum(per_path.values()) == sum(per_process.values()) == H_total.

    Output ordering matches sopt's [c, d, a, b] bra/ket convention so that
    `Heff.reshape(n_k**2, n_k**2)` plugs straight into `spin12_map`.
    """
    l2_proj = project_W(l2, W)

    n_k = l2_proj[(1, 1, "fn_p6")]["V_plus"].shape[2]
    H = np.zeros((n_k,) * 4, dtype=DTYPE_COMPLEX)
    per_path: dict[tuple[str, int, str | None], NDArray[np.complexfloating]] = {}
    per_process: dict[str, NDArray[np.complexfloating]] = {
        proc: np.zeros((n_k,) * 4, dtype=DTYPE_COMPLEX) for proc in ("P1", "P2", "P3", "P4")
    }

    E_np1 = energies["f_np1"]
    E_nm1 = energies["f_nm1"]
    E_p5  = energies["p_5"]
    E_p4  = energies["p_4"]
    E_0   = float(energies.get("E_0", 0.0))

    def _accumulate(proc: str, idx: int, pattern: str | None, contribution):
        H[:] += contribution
        per_process[proc] += contribution
        if return_per_path:
            per_path[(proc, idx, pattern)] = contribution

    # Process 1: alternating single-ligand (4 paths).
    for idx, (f_first, lig, sign_const) in enumerate(_PROCESS1_PATHS):
        f_other = 3 - f_first
        V_1 = l2_proj[(f_first, lig, "fn_p6")  ]["V_plus"]
        V_2 = l2_proj[(f_other, lig, "fnm1_p6")]["V_plus"]
        V_3 = l2_proj[(f_other, lig, "fnm1_p6")]["V_plus"]
        V_4 = l2_proj[(f_first, lig, "fn_p6")  ]["V_plus"]
        H_native = _path_amplitude_process1(V_1, V_2, V_3, V_4, E_np1, E_nm1, E_p5, E_0=E_0)
        contribution = _cluster_sign(n_ele, sign_const) * _to_standard(
            H_native, r1=f_first, r2=f_other, r3=f_other, r4=f_first,
        )
        _accumulate("P1", idx, None, contribution)

    # Process 2: onion single-ligand (8 paths).
    for idx, (r1, r2, r3, r4, lig, sign_const) in enumerate(_PROCESS2_PATHS):
        V_1 = l2_proj[(r1, lig, "fn_p6")]["V_plus"]
        V_2 = l2_proj[(r2, lig, "fn_p5")]["V_plus"]
        V_3 = l2_proj[(r3, lig, "fn_p5")]["V_plus"]
        V_4 = l2_proj[(r4, lig, "fn_p6")]["V_plus"]
        pattern = "A" if r3 == r1 else "B"
        H_native = _path_amplitude_process2(
            V_1, V_2, V_3, V_4, E_np1, E_p5, E_p4, pattern=pattern, E_0=E_0,
        )
        contribution = _cluster_sign(n_ele, sign_const) * _to_standard(
            H_native, r1=r1, r2=r2, r3=r3, r4=r4,
        )
        _accumulate("P2", idx, pattern, contribution)

    # Process 3: alternating cross-ligand (4 paths).
    for idx, (f_first, lig_first, lig_other, sign_const) in enumerate(_PROCESS3_PATHS):
        f_other = 3 - f_first
        V_1 = l2_proj[(f_first, lig_first, "fn_p6")  ]["V_plus"]
        V_2 = l2_proj[(f_other, lig_first, "fnm1_p6")]["V_plus"]
        V_3 = l2_proj[(f_other, lig_other, "fnm1_p6")]["V_plus"]
        V_4 = l2_proj[(f_first, lig_other, "fn_p6")  ]["V_plus"]
        H_native = _path_amplitude_process3(V_1, V_2, V_3, V_4, E_np1, E_nm1, E_p5, E_0=E_0)
        contribution = _cluster_sign(n_ele, sign_const) * _to_standard(
            H_native, r1=f_first, r2=f_other, r3=f_other, r4=f_first,
        )
        _accumulate("P3", idx, None, contribution)

    # Process 4: onion crossed double-ligand (8 paths).
    for idx, (r_a, lig_a, r_b, lig_b, r_c, lig_c, r_d, lig_d, sign_const) in enumerate(_PROCESS4_PATHS):
        V_1 = l2_proj[(r_a, lig_a, "fn_p6")]["V_plus"]
        V_2 = l2_proj[(r_b, lig_b, "fn_p6")]["V_plus"]
        V_3 = l2_proj[(r_c, lig_c, "fn_p6")]["V_plus"]
        V_4 = l2_proj[(r_d, lig_d, "fn_p6")]["V_plus"]
        pattern = "A" if r_c == r_a else "B"
        H_native = _path_amplitude_process4(
            V_1, V_2, V_3, V_4, E_np1, E_p5, pattern=pattern, E_0=E_0,
        )
        contribution = _cluster_sign(n_ele, sign_const) * _to_standard(
            H_native, r1=r_a, r2=r_b, r3=r_c, r4=r_d,
        )
        _accumulate("P4", idx, pattern, contribution)

    if return_per_path:
        return {
            "H_total":     H,
            "per_path":    per_path,
            "per_process": per_process,
        }
    return H
