"""
Decompose 14x14 f-shell h_local into CEF and SOC parts.

Usage:
    python w90_decompose_h_local.py <h_local.dat>

Reads the h_local output from w90_extract, prints ζ, saves h_cef.
"""

from __future__ import annotations

from functools import lru_cache
import sys

import numpy as np
from sympy.physics.wigner import wigner_3j


def _spin_block_indices(n_orb):
    """Return spin block indices for internal interleaved (dn, up) ordering."""
    dn_idx = np.arange(0, 2 * n_orb, 2)
    up_idx = np.arange(1, 2 * n_orb, 2)
    return dn_idx, up_idx


def _build_l_operators(ell):
    """Build orbital angular-momentum matrices in complex-harmonic basis."""
    m_vals = np.arange(-ell, ell + 1, dtype=float)
    lz = np.diag(m_vals).astype(complex)
    lp = np.zeros((2 * ell + 1, 2 * ell + 1), dtype=complex)
    for col, m in enumerate(range(-ell, ell + 1)):
        if m == ell:
            continue
        row = col + 1
        lp[row, col] = np.sqrt(ell * (ell + 1) - m * (m + 1))
    lm = lp.conj().T
    return lz, lp, lm


def _build_soc_matrix(ell, zeta):
    """Build zeta * L.S in internal interleaved (dn, up) ordering."""
    lz, lp, lm = _build_l_operators(ell)
    dim = 2 * (2 * ell + 1)
    soc = np.zeros((dim, dim), dtype=complex)
    dn_idx, up_idx = _spin_block_indices(2 * ell + 1)

    soc[np.ix_(dn_idx, dn_idx)] = -0.5 * zeta * lz
    soc[np.ix_(up_idx, up_idx)] = 0.5 * zeta * lz
    soc[np.ix_(dn_idx, up_idx)] = 0.5 * zeta * lp
    soc[np.ix_(up_idx, dn_idx)] = 0.5 * zeta * lm
    return soc


def decompose_h_local(h_local, rtol=1e-3):
    """
    Decompose 14x14 h_local = H_CEF ⊗ I₂ + ζ L·S.

    Returns (zeta, h_cef) where h_cef is 7x7 complex-harmonic.
    Raises ValueError if the L·S model is inconsistent.
    """
    h_local = np.asarray(h_local, dtype=complex)
    if h_local.shape != (14, 14):
        raise ValueError(f"h_local must be (14,14), got {h_local.shape}")
    if not np.allclose(h_local, h_local.conj().T, atol=1e-10):
        raise ValueError("h_local is not Hermitian")

    ell = 3
    dn, up = _spin_block_indices(7)
    h_uu, h_dd = h_local[np.ix_(up, up)], h_local[np.ix_(dn, dn)]
    h_du, h_ud = h_local[np.ix_(dn, up)], h_local[np.ix_(up, dn)]

    # ζ from spin-diagonal: (h_↑↑ - h_↓↓)_{mm} = ζ·m  (must be real)
    m_vals = np.arange(-ell, ell + 1, dtype=float)
    mask = m_vals != 0
    diag_diff = np.diag(h_uu - h_dd)[mask]
    if np.any(np.abs(diag_diff.imag) > rtol * np.abs(diag_diff)):
        ratio = np.max(np.abs(diag_diff.imag) / np.abs(diag_diff))
        raise ValueError(
            f"(h_↑↑ - h_↓↓) diagonal has non-negligible imaginary part "
            f"(max |Im/|z||={ratio:.2e}, rtol={rtol:.0e})")
    zeta_diag = diag_diff.real / m_vals[mask]
    if not np.allclose(zeta_diag, zeta_diag[0], rtol=rtol):
        spread = (zeta_diag.max() - zeta_diag.min()) / np.abs(np.mean(zeta_diag))
        raise ValueError(
            f"ζ from spin-diagonal inconsistent (spread={spread:.2e}, rtol={rtol:.0e}): "
            f"{zeta_diag}")

    # ζ from spin-flip: h_↓↑[m+1,m] = (ζ/2)·√(ℓ(ℓ+1)-m(m+1))  (must be real)
    #                    h_↑↓[m,m+1] = (ζ/2)·√(ℓ(ℓ+1)-m(m+1))  (must be real)
    ll1 = ell * (ell + 1)
    coeffs = np.sqrt([ll1 - m * (m + 1) for m in range(-ell, ell)])
    du_elements = np.array([h_du[c + 1, c] for c in range(2 * ell)])
    ud_elements = np.array([h_ud[c, c + 1] for c in range(2 * ell)])
    if np.any(np.abs(du_elements.imag) > rtol * np.abs(du_elements)):
        ratio = np.max(np.abs(du_elements.imag) / np.abs(du_elements))
        raise ValueError(
            f"h_↓↑ L+ elements have non-negligible imaginary part "
            f"(max |Im/|z||={ratio:.2e}, rtol={rtol:.0e})")
    if np.any(np.abs(ud_elements.imag) > rtol * np.abs(ud_elements)):
        ratio = np.max(np.abs(ud_elements.imag) / np.abs(ud_elements))
        raise ValueError(
            f"h_↑↓ L- elements have non-negligible imaginary part "
            f"(max |Im/|z||={ratio:.2e}, rtol={rtol:.0e})")
    zeta_du = 2.0 * du_elements.real / coeffs
    zeta_ud = 2.0 * ud_elements.real / coeffs
    if not np.allclose(zeta_du, zeta_du[0], rtol=rtol):
        spread = (zeta_du.max() - zeta_du.min()) / np.abs(np.mean(zeta_du))
        raise ValueError(
            f"ζ from h_↓↑ inconsistent (spread={spread:.2e}, rtol={rtol:.0e}): {zeta_du}")
    if not np.allclose(zeta_ud, zeta_ud[0], rtol=rtol):
        spread = (zeta_ud.max() - zeta_ud.min()) / np.abs(np.mean(zeta_ud))
        raise ValueError(
            f"ζ from h_↑↓ inconsistent (spread={spread:.2e}, rtol={rtol:.0e}): {zeta_ud}")
    if not np.allclose(zeta_du, zeta_ud, rtol=rtol):
        diff = np.max(np.abs(zeta_du - zeta_ud)) / np.abs(np.mean(zeta_du))
        raise ValueError(
            f"ζ from h_↓↑ vs h_↑↓ mismatch (max_rel_diff={diff:.2e}, rtol={rtol:.0e})")

    # cross-check all three channels
    zeta = float(np.mean(zeta_diag))
    zeta_offdiag = float(np.mean(zeta_du))
    if not np.isclose(zeta, zeta_offdiag, rtol=rtol):
        diff = abs(zeta - zeta_offdiag) / abs(zeta)
        raise ValueError(
            f"ζ diagonal ({zeta:.6f}) vs off-diagonal ({zeta_offdiag:.6f}) "
            f"mismatch (rel_diff={diff:.2e}, rtol={rtol:.0e})")

    h_cef = 0.5 * (h_uu + h_dd)

    # total residual: h_local vs H_CEF ⊗ I₂ + ζ L·S
    h_model = np.kron(h_cef, np.eye(2, dtype=complex)) + _build_soc_matrix(ell, zeta)
    residual = np.linalg.norm(h_local - h_model) / np.linalg.norm(h_local)
    if residual > rtol:
        raise ValueError(f"model residual too large: {residual:.2e}")

    return zeta, h_cef


def _build_ckq_matrix(ell, k, q):
    """Build Racah C_q^(k) matrix in the |ell, m> basis with m=-ell..ell."""
    dim = 2 * ell + 1
    ckq = np.zeros((dim, dim), dtype=complex)
    reduced = ((-1) ** ell) * (2 * ell + 1) * float(wigner_3j(ell, k, ell, 0, 0, 0))
    m_vals = list(range(-ell, ell + 1))

    for row, m_prime in enumerate(m_vals):
        for col, m in enumerate(m_vals):
            three_j = float(wigner_3j(ell, k, ell, -m_prime, q, m))
            ckq[row, col] = ((-1) ** (ell - m_prime)) * three_j * reduced
    return ckq


@lru_cache(maxsize=None)
def build_tensor_basis(ell=3):
    """Return cached Racah tensor basis matrices for k=2,4,6."""
    return {
        (k, q): _build_ckq_matrix(ell, k, q)
        for k in (2, 4, 6)
        for q in range(-k, k + 1)
    }


def _relative_residual(target, fit):
    """Relative Frobenius residual with zero-safe denominator."""
    denom = np.linalg.norm(target)
    if denom < 1e-12:
        return 0.0 if np.linalg.norm(target - fit) < 1e-12 else np.inf
    return np.linalg.norm(target - fit) / denom


def decompose_orbital_tensors(h_cef, ell=3):
    """
    Expand traceless h_cef onto the orbital Racah tensor basis.

    Returns dict with complex coefficients a_kq, fit residual, and removed trace.
    """
    h_cef = np.asarray(h_cef, dtype=complex)
    dim = 2 * ell + 1
    if h_cef.shape != (dim, dim):
        raise ValueError(f"h_cef must be ({dim},{dim}), got {h_cef.shape}")
    if not np.allclose(h_cef, h_cef.conj().T, atol=1e-10):
        raise ValueError("h_cef is not Hermitian")

    trace_avg = complex(np.trace(h_cef) / dim)
    h_0 = h_cef - trace_avg * np.eye(dim, dtype=complex)

    basis = build_tensor_basis(ell=ell)
    keys = sorted(basis)
    design = np.column_stack([basis[key].reshape(-1) for key in keys])
    coeffs_vec, _, _, _ = np.linalg.lstsq(design, h_0.reshape(-1), rcond=None)
    fit = sum(coeffs_vec[idx] * basis[key] for idx, key in enumerate(keys))

    return {
        "a_kq": {key: coeffs_vec[idx] for idx, key in enumerate(keys)},
        "residual": _relative_residual(h_0, fit),
        "trace_avg": trace_avg,
        "fit_matrix": fit,
    }


def _unit_tensor_rme_fn(n_ele, ell=3):
    """
    Return ⟨αLS||U^(k)||αLS⟩ for f^n ground LS term.

    Hardcoded from Nielson-Koster tables (computed via Fock space projection).
    Particle-hole symmetry: U_{14-n}(k) = -U_n(k) for even k.
    Half-shell symmetry: U_{7-n}(k) = -U_n(k).
    """
    if ell != 3:
        raise NotImplementedError("Unit tensor RME only implemented for f-shell (ell=3)")
    # Nielson-Koster RME table for f^n ground LS terms, k = 2, 4, 6
    #   n=1 ²F, n=2 ³H, n=3 ⁴I, n=4 ⁵I, n=5 ⁶H, n=6 ⁷F, n=7 ⁸S
    #   n=8..13 from particle-hole: U_{14-n}(k) = -U_n(k)
    _NK_TABLE = {
        1:  { 2:  1.00000000, 4:  1.00000000, 6:  1.00000000},
        2:  { 2:  1.06532654, 4: -0.90851353, 6: -1.16155342},
        3:  { 2:  0.44381268, 4: -0.63708473, 6:  1.78266170},
        4:  { 2: -0.44381268, 4:  0.63708473, 6: -1.78266170},
        5:  { 2: -1.06532654, 4:  0.90851353, 6:  1.16155342},
        6:  { 2: -1.00000000, 4: -1.00000000, 6: -1.00000000},
        7:  { 2:  0.00000000, 4:  0.00000000, 6:  0.00000000},
        8:  { 2:  1.00000000, 4:  1.00000000, 6:  1.00000000},
        9:  { 2:  1.06532654, 4: -0.90851353, 6: -1.16155342},
        10: { 2:  0.44381268, 4: -0.63708473, 6:  1.78266170},
        11: { 2: -0.44381268, 4:  0.63708473, 6: -1.78266170},
        12: { 2: -1.06532654, 4:  0.90851353, 6:  1.16155342},
        13: { 2: -1.00000000, 4: -1.00000000, 6: -1.00000000},
    }
    if n_ele not in _NK_TABLE:
        raise ValueError(f"n_ele={n_ele} out of range [1, 13]")
    table = _NK_TABLE[n_ele]
    return lambda k: table[k]


def _compute_lambda_k(k, ell, L, S, J, unit_rme_func):
    """
    Analytical operator equivalence factor Λ_k.

    Λ_k = ⟨ℓ||C^(k)||ℓ⟩ × ⟨αLS||U^(k)||αLS⟩ × (-1)^(L+S+J+k) × (2J+1) × {L k L; J S J}
    """
    from sympy.physics.wigner import wigner_6j

    # ⟨ℓ||C^(k)||ℓ⟩
    ell_rme = ((-1) ** ell) * (2 * ell + 1) * float(wigner_3j(ell, k, ell, 0, 0, 0))

    # ⟨αLS||U^(k)||αLS⟩
    u_rme = unit_rme_func(k)

    # 6j symbol {L k L; J S J}
    sixj = float(wigner_6j(L, k, L, J, S, J))

    phase = (-1) ** int(round(L + S + J + k))

    return ell_rme * u_rme * phase * (2 * J + 1) * sixj


def project_akq_to_j(a_kq, J0, ell, L, S, unit_rme_func):
    """
    Build H_CEF in the |J,M⟩ basis analytically from orbital a_kq.

    Uses: H[M',M] = Σ_{k,q} a_kq × Λ_k × (-1)^(J-M') × (J k J; -M' q M)
    """
    dim_j = int(round(2 * J0 + 1))
    h_j = np.zeros((dim_j, dim_j), dtype=complex)
    m_vals = [J0 - i for i in range(dim_j)]  # J, J-1, ..., -J

    lambda_cache = {}
    for (k, q), coeff in a_kq.items():
        if abs(coeff) < 1e-30:
            continue
        if k not in lambda_cache:
            lambda_cache[k] = _compute_lambda_k(k, ell, L, S, J0, unit_rme_func)
        lam = lambda_cache[k]

        for row, mp in enumerate(m_vals):
            for col, m in enumerate(m_vals):
                tj = float(wigner_3j(J0, k, J0, -mp, q, m))
                h_j[row, col] += coeff * lam * ((-1) ** int(round(J0 - mp))) * tj

    return h_j


def convert_akq_to_bkq(a_kq, n_ele, J0, L0, S0, ell=3, symmetry="C3v", mode_q3="sin"):
    """
    Convert Scheme A a_kq to Stevens B parameters analytically.

    Uses the Wigner-Eckart theorem to project a_kq directly into the J0
    subspace, then fits to Stevens templates.  No Fock space needed.

    Parameters
    ----------
    a_kq : dict mapping (k, q) to complex coefficient (from decompose_orbital_tensors)
    n_ele : number of f electrons (1 or 13 supported analytically)
    J0, L0, S0 : ground-state quantum numbers
    ell : orbital angular momentum (3 for f)
    symmetry : "Oh" or "C3v"
    mode_q3 : tesseral mode for q=3 Stevens terms

    Returns
    -------
    dict with "B_params", "H_cef_j0", "stevens_fit_residual"
    """
    unit_rme = _unit_tensor_rme_fn(n_ele, ell=ell)

    # Build H_CEF in |J,M⟩ basis from Wigner-Eckart theorem
    h_j = project_akq_to_j(a_kq, J0, ell, L0, S0, unit_rme)

    # Remove trace (Stevens operators are traceless)
    dim_j = h_j.shape[0]
    trace_avg = complex(np.trace(h_j) / dim_j)
    h_j_0 = h_j - trace_avg * np.eye(dim_j, dtype=complex)

    # Fit to Stevens templates
    templates = _build_stevens_templates(J0, symmetry=symmetry, mode_q3=mode_q3)
    names, coeffs, h_fit = _real_least_squares_coeffs(templates, h_j_0)

    return {
        "B_params": {name: float(coeffs[idx]) for idx, name in enumerate(names)},
        "H_cef_j0": h_j,
        "stevens_fit_residual": _relative_residual(h_j_0, h_fit),
    }


def _build_stevens_templates(J0, symmetry="Oh", mode_q3="cos"):
    """Build symmetry-specific Stevens fitting templates in the |J,M> basis."""
    from fexchange.core.stevens import build_cef_stevens_operators

    ops = build_cef_stevens_operators(J0, symmetry=symmetry, mode_q3=mode_q3)
    if symmetry == "Oh":
        return {
            "B4": ops["O40"] + 5.0 * ops["O44c"],
            "B6": ops["O60"] - 21.0 * ops["O64c"],
        }
    if symmetry == "C3v":
        return {
            "B20": ops["O20"],
            "B40": ops["O40"],
            "B60": ops["O60"],
            "B66": ops["O66"],
            "B43": ops["O43_eta"],
            "B63": ops["O63_eta"],
        }
    raise ValueError(f"Unknown symmetry: {symmetry!r}")


def _real_least_squares_coeffs(template_map, target):
    """Solve for real coefficients against complex Hermitian templates."""
    names = list(template_map)
    design = np.column_stack([template_map[name].reshape(-1) for name in names])
    design_r = np.vstack([design.real, design.imag])
    target_r = np.concatenate([target.reshape(-1).real, target.reshape(-1).imag])
    coeffs, _, _, _ = np.linalg.lstsq(design_r, target_r, rcond=None)
    fit = sum(coeffs[i] * template_map[n] for i, n in enumerate(names))
    return names, coeffs, fit


def extract_cef_stevens_fit(
    h_cef,
    n_ele,
    zeta,
    symmetry="Oh",
    mode_q3="cos",
):
    """
    Project h_cef into the SOC-lowest J0 subspace and fit Stevens parameters.

    Slater integrals fixed (F2=1, F4=F6=0) — only affect J0 selection,
    which is insensitive to their values.
    """
    h_cef = np.asarray(h_cef, dtype=complex)
    if h_cef.shape != (7, 7):
        raise ValueError(f"h_cef must be (7,7), got {h_cef.shape}")

    from fexchange.core.fermion import one_body_operator_matrix
    from fexchange.spectrum.lsms import build_lsms
    from fexchange.spectrum.lsjm import build_lsjm, select_soc_lowest_subspace
    from fexchange.utils.constants import N_ORB

    h_1b = np.kron(h_cef, np.eye(2))
    h_cef_fock = one_body_operator_matrix(h_1b, n_ele, N_ORB)

    lsms = build_lsms(n_ele, N_ORB, r42=0, r62=0)
    lsjm = build_lsjm(lsms, N_ORB)
    soc0 = select_soc_lowest_subspace(lsjm, F2=1, F4=0, F6=0, zeta=zeta)

    U_j0 = soc0["U_n_soc0"]
    h_cef_j0 = U_j0.conj().T @ h_cef_fock @ U_j0

    fock_norm = np.linalg.norm(h_cef_fock)
    j0_weight = np.linalg.norm(h_cef_j0) / fock_norm if fock_norm > 0 else 1.0

    dim = h_cef_j0.shape[0]
    trace_avg = complex(np.trace(h_cef_j0) / dim)
    h_j0_0 = h_cef_j0 - trace_avg * np.eye(dim, dtype=complex)

    templates = _build_stevens_templates(soc0["J0"], symmetry=symmetry, mode_q3=mode_q3)
    names, coeffs, h_fit_0 = _real_least_squares_coeffs(templates, h_j0_0)
    h_fit = h_fit_0 + trace_avg * np.eye(dim, dtype=complex)

    eigvals = np.linalg.eigvalsh(h_cef_j0)
    eigvals_fit = np.linalg.eigvalsh(h_fit)

    return {
        "alpha0": soc0["alpha0"],
        "L0": soc0["L0"],
        "S0": soc0["S0"],
        "J0": soc0["J0"],
        "n_j": soc0["n_j"],
        "j0_weight": j0_weight,
        "trace_avg_j0": trace_avg,
        "stevens_fit_residual": _relative_residual(h_j0_0, h_fit_0),
        "B_params": {name: float(coeffs[i]) for i, name in enumerate(names)},
        "H_cef_j0": h_cef_j0,
        "H_cef_j0_fit": h_fit,
        "eigvals_meV": eigvals,
        "eigvals_fit_meV": eigvals_fit,
        "cef_splitting_meV": eigvals - eigvals.min(),
        "cef_splitting_fit_meV": eigvals_fit - eigvals_fit.min(),
    }


def extract_local_stevens_fit(
    h_local,
    n_ele,
    zeta,
    symmetry="Oh",
    mode_q3="cos",
):
    """
    Project a full 14x14 local one-body Hamiltonian directly into the SOC-lowest
    J0 subspace and fit Stevens parameters there.

    This route is intended for SOC-included ``h_local`` matrices, where fitting
    in the J0 manifold avoids errors from forcing an approximate
    ``h_local = h_cef ⊗ I + zeta L·S`` separation at the orbital level.
    """
    h_local = np.asarray(h_local, dtype=complex)
    if h_local.shape != (14, 14):
        raise ValueError(f"h_local must be (14,14), got {h_local.shape}")

    from fexchange.core.fermion import one_body_operator_matrix
    from fexchange.spectrum.lsms import build_lsms
    from fexchange.spectrum.lsjm import build_lsjm, select_soc_lowest_subspace
    from fexchange.utils.constants import N_ORB

    h_local_fock = one_body_operator_matrix(h_local, n_ele, N_ORB)

    lsms = build_lsms(n_ele, N_ORB, r42=0, r62=0)
    lsjm = build_lsjm(lsms, N_ORB)
    soc0 = select_soc_lowest_subspace(lsjm, F2=1, F4=0, F6=0, zeta=zeta)

    U_j0 = soc0["U_n_soc0"]
    h_local_j0 = U_j0.conj().T @ h_local_fock @ U_j0

    fock_norm = np.linalg.norm(h_local_fock)
    j0_weight = np.linalg.norm(h_local_j0) / fock_norm if fock_norm > 0 else 1.0

    dim = h_local_j0.shape[0]
    trace_avg = complex(np.trace(h_local_j0) / dim)
    h_j0_0 = h_local_j0 - trace_avg * np.eye(dim, dtype=complex)

    templates = _build_stevens_templates(soc0["J0"], symmetry=symmetry, mode_q3=mode_q3)
    names, coeffs, h_fit_0 = _real_least_squares_coeffs(templates, h_j0_0)
    h_fit = h_fit_0 + trace_avg * np.eye(dim, dtype=complex)

    eigvals = np.linalg.eigvalsh(h_local_j0)
    eigvals_fit = np.linalg.eigvalsh(h_fit)

    return {
        "alpha0": soc0["alpha0"],
        "L0": soc0["L0"],
        "S0": soc0["S0"],
        "J0": soc0["J0"],
        "n_j": soc0["n_j"],
        "j0_weight": j0_weight,
        "trace_avg_j0": trace_avg,
        "stevens_fit_residual": _relative_residual(h_j0_0, h_fit_0),
        "B_params": {name: float(coeffs[i]) for i, name in enumerate(names)},
        "H_local_j0": h_local_j0,
        "H_cef_j0": h_local_j0,
        "H_cef_j0_fit": h_fit,
        "eigvals_meV": eigvals,
        "eigvals_fit_meV": eigvals_fit,
        "cef_splitting_meV": eigvals - eigvals.min(),
        "cef_splitting_fit_meV": eigvals_fit - eigvals_fit.min(),
    }


def _format_scalar(value):
    """Format scalar output compactly for CLI summaries."""
    value = complex(value)
    if abs(value.imag) < 1e-10:
        return f"{value.real:+.6f}"
    return f"{value.real:+.6f}{value.imag:+.6f}j"


def _print_scheme_a(sa):
    print("=== Scheme A: orbital tensor decomposition ===")
    print(f"  residual: {sa['residual']:.3e}")
    print(f"  trace:    {sa['trace_avg'].real:.6f} meV")
    for (k, q), v in sorted(sa["a_kq"].items()):
        if abs(v) > 1e-10:
            print(f"  a({k},{q:+d}) = {_format_scalar(v)}")


def _print_scheme_b(sb, symmetry):
    print(f"=== Scheme B: Stevens fit ({symmetry}) ===")
    print(f"  ground: alpha={sb['alpha0']}, L={sb['L0']}, S={sb['S0']}, J0={sb['J0']}")
    print(f"  J0 weight: {sb['j0_weight']:.4f},  residual: {sb['stevens_fit_residual']:.3e}")
    for name, val in sb["B_params"].items():
        print(f"  {name} = {val:+.6f} meV")
    for i, e in enumerate(sb["cef_splitting_meV"]):
        print(f"  level {i}: {e:.4f} meV")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Decompose h_local into CEF + SOC")
    p.add_argument("--h_local_dat", required=True)
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--n_ele", type=int)
    p.add_argument("--zeta_soc", type=float)
    p.add_argument("--symmetry", choices=("Oh", "C3v"), default="Oh")
    p.add_argument("--mode_q3", choices=("cos", "sin"), default="cos")
    p.add_argument(
        "--direct_local_fit",
        action="store_true",
        help="Fit Stevens parameters directly in the SOC-lowest J0 manifold",
    )
    p.add_argument("--write_cef_config", type=str)
    args = p.parse_args()

    h_local = np.loadtxt(args.h_local_dat, dtype=complex)
    zeta, h_cef = decompose_h_local(h_local, rtol=args.rtol)

    out_path = args.h_local_dat.replace("h_local", "h_cef")
    if out_path == args.h_local_dat:
        out_path = args.h_local_dat.rsplit(".", 1)[0] + "_cef.dat"
    np.savetxt(out_path, h_cef, fmt="%.12f", delimiter="  ")

    print(f"=== SOC: zeta = {zeta:.6f} meV ===\n")
    _print_scheme_a(decompose_orbital_tensors(h_cef))

    if args.n_ele is not None:
        zeta_fit = args.zeta_soc or zeta
        if args.direct_local_fit:
            sb = extract_local_stevens_fit(
                h_local,
                n_ele=args.n_ele,
                zeta=zeta_fit,
                symmetry=args.symmetry,
                mode_q3=args.mode_q3,
            )
        else:
            sb = extract_cef_stevens_fit(
                h_cef, n_ele=args.n_ele,
                zeta=zeta_fit,
                symmetry=args.symmetry, mode_q3=args.mode_q3,
            )
        print()
        _print_scheme_b(sb, args.symmetry)

        if args.write_cef_config:
            lines = [
                f'point_group = "{args.symmetry}"',
                f"J = {sb['J0']}",
                f'mode_q3 = "{args.mode_q3}"',
                "",
            ]
            for name, val in sb["B_params"].items():
                lines.append(f"{name} = {val}")
            with open(args.write_cef_config, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"\nSaved config: {args.write_cef_config}")

    print(f"Saved h_cef: {out_path}")
