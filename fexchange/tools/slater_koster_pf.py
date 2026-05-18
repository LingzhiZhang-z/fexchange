"""
Slater-Koster p-f and effective f-p-f hopping integrals in the cubic basis.

Reference: Takegahara, Aoki & Yanase, J. Phys. C 13 (1980) 583.

f-orbital ordering: ξ, η, ζ, A₂ᵤ, α, β, γ
    ξ = T₂ᵤ x(y²−z²)
    η = T₂ᵤ y(z²−x²)
    ζ = T₂ᵤ z(x²−y²)
    A = A₂ᵤ  xyz
    α = T₁ᵤ x(5x²−3r²)
    β = T₁ᵤ y(5y²−3r²)
    γ = T₁ᵤ z(5z²−3r²)

p-orbital ordering: x, y, z
complex f ordering: Y₃,-3, Y₃,-2, Y₃,-1, Y₃,0, Y₃,1, Y₃,2, Y₃,3

The primary purpose of this module is to provide the general Slater-Koster
formulae:
    - hopping_pf(...)
    - hopping_fpf(...)

The PRB A₂PrO₃ transfer matrices are included as a literature preset built on
top of those conventions. Command-line outputs use the fexchange text-block
format, i.e. ``[t_mu]`` / ``[t_*]`` headers followed by ``real imag`` rows.

Usage:
    python slater_koster_pf.py --pf_sigma 1.0 --ratio 0.3 --delta_pf 2.0 --output t_mu.txt
    python slater_koster_pf.py literature
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_BASIS_CUBIC_F = ("xi", "eta", "zeta", "A", "alpha", "beta", "gamma")
_BASIS_COMPLEX_F = ("Y3,-3", "Y3,-2", "Y3,-1", "Y3,0", "Y3,1", "Y3,2", "Y3,3")

_SQRT6 = np.sqrt(6.0)
_SQRT10 = np.sqrt(10.0)
_SQRT15 = np.sqrt(15.0)


# ---------------------------------------------------------------------------
# Core Slater-Koster p-f / f-p-f formulae
# ---------------------------------------------------------------------------

def _eval_px(l: float, m: float, n: float) -> tuple[np.ndarray, np.ndarray]:
    """σ and π coefficient arrays for E_{px, f_λ}, shape (7,) each.

    Returns (sigma_coeffs, pi_coeffs) in f-orbital order [ξ,η,ζ,A,α,β,γ].
    """
    l2, m2, n2 = l * l, m * m, n * n

    sigma = np.array([
        0.5 * _SQRT15 * l2 * (m2 - n2),            # ξ
        0.5 * _SQRT15 * l * m * (n2 - l2),         # η
        0.5 * _SQRT15 * l * n * (l2 - m2),         # ζ
        _SQRT15 * l2 * m * n,                       # A
        0.5 * l2 * (5.0 * l2 - 3.0),              # α
        0.5 * l * m * (5.0 * m2 - 3.0),            # β
        0.5 * l * n * (5.0 * n2 - 3.0),            # γ
    ])

    pi = np.array([
        _SQRT10 / 4.0 * (n2 - m2) * (3.0 * l2 - 1.0),                   # ξ
        -_SQRT10 / 4.0 * l * m * (3.0 * (n2 - l2) + 2.0),               # η
        -_SQRT10 / 4.0 * l * n * (3.0 * (l2 - m2) - 2.0),               # ζ
        -_SQRT10 / 2.0 * m * n * (3.0 * l2 - 1.0),                      # A
        _SQRT6 / 4.0 * (1.0 - l2) * (5.0 * l2 - 1.0),                  # α
        -_SQRT6 / 4.0 * l * m * (5.0 * m2 - 1.0),                       # β
        -_SQRT6 / 4.0 * l * n * (5.0 * n2 - 1.0),                       # γ
    ])

    return sigma, pi


# C₃ rotation along [111]:  x → y → z → x,  (l,m,n) → (n,l,m).
#
# All cubic harmonics cycle cleanly under C₃ (proper rotation, no extra signs):
#   T₂ᵤ:  ξ → η → ζ → ξ        T₁ᵤ:  α → β → γ → α        A₂ᵤ:  A → A
#
# Physical invariance of the two-centre integral:
#   E_{px, f_j}(l,m,n) = E_{py, cyc(j)}(n,l,m) = E_{pz, cyc²(j)}(m,n,l)
#
# Rearranging:
#   E_{py, f_j}(l,m,n) = E_{px, cyc⁻¹(j)}(m, n, l)     — no sign
#   E_{pz, f_j}(l,m,n) = E_{px, cyc⁻²(j)}(n, l, m)     — no sign
#
# Indices: ξ=0, η=1, ζ=2, A=3, α=4, β=5, γ=6

_CYC_INV = np.array([2, 0, 1, 3, 6, 4, 5], dtype=int)   # cyc⁻¹: ξ←ζ, η←ξ, ζ←η, A←A, α←γ, β←α, γ←β
_CYC_INV2 = np.array([1, 2, 0, 3, 5, 6, 4], dtype=int)  # cyc⁻²: ξ←η, η←ζ, ζ←ξ, A←A, α←β, β←γ, γ←α




def hopping_pf(l: float, m: float, n: float,
               pf_sigma: float, pf_pi: float) -> np.ndarray:
    """Compute the 3×7 p-f hopping matrix H_{pf}(l,m,n).

    Parameters
    ----------
    l, m, n : float
        Direction cosines of the bond vector (must satisfy l²+m²+n²=1).
    pf_sigma, pf_pi : float
        Slater-Koster two-centre integral parameters (pfσ) and (pfπ).

    Returns
    -------
    H : ndarray, shape (3, 7)
        Hopping matrix.  Rows = p_x, p_y, p_z.
        Cols = f_ξ, f_η, f_ζ, f_A₂ᵤ, f_α, f_β, f_γ.
    """
    H = np.empty((3, 7))

    # --- Row 0: p_x ---
    s0, p0 = _eval_px(l, m, n)
    H[0, :] = s0 * pf_sigma + p0 * pf_pi

    # --- Row 1: p_y  (C₃ once: evaluate p_x at (m,n,l), permute f indices) ---
    s1, p1 = _eval_px(m, n, l)
    H[1, :] = s1[_CYC_INV] * pf_sigma + p1[_CYC_INV] * pf_pi

    # --- Row 2: p_z  (C₃ twice: evaluate p_x at (n,l,m), permute f indices) ---
    s2, p2 = _eval_px(n, l, m)
    H[2, :] = s2[_CYC_INV2] * pf_sigma + p2[_CYC_INV2] * pf_pi

    return H


def hopping_fpf(
    l1: float,
    m1: float,
    n1: float,
    l2: float,
    m2: float,
    n2: float,
    pf_sigma: float,
    pf_pi: float,
    delta_pf: float,
) -> np.ndarray:
    """Compute the effective 7×7 f-p-f hopping in the cubic basis."""
    h_pf1 = hopping_pf(l1, m1, n1, pf_sigma, pf_pi)
    h_pf2 = hopping_pf(l2, m2, n2, pf_sigma, pf_pi)
    return h_pf1.conj().T @ h_pf2 / delta_pf


def fpf_cubic_7x7(
    pf_sigma: float,
    pf_pi: float,
    delta_pf: float,
) -> np.ndarray:
    """Effective 7x7 cubic hopping for the A2PrO3 z bond geometry."""
    return (
        hopping_fpf(1, 0, 0, 0, 1, 0, pf_sigma, pf_pi, delta_pf)
        + hopping_fpf(0, -1, 0, -1, 0, 0, pf_sigma, pf_pi, delta_pf)
    )


# ---------------------------------------------------------------------------
# Basis transforms and spinor lifting
# ---------------------------------------------------------------------------

def _build_U_c2y(spinor=False):
    """
    Cubic-harmonic -> complex-harmonic unitary for f orbitals.
    Cubic order (rows): ξ, η, ζ, A, α, β, γ.
    Complex order (cols): m = -3, -2, -1, 0, 1, 2, 3.

    Each row is a cubic orbital expressed in Y_lm (Takegahara 1980 Table 1).
    """
    sq2 = np.sqrt(2.0)
    sq3 = np.sqrt(3.0)
    sq5 = np.sqrt(5.0)
    tmp = np.array([
        #  m=-3        m=-2    m=-1        m=0   m=+1        m=+2    m=+3
        [-sq3/4,    0,     -sq5/4,    0,    sq5/4,    0,      sq3/4  ],  # ξ = (√3 Y₃₃+√5 Y₃₁−√5 Y₃₋₁−√3 Y₃₋₃)/4
        [-1j*sq3/4, 0,   1j*sq5/4,   0,   1j*sq5/4,  0,  -1j*sq3/4 ],  # η = i(−√3 Y₃₃+√5 Y₃₁+√5 Y₃₋₁−√3 Y₃₋₃)/4
        [ 0,      1/sq2,   0,        0,    0,       1/sq2,   0       ],  # ζ = (Y₃₋₂+Y₃₂)/√2
        [ 0,    1j/sq2,    0,        0,    0,      -1j/sq2,  0       ],  # A = i(Y₃₋₂−Y₃₂)/√2
        [ sq5/4,    0,    -sq3/4,    0,    sq3/4,    0,     -sq5/4   ],  # α = (−√5 Y₃₃+√3 Y₃₁−√3 Y₃₋₁+√5 Y₃₋₃)/4
        [-1j*sq5/4, 0,  -1j*sq3/4,  0,  -1j*sq3/4,  0,  -1j*sq5/4  ],  # β = i(−√5 Y₃₃−√3 Y₃₁−√3 Y₃₋₁−√5 Y₃₋₃)/4
        [ 0,        0,    0,         1,   0,          0,    0         ],  # γ = Y₃₀
    ], dtype=complex)
    return np.kron(tmp.T, np.eye(2)) if spinor else tmp.T


def _expand_to_spinor_interleaved(mat: np.ndarray) -> np.ndarray:
    """Expand a 7x7 orbital matrix to 14x14 interleaved (dn,up) spinor form."""
    n = mat.shape[0]
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        for j in range(n):
            out[2 * i, 2 * j] = mat[i, j]
            out[2 * i + 1, 2 * j + 1] = mat[i, j]
    return out


def to_complex_basis_spinor(h_cubic_spinor: np.ndarray) -> np.ndarray:
    """Transform a 14x14 spinor matrix from cubic to complex spherical basis."""
    u = _build_U_c2y(spinor=True)
    return u @ h_cubic_spinor @ u.conj().T


def fpf_complex_spinor_14x14(
    pf_sigma: float,
    pf_pi: float,
    delta_pf: float,
) -> np.ndarray:
    """Effective 14x14 f-f hopping in the fexchange complex spinor basis."""
    cubic_7x7 = fpf_cubic_7x7(pf_sigma, pf_pi, delta_pf)
    return to_complex_basis_spinor(_expand_to_spinor_interleaved(cubic_7x7))


def expand_pf_to_spinor_interleaved(h_pf: np.ndarray) -> np.ndarray:
    """Expand a 3x7 p-f hopping block to 6x14 interleaved spinor layout."""
    out = np.zeros((6, 14), dtype=complex)
    for p in range(3):
        for f in range(7):
            out[2 * p, 2 * f] = h_pf[p, f]
            out[2 * p + 1, 2 * f + 1] = h_pf[p, f]
    return out


def fp_complex_spinor_block(
    l: float,
    m: float,
    n: float,
    pf_sigma: float,
    pf_pi: float,
) -> np.ndarray:
    """Return a 14x6 ``t_f_lig`` block for the downfold input contract.

    The ligand side remains in the raw p spinor order used by
    ``hopping_pf``: px, py, pz with interleaved spins. The f side is converted
    to the fexchange complex spinor basis.
    """
    h_pf = hopping_pf(l, m, n, pf_sigma, pf_pi)
    t_f_lig_cubic = expand_pf_to_spinor_interleaved(h_pf).conj().T
    return _build_U_c2y(spinor=True) @ t_f_lig_cubic


def slater_koster_hopping_fp_blocks(
    pf_sigma: float,
    pf_pi: float,
) -> list[tuple[str, np.ndarray]]:
    """Build f-p hopping blocks for the A2PrO3 z-bond downfold input."""
    hopping_fp = [
        ("t_f1_lig1", fp_complex_spinor_block(1, 0, 0, pf_sigma, pf_pi)),
        ("t_f1_lig2", fp_complex_spinor_block(0, -1, 0, pf_sigma, pf_pi)),
        ("t_f2_lig1", fp_complex_spinor_block(0, 1, 0, pf_sigma, pf_pi)),
        ("t_f2_lig2", fp_complex_spinor_block(-1, 0, 0, pf_sigma, pf_pi)),
    ]
    return hopping_fp


def write_blocks_txt(path: str | Path, blocks: list[tuple[str, np.ndarray]]) -> Path:
    """Write fexchange multi-block text: ``[key]`` followed by ``real imag`` rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, mat in blocks:
        lines.append(f"[{key}]")
        flat = np.asarray(mat, dtype=complex).reshape(-1)
        lines.extend(f"{val.real:.12e} {val.imag:.12e}" for val in flat)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_fexchange_t_mu(path: str | Path, t_mu: np.ndarray) -> Path:
    """Write a SOPT-compatible ``[t_mu]`` text block."""
    return write_blocks_txt(path, [("t_mu", np.asarray(t_mu, dtype=complex))])


_A2PRO3_TRANSFER_LOWER_TRIANGLES = {
    "K": [
        [12.3],
        [-7.16 + 0.15j, 12.3],
        [-1.04 + 0.12j, -1.04 + 0.12j, -84.1],
        [5.30 - 0.09j, -5.30 - 0.09j, 0.02 + 0.68j, -30.0],
        [-55.9 + 0.17j, 12.4 + 0.25j, 20.5 - 0.27j, -10.6 + 0.22j, 123.0],
        [-12.4 + 0.25j, 55.9 + 0.17j, -20.5 - 0.27j, -10.6 - 0.22j, -49.1 - 0.88j, 123.0],
        [3.21 + 0.49j, -3.21 + 0.49j, -0.01 - 0.17j, -7.32, 7.53 + 0.53j, 7.53 - 0.53j, 44.2],
    ],
    "RB": [
        [9.15],
        [-11.5 + 0.09j, 9.15],
        [-0.89 - 0.21j, -0.89 + 0.21j, -79.4],
        [8.48 - 0.03j, -8.48 - 0.03j, -0.63j, -22.5],
        [-47.1 + 0.06j, 12.1 + 0.24j, 35.0 - 0.34j, -14.3 + 0.14j, 110.0],
        [-12.1 + 0.24j, 47.1 + 0.06j, -35.0 - 0.34j, -14.3 - 0.14j, -46.7 - 0.68j, 110.0],
        [5.98 + 0.49j, -5.98 + 0.49j, 0.02j, -1.25, 11.2 + 0.55j, 11.2 - 0.55j, 40.3],
    ],
    "CS": [
        [5.22],
        [-16.04 + 0.04j, 5.22],
        [-0.09 - 0.34j, -0.09 + 0.34j, -68.2],
        [12.1 + 0.05j, -12.1 + 0.05j, -0.54j, -12.5],
        [-37.3 + 0.05j, 9.46 + 0.26j, 50.6 - 0.36j, -16.5 + 0.10j, 91.6],
        [-9.46 + 0.26j, 37.3 + 0.05j, -50.6 - 0.36j, -16.5 - 0.10j, -35.4 - 0.42j, 91.6],
        [7.80 + 0.45j, -7.80 + 0.45j, 0.23j, 5.20, 16.4 + 0.48j, 16.4 - 0.48j, 33.1],
    ],
}


def _hermitian_from_lower_triangle(rows: list[list[complex]]) -> np.ndarray:
    h = np.zeros((7, 7), dtype=complex)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            h[i, j] = value
            if i != j:
                h[j, i] = np.conj(value)
    return h


def _write_literature(output: str | None) -> None:
    base = Path(output) if output else Path("literature_t_mu")
    stem = base.stem if base.suffix else base.name
    labels = {"K": "K", "RB": "Rb", "CS": "Cs"}
    for key, label in labels.items():
        cubic_7x7 = _hermitian_from_lower_triangle(_A2PRO3_TRANSFER_LOWER_TRIANGLES[key])
        complex_14x14 = to_complex_basis_spinor(_expand_to_spinor_interleaved(cubic_7x7))
        write_fexchange_t_mu(base.parent / f"{stem}_{label}.txt", complex_14x14)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute z-bond Slater-Koster hopping integrals.")
    parser.add_argument("mode", nargs="?", choices=("literature",))
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--pf_sigma", type=float)
    parser.add_argument("--pf_pi", type=float)
    parser.add_argument(
        "--ratio",
        type=float,
        help="Set pf_pi = -ratio * pf_sigma when --pf_pi is omitted.",
    )
    parser.add_argument("--delta_pf", type=float)
    parser.add_argument("--downfold-hopping-fp", help="Write NSOC downfold f-p hopping blocks to this path.")

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.mode == "literature":
        _write_literature(args.output)
        return 0

    if args.pf_sigma is None or args.delta_pf is None:
        parser.error("parameter mode requires --pf_sigma and --delta_pf")
    if args.pf_pi is None:
        if args.ratio is None:
            parser.error("parameter mode requires --pf_pi or --ratio")
        args.pf_pi = -float(args.ratio) * float(args.pf_sigma)

    if args.downfold_hopping_fp:
        hopping_fp = slater_koster_hopping_fp_blocks(args.pf_sigma, args.pf_pi)
        write_blocks_txt(args.downfold_hopping_fp, hopping_fp)

    output = Path(args.output) if args.output else Path("./t_mu")
    complex_14x14 = fpf_complex_spinor_14x14(args.pf_sigma, args.pf_pi, args.delta_pf)
    target = output if output.suffix else output.with_suffix(".txt")
    write_fexchange_t_mu(target, complex_14x14)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
