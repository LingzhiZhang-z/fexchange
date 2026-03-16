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

The literature transfer matrices included below are for A₂PrO₃ z bonds in
the cubic harmonic basis and are transformed here to the complex spherical
harmonic basis by a unitary change of basis.
"""

import numpy as np

_BASIS_CUBIC_F = ("xi", "eta", "zeta", "A", "alpha", "beta", "gamma")
_BASIS_COMPLEX_F = ("Y3,-3", "Y3,-2", "Y3,-1", "Y3,0", "Y3,1", "Y3,2", "Y3,3")

_SQRT6 = np.sqrt(6.0)
_SQRT10 = np.sqrt(10.0)
_SQRT15 = np.sqrt(15.0)

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


def to_complex_basis(h_cubic: np.ndarray) -> np.ndarray:
    """Transform a 7×7 matrix from cubic to complex spherical harmonic basis."""
    u = _build_U_c2y()
    return u @ h_cubic @ u.conj().T


def _build_hermitian_from_lower_triangle(rows: list[list[complex]]) -> np.ndarray:
    if len(rows) != 7:
        raise ValueError(f"expected 7 lower-triangle rows, got {len(rows)}")

    h = np.zeros((7, 7), dtype=complex)
    for i, row in enumerate(rows):
        if len(row) != i + 1:
            raise ValueError(f"row {i} must have length {i + 1}, got {len(row)}")
        for j, value in enumerate(row):
            h[i, j] = value
            if i != j:
                h[j, i] = np.conj(value)
    return h


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


def literature_a2pro3_transfer_cubic(a_site: str) -> np.ndarray:
    """Return the A₂PrO₃ z-bond nearest-neighbor transfer matrix in cubic basis."""
    key = a_site.strip().upper()
    if key not in _A2PRO3_TRANSFER_LOWER_TRIANGLES:
        allowed = ", ".join(sorted(_A2PRO3_TRANSFER_LOWER_TRIANGLES))
        raise ValueError(f"unknown A-site '{a_site}'. Expected one of: {allowed}")

    return _build_hermitian_from_lower_triangle(_A2PRO3_TRANSFER_LOWER_TRIANGLES[key])


def literature_a2pro3_transfer_complex(a_site: str) -> np.ndarray:
    """Return the A₂PrO₃ z-bond transfer matrix in complex spherical basis."""
    return to_complex_basis(literature_a2pro3_transfer_cubic(a_site))


def _format_matrix(matrix: np.ndarray) -> str:
    with np.printoptions(precision=4, suppress=True, linewidth=140):
        return np.array2string(np.asarray(matrix), separator=", ")


if __name__ == "__main__":
    import sys

    usage = (
        f"Usage:\n"
        f"  python {sys.argv[0]} <pf_sigma> <pf_pi> <delta_pf>\n"
        f"  python {sys.argv[0]} literature"
    )

    if len(sys.argv) == 2 and sys.argv[1] == "literature":
        for a_site in ("K", "Rb", "Cs"):
            h_cubic = literature_a2pro3_transfer_cubic(a_site)
            h_complex = literature_a2pro3_transfer_complex(a_site)
            print(f"t_lit_{a_site}_cubic:")
            print(_format_matrix(h_cubic))
            print()
            print(f"t_lit_{a_site}_complex:")
            print(_format_matrix(h_complex))
            if a_site != "Cs":
                print()
        sys.exit(0)

    if len(sys.argv) != 4:
        print(usage)
        sys.exit(1)

    try:
        pf_sigma = float(sys.argv[1])
        pf_pi = float(sys.argv[2])
        delta_pf = float(sys.argv[3])
    except ValueError:
        print(usage)
        sys.exit(1)

    # Geometry: f1=(0,0,0), f2=(1,-1,0), ligands at (1,0,0) and (0,-1,0),
    # with direction convention d = R_p - R_f for the two f-p-f paths.
    h_eff_cubic = (
        hopping_fpf(1, 0, 0, 0, 1, 0, pf_sigma, pf_pi, delta_pf)
        + hopping_fpf(0, -1, 0, -1, 0, 0, pf_sigma, pf_pi, delta_pf)
    )
    h_eff_complex = to_complex_basis(h_eff_cubic)

    print("t_eff_cubic:")
    print(_format_matrix(h_eff_cubic))
    print()
    print("t_eff_complex:")
    print(_format_matrix(h_eff_complex))
