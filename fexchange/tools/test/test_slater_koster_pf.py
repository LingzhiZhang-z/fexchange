"""
Symmetry tests for Slater-Koster p-f hopping integrals.

Verification strategy: the hopping matrix H(d) must transform correctly
under all generators of O_h.  For any point-group operation R:

    H(R·d) = D_p(R) · H(d) · D_f(R)^T

where D_p (3×3) and D_f (7×7) are the representation matrices of R
acting on p and f cubic harmonics respectively.  These matrices are
read directly from the cubic harmonic definitions and are completely
independent of the Wigner rotation derivation.
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from slater_koster_pf import hopping_pf

PF_SIGMA, PF_PI = 1.7, -0.9
N_RANDOM = 1000
TOL = 1e-13


def _random_directions(n, seed=42):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


# ---------------------------------------------------------------------------
# Representation matrices D_f(R) for f cubic harmonics
# Ordering: ξ=0, η=1, ζ=2, A=3, α=4, β=5, γ=6
# D_f[i', j] = coefficient of f_{i'} in R · f_j
# ---------------------------------------------------------------------------

def _D_f_C3_111():
    """C₃ along [111]: x→y→z→x.  ξ→η→ζ→ξ, α→β→γ→α, A→A."""
    D = np.zeros((7, 7))
    D[1, 0] = 1   # ξ → η
    D[2, 1] = 1   # η → ζ
    D[0, 2] = 1   # ζ → ξ
    D[3, 3] = 1   # A → A
    D[5, 4] = 1   # α → β
    D[6, 5] = 1   # β → γ
    D[4, 6] = 1   # γ → α
    return D


def _D_f_C4z():
    """C₄z (90° about z): x→y, y→−x, z→z."""
    D = np.zeros((7, 7))
    D[1, 0] = -1   # ξ = x(y²−z²) → y(x²−z²) = −η
    D[0, 1] = 1    # η = y(z²−x²) → (−x)(z²−y²) = x(y²−z²) = ξ
    D[2, 2] = -1   # ζ = z(x²−y²) → z(y²−x²) = −ζ
    D[3, 3] = -1   # A = xyz → y(−x)z = −A
    D[5, 4] = 1    # α = x(5x²−3r²) → y(5y²−3r²) = β
    D[4, 5] = -1   # β = y(5y²−3r²) → (−x)(5x²−3r²) = −α
    D[6, 6] = 1    # γ = z(5z²−3r²) → γ
    return D


def _D_f_C4x():
    """C₄x (90° about x): y→z, z→−y, x→x."""
    D = np.zeros((7, 7))
    D[0, 0] = -1   # ξ = x(y²−z²) → x(z²−y²) = −ξ
    D[2, 1] = -1   # η = y(z²−x²) → z(y²−x²) = −z(x²−y²) = −ζ
    D[1, 2] = 1    # ζ = z(x²−y²) → (−y)(x²−z²) = y(z²−x²) = η
    D[3, 3] = -1   # A = xyz → x·z·(−y) = −A
    D[4, 4] = 1    # α → α
    D[6, 5] = 1    # β = y(5y²−3r²) → z(5z²−3r²) = γ
    D[5, 6] = -1   # γ = z(5z²−3r²) → (−y)(5y²−3r²) = −β
    return D


def _D_f_mirror_xy():
    """Mirror σ_xy: z→−z, x→x, y→y."""
    return np.diag([1., 1., -1., -1., 1., 1., -1.])


# ---------------------------------------------------------------------------
# Representation matrices D_p(R) for p cubic harmonics (= rotation on xyz)
# ---------------------------------------------------------------------------

def _D_p_C3_111():
    return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

def _D_p_C4z():
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

def _D_p_C4x():
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

def _D_p_mirror_xy():
    return np.diag([1., 1., -1.])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_SYMMETRY_OPS = [
    ("C3_111",    _D_p_C3_111,    _D_f_C3_111,    lambda l, m, n: (n, l, m)),
    ("C4z",       _D_p_C4z,       _D_f_C4z,       lambda l, m, n: (-m, l, n)),
    ("C4x",       _D_p_C4x,       _D_f_C4x,       lambda l, m, n: (l, -n, m)),
    ("mirror_xy", _D_p_mirror_xy, _D_f_mirror_xy, lambda l, m, n: (l, m, -n)),
]


@pytest.mark.parametrize("name,Dp_fn,Df_fn,rot_d", _SYMMETRY_OPS)
def test_point_group_symmetry(name, Dp_fn, Df_fn, rot_d):
    """H(R·d) = D_p(R) · H(d) · D_f(R)^T for each generator of O_h."""
    Dp = Dp_fn()
    Df = Df_fn()
    max_err = 0.0
    for d in _random_directions(N_RANDOM):
        l, m, n = d
        H = hopping_pf(l, m, n, PF_SIGMA, PF_PI)
        lr, mr, nr = rot_d(l, m, n)
        H_rot = hopping_pf(lr, mr, nr, PF_SIGMA, PF_PI)
        H_pred = Dp @ H @ Df.T
        max_err = max(max_err, np.max(np.abs(H_rot - H_pred)))
    assert max_err < TOL, f"{name}: max err {max_err:.2e}"


def test_orthogonality():
    """When all SK params = 1, each p row satisfies Σ_λ |E_{p,λ}|² = 1."""
    max_err = 0.0
    for d in _random_directions(N_RANDOM):
        H = hopping_pf(*d, 1.0, 1.0)
        for i in range(3):
            err = abs(np.sum(H[i, :] ** 2) - 1.0)
            max_err = max(max_err, err)
    assert max_err < TOL, f"orthogonality: max err {max_err:.2e}"


def test_known_values_bond_along_z():
    """For bond along z, only σ channel couples p_z to f_γ."""
    H = hopping_pf(0, 0, 1, 1.0, 0.0)
    # p_z (row 2) to f_γ (col 6): pure σ, coefficient = 1
    assert abs(H[2, 6] - 1.0) < TOL
    # p_x, p_y have zero σ coupling to any f orbital along z
    assert np.max(np.abs(H[0, :])) < TOL
    assert np.max(np.abs(H[1, :])) < TOL


def test_known_values_bond_along_x():
    """For bond along x, only σ channel couples p_x to f_α."""
    H = hopping_pf(1, 0, 0, 1.0, 0.0)
    assert abs(H[0, 4] - 1.0) < TOL
    assert np.max(np.abs(H[1, :])) < TOL
    assert np.max(np.abs(H[2, :])) < TOL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
