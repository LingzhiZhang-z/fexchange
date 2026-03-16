"""
Independent Wannier90 extraction tool.

Extract hopping matrix (t_mu) and onsite Hamiltonian (h_local)
from Wannier90 hr.dat output.

Usage:
    # Programmatic
    from w90_extract import w90_extract
    result = w90_extract(
        hr_path="wannier90_hr.dat",
        n_orb_per_atom=[7, 7, 3, 3],
        f_sites=[(0, [0,0,0]), (1, [1,0,0])],
        ligands=[(2, [0,0,0]), (3, [0,0,0])],
    )

    # From config file
    from w90_extract import w90_extract_from_toml
    result = w90_extract_from_toml("config.toml")

    # Command line
    python w90_extract.py config.toml
"""

from __future__ import annotations

import sys
import tomllib
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Energy unit conversion
# ---------------------------------------------------------------------------

_ENERGY_SCALE = {
    "eV": 1.0,
    "meV": 1e-3,
    "Ha": 27.211386245988,
    "Ry": 13.605693122994,
}

# ---------------------------------------------------------------------------
# hr.dat parser
# ---------------------------------------------------------------------------

def read_hr(path):
    """
    Parse wannier90_hr.dat -> dict keyed by R-vector tuples.

    Returns (H_R, num_wann) where H_R[(r1,r2,r3)] is a (num_wann, num_wann)
    complex matrix, already divided by degeneracy weight.
    """
    path = str(path)
    num_wann = int(np.loadtxt(path, skiprows=1, max_rows=1))
    nrpts = int(np.loadtxt(path, skiprows=2, max_rows=1))

    n_wt_lines = nrpts // 15
    weights = np.loadtxt(path, skiprows=3, max_rows=n_wt_lines).flatten() if n_wt_lines > 0 else np.array([])
    if nrpts % 15 != 0:
        extra = np.atleast_1d(np.loadtxt(path, skiprows=3 + n_wt_lines, max_rows=1))
        weights = np.concatenate((weights, extra))
        n_wt_lines += 1

    raw = np.loadtxt(path, skiprows=3 + n_wt_lines)
    block = num_wann * num_wann

    H_R = {}
    for i in range(nrpts):
        rows = raw[block * i : block * (i + 1)]
        R = tuple(rows[0, 0:3].astype(int))
        m_idx = rows[:, 3].astype(int) - 1
        n_idx = rows[:, 4].astype(int) - 1
        mat = np.zeros((num_wann, num_wann), dtype=complex)
        mat[m_idx, n_idx] = (rows[:, 5] + 1j * rows[:, 6]) / weights[i]
        H_R[R] = mat

    return H_R, num_wann


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def _extract_block(H_R, rows, cols, R):
    """Extract sub-block H_R[R][rows, cols] with Hermitian fallback."""
    R = tuple(int(x) for x in R)
    if R in H_R:
        return H_R[R][np.ix_(rows, cols)].copy()
    R_neg = tuple(-x for x in R)
    if R_neg in H_R:
        return H_R[R_neg][np.ix_(cols, rows)].conj().T.copy()
    raise KeyError(f"R-vector {R} not found in H_R (no Hermitian pair either)")


def _expand_to_spinor(mat):
    """
    Expand orbital-only matrix (n x n) to spinor (2n x 2n).
    Interleaved order: (m0_dn, m0_up, m1_dn, m1_up, ...).
    spin-up = mat, spin-down = conj(mat), spin-flip = 0.

    Note: the conjugation for spin-down is harmless for nsoc (real-harmonic
    matrices are real), but is kept for consistency with reference literature
    results that depend on this convention.
    """
    n = mat.shape[0]
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    for i in range(n):
        for j in range(n):
            out[2 * i, 2 * j] = mat[i, j].conj()        # spin down
            out[2 * i + 1, 2 * j + 1] = mat[i, j]       # spin up
    return out


def _build_U_r2c(spinor=False):
    """
    Real-harmonic -> complex-harmonic unitary for f orbitals.
    Real order: m = 0, 1, -1, 2, -2, 3, -3.
    Complex order: m = -3, -2, -1, 0, 1, 2, 3.
    """
    sq2 = np.sqrt(2.0)
    tmp = np.array([
        [0,       0,       0,       1, 0,        0,       0      ],
        [0,       0,       1/sq2,   0, -1/sq2,   0,       0      ],
        [0,       0,       1j/sq2,  0, 1j/sq2,   0,       0      ],
        [0,       1/sq2,   0,       0, 0,         1/sq2,  0      ],
        [0,       1j/sq2,  0,       0, 0,        -1j/sq2, 0      ],
        [1/sq2,   0,       0,       0, 0,         0,      -1/sq2 ],
        [1j/sq2,  0,       0,       0, 0,         0,      1j/sq2 ],
    ], dtype=complex)
    return np.kron(tmp.T, np.eye(2)) if spinor else tmp.T

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

def _ud_to_du_perm(dim):
    if dim % 2 != 0:
        raise ValueError("ud->du permutation requires even dimension")
    n = dim // 2
    return [idx for m in range(n) for idx in (2 * m + 1, 2 * m)]


def _permute_ud_to_du(mat):
    """Reorder Wannier90 spinor from (up,down) per m to internal (down,up) per m."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] % 2 != 0:
        raise ValueError("ud->du permutation requires an even square matrix")
    perm = _ud_to_du_perm(mat.shape[0])
    return mat[np.ix_(perm, perm)].copy()


def _extract_w90_blocks(H_R, n_orb_per_atom, f_sites, ligands):
    """
    Extract raw blocks from Wannier90 H_R data.

    Returns (h_local_raw, t_ff, ligand_hops, eps_i, eps_j) where:
    - h_local_raw: (f_dim, f_dim) averaged onsite block
    - t_ff: (f_dim, f_dim) direct f-f hopping
    - ligand_hops: list of (t_i, t_j, eps_l) tuples per ligand
    - eps_i, eps_j: (f_dim,) onsite diagonal energies
    All in original Wannier90 ordering (no spin permutation).
    """
    offsets = [0]
    for n in n_orb_per_atom:
        offsets.append(offsets[-1] + n)

    def orb(atom):
        return list(range(offsets[atom], offsets[atom + 1]))

    (ai, ci), (aj, cj) = f_sites
    fi, fj = orb(ai), orb(aj)
    if len(fi) not in (7, 14):
        raise ValueError(
            f"f-site {ai} has {len(fi)} orbitals; expected 7 (nsoc) or 14 (soc)")
    if len(fj) not in (7, 14):
        raise ValueError(
            f"f-site {aj} has {len(fj)} orbitals; expected 7 (nsoc) or 14 (soc)")
    R_ij = tuple(int(cj[k]) - int(ci[k]) for k in range(3))

    H0 = H_R[(0, 0, 0)]
    eps = np.real(np.diag(H0))

    h_local_raw = 0.5 * (H0[np.ix_(fi, fi)] + H0[np.ix_(fj, fj)])
    t_ff = _extract_block(H_R, fi, fj, R_ij)

    eps_i = eps[fi]
    eps_j = eps[fj]

    ligand_hops = []
    for lig_atom, lig_cell in (ligands or []):
        li = orb(lig_atom)
        R_io = tuple(int(lig_cell[k]) - int(ci[k]) for k in range(3))
        R_jo = tuple(int(lig_cell[k]) - int(cj[k]) for k in range(3))
        t_i = _extract_block(H_R, fi, li, R_io)
        t_j = _extract_block(H_R, fj, li, R_jo)
        ligand_hops.append((t_i, t_j, eps[li]))

    return h_local_raw, t_ff, ligand_hops, eps_i, eps_j


def _compute_fpf(t_ff, ligand_hops, eps_i, eps_j):
    """
    Combine direct f-f hopping with f-p-f second-order corrections.

    Parameters
    ----------
    t_ff : (f_dim, f_dim) direct hopping matrix
    ligand_hops : list of (t_i, t_j, eps_l) from extraction phase
    eps_i, eps_j : (f_dim,) onsite diagonal energies for f-sites i, j

    Returns
    -------
    t_mu_raw : (f_dim, f_dim) total hopping = t_ff + sum(t_lig)
    """
    f_dim = t_ff.shape[0]
    t_lig = np.zeros((f_dim, f_dim), dtype=complex)

    for t_i, t_j, eps_l in ligand_hops:
        du = eps_i[:, None] - eps_l[None, :]
        dv = eps_j[:, None] - eps_l[None, :]

        denom = du[:, None, :] + dv[None, :, :]
        safe = np.abs(denom) > 1e-12
        n_degen = int(np.sum(~safe))
        if n_degen > 0:
            warnings.warn(
                f"_compute_fpf: {n_degen} near-degenerate denominator(s) "
                f"dropped (|eps_i + eps_j - 2*eps_l| < 1e-12)")
        delta = np.where(safe,
                         2.0 * du[:, None, :] * dv[None, :, :] / np.where(safe, denom, 1.0),
                         np.inf)

        numer = t_i[:, None, :] * np.conj(t_j[None, :, :])
        t_lig += np.sum(numer / delta, axis=2)

    return t_ff + t_lig


def _finalize(t_mu_raw, h_local_raw, is_soc):
    """
    Apply spin expansion (without_soc only) and real→complex basis transformation.

    Assumes input is already a full spinor matrix (is_soc=True) or an
    orbital-only matrix that needs expanding (is_soc=False).

    Returns (t_mu, h_local), both 14×14 complex.
    """
    if not is_soc:
        t_mu_raw = _expand_to_spinor(t_mu_raw)
        h_local_raw = _expand_to_spinor(h_local_raw)

    U = _build_U_r2c(spinor=True)
    return U @ t_mu_raw @ U.T.conj(), U @ h_local_raw @ U.T.conj()


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def w90_extract(hr_path, n_orb_per_atom, f_sites, ligands=None,
                spinor=False, energy_unit="eV"):
    """
    Extract hopping matrix and onsite Hamiltonian from Wannier90 hr.dat.
    """
    if energy_unit not in _ENERGY_SCALE:
        raise ValueError(f"Unknown energy_unit: {energy_unit!r}. Choose from {list(_ENERGY_SCALE)}")

    H_R, num_wann = read_hr(hr_path)
    scale = _ENERGY_SCALE[energy_unit]
    if scale != 1.0:
        H_R = {R: scale * mat for R, mat in H_R.items()}

    spin_mult = 2 if spinor else 1
    norbs = [ n_orb * spin_mult for n_orb in n_orb_per_atom ]
    # Phase 1: Extract raw blocks from Wannier90 data
    h_local_raw, t_ff, ligand_hops, eps_i, eps_j = _extract_w90_blocks(
        H_R, norbs, f_sites, ligands)

    # Phase 2: Combine direct + second-order hopping
    t_mu_raw = _compute_fpf(t_ff, ligand_hops, eps_i, eps_j)

    # Wannier90 SOC convention: reorder spinor from (up,down) to internal (down,up)
    if spinor:
        t_mu_raw = _permute_ud_to_du(t_mu_raw)
        h_local_raw = _permute_ud_to_du(h_local_raw)

    # Remove trace from h_local (pure CEF + SOC, no constant shift)
    n = h_local_raw.shape[0]
    h_local_raw -= np.eye(n) * (np.trace(h_local_raw) / n)

    # Phase 3: Spin expansion (without_soc) + basis transformation
    t_mu, h_local = _finalize(t_mu_raw, h_local_raw, spinor)

    return {"t_mu": t_mu, "h_local": h_local}


# ---------------------------------------------------------------------------
# TOML config entry point
# ---------------------------------------------------------------------------

def w90_extract_from_toml(toml_path):
    """Load config from TOML file, extract, and save results as txt."""
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    fi = cfg["f_site_i"]
    fj = cfg["f_site_j"]
    f_sites = [(fi["atom"], fi["cell"]), (fj["atom"], fj["cell"])]

    ligands = None
    if "ligands" in cfg:
        ligands = [(lig["atom"], lig["cell"]) for lig in cfg["ligands"]]

    result = w90_extract(
        hr_path=cfg["hr_path"],
        n_orb_per_atom=cfg["n_orb_per_atom"],
        f_sites=f_sites,
        ligands=ligands,
        spinor=cfg.get("spinor", False),
        energy_unit=cfg.get("energy_unit", "eV"),
    )

    prefix = cfg.get("output", "w90_output")
    is_spinor = cfg.get("spinor", False)
    U_r2c = _build_U_r2c(spinor=is_spinor)
    U_c2y = _build_U_c2y(spinor=is_spinor)

    t_meV = result["t_mu"] * 1000
    h_meV = result["h_local"] * 1000

    np.savetxt(f"{prefix}_t_mu.dat", t_meV, fmt="%.12f")
    np.savetxt(f"{prefix}_t_mu_real.dat", U_r2c.conj().T @ t_meV @ U_r2c, fmt="%.12f")
    np.savetxt(f"{prefix}_t_mu_cubic.dat", U_c2y.conj().T @ t_meV @ U_c2y, fmt="%.12f")

    np.savetxt(f"{prefix}_h_local.dat", h_meV, fmt="%.12f")
    np.savetxt(f"{prefix}_h_local_real.dat", U_r2c.conj().T @ h_meV @ U_r2c, fmt="%.12f")
    np.savetxt(f"{prefix}_h_local_cubic.dat", U_c2y.conj().T @ h_meV @ U_c2y, fmt="%.12f")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python w90_extract.py config.toml")
        sys.exit(1)
    w90_extract_from_toml(sys.argv[1])
