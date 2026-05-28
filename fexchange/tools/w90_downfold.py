"""Second-order p-shell downfold: f-p hopping -> effective f-f hopping.

This tool follows ``standards/fopt/downfold.md``.  The p shell is closed in the
initial and final states.  For each ligand, the folded p part is written in the
hole basis

    |mu> = p_mu |Omega>,       |eta> = sum_mu Q[mu, eta] |mu>

with positive ligand-resolved p5 denominators

    E_p5(lig, eta) = Delta_lig + lambda_lig * c_eta.

The orbital-index propagator used by the f-only hopping is

    G_p[mu, nu] = sum_eta conj(Q[mu, eta]) * Q[nu, eta] / E_p5[eta].

The effective directed hopping is then

    t_eff = t_direct + sum_lig t_f1_lig @ G_p(lig) @ t_f2_lig.conj().T.

For NSOC, ``Q = I`` and ``G_p = I / Delta``.  For SOC, ``Q`` is the fixed
Clebsch-Gordan gauge in the basis ``m=-1,0,+1`` with interleaved ``down,up``
spin; equivalently it diagonalizes the hole SOC matrix ``-(L.S).T``.

Current I/O:
  direct_t_mu_in  optional block [t_mu] 14x14; omitted means zero direct term
  hopping_fp_in   blocks [t_f1_lig1] [t_f1_lig2] [t_f2_lig1] [t_f2_lig2]
  delta_lig1      positive p5 denominator center for ligand 1
  delta_lig2      positive p5 denominator center for ligand 2
  lambda_lig1     p-hole SOC scale for ligand 1, same unit as Delta
  lambda_lig2     p-hole SOC scale for ligand 2, same unit as Delta
  output          block [t_mu] 14x14 = direct + folded ligand correction
"""

from __future__ import annotations

import argparse
import logging
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.io.matrix import load_txt_blocks
from fexchange.utils.errors import InputError, IOError_, NumError, PhysError

logger = logging.getLogger("fexchange")

N_F_SPINOR: int = 14   # 2 × (2ℓ + 1) for ℓ = 3
N_P_SPINOR: int = 6    # 2 × (2ℓ + 1) for ℓ = 1


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_fp_blocks(path: Path) -> dict[tuple[str, str], NDArray[np.complexfloating]]:
    expected = {
        f"t_{f}_{lig}": N_F_SPINOR * N_P_SPINOR
        for f in ("f1", "f2")
        for lig in ("lig1", "lig2")
    }
    raw = load_txt_blocks(path, expected)
    return {
        (f, lig): raw[f"t_{f}_{lig}"].reshape(N_F_SPINOR, N_P_SPINOR)
        for f in ("f1", "f2")
        for lig in ("lig1", "lig2")
    }


def _load_t_mu_block(path: Path) -> NDArray[np.complexfloating]:
    raw = load_txt_blocks(path, {"t_mu": N_F_SPINOR * N_F_SPINOR})
    return raw["t_mu"].reshape(N_F_SPINOR, N_F_SPINOR)


# ---------------------------------------------------------------------------
# Fixed p-ligand model
# ---------------------------------------------------------------------------

def _p_angular_momentum_complex_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angular momentum matrices for p complex harmonics ordered m=-1,0,+1."""
    m_vals = np.array([-1.0, 0.0, 1.0], dtype=float)
    lz = np.diag(m_vals).astype(complex)
    l_plus = np.zeros((N_P_SPINOR // 2, N_P_SPINOR // 2), dtype=complex)
    ell = 1.0
    for col, m in enumerate(m_vals):
        m_to = m + 1.0
        if m_to > ell:
            continue
        row = int(np.where(np.isclose(m_vals, m_to))[0][0])
        l_plus[row, col] = np.sqrt(ell * (ell + 1.0) - m * (m + 1.0))
    l_minus = l_plus.conj().T
    lx = 0.5 * (l_plus + l_minus)
    ly = (l_plus - l_minus) / (2.0j)
    return lx, ly, lz


def _spin_half_down_up_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spin-1/2 matrices in the local spin order down, up."""
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, 1j], [-1j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=complex)
    return sx, sy, sz


def _p_l_dot_s_complex_basis() -> np.ndarray:
    lx, ly, lz = _p_angular_momentum_complex_basis()
    sx, sy, sz = _spin_half_down_up_basis()
    return np.kron(lx, sx) + np.kron(ly, sy) + np.kron(lz, sz)


def fixed_p_ligand_hamiltonian(delta_lig: float, lambda_p: float = 0.0) -> np.ndarray:
    """Return the electron p-ligand Hamiltonian in m=-1,0,+1, down/up order.

    This public helper is kept for checking the SOC convention.  The downfold
    path below uses the equivalent p5 hole propagator directly.
    """
    delta = float(delta_lig)
    lam = float(lambda_p)
    if not np.isfinite(delta) or not np.isfinite(lam):
        raise InputError(
            "FXE-INPUT-003",
            "fixed ligand parameters must be finite",
            actual={"delta_lig": delta, "lambda_p": lam},
        )
    l_dot_s = _p_l_dot_s_complex_basis()
    return -delta * np.eye(N_P_SPINOR, dtype=complex) + lam * l_dot_s


# ---------------------------------------------------------------------------
# p5 hole propagator
# ---------------------------------------------------------------------------

def _p5_hole_soc_analytic_q() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(c_eta, Q)`` for ``-(L.S).T`` in the fixed p-hole basis."""
    sq13 = np.sqrt(1.0 / 3.0)
    sq23 = np.sqrt(2.0 / 3.0)
    c_eta = np.array([-0.5, -0.5, -0.5, -0.5, 1.0, 1.0], dtype=float)
    q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, sq13, 0.0, 0.0, sq23, 0.0],
            [0.0, sq23, 0.0, 0.0, -sq13, 0.0],
            [0.0, 0.0, sq23, 0.0, 0.0, -sq13],
            [0.0, 0.0, sq13, 0.0, 0.0, sq23],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=complex,
    )
    return c_eta, q


def _p5_hole_propagator(
    delta_lig: float,
    lambda_p: float,
    *,
    degenerate_tol: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating], NDArray[np.floating]]:
    """Build ``G_p[mu,nu] = sum_eta conj(Q[mu,eta]) Q[nu,eta] / E_p5[eta]``."""
    delta = float(delta_lig)
    lam = float(lambda_p)
    if not np.isfinite(delta) or not np.isfinite(lam):
        raise InputError(
            "FXE-INPUT-003",
            "fixed ligand parameters must be finite",
            actual={"delta_lig": delta, "lambda_p": lam},
        )

    if abs(lam) == 0.0:
        c_eta = np.zeros(N_P_SPINOR, dtype=float)
        q = np.eye(N_P_SPINOR, dtype=complex)
    else:
        c_eta, q = _p5_hole_soc_analytic_q()

    e_p5 = delta + lam * c_eta
    negative = [int(i) for i, x in enumerate(e_p5) if x < 0.0]
    if negative:
        raise PhysError(
            "FXE-PHYS-001",
            "p5 downfold: negative p5 denominator",
            actual={
                "delta_lig": delta,
                "lambda_p": lam,
                "p5_energies": [float(x) for x in e_p5],
                "negative_indices": negative,
            },
        )
    if np.any(np.abs(e_p5) < degenerate_tol):
        bad_idx = [int(i) for i, x in enumerate(e_p5) if abs(x) < degenerate_tol]
        raise NumError(
            "FXE-NUM-001",
            "p5 downfold: p5 denominator below tolerance",
            expected={"degenerate_tol": float(degenerate_tol)},
            actual={
                "delta_lig": delta,
                "lambda_p": lam,
                "p5_energies": [float(x) for x in e_p5],
                "bad_indices": bad_idx,
            },
        )

    g_p = (q.conj() * (1.0 / e_p5)) @ q.T
    return g_p, e_p5, c_eta


# ---------------------------------------------------------------------------
# Block writer (multi-block [key] format)
# ---------------------------------------------------------------------------

def _write_blocks_txt(path: Path, blocks: list[tuple[str, NDArray[np.complexfloating]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for key, mat in blocks:
        flat = np.asarray(mat, dtype=complex).reshape(-1)
        out.append(f"[{key}]")
        out.extend(f"{val.real:.12e} {val.imag:.12e}" for val in flat)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def downfold_to_ff_fixed_ligand(
    *,
    hopping_fp_in: str | Path,
    output: str | Path,
    delta_lig1: float,
    delta_lig2: float,
    lambda_lig1: float,
    lambda_lig2: float,
    direct_t_mu_in: str | Path | None = None,
    degenerate_tol: float = 1e-6,
) -> dict[str, Any]:
    """Downfold with the fixed p5 hole propagator.

    This path does not read onsite input.  It uses independent positive
    ``Delta_1``/``Delta_2`` values and independent p-hole SOC splittings
    ``lambda_lig1 * c_eta`` / ``lambda_lig2 * c_eta``.
    If ``direct_t_mu_in`` is supplied, that direct f-f ``[t_mu]`` is added to
    the folded ligand correction; otherwise the direct term is zero.
    """
    fp_path = Path(hopping_fp_in)
    out_path = Path(output)
    if not fp_path.exists():
        raise IOError_("FXE-IO-001", f"hopping_fp_in missing: {fp_path}", paths={"path": str(fp_path)})

    t_fp = _load_fp_blocks(fp_path)
    delta_by_lig = {
        "lig1": float(delta_lig1),
        "lig2": float(delta_lig2),
    }
    lambda_by_lig = {
        "lig1": float(lambda_lig1),
        "lig2": float(lambda_lig2),
    }

    per_ligand: dict[str, dict[str, Any]] = {}
    direct_t_mu_path = None if direct_t_mu_in is None else Path(direct_t_mu_in)
    if direct_t_mu_path is None:
        t_direct = np.zeros((N_F_SPINOR, N_F_SPINOR), dtype=complex)
    else:
        if not direct_t_mu_path.exists():
            raise IOError_(
                "FXE-IO-001",
                f"direct_t_mu_in missing: {direct_t_mu_path}",
                paths={"path": str(direct_t_mu_path)},
            )
        t_direct = _load_t_mu_block(direct_t_mu_path)

    t_correction = np.zeros((N_F_SPINOR, N_F_SPINOR), dtype=complex)

    for lig in ("lig1", "lig2"):
        g_p, e_p5, c_eta = _p5_hole_propagator(
            delta_by_lig[lig],
            lambda_by_lig[lig],
            degenerate_tol=degenerate_tol,
        )
        per_ligand[lig] = {
            "delta_lig": delta_by_lig[lig],
            "lambda_lig": lambda_by_lig[lig],
            "p5_energies": [float(x) for x in e_p5],
            "p5_soc_coefficients": [float(x) for x in c_eta],
        }
        contrib = t_fp[("f1", lig)] @ g_p @ t_fp[("f2", lig)].conj().T
        per_ligand[lig]["contrib_norm"] = float(np.linalg.norm(contrib))
        t_correction += contrib

    t_eff = t_direct + t_correction

    _write_blocks_txt(out_path, [("t_mu", t_eff)])

    meta = {
        "hopping_fp_in": str(fp_path),
        "direct_t_mu_in": (None if direct_t_mu_path is None else str(direct_t_mu_path)),
        "output": str(out_path),
        "ligand_model": "fixed_p5_hole",
        "delta_mode": "manual",
        "delta_lig1": float(delta_lig1),
        "delta_lig2": float(delta_lig2),
        "lambda_lig1": float(lambda_lig1),
        "lambda_lig2": float(lambda_lig2),
        "degenerate_tol": float(degenerate_tol),
        "direct_norm": float(np.linalg.norm(t_direct)),
        "correction_norm": float(np.linalg.norm(t_correction)),
        "per_ligand": per_ligand,
    }
    logger.info(
        "downfold_to_ff_fixed_ligand: wrote %s; ||T_eff||_F = %.6e",
        out_path,
        float(np.linalg.norm(t_eff)),
    )
    return {"t_mu": t_eff, "per_ligand": per_ligand, "meta": meta}


# ---------------------------------------------------------------------------
# TOML CLI
# ---------------------------------------------------------------------------

def downfold_from_toml(toml_path: str | Path) -> dict[str, Any]:
    """Load config and run the current no-onsite p5-hole downfold path.

    Config schema::

        # Current offsite/manual path:
        hopping_fp_in = "out/hopping_fp.txt"
        direct_t_mu_in = "out/direct_t_mu.txt"  # optional; omitted => zero
        output = "out/hopping_ff_downfold.txt"
        delta_lig1 = 1.0
        delta_lig2 = 1.0
        lambda_lig1 = 0.0
        lambda_lig2 = 0.0
        degenerate_tol = 1e-6  # optional
    """
    path = Path(toml_path)
    if not path.exists():
        raise IOError_("FXE-IO-001", f"config not found: {path}", paths={"path": str(path)})
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    required = (
        "hopping_fp_in",
        "output",
        "delta_lig1",
        "delta_lig2",
        "lambda_lig1",
        "lambda_lig2",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise InputError(
            "FXE-INPUT-003",
            f"config missing fields: {missing}",
            expected={"required": list(required)},
            actual={"present": sorted(cfg.keys())},
        )
    return downfold_to_ff_fixed_ligand(
        hopping_fp_in=cfg["hopping_fp_in"],
        output=cfg["output"],
        delta_lig1=float(cfg["delta_lig1"]),
        delta_lig2=float(cfg["delta_lig2"]),
        lambda_lig1=float(cfg["lambda_lig1"]),
        lambda_lig2=float(cfg["lambda_lig2"]),
        direct_t_mu_in=cfg.get("direct_t_mu_in"),
        degenerate_tol=float(cfg.get("degenerate_tol", 1e-6)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Downfold f-p hopping blocks into an effective f-f hopping block.")
    parser.add_argument("config", help="TOML config path")
    args = parser.parse_args(argv)
    downfold_from_toml(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Numerical invariants (verified in tests):
#  1. f-basis unitary:  T_eff -> U_f · T_eff · U_f^†.
#  2. p-hole basis unitary:  cancels between t and G_p.
#  3. lambda_lig = 0:  G_p = (1/Delta) · I_6 exactly.
#  4. T_eff is the off-diagonal block H[f1, f2], not required Hermitian;
#     full H = T_eff cdag_f1 c_f2 + h.c. is Hermitian.
