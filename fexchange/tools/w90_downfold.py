"""Second-order p-shell downfold: f-p hopping -> effective f-f hopping.

Formula (single-particle Löwdin / 2nd-order PT, basis-independent):

    T_eff(f_1, f_2)  =  T_direct  +  Σ      T_{lig_k → f_1}  ·  G_k  ·  T_{lig_k → f_2}^†
                                      k∈{lig1,lig2}

    G_k  =  (0 · I − h_lig_k)^{-1}
         =  U_k · diag( 1 / (0 − E_a) ) · U_k^†          (h_lig_k = U_k diag(E_a) U_k^†)

    T_{lig_k → f_i}[α, a] = ⟨f_i,α | H | lig_k,a⟩ = t_fp[(f_i,lig_k)]   (14×6)
    reverse (f → lig) = Hermitian conjugate  →  T_{lig_k → f_j}^†       (6×14)

SOC ligand h_lig_k = E_p·I + λ_p·L·S: eigvals E_p−λ_p (J=1/2 ×2),
E_p+λ_p/2 (J=3/2 ×4); the two J channels get 1/(Δ+λ_p), 1/(Δ−λ_p/2)
automatically.  NSOC = λ_p=0 limit (h_lig_k = E_p·I → G_k = (1/Δ)·I);
same code path, no mode switch.

Closed-shell p^6 assumed: all eigvals below E_f (Löwdin convergence at
z = E_f); enforced as a sanity check.

Current I/O:
  direct_t_mu_in  optional block [t_mu] 14×14; omitted means zero direct term
  hopping_fp_in   blocks [t_f1_lig1] [t_f1_lig2] [t_f2_lig1] [t_f2_lig2]
  delta_lig1      manual denominator for ligand 1
  delta_lig2      manual denominator for ligand 2
  output          block [t_mu] 14×14 = direct + folded ligand correction
"""

from __future__ import annotations

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

def _p_angular_momentum_real_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angular momentum matrices for p_x, p_y, p_z real harmonics."""
    lx = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1j], [0.0, 1j, 0.0]],
        dtype=complex,
    )
    ly = np.array(
        [[0.0, 0.0, 1j], [0.0, 0.0, 0.0], [-1j, 0.0, 0.0]],
        dtype=complex,
    )
    lz = np.array(
        [[0.0, -1j, 0.0], [1j, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=complex,
    )
    return lx, ly, lz


def _spin_half_down_up_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spin-1/2 matrices in the local spin order down, up."""
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, 1j], [-1j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=complex)
    return sx, sy, sz


def fixed_p_ligand_hamiltonian(delta_lig: float, lambda_p: float = 0.0) -> np.ndarray:
    """Return the fixed 6x6 p-ligand Hamiltonian in px/py/pz, down/up order.

    The f reference is zero, so the p-shell center is ``-delta_lig``.  The SOC
    term is ``lambda_p * L.S``.  Its eigenvalues are ``-delta_lig - lambda_p``
    for the j=1/2 doublet and ``-delta_lig + lambda_p/2`` for the j=3/2 quartet.
    """
    delta = float(delta_lig)
    lam = float(lambda_p)
    if not np.isfinite(delta) or not np.isfinite(lam):
        raise InputError(
            "FXE-INPUT-003",
            "fixed ligand parameters must be finite",
            actual={"delta_lig": delta, "lambda_p": lam},
        )
    lx, ly, lz = _p_angular_momentum_real_basis()
    sx, sy, sz = _spin_half_down_up_basis()
    l_dot_s = np.kron(lx, sx) + np.kron(ly, sy) + np.kron(lz, sz)
    return -delta * np.eye(N_P_SPINOR, dtype=complex) + lam * l_dot_s


# ---------------------------------------------------------------------------
# Resolvent G_k
# ---------------------------------------------------------------------------

def _ligand_resolvent(
    h_lig: NDArray[np.complexfloating],
    e_f: float,
    *,
    degenerate_tol: float,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """G_k = (E_f · I − h_lig)^{-1}, computed via eigendecomposition:

        G_k = Σ_a |a⟩^{sp} (1 / (E_f − E_a^{sp})) ⟨a|^{sp}

    where ``|a⟩^{sp}`` and ``E_a^{sp}`` are the single-particle eigenvectors
    and eigenvalues of ``h_lig`` (the W90 6×6 ligand block).

    * SOC ligand ``h_lig = E_p·I + λ_p·L·S``: eigvals split into
      ``E_p − λ_p`` (J=1/2 doublet) and ``E_p + λ_p/2`` (J=3/2 quartet);
      each subspace gets its own ``1/(Δ ± λ_p/2)`` factor automatically.
    * NSOC ligand is the ``λ_p = 0`` limit: ``h_lig = E_p·I``, 6 degenerate
      eigvals, ``G_k → (1/Δ_k)·I_6``.  Falls out of the *same* code path
      exactly (no separate branch).

    Many-body p^5 picture gives the *same* ``G_k`` numerically: U column
    vectors and denominators ``E_f − E_a^{sp}`` transfer one-to-one, with
    the multi-body p^6 energy offset cancelling in the PT denominator.

    Returns ``(G_k, eigvals)`` where ``eigvals`` is the single-particle
    spectrum (real, sorted ascending).  Raises ``FXE-NUM-001`` if any
    ``|E_f − E_a^{sp}| < degenerate_tol``.  Raises ``FXE-PHYS-001`` if any
    eigval lies above E_f (Löwdin partitioning convergence condition).
    """
    h_sym = 0.5 * (h_lig + h_lig.conj().T)  # symmetrize for eigh stability
    eigvals, eigvecs = np.linalg.eigh(h_sym)
    denom = e_f - eigvals
    above = [int(i) for i, d in enumerate(denom) if d < 0.0]
    if above:
        raise PhysError(
            "FXE-PHYS-001",
            "SOC downfold: some ligand eigvals lie above E_f — breaks the "
            "Löwdin partitioning assumption that all 6 single-particle "
            "ligand states are below E_f (closed-shell p^6)",
            actual={
                "e_f": float(e_f),
                "eigvals": [float(x) for x in eigvals],
                "above_indices": above,
                "above_eigvals": [float(eigvals[i]) for i in above],
            },
        )
    if np.any(np.abs(denom) < degenerate_tol):
        bad_idx = [int(i) for i, d in enumerate(denom) if abs(d) < degenerate_tol]
        raise NumError(
            "FXE-NUM-001",
            "SOC downfold: degenerate ligand denominator |E_f − E_a| < tol",
            expected={"degenerate_tol": float(degenerate_tol)},
            actual={
                "e_f": float(e_f),
                "eigvals": [float(x) for x in eigvals],
                "bad_indices": bad_idx,
            },
        )
    g = (eigvecs * (1.0 / denom)) @ eigvecs.conj().T
    return g, eigvals


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
    direct_t_mu_in: str | Path | None = None,
    lambda_p: float = 0.0,
    degenerate_tol: float = 1e-6,
) -> dict[str, Any]:
    """Downfold with the analytic fixed p-ligand Hamiltonian.

    This path does not read onsite input.  It uses
    ``h_lig_k = -Delta_k * I + lambda_p * L.S`` with independent
    ``Delta_1``/``Delta_2`` values and the fixed f reference ``E_f = 0``.
    If ``direct_t_mu_in`` is supplied, that direct f-f ``[t_mu]`` is added to
    the folded ligand correction; otherwise the direct term is zero.
    """
    fp_path = Path(hopping_fp_in)
    out_path = Path(output)
    if not fp_path.exists():
        raise IOError_("FXE-IO-001", f"hopping_fp_in missing: {fp_path}", paths={"path": str(fp_path)})

    t_fp = _load_fp_blocks(fp_path)
    e_f = 0.0
    delta_by_lig = {
        "lig1": float(delta_lig1),
        "lig2": float(delta_lig2),
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
        h_lig = fixed_p_ligand_hamiltonian(delta_by_lig[lig], lambda_p)
        g_k, eigvals = _ligand_resolvent(
            h_lig,
            e_f,
            degenerate_tol=degenerate_tol,
        )
        per_ligand[lig] = {
            "eigvals": [float(x) for x in eigvals],
            "deltas": [float(e_f - x) for x in eigvals],
            "delta_lig": delta_by_lig[lig],
        }
        contrib = t_fp[("f1", lig)] @ g_k @ t_fp[("f2", lig)].conj().T
        per_ligand[lig]["contrib_norm"] = float(np.linalg.norm(contrib))
        t_correction += contrib

    t_eff = t_direct + t_correction

    _write_blocks_txt(out_path, [("t_mu", t_eff)])

    meta = {
        "hopping_fp_in": str(fp_path),
        "direct_t_mu_in": (None if direct_t_mu_path is None else str(direct_t_mu_path)),
        "output": str(out_path),
        "ligand_model": "fixed_p_soc",
        "delta_mode": "manual",
        "delta_lig1": float(delta_lig1),
        "delta_lig2": float(delta_lig2),
        "lambda_p": float(lambda_p),
        "e_f": e_f,
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
    """Load config and run the current no-onsite downfold path.

    Config schema::

        # Current offsite/manual path:
        hopping_fp_in = "out/hopping_fp.txt"
        direct_t_mu_in = "out/direct_t_mu.txt"  # optional; omitted => zero
        output = "out/hopping_ff_downfold.txt"
        delta_lig1 = 1.0
        delta_lig2 = 1.0
        lambda_p = 0.0         # optional
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
        direct_t_mu_in=cfg.get("direct_t_mu_in"),
        lambda_p=float(cfg.get("lambda_p", 0.0)),
        degenerate_tol=float(cfg.get("degenerate_tol", 1e-6)),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m fexchange.tools.w90_downfold config.toml", file=sys.stderr)
        sys.exit(1)
    downfold_from_toml(sys.argv[1])

# Numerical invariants (verified in tests):
#  1. f-basis unitary:  T_eff -> U_f · T_eff · U_f^†.
#  2. p-basis unitary:  cancels between t and G_k (no effect).
#  3. λ_p = 0 (h_lig = scalar·I):  G_k = (1/Δ)·I_6 exactly (np.array_equal).
#  4. T_eff is the off-diagonal block H[f1, f2], not required Hermitian;
#     full H = T_eff c†_f1 c_f2 + h.c. is Hermitian.
