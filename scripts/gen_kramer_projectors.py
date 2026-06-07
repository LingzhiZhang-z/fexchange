"""Generate Ce YbOCl-baseline + 38 θ-scan kramer projectors for Ce/Sm J=5/2.

Outputs (relative to repo root, default to `data/data-DFT/kramer/`, override via --out):
- f1_J5_2_projector_YbOCl.txt        — Ce baseline (was missing)
- f{1,5}_J5_2_C3v_theta{NNN:03d}_projector.txt  — 19 θ values × 2 = 38 files
- table_theta_scan_g.txt             — g-tensor calibration per θ

Convention (§7.2 of multi-system prompt v9+):
  |ψ_+⟩ = cos θ |−1/2⟩ + i sin θ |+5/2⟩
  |ψ_-⟩ = cos θ |+1/2⟩ + i sin θ |−5/2⟩  (T-conjugate)
Basis order: [|−5/2⟩, |−3/2⟩, |−1/2⟩, |+1/2⟩, |+3/2⟩, |+5/2⟩] (ascending M,
matching fexchange core/space_j and pipeline._load_stevens).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from fexchange.tools.cef_states import (
    _extract_ground_doublet,
    _write_projector_txt,
    compute_cef_states,
)

YBOCL_B_PARAMS_MEV = {
    "B20": 7.766480502269e-01,
    "B40": 1.364717720245e-02,
    "B60": 4.637748236716e-04,
    "B66": -4.319221038524e-03,
    "B43": 2.514853991968e-01,
    "B63": -6.168010075996e-03,
}

THETA_DEG_GRID = list(range(0, 91, 5))  # 0, 5, 10, ..., 90 → 19 points
J = 2.5
N_J = 6


def _write_projector_from_W(
    path: Path,
    W: np.ndarray,
    kramer_name: str,
    ground_irrep: str = "Gamma4",
) -> None:
    """Write a (n_j × 2) W matrix in fexchange projector.txt format."""
    lines = [
        "# schema_version fxe.cef_projector.v1",
        "# standard_version 2026-02",
        f"# kramer_name {kramer_name}",
        "# point_group C3v",
        f"# J {float(J):.12e}",
        "# basis_id complex_spherical_j_v1",
        "# orbital_order_id f7_m-3..3_v1",
        f"# ground_irrep {ground_irrep}",
        "# doublet_type kramers",
        "",
    ]
    for idx in range(W.shape[1]):
        lines.append(f"[W_state_{idx}]")
        lines.extend(f"{val.real:.12e} {val.imag:.12e}" for val in W[:, idx])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def gen_ce_yboctl_baseline(out_dir: Path) -> None:
    """Ce J=5/2 with YbOCl-fitted CEF B params (same form as f5_J5_2_projector_YbOCl.txt)."""
    result = compute_cef_states(
        point_group="C3v",
        J=J,
        B_params=YBOCL_B_PARAMS_MEV,
        mode_q3="sin",
    )
    doublet = _extract_ground_doublet(result)
    path = out_dir / "f1_J5_2_projector_YbOCl.txt"
    _write_projector_txt(
        path,
        doublet,
        result=result,
        kramer_name="YbOClCEF_f1_J5_2",
        basis_id="complex_spherical_j_v1",
        orbital_order_id="f7_m-3..3_v1",
    )
    g = np.asarray(doublet["g_tensor"], dtype=np.complex128).real
    print(f"[Ce baseline] wrote {path}")
    print(f"  g-tensor diag: gx={g[0,0]:+.4f}  gy={g[1,1]:+.4f}  gz={g[2,2]:+.4f}")


def gen_theta_W(theta_deg: float) -> np.ndarray:
    """Construct W (6×2) for given θ per §7.2 convention.

    Basis (ascending M) [|−5/2⟩, |−3/2⟩, |−1/2⟩, |+1/2⟩, |+3/2⟩, |+5/2⟩]:
      ψ_+ = cos θ |−1/2⟩ + i sin θ |+5/2⟩
      ψ_- = cos θ |+1/2⟩ + i sin θ |−5/2⟩   (T-conjugate)
    """
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    W = np.zeros((N_J, 2), dtype=np.complex128)
    W[2, 0] = c        # |−1/2⟩ component of ψ_+  (ascending index 2)
    W[5, 0] = 1j * s   # |+5/2⟩ component of ψ_+  (ascending index 5)
    W[3, 1] = c        # |+1/2⟩ component of ψ_-  (ascending index 3)
    W[0, 1] = 1j * s   # |−5/2⟩ component of ψ_-  (ascending index 0)
    return W


def _compute_g_tensor_from_W(W: np.ndarray, J_val: float = J) -> tuple[float, float, float]:
    """g-tensor diagonal entries from W in |J,M⟩ basis (ascending M).

    For pseudospin-1/2: g_α = 2 ⟨ψ_+| 2 g_J J_α |ψ_+⟩ for α = x,y,z (signed).
    g_J for J=5/2 multiplet — use Landé factor 6/7 for Ce ²F_{5/2}, 2/7 for Sm ⁶H_{5/2}.
    Here we just report g_J · (effective spin operator) projection without Landé multiplication,
    to be applied per RE downstream.
    """
    # J_z, J_+, J_- in basis (ascending M): diag J_z = [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2]
    Mvals = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    Jz = np.diag(Mvals).astype(np.complex128)
    Jp = np.zeros((N_J, N_J), dtype=np.complex128)
    Jm = np.zeros((N_J, N_J), dtype=np.complex128)
    for i in range(N_J - 1):
        # J_+ |M⟩ = sqrt(J(J+1)-M(M+1)) |M+1⟩. Ascending: index i has M=Mvals[i],
        # and |M+1⟩ is index i+1, so J_+ connects i → i+1.
        M_from = Mvals[i]
        coeff = math.sqrt(J_val * (J_val + 1) - M_from * (M_from + 1))
        Jp[i + 1, i] = coeff
        Jm[i, i + 1] = coeff
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    # g components: 2 |⟨ψ_+| J_α |ψ_+⟩|  (without Landé factor — caller multiplies)
    a = W[:, 0]
    gz = 2 * (a.conj() @ Jz @ a).real
    gx_off = 2 * (a.conj() @ Jx @ W[:, 1]).real
    gx_off_im = 2 * (a.conj() @ Jx @ W[:, 1]).imag
    gx = 2 * abs(a.conj() @ Jx @ W[:, 1])
    gy = 2 * abs(a.conj() @ Jy @ W[:, 1])
    return gx, gy, gz


def gen_theta_projectors(out_dir: Path, table_lines: list[str]) -> None:
    for n_label, n_int in (("f1", 1), ("f5", 5)):
        for theta_deg in THETA_DEG_GRID:
            W = gen_theta_W(float(theta_deg))
            name = f"{n_label}_J5_2_C3v_theta{theta_deg:03d}_projector"
            path = out_dir / f"{name}.txt"
            _write_projector_from_W(
                path,
                W,
                kramer_name=name,
                ground_irrep="Gamma4",
            )
            gx_2J, gy_2J, gz_2J = _compute_g_tensor_from_W(W)
            table_lines.append(
                f"{name:<48} {theta_deg:3d}  "
                f"gx_2J={gx_2J:+.6f}  gy_2J={gy_2J:+.6f}  gz_2J={gz_2J:+.6f}"
            )
    print(f"[θ scan] wrote {len(THETA_DEG_GRID) * 2} files to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/kramer_out"),
        help="Output directory for kramer projector files",
    )
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_ce_yboctl_baseline(out_dir)

    table_lines = [
        "# θ-scan g-tensor calibration table (without Landé factor g_J)",
        "# Applied formula: g_α^pseudospin = g_J × 2|<ψ_+| J_α |ψ_∓>|  (for α=x,y); g_z = 2 g_J <ψ_+| J_z |ψ_+>",
        "# Ce: g_J = 6/7;  Sm: g_J = 2/7  (Landé factor for ²F_{5/2} and ⁶H_{5/2})",
        "# Columns: name θ(deg) gx_2J gy_2J gz_2J  (values before Landé multiplication)",
        "",
    ]
    gen_theta_projectors(out_dir, table_lines)

    table_path = out_dir / "table_theta_scan_g.txt"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"[table] wrote {table_path}")


if __name__ == "__main__":
    main()
