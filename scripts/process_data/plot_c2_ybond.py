#!/usr/bin/env python3
"""Symmetry-correct C2m REX3 plots: the doublet has only C2 (axis [10-1]=y-bond),
so report the symmetry-matched y-bond for Yb/Er, and [111]-Ising for Dy.

  Yb/Er  -> y-bond J/K/Gamma/Gamma' (Heisenberg-dominated for Yb)
  Dy     -> rank-1 [111]-Ising, plot J_parallel = leading eigenvalue of the tensor

U=4,6,8 overlaid with the 1/U collapse (curves scaled by U/8), sopt branch.

Usage: python scripts/process_data/plot_c2_ybond.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_bond_exchange import decompose

DST = Path("/work/k0282/k028230/code/fexchange/outputs_organized/REX3-C2m")
OUT = Path("/work/k0282/k028230/code/fexchange/jkgamma_figures/REX3_c2")
US = (4, 6, 8)
REFERENCE_U = 8
GREEN, RED, ORANGE, BLUE = "#2e8b3d", "#e8413a", "#f5821f", "#3d4eb8"
LW = {4: 1.4, 6: 1.6, 8: 3.0}


def sweep(mat, run, bond, U):
    f = DST / mat / bond / "sopt" / run / f"U_{U:02d}.txt"
    rs, Ms = [], []
    for ln in f.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        rs.append(float(p[0]))
        Ms.append(np.array([float(x) for x in p[3:12]]).reshape(3, 3) * 1e3)  # meV
    return np.array(rs), Ms


def panel_kitaev(ax, mat, run, title):
    for U in US:
        r, Ms = sweep(mat, run, "y", U)
        s = U / REFERENCE_U
        for key, c in [("J", GREEN), ("K", RED), ("Gamma", ORANGE), ("Gamma_prime", BLUE)]:
            y = np.array([decompose(M, "y")[key] * s for M in Ms])  # M already in meV
            ax.plot(r, y, color=c, lw=LW[U])
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(title)
    for key, c in [("J", GREEN), ("K", RED), ("Γ", ORANGE), ("Γ'", BLUE)]:
        ax.plot([], [], color=c, label=key)
    ax.legend(fontsize=9, loc="upper left")


def panel_ising(ax, mat, run, title):
    for U in US:
        r, Ms = sweep(mat, run, "z", U)
        s = U / REFERENCE_U
        lam = np.array([sorted(np.linalg.eigvalsh((M + M.T) / 2), key=abs)[-1] * s for M in Ms])
        ax.plot(r, lam, color=RED, lw=LW[U])
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(title)
    ax.plot([], [], color=RED, label=r"$J_\parallel$ ([111] Ising)")
    ax.legend(fontsize=9, loc="upper right")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panel_kitaev(axes[0], "YbCl3", "YbCl3_exp_baseline_Yb_J7_2_projector", "YbCl3  y-bond (Heisenberg-dom.)")
    panel_kitaev(axes[1], "ErCl3-NC", "YbCl3_exp_baseline_Er_J15_2_projector", "ErCl3  y-bond")
    panel_ising(axes[2], "DyCl3", "YbCl3_exp_baseline_Dy_J15_2_projector", "DyCl3  [111]-Ising")
    for ax in axes:
        ax.set_xlabel(r"$J_H/U$")
        ax.set_xlim(0, 0.4)
    axes[0].set_ylabel("coupling (meV)   [U=8; 4,6 scaled ×U/8]")
    fig.tight_layout()
    out = OUT / "REX3-C2m_c2corrected_sopt.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
