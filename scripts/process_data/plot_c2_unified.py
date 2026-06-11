#!/usr/bin/env python3
"""Unified C2m REX3 exchange from the 3 bonds: the single C2 doublet shares ONE
anisotropy axis across all bonds, so the unified NN model is the bond-averaged
tensor  Mbar = (Jx+Jy+Jz)/3  (all bonds are in the same doublet gauge frame).

Reported, vs Jh/U (U=4,6,8 overlaid, 1/U collapse):
  * J_iso = trace(Mbar)/3            -- Heisenberg part (frame-independent)
  * lam1 >= lam2 >= lam3             -- principal couplings of Mbar (anisotropy)
  * n1                               -- axis of lam1 (the shared anisotropy axis)
Heisenberg <=> lam1=lam2=lam3 ;  Ising <=> one lam dominates.

Usage: python scripts/process_data/plot_c2_unified.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DST = Path("/work/k0282/k028230/code/fexchange/outputs_organized/REX3-C2m")
OUT = Path("/work/k0282/k028230/code/fexchange/jkgamma_figures/REX3_c2")
US = (4, 6, 8)
MATS = [("YbCl3", "YbCl3_exp_baseline_Yb_J7_2_projector", "Yb"),
        ("ErCl3-NC", "YbCl3_exp_baseline_Er_J15_2_projector", "Er"),
        ("DyCl3", "YbCl3_exp_baseline_Dy_J15_2_projector", "Dy")]
AXES_NAMED = {(1, 1, 1): "[111] (C3 axis)", (1, -2, 1): "[1-21] (in-plane)",
              (1, 0, -1): "[10-1] (C2/y-bond)"}


def named_axis(n):
    n = n / np.max(np.abs(n))
    best, lab = 9, "?"
    for v, name in AXES_NAMED.items():
        u = np.array(v, float); u /= np.linalg.norm(u)
        d = 1 - abs(n @ u / np.linalg.norm(n))
        if d < best:
            best, lab = d, name
    return lab


def sweep(mat, run):
    """ratio -> bond-averaged symmetric tensor Mbar (meV) and per-bond J_iso list."""
    bonds = {}
    for b in ("x", "y", "z"):
        f = DST / mat / b / "sopt" / run / "U_08.txt"
        for ln in f.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split(); r = round(float(p[0]), 2)
            M = np.array([float(x) for x in p[3:12]]).reshape(3, 3) * 1e3
            bonds.setdefault(r, {})[b] = (M + M.T) / 2
    out = {}
    for r, bm in bonds.items():
        Ms = list(bm.values())
        Mbar = sum(Ms) / 3
        iso = [np.trace(M) / 3 for M in Ms]
        out[r] = (Mbar, np.mean(iso), np.std(iso))
    return dict(sorted(out.items()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    print(f"{'mat':>5} {'Jh/U':>5} {'J_iso':>7} {'lam1':>7} {'lam2':>7} {'lam3':>7} {'spread%':>7}  axis(lam1)")
    for ax, (mat, run, ion) in zip(axes, MATS):
        s = sweep(mat, run)
        rs = np.array(sorted(s))
        Jiso = np.array([s[r][1] for r in rs])
        spr = np.array([s[r][2] for r in rs])
        lams = np.array([sorted(np.linalg.eigvalsh(s[r][0]), reverse=True) for r in rs])
        # leading-eig axis at Jh/U=0.10
        Mb = s[round(0.10, 2)][0]
        w, V = np.linalg.eigh(Mb)
        n1 = V[:, np.argmax(np.abs(w))]
        ax.plot(rs, lams[:, 0], color="#e8413a", lw=2.4, label=r"$\lambda_1$ (along %s)" % named_axis(n1))
        ax.plot(rs, lams[:, 1], color="#f5821f", lw=2.0, label=r"$\lambda_2$")
        ax.plot(rs, lams[:, 2], color="#3d4eb8", lw=2.0, label=r"$\lambda_3$")
        ax.plot(rs, Jiso, color="#2e8b3d", lw=2.8, ls="--", label=r"$J_{\rm iso}$ (Heisenberg)")
        ax.fill_between(rs, Jiso - spr, Jiso + spr, color="#2e8b3d", alpha=0.15)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_title(f"{mat}  (unified from x,y,z)")
        ax.set_xlabel(r"$J_H/U$"); ax.set_xlim(0, 0.4)
        ax.legend(fontsize=8, loc="best")
        for r in (0.0, 0.1, 0.2, 0.4):
            Mb = s[round(r, 2)][0]; J = s[round(r, 2)][1]; sp = s[round(r, 2)][2]
            lm = sorted(np.linalg.eigvalsh(Mb), reverse=True)
            w, V = np.linalg.eigh(Mb); n = V[:, np.argmax(np.abs(w))]
            print(f"{ion:>5} {r:>5.2f} {J:>7.3f} {lm[0]:>7.3f} {lm[1]:>7.3f} {lm[2]:>7.3f} "
                  f"{100*sp/abs(J) if J else 0:>6.0f}  {named_axis(n)}")
    axes[0].set_ylabel("unified coupling (meV)   U=8")
    fig.tight_layout()
    out = OUT / "REX3-C2m_unified_sopt.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
