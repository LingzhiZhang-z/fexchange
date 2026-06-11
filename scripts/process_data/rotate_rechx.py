#!/usr/bin/env python3
"""Rotate REChX organized exchange tensors into the cubic Kitaev frame.

Two-step rotation J -> R J R^T with R = R(111, theta) . U_geom:

  1. U_geom: fixed geometric crystal -> cubic (100->11-2, 010->1-10, 001->111).
  2. R(111, theta): rotation about the trigonal 111 axis that puts the NN z-bond's
     bond vector along the cubic [1,1,0] direction (the standard Kitaev convention),
     giving a clean, uniquely-defined 4-parameter J/K/Gamma/Gamma'. "Make Jxx=Jyy"
     alone is ambiguous -- two theta 180 deg apart both satisfy it (the ABC/ACB
     stacking ambiguity); the bond geometry picks the right one.

theta is fixed by the stacking polytype, a per-material parameter:
  ABC stacking -> 90 deg,   ACB stacking -> 270 deg.
Currently REOCl and RESI are ABC, REOF is ACB; add a material key to STACKING to
override a family default. The same R is applied to every bond, U and jh of a
material, so the curves stay smooth and comparable across materials.

Output mirrors the input format under outputs_organized/REChX_cubic/.

Usage:  python scripts/process_data/rotate_rechx.py <organized_root> [material]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FAMS = ("REOCl", "REOF", "RESI")
# stacking by family; add per-material keys (e.g. "DyOF": "ACB") to override a default
STACKING = {"REOCl": "ABC", "REOF": "ACB", "RESI": "ABC"}
THETA = {"ABC": np.pi / 2.0, "ACB": 3.0 * np.pi / 2.0}             # aligns z-bond vector to cubic [1,1,0]
U = np.column_stack([np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0),    # crystal 100 -> cubic 11-2
                     np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0),    # crystal 010 -> cubic 1-10
                     np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)])    # crystal 001 -> cubic 111
N111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def r_about(axis: np.ndarray, th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def tensor(line: str) -> np.ndarray:
    return np.array([float(x) for x in line.split()[3:12]]).reshape(3, 3)


def rotation(fam: str, mat: str) -> np.ndarray:
    """R(111, theta) . U for a material; theta from its stacking (material overrides family)."""
    stack = STACKING.get(mat, STACKING.get(fam, "ABC"))
    return r_about(N111, THETA[stack]) @ U


def main() -> int:
    if not 2 <= len(sys.argv) <= 3:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    # optional scope (e.g. "REOF/ErOF") to re-rotate one material; default = all REChX
    bases = [root / sys.argv[2]] if len(sys.argv) == 3 else [root / fam for fam in FAMS]
    cache: dict = {}
    n = 0
    for base in bases:
        for f in sorted(base.rglob("U_*.txt")):
            if "REChX_cubic" in f.parts:
                continue
            fam, mat = f.parts[-6], f.parts[-5]                   # one rotation per material
            R = cache.get((fam, mat))
            if R is None:
                R = cache[(fam, mat)] = rotation(fam, mat)
            out = root / "REChX_cubic" / f.relative_to(root)
            out.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.startswith("#") or not ln.strip():
                    rows.append(ln)
                    continue
                p = ln.split()
                Jr = (R @ tensor(ln) @ R.T).ravel()
                rows.append(f"{p[0]} {p[1]} {p[2]} " + " ".join(f"{v:.12g}" for v in Jr))
            out.write_text("\n".join(rows) + "\n", encoding="utf-8")
            n += 1
    print(f"# rotated {n} files into {root}/REChX_cubic (stacking {STACKING})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
