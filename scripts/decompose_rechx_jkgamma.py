#!/usr/bin/env python3
"""Batch trigonal J-K-Gamma decomposition of REChX exchange tensors.

For every L3/exchange.txt under <root>, take the 3x3 J_mu (in the crystallographic
frame, where z = the c-axis = trigonal [111]), rotate about c by the angle that
equalizes the two in-plane diagonals (Jaa = Jbb), and read off a trigonal
J-K-Gamma model plus the DM along c:

    theta* = 1/2 * atan2( (Jxx-Jyy)/2 , (Jxy+Jyx)/2 )   # makes Jaa=Jbb
    J      = (Jaa+Jbb)/2        # in-plane Heisenberg        (invariant about c)
    K      =  Jcc - J           # c-axis (111) anisotropy     (invariant about c)
    Gamma  =  Jab               # in-plane off-diagonal in the equalized frame
                                #   |Gamma| = sqrt(((Jxx-Jyy)/2)^2 + Jxy^2)
    D_c    = (Jab-Jba)/2        # DM along c (111)            (invariant about c)

So the whole in-plane symmetric anisotropy (both Jxx-Jyy and Jxy) is folded into a
single Gamma, giving the canonical Jaa=Jbb form.

FRAME / NAMING (read before trusting numbers):
  * Assumes J_mu's z is the crystallographic c-axis -- true for a uniaxial C3v
    Kramers doublet, whose pseudospin quantizes along c. If that is not your spin
    frame, equalize about the correct axis instead.
  * c = 111, so K = Jcc-J is the c-AXIAL anisotropy = (cubic Gamma + 2 Gamma'),
    NOT the cubic Kitaev K. This is the trigonal model on purpose.

Usage:  python scripts/decompose_rechx_jkgamma.py <output_root>
        (globs <root>/*/L3/exchange.txt, falling back to <root>/**/exchange.txt)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_KEYS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]


def read_J_mu(path: Path) -> np.ndarray:
    """Parse the 'total' row of an L3 exchange.txt into a 3x3 tensor."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("total"):
            return np.array([float(x) for x in line.split()[2:11]]).reshape(3, 3)
    raise ValueError(f"no 'total' row in {path}")


def jkgamma(J: np.ndarray) -> dict[str, float]:
    """Trigonal J-K-Gamma + DM, rotating about c (=z=111) so that Jaa=Jbb."""
    A = 0.5 * (J[0, 0] - J[1, 1])
    B = 0.5 * (J[0, 1] + J[1, 0])
    th = 0.5 * np.arctan2(A, B)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    M = R @ J @ R.T
    Jh = 0.5 * (M[0, 0] + M[1, 1])
    Dperp = float(np.hypot(0.5 * (M[1, 2] - M[2, 1]), 0.5 * (M[2, 0] - M[0, 2])))
    return {
        "theta_deg": float(np.degrees(th)),
        "J": float(Jh),
        "K": float(M[2, 2] - Jh),
        "Gamma": float(0.5 * (M[0, 1] + M[1, 0])),
        "Gamma_prime": float(0.25 * (M[0, 2] + M[2, 0] + M[1, 2] + M[2, 1])),
        "D_c": float(0.5 * (M[0, 1] - M[1, 0])),
        "D_perp": Dperp,
    }


def run_name_of(exchange_txt: Path) -> str:
    """<run_name>/L3/exchange.txt -> run_name."""
    return exchange_txt.parent.parent.name


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    files = sorted(root.glob("*/L3/exchange.txt")) or sorted(root.rglob("exchange.txt"))
    if not files:
        print(f"no exchange.txt under {root}")
        return 1

    rows = []
    for f in files:
        run = run_name_of(f)
        material, _, bond = run.partition("_FOPT_")
        d = jkgamma(read_J_mu(f))
        rows.append((material or run, bond or run, d))

    hdr = f"{'material':<12}{'bond':<10}{'J':>10}{'K':>10}{'Gamma':>10}{'Gamma_p':>10}{'D_c':>10}{'theta':>8}"
    print(hdr)
    print("-" * len(hdr))
    for material, bond, d in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{material:<12}{bond:<10}{d['J']:>10.4f}{d['K']:>10.4f}{d['Gamma']:>10.4f}"
              f"{d['Gamma_prime']:>10.4f}{d['D_c']:>10.4f}{d['theta_deg']:>8.1f}")
    print(f"\n# {len(rows)} tensors processed under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
