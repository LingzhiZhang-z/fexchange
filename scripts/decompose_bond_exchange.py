#!/usr/bin/env python3
"""Per-bond Kitaev decomposition of an exchange tensor, including DM.

Each NN bond carries its Kitaev interaction on its own cubic axis. We put that
axis on 'c' by a CYCLIC relabel of the cubic axes (x,y,z), then read off the
J-K-Gamma-Gamma'-D parameters in that common (a,b,c) frame:

    bond    (a,b,c)      c = Kitaev axis
    z       (x, y, z)    z
    x       (y, z, x)    x
    y       (z, x, y)    y

The relabel MUST be cyclic (an even permutation, det=+1): J/K/Gamma/Gamma' are
symmetric under a<->b, but the DM component D_c=(Jab-Jba)/2 is antisymmetric, so
a non-cyclic swap would flip every DM sign. Cyclic keeps the D signs consistent
across the three bonds.

In the bond frame the symmetric tensor is the canonical Z-bond form
    [[ J,     Gamma,  Gamma'],
     [ Gamma, J,      Gamma'],
     [ Gamma',Gamma', J+K  ]]
so
    J  = (Jaa + Jbb) / 2              # Heisenberg
    K  =  Jcc - J                     # Kitaev (c-axis anisotropy)
    G  = (Jab + Jba) / 2              # Gamma  (symmetric, a-b plane)
    G' = (Jac+Jca+Jbc+Jcb) / 4        # Gamma' (symmetric, involving c)
and the antisymmetric part is the DM vector
    D_c = (Jab - Jba) / 2
    D_a = (Jbc - Jcb) / 2
    D_b = (Jca - Jac) / 2

FRAME ASSUMPTION: the input tensor must already be in the cubic (Kitaev) x,y,z
frame. J_mu from L3/exchange.txt is in the crystallographic frame (the spins come
from the cef Kramers doublet), so rotate it to cubic first -- the same
crystallographic->cubic rotation scripts/bond/enumerate.py uses, per material
from the .wout lattice. That rotation is NOT done here yet.

Usage:  python scripts/decompose_bond_exchange.py <L3/exchange.txt> <z|x|y>
"""
from __future__ import annotations

import sys

import numpy as np

# bond Kitaev axis -> (a,b,c): cyclic relabel of cubic x=0,y=1,z=2, with c = axis
PERM = {"z": (0, 1, 2), "x": (1, 2, 0), "y": (2, 0, 1)}
_KEYS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]


def read_J_mu(path: str) -> np.ndarray:
    """Parse the 'total' row of an L3 exchange.txt into a 3x3 tensor (cubic-frame assumed)."""
    for line in open(path, encoding="utf-8"):
        if line.startswith("total"):
            return np.array([float(x) for x in line.split()[2:11]]).reshape(3, 3)
    raise ValueError(f"no 'total' row in {path}")


def decompose(J: np.ndarray, bond: str) -> dict[str, float]:
    """J-K-Gamma-Gamma'-D in the bond's own frame (c = bond Kitaev axis)."""
    if bond not in PERM:
        raise ValueError(f"bond must be z|x|y, got {bond!r}")
    p = list(PERM[bond])
    M = J[np.ix_(p, p)]                         # axes now (a,b,c), c = Kitaev axis
    Jh = 0.5 * (M[0, 0] + M[1, 1])
    return {
        "J": float(Jh),
        "K": float(M[2, 2] - Jh),
        "Gamma": float(0.5 * (M[0, 1] + M[1, 0])),
        "Gamma_prime": float(0.25 * (M[0, 2] + M[2, 0] + M[1, 2] + M[2, 1])),
        "D_a": float(0.5 * (M[1, 2] - M[2, 1])),
        "D_b": float(0.5 * (M[2, 0] - M[0, 2])),
        "D_c": float(0.5 * (M[0, 1] - M[1, 0])),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    out = decompose(read_J_mu(sys.argv[1]), sys.argv[2].lower())
    print(f"# {sys.argv[1]}  bond={sys.argv[2]}  (c = Kitaev axis; input assumed cubic frame)")
    for k, v in out.items():
        print(f"{k:>12} = {v:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
