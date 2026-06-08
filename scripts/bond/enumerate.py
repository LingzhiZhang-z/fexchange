#!/usr/bin/env python3
"""For each RE atom: its NN (and, for REChX, NNN) RE neighbours, the R-vector
of each neighbour, and the bridging ligand(s) between them.

REX3 (path contains 'REX3') only needs the 3 NN; REChX needs 3 NN + 6 NNN.

Usage:
  python scripts/bond/enumerate.py <material>.wout     # one system
  python scripts/bond/enumerate.py data/data-DFT       # every system below
"""

import sys
from pathlib import Path

import numpy as np

RARE_EARTH = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
              "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
BRIDGE_CUTOFF = 3.5            # A: a ligand bonded to a RE sits within this distance
CELLS = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]


def read_wout(path):
    """Return (lattice 3x3, atoms) where atoms = [(element, frac_xyz), ...]."""
    lines = open(path).read().splitlines()
    lattice, atoms = [], []
    for i, line in enumerate(lines):
        if "Lattice Vectors (Ang)" in line:                  # 3 lines of a1, a2, a3
            lattice = [[float(x) for x in lines[i + k].split()[-3:]] for k in (1, 2, 3)]
        p = line.replace("|", " ").split()                   # atom row: El n fx fy fz cx cy cz
        if len(p) == 8 and p[0].isalpha() and p[1].isdigit():
            atoms.append((p[0], np.array([float(p[2]), float(p[3]), float(p[4])])))
    return np.array(lattice), atoms


def distance(lattice, frac_a, frac_b):
    return float(np.linalg.norm((frac_b - frac_a) @ lattice))


def angle(a, b, c):
    """RE-X-RE angle in degrees at the ligand b (a, b, c are cartesian points)."""
    u, v = a - b, c - b
    return np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))


def is_rex3(path):
    return "REX3" in str(path)


def rex3_nn_label(v):
    """REX3 convention: x/y/z bond has the corresponding Cartesian component ~0."""
    return "xyz"[int(np.argmin(np.abs(v)))]


def rechx_nn_label(lattice, v):
    """REChX: rotate in-plane bonds to the cubic frame, then label by zero component."""
    a = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)   # current 100 -> cubic 11-2
    b = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)   # current 010 -> cubic 1-10
    u = v[0] * a + v[1] * b                         # ignore out-of-plane c component
    return "xyz"[int(np.argmin(np.abs(u)))]


def nn_label(lattice, v, rex3):
    return rex3_nn_label(v) if rex3 else rechx_nn_label(lattice, v)


def nnn_angle(lattice, v):
    a1, a2 = lattice[0], lattice[1]
    n = np.cross(a1, a2)
    n = n / np.linalg.norm(n)
    return np.degrees(np.arctan2(np.cross(a1, v) @ n, a1 @ v)) % 360


def bond_tag(path, shell, lattice, v):
    rex3 = is_rex3(path)
    if shell.strip() == "NN":
        axis = nn_label(lattice, v, rex3)
        return axis if rex3 else f"J1-{axis}"
    angle_deg = int(round(nnn_angle(lattice, v))) % 360
    return f"J2-{angle_deg:03d}"


def bridges(lattice, atoms, ligands, i, j, cell):
    ri = atoms[i][1] @ lattice
    rj = (atoms[j][1] + np.array(cell)) @ lattice
    found = []
    for l in ligands:
        for c in CELLS:
            rl = (atoms[l][1] + np.array(c)) @ lattice
            di, dj = np.linalg.norm(rl - ri), np.linalg.norm(rl - rj)
            if di < BRIDGE_CUTOFF and dj < BRIDGE_CUTOFF:
                found.append((atoms[l][0], l, c, di, dj, angle(ri, rl, rj)))
    return found


def print_structure(lattice, atoms, re, ligands):
    print("\n-- lattice vectors (Ang) --")
    for i, v in enumerate(lattice, start=1):
        print(f"  a{i}: {v[0]: .10f} {v[1]: .10f} {v[2]: .10f}")

    print("\n-- atoms (fractional, cartesian Ang) --")
    for title, indices in (("RE", re), ("ligand", ligands)):
        print(f"  {title}:")
        for i in indices:
            el, f = atoms[i]
            r = f @ lattice
            print(
                f"    #{i:2d} {el:<2}  "
                f"frac= {f[0]: .10f} {f[1]: .10f} {f[2]: .10f}  "
                f"cart= {r[0]: .10f} {r[1]: .10f} {r[2]: .10f}"
            )


def process(path):
    lattice, atoms = read_wout(path)
    re = [k for k, (el, _) in enumerate(atoms) if el in RARE_EARTH]
    ligands = [k for k, (el, _) in enumerate(atoms) if el not in RARE_EARTH]
    print_structure(lattice, atoms, re, ligands)
    # REX3 has an interlayer shell between NN and NNN, and only NN matters there;
    # for REChX the nearest 3 + next 6 are the in-plane NN + NNN.
    shells = (("NN ", slice(0, 3)),) if is_rex3(path) else (("NN ", slice(0, 3)), ("NNN", slice(3, 9)))

    for i in re:
        # every RE neighbour (other atom in any of the 27 nearby cells), sorted by distance
        neigh = []
        for j in re:
            for cell in CELLS:
                if i == j and cell == (0, 0, 0):
                    continue
                d = distance(lattice, atoms[i][1], atoms[j][1] + np.array(cell))
                neigh.append((d, j, cell))
        neigh.sort()

        print(f"\nRE #{i} ({atoms[i][0]}):")
        for shell, sel in shells:
            for d, j, cell in neigh[sel]:
                v = (atoms[j][1] + np.array(cell) - atoms[i][1]) @ lattice
                tag = f"[{bond_tag(path, shell, lattice, v)}]"
                print(f"  {shell} {tag:6} -> RE #{j}  R={cell}  {d:.3f} A")
                for el, l, c, di, dj, ang in bridges(lattice, atoms, ligands, i, j, cell):
                    print(f"        bridge {el}#{l} R={c}:  RE-X = {di:.2f}/{dj:.2f} A,  angle = {ang:.0f} deg")

    # inverse map: each ligand and the RE it bonds to (within BRIDGE_CUTOFF)
    print("\n-- ligand -> connected RE --")
    for l in ligands:
        nb = sorted((distance(lattice, atoms[l][1], atoms[j][1] + np.array(c)), j, c)
                    for j in re for c in CELLS)
        nb = [x for x in nb if x[0] < BRIDGE_CUTOFF]
        conn = "  ".join(f"RE#{j} R={c} {d:.2f}A" for d, j, c in nb)
        print(f"  {atoms[l][0]}#{l}: {conn}")


def main(target):
    target = Path(target)
    wouts = [target] if target.suffix == ".wout" else sorted(
        w for w in target.rglob("*.wout") if ".pp." not in w.name
    )
    for w in wouts:
        print(f"\n{'='*70}\n{w}\n{'='*70}")
        process(w)


if __name__ == "__main__":
    main(sys.argv[1])
