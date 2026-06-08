#!/usr/bin/env python3
"""Build tight-binding models from the enumerate_nn_bonds output of ONE material
and compute their band structures along the .win k-path. Two models are written:

(1) pruned (full num_wann dimension). From a zero H(R), fill only:
      - onsite    : diagonal block of every atom at R=(0,0,0)
      - RE-RE     : every (RE_i, RE_j, R) from the NN/NNN lines
      - RE-ligand : every (ligand, RE, R) from the 'ligand -> connected RE' table
    each kept block (A,B,R) mirrored to (B,A,-R) for Hermiticity; rest zero.

(2) downfold (RE-only, dim = n_RE * 14). Each bridging ligand is folded into an
    effective RE-RE hopping (w90_downfold algorithm, lambda=0 so G_p = I/Delta):
      t_eff = t_direct + sum_lig (1/Delta_lig) t_iL @ t_jL^dagger
      Delta_lig = <f onsite trace> - <ligand onsite trace>, from H(0).

Bands are H(k) = sum_R H(R) exp(2j pi k.R), diagonalised along the .win kpoint_path.
Plain-text gnuplot output (k_dist energy, one band per block).

Usage:
  python scripts/bond/enumerate.py <mat>.wout > bonds.txt
  python scripts/tb_from_enumerate.py bonds.txt
"""

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "bond"))
from enumerate import RARE_EARTH, read_wout                   # noqa: E402
from fexchange.tools.w90_extract import read_w90_hr           # noqa: E402

F_ORB, P_ORB = 14, 6   # SOC spinor orbitals: RE f = 7x2, ligand p = 3x2


def parse_input(text):
    """Return (wout, pairs, re_bonds).
      pairs    : {(atom_A, atom_B, R)} for the explicit/pruned model
                 (RE-RE from NN/NNN lines, ligand-RE from the connection table).
      re_bonds : [{'i','j','R','bridges':[(lig, cell), ...]}] for the downfold."""
    lines = text.splitlines()
    wout = next(ln.strip() for ln in lines if ln.strip().endswith(".wout"))
    pairs, re_bonds = set(), []
    i, cur, in_table = None, None, False
    for ln in lines:
        if "ligand -> connected RE" in ln:
            in_table = True
        if in_table:
            m = re.match(r"\s*\w+#(\d+):", ln)
            if m:
                lig = int(m.group(1))
                for j, cell in re.findall(r"RE#(\d+) R=\(([^)]+)\)", ln):
                    pairs.add((lig, int(j), tuple(int(x) for x in cell.split(","))))
            continue
        h = re.match(r"\s*RE #(\d+) ", ln)
        if h and "->" not in ln:
            i, cur = int(h.group(1)), None
            continue
        b = re.match(r"\s*(?:NN|NNN)\s+\[[^\]]+\]\s+-> RE #(\d+)\s+R=\(([^)]+)\)", ln)
        if b and i is not None:
            R = tuple(int(x) for x in b.group(2).split(","))
            pairs.add((i, int(b.group(1)), R))
            cur = {"i": i, "j": int(b.group(1)), "R": R, "bridges": []}
            re_bonds.append(cur)
            continue
        g = re.match(r"\s*bridge \w+#(\d+) R=\(([^)]+)\)", ln)
        if g and cur is not None:
            cur["bridges"].append((int(g.group(1)), tuple(int(x) for x in g.group(2).split(","))))
    return wout, pairs, re_bonds


def read_win_kpath(win_path):
    """Return (segments, n_per_segment); each segment is (label1, k1, label2, k2)."""
    lines = Path(win_path).read_text().splitlines()
    npts = 60
    segs, inside = [], False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("bands_num_points"):
            npts = int(s.split()[-1])
        if s.lower().startswith("begin kpoint_path"):
            inside = True
            continue
        if s.lower().startswith("end kpoint_path"):
            break
        if inside and s:
            t = s.split()
            segs.append((t[0], np.array(t[1:4], float), t[4], np.array(t[5:8], float)))
    return segs, npts


def atom_blocks(wout_path, num_wann):
    lattice, atoms = read_wout(wout_path)
    blocks, off = {}, 0
    for k, (el, _) in enumerate(atoms):
        n = F_ORB if el in RARE_EARTH else P_ORB
        blocks[k] = list(range(off, off + n))
        off += n
    if off != num_wann:
        raise ValueError(f"orbital total {off} != num_wann {num_wann}")
    return lattice, atoms, blocks


def block(H, R, rows, cols):
    if R in H:
        return H[R][np.ix_(rows, cols)]
    return H[tuple(-x for x in R)][np.ix_(cols, rows)].conj().T


def build_pruned(H, atoms, blocks, pairs, num_wann):
    """Pruned H(R): onsite + kept pairs, mirrored to keep it Hermitian."""
    pruned = {}

    def put(R, rows, cols, mat):
        pruned.setdefault(R, np.zeros((num_wann, num_wann), complex))[np.ix_(rows, cols)] = mat

    for a in range(len(atoms)):                                    # onsite
        put((0, 0, 0), blocks[a], blocks[a], block(H, (0, 0, 0), blocks[a], blocks[a]))
    for a, b, R in pairs:
        mat = block(H, R, blocks[a], blocks[b])
        put(R, blocks[a], blocks[b], mat)
        put(tuple(-x for x in R), blocks[b], blocks[a], mat.conj().T)
    return pruned


def build_downfolded(H, atoms, blocks, re_bonds):
    """RE-only effective model: fold each bridging ligand into the RE-RE hopping.
        t_eff = t_direct + sum_lig (1/Delta_lig) * t_iL @ t_jL^dagger     (lambda=0)
        Delta_lig = <f_i,f_j onsite trace> - <ligand onsite trace>, from H(0).
    Returns the effective H(R) dict on the RE-only basis (dim = n_RE * 14)."""
    re_atoms = [k for k, (el, _) in enumerate(atoms) if el in RARE_EARTH]
    pos = {k: n for n, k in enumerate(re_atoms)}
    rb = {k: list(range(pos[k] * F_ORB, pos[k] * F_ORB + F_ORB)) for k in re_atoms}
    dim = len(re_atoms) * F_ORB
    H0 = H[(0, 0, 0)]

    def onsite_avg(blk):
        return float(np.trace(H0[np.ix_(blk, blk)]).real) / len(blk)

    eff = {}

    def put(R, rows, cols, mat):
        eff.setdefault(R, np.zeros((dim, dim), complex))[np.ix_(rows, cols)] = mat

    for k in re_atoms:                                             # RE onsite (f block)
        put((0, 0, 0), rb[k], rb[k], block(H, (0, 0, 0), blocks[k], blocks[k]))
    for bd in re_bonds:
        i, j, R = bd["i"], bd["j"], bd["R"]
        t = block(H, R, blocks[i], blocks[j])                      # direct f-f, 14x14
        e_f = 0.5 * (onsite_avg(blocks[i]) + onsite_avg(blocks[j]))
        for lig, cL in bd["bridges"]:
            delta = e_f - onsite_avg(blocks[lig])
            t_iL = block(H, cL, blocks[i], blocks[lig])            # 14x6  RE_i(0)->L(cL)
            t_jL = block(H, tuple(cL[a] - R[a] for a in range(3)), blocks[j], blocks[lig])
            t = t + (t_iL @ t_jL.conj().T) / delta                 # G_p = I/Delta
        put(R, rb[i], rb[j], t)
        put(tuple(-x for x in R), rb[j], rb[i], t.conj().T)
    return eff


def kpath(segs, npts, lattice):
    B = 2 * np.pi * np.linalg.inv(lattice).T                       # reciprocal vectors (rows)
    kfracs, dist, ticks = [], [], []
    d = 0.0
    for s, (l1, k1, l2, k2) in enumerate(segs):
        if s == 0:
            ticks.append((0.0, l1))
        c1, c2 = k1 @ B, k2 @ B
        seglen = np.linalg.norm(c2 - c1)
        for n in range(npts):
            f = n / (npts - 1)
            kfracs.append(k1 + f * (k2 - k1))
            dist.append(d + f * seglen)
        d += seglen
        ticks.append((d, l2))
    return np.array(kfracs), np.array(dist), ticks


def bands(Hdict, kfracs):
    Rs = np.array(list(Hdict.keys()))
    mats = np.array(list(Hdict.values()))
    out = []
    for k in kfracs:
        phase = np.exp(2j * np.pi * (Rs @ k))
        Hk = np.tensordot(phase, mats, axes=(0, 0))
        out.append(np.linalg.eigvalsh(Hk))
    return np.array(out)


def write_bands(path, dist, ev, ticks):
    """gnuplot format: two columns 'k_dist energy', one band per block, blocks
    separated by a blank line (plot 'file' w l draws each band as one line)."""
    with path.open("w") as f:
        f.write("# ticks: " + "  ".join(f"{lbl}={d:.4f}" for d, lbl in ticks) + "\n")
        f.write("# k_dist(1/Ang)  energy(eV)\n")
        for b in range(ev.shape[1]):
            for d, e in zip(dist, ev[:, b]):
                f.write(f"{d:.6f} {e:.6f}\n")
            f.write("\n")


def main(path):
    wout, pairs, re_bonds = parse_input(Path(path).read_text())
    hr_path = next(Path(wout).parent.glob("*_hr.dat"))
    win_path = next(Path(wout).parent.glob("*.win"))
    H, num_wann = read_w90_hr(hr_path)
    lattice, atoms, blocks = atom_blocks(wout, num_wann)

    segs, npts = read_win_kpath(win_path)
    kfracs, dist, ticks = kpath(segs, npts, lattice)
    stem = Path(wout).stem.split(".")[0]
    out_dir = REPO / "outputs"
    out_dir.mkdir(exist_ok=True)

    pruned = build_pruned(H, atoms, blocks, pairs, num_wann)
    write_bands(out_dir / f"{stem}_bands.txt", dist, bands(pruned, kfracs), ticks)

    eff = build_downfolded(H, atoms, blocks, re_bonds)
    write_bands(out_dir / f"{stem}_bands_downfold.txt", dist, bands(eff, kfracs), ticks)

    print(f"pruned dim={num_wann} (kept pairs={len(pairs)})  "
          f"downfold dim={eff[(0, 0, 0)].shape[0]} (RE-RE bonds={len(re_bonds)})  k-points={len(kfracs)}")
    print(f"wrote {out_dir / (stem + '_bands.txt')}")
    print(f"wrote {out_dir / (stem + '_bands_downfold.txt')}")


if __name__ == "__main__":
    main(sys.argv[1])
