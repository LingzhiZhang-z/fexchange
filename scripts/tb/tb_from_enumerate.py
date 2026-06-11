#!/usr/bin/env python3
"""Build tight-binding models from the enumerate_nn_bonds output of ONE material
and compute their band structures along the .win k-path. Three models are written:

(1) pruned (full num_wann dimension). From a zero H(R), fill only:
      - onsite    : diagonal block of every atom at R=(0,0,0)
      - RE-RE     : every (RE_i, RE_j, R) from the NN/NNN lines
      - RE-ligand : every (ligand, RE, R) from the 'ligand -> connected RE' table
    each kept block (A,B,R) mirrored to (B,A,-R) for Hermiticity; rest zero.

(2) downfold (RE-only, dim = n_RE * 14). Each bridging ligand is folded into an
    effective RE-RE hopping with the fixed-p5 propagator from w90_downfold:
      t_eff = t_direct + sum_lig t_iL @ G_p(Delta_lig, lambda_lig) @ t_jL^dagger
      Delta_lig = <f onsite trace> - <ligand onsite trace>, from H(0).

(3) pruned_bondpp (full num_wann dimension). Same as pruned, plus the p-p
    hopping between the two bridge ligands on each enumerated RE-RE bond.
    Onsite ligand self blocks are not replaced, so --onsite-mode soc-delta
    still controls the ligand onsite energy.

Bands are H(k) = sum_R H(R) exp(2j pi k.R), diagonalised along the .win kpoint_path.
Plain-text gnuplot output (k_dist energy, one band per block).

Usage:
  python scripts/bond/enumerate.py <mat>.wout > bonds.txt
  python scripts/tb_from_enumerate.py bonds.txt

By default the downfold model uses the same ligand lambda_p inputs as
scripts/sweep-dft/make_inputs.py: Br=0.270838611, I=0.570592185, else zero.
Use --lambda-p to override all bridge ligands with one value.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "bond"))
sys.path.insert(0, str(REPO / "scripts" / "w90"))
from enumerate import RARE_EARTH, read_wout                   # noqa: E402
from prepare_downfold import lambda_for_downfold, ls_operator, soc_diagnostics  # noqa: E402
from fexchange.tools.w90_downfold import (                     # noqa: E402
    _p5_hole_propagator,
    fixed_p_ligand_hamiltonian,
)
from fexchange.tools.w90_extract import (                      # noqa: E402
    W90_F_REAL_ORDER,
    W90_P_REAL_ORDER,
    _shell_transform,
    read_w90_hr,
)

F_ORB, P_ORB = 14, 6   # SOC spinor orbitals: RE f = 7x2, ligand p = 3x2
CALC_RE = Path("/home/k0282/k028230/calculation/RE")
ZETA_BY_RE = {
    "Ce": 0.089441917,
    "Nd": 0.124237963,
    "Sm": 0.163421194,
    "Dy": 0.261607194,
    "Er": 0.319688417,
    "Yb": 0.390238722,
}
LAMBDA_P_BY_ELEMENT = {
    "Br": 0.270838611,
    "I": 0.570592185,
}


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


def read_w90_wsvec(path, *, needed_keys=None):
    """Read Wannier90 *_wsvec.dat into {(R, row, col): [T, ...]}.

    ``row`` and ``col`` are zero-based Wannier indices.  The phase for one
    matrix element is the average over exp(2 pi i k . (R + T)).
    """
    if path is None:
        return None
    ws_path = Path(path)
    if not ws_path.is_file():
        return None
    needed = set(needed_keys) if needed_keys is not None else None
    out = {}
    with ws_path.open(encoding="utf-8", errors="ignore") as handle:
        line_no = 0
        for line in handle:
            line_no += 1
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) != 5:
                raise ValueError(f"{ws_path}:{line_no}: expected 'R1 R2 R3 m n'")
            R = tuple(int(x) for x in parts[:3])
            row = int(parts[3]) - 1
            col = int(parts[4]) - 1
            key = (R, row, col)
            nvec_line = next(handle)
            line_no += 1
            nvec = int(nvec_line.split()[0])
            keep = needed is None or key in needed
            vecs = []
            for _ in range(nvec):
                vec_line = next(handle)
                line_no += 1
                if keep:
                    vec_parts = vec_line.split()
                    if len(vec_parts) != 3:
                        raise ValueError(f"{ws_path}:{line_no}: expected wsvec triplet")
                    vecs.append(tuple(int(x) for x in vec_parts))
            if keep:
                out[key] = tuple(vecs)
    return out


def wsvec_sidecar(hr_path):
    """Return the canonical sidecar path for a Wannier90 *_hr.dat file."""
    text = str(hr_path)
    if text.endswith("_hr.dat"):
        return Path(text.removesuffix("_hr.dat") + "_wsvec.dat")
    return Path(text + "_wsvec.dat")


def resolve_wsvec_path(hr_path, wout_path):
    """Locate wsvec next to data copy, then in the original calculation tree."""
    sidecar = wsvec_sidecar(hr_path)
    if sidecar.is_file():
        return sidecar

    stem = sidecar.name
    group, material = material_context_from_wout(wout_path)
    calc_root = calc_wannier_root(group)
    candidates = sorted(calc_root.glob(f"**/wannier90/{stem}"))
    candidates = [path for path in candidates if wsvec_candidate_ok(path, group, material)]
    return candidates[0] if candidates else None


def material_context_from_wout(wout_path):
    parts = Path(wout_path).parts
    for marker in ("data-DFT", "data-DFT-position"):
        if marker not in parts:
            continue
        i = parts.index(marker)
        if i + 2 < len(parts):
            return parts[i + 1], parts[i + 2]
    # enumerate_nn_bonds currently writes the data/data-DFT path in the first
    # line, so reaching this usually means the input was hand-written.
    return "", Path(wout_path).stem.split(".")[0]


def calc_wannier_root(group):
    if group == "REOF":
        return CALC_RE / "REChX" / "REChX_SCF_SOC_ACB"
    if group in {"REOCl", "RESI"}:
        return CALC_RE / "REChX" / "REChX_SCF_SOC_ABC"
    if group == "REX3-C2m":
        return CALC_RE / "REX3" / "REX3_SOC" / "REX3_C2m"
    if group == "REX3-R3":
        return CALC_RE / "REX3" / "REX3_SOC" / "REX3_R3"
    return CALC_RE


def wsvec_candidate_ok(path, group, material):
    parts = set(path.parts)
    wants_nc = material.endswith("-NC")
    has_nc = any(part.endswith("-NC") for part in path.parts)
    if group == "REX3-C2m" and wants_nc != has_nc:
        return False
    if group == "RESI" and "I-nore" in parts:
        return False
    return True


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


def build_pruned(
    H,
    atoms,
    blocks,
    pairs,
    num_wann,
    *,
    lambda_p,
    lambda_source,
    zeta_f,
    zeta_source,
    onsite_mode,
    hermitian_atol,
    drop_c_axis=False,
):
    """Pruned H(R): onsite + kept pairs, mirrored to keep it Hermitian."""
    pruned = {}
    H0 = H[(0, 0, 0)]
    U = {
        "f": _shell_transform("f", W90_F_REAL_ORDER, spinor_input=True),
        "p": _shell_transform("p", W90_P_REAL_ORDER, spinor_input=True),
    }

    def onsite_avg(blk):
        return float(np.trace(H0[np.ix_(blk, blk)]).real) / len(blk)

    def to_fxe(mat, row_shell, col_shell):
        return U[row_shell] @ mat @ U[col_shell].conj().T

    def from_fxe(mat, shell):
        return U[shell].conj().T @ mat @ U[shell]

    f_center = float(np.mean([onsite_avg(blocks[a]) for a, (el, _) in enumerate(atoms) if el in RARE_EARTH]))
    lambda_cache = {}
    zeta_cache = {}

    def lambda_for_ligand(lig):
        if lig in lambda_cache:
            return lambda_cache[lig]["value"]
        if lambda_p is not None:
            value, source = float(lambda_p), "override"
        elif lambda_source == "sweep-input":
            value, source = LAMBDA_P_BY_ELEMENT.get(atoms[lig][0], 0.0), "sweep-input"
        elif lambda_source == "onsite-fit":
            h_lig = to_fxe(block(H, (0, 0, 0), blocks[lig], blocks[lig]), "p", "p")
            fitted = soc_diagnostics(h_lig, ell=1, hermitian_atol=hermitian_atol)["lambda_eV"]
            value, source = lambda_for_downfold(atoms[lig][0], fitted)
        else:
            raise ValueError(f"unknown lambda_source: {lambda_source}")
        lambda_cache[lig] = {"value": float(value), "source": source, "element": atoms[lig][0]}
        return float(value)

    def zeta_for_re(atom):
        if atom in zeta_cache:
            return zeta_cache[atom]["value"]
        element = atoms[atom][0]
        if zeta_f is not None:
            value, source = float(zeta_f), "override"
        elif zeta_source == "sweep-input":
            value, source = ZETA_BY_RE[element], "sweep-input"
        elif zeta_source == "onsite-fit":
            h_re = to_fxe(block(H, (0, 0, 0), blocks[atom], blocks[atom]), "f", "f")
            value = soc_diagnostics(h_re, ell=3, hermitian_atol=hermitian_atol)["lambda_eV"]
            source = "onsite-fit"
        else:
            raise ValueError(f"unknown zeta_source: {zeta_source}")
        zeta_cache[atom] = {"value": float(value), "source": source, "element": element}
        return float(value)

    def put(R, rows, cols, mat):
        pruned.setdefault(R, np.zeros((num_wann, num_wann), complex))[np.ix_(rows, cols)] = mat

    for a in range(len(atoms)):                                    # onsite
        element = atoms[a][0]
        if onsite_mode == "hr":
            onsite = block(H, (0, 0, 0), blocks[a], blocks[a])
        elif onsite_mode == "soc-delta" and element in RARE_EARTH:
            onsite = from_fxe(zeta_for_re(a) * ls_operator(3), "f")
        elif onsite_mode == "soc-delta":
            delta = f_center - onsite_avg(blocks[a])
            onsite = from_fxe(fixed_p_ligand_hamiltonian(delta, lambda_for_ligand(a)), "p")
        else:
            raise ValueError(f"unknown onsite_mode: {onsite_mode}")
        put((0, 0, 0), blocks[a], blocks[a], onsite)
    for a, b, R in pairs:
        if drop_c_axis and R[2] != 0:
            continue
        mat = block(H, R, blocks[a], blocks[b])
        put(R, blocks[a], blocks[b], mat)
        put(tuple(-x for x in R), blocks[b], blocks[a], mat.conj().T)
    return pruned, {"lambda": lambda_cache, "zeta": zeta_cache}


def build_pruned_bondpp(H, blocks, pruned, re_bonds, num_wann, *, drop_c_axis=False):
    """Add p-p hopping only between bridge ligands on each enumerated RE-RE bond."""
    out = {R: mat.copy() for R, mat in pruned.items()}
    added = 0
    seen: set[tuple[tuple[int, int, int], int, int]] = set()

    def put(R, rows, cols, mat):
        out.setdefault(R, np.zeros((num_wann, num_wann), complex))[np.ix_(rows, cols)] = mat

    for bond in re_bonds:
        bridges = bond["bridges"]
        for left in range(len(bridges)):
            for right in range(left + 1, len(bridges)):
                lig_a, cell_a = bridges[left]
                lig_b, cell_b = bridges[right]
                R = tuple(cell_b[idx] - cell_a[idx] for idx in range(3))
                if lig_a == lig_b and R == (0, 0, 0):
                    continue
                if drop_c_axis and R[2] != 0:
                    continue
                key = (R, lig_a, lig_b)
                mirror_key = (tuple(-x for x in R), lig_b, lig_a)
                if key in seen or mirror_key in seen:
                    continue
                block_mat = block(H, R, blocks[lig_a], blocks[lig_b])
                if not np.any(np.abs(block_mat) > 1.0e-12):
                    continue
                put(R, blocks[lig_a], blocks[lig_b], block_mat)
                put(tuple(-x for x in R), blocks[lig_b], blocks[lig_a], block_mat.conj().T)
                seen.add(key)
                seen.add(mirror_key)
                added += 1
    return out, added


def build_downfolded(
    H,
    atoms,
    blocks,
    re_bonds,
    *,
    lambda_p,
    lambda_source,
    zeta_f,
    zeta_source,
    onsite_mode,
    degenerate_tol,
    hermitian_atol,
    drop_c_axis=False,
):
    """RE-only effective model: fold each bridging ligand into the RE-RE hopping.
        t_eff = t_direct + sum_lig t_iL @ G_p(Delta_lig, lambda_lig) @ t_jL^dagger
        Delta_lig = <f_i,f_j onsite trace> - <ligand onsite trace>, from H(0).
    Returns the effective H(R) dict on the RE-only basis (dim = n_RE * 14)."""
    re_atoms = [k for k, (el, _) in enumerate(atoms) if el in RARE_EARTH]
    pos = {k: n for n, k in enumerate(re_atoms)}
    rb = {k: list(range(pos[k] * F_ORB, pos[k] * F_ORB + F_ORB)) for k in re_atoms}
    dim = len(re_atoms) * F_ORB
    H0 = H[(0, 0, 0)]
    U = {
        "f": _shell_transform("f", W90_F_REAL_ORDER, spinor_input=True),
        "p": _shell_transform("p", W90_P_REAL_ORDER, spinor_input=True),
    }

    def onsite_avg(blk):
        return float(np.trace(H0[np.ix_(blk, blk)]).real) / len(blk)

    def to_fxe(mat, row_shell, col_shell):
        return U[row_shell] @ mat @ U[col_shell].conj().T

    lambda_cache = {}
    zeta_cache = {}

    def lambda_for_ligand(lig):
        if lig in lambda_cache:
            return lambda_cache[lig]
        if lambda_p is not None:
            value, source = float(lambda_p), "override"
        elif lambda_source == "sweep-input":
            value, source = LAMBDA_P_BY_ELEMENT.get(atoms[lig][0], 0.0), "sweep-input"
        elif lambda_source == "onsite-fit":
            h_lig = to_fxe(block(H, (0, 0, 0), blocks[lig], blocks[lig]), "p", "p")
            fitted = soc_diagnostics(h_lig, ell=1, hermitian_atol=hermitian_atol)["lambda_eV"]
            value, source = lambda_for_downfold(atoms[lig][0], fitted)
        else:
            raise ValueError(f"unknown lambda_source: {lambda_source}")
        lambda_cache[lig] = {"value": float(value), "source": source, "element": atoms[lig][0]}
        return lambda_cache[lig]

    def zeta_for_re(atom):
        if atom in zeta_cache:
            return zeta_cache[atom]
        element = atoms[atom][0]
        if zeta_f is not None:
            value, source = float(zeta_f), "override"
        elif zeta_source == "sweep-input":
            value, source = ZETA_BY_RE[element], "sweep-input"
        elif zeta_source == "onsite-fit":
            h_re = to_fxe(block(H, (0, 0, 0), blocks[atom], blocks[atom]), "f", "f")
            value = soc_diagnostics(h_re, ell=3, hermitian_atol=hermitian_atol)["lambda_eV"]
            source = "onsite-fit"
        else:
            raise ValueError(f"unknown zeta_source: {zeta_source}")
        zeta_cache[atom] = {"value": float(value), "source": source, "element": element}
        return zeta_cache[atom]

    eff = {}

    def put(R, rows, cols, mat):
        eff.setdefault(R, np.zeros((dim, dim), complex))[np.ix_(rows, cols)] = mat

    for k in re_atoms:                                             # RE onsite (f block)
        if onsite_mode == "hr":
            onsite = to_fxe(block(H, (0, 0, 0), blocks[k], blocks[k]), "f", "f")
        elif onsite_mode == "soc-delta":
            onsite = zeta_for_re(k)["value"] * ls_operator(3)
        else:
            raise ValueError(f"unknown onsite_mode: {onsite_mode}")
        put((0, 0, 0), rb[k], rb[k], onsite)
    for bd in re_bonds:
        i, j, R = bd["i"], bd["j"], bd["R"]
        if drop_c_axis and R[2] != 0:
            continue
        t = to_fxe(block(H, R, blocks[i], blocks[j]), "f", "f")    # direct f-f, 14x14
        e_f = 0.5 * (onsite_avg(blocks[i]) + onsite_avg(blocks[j]))
        for lig, cL in bd["bridges"]:
            j_to_lig = tuple(cL[a] - R[a] for a in range(3))
            if drop_c_axis and (cL[2] != 0 or j_to_lig[2] != 0):
                continue
            delta = e_f - onsite_avg(blocks[lig])
            lam = lambda_for_ligand(lig)["value"]
            g_p, _, _ = _p5_hole_propagator(delta, lam, degenerate_tol=degenerate_tol)
            t_iL = to_fxe(block(H, cL, blocks[i], blocks[lig]), "f", "p")  # RE_i(0)->L(cL)
            t_jL = to_fxe(block(H, j_to_lig, blocks[j], blocks[lig]), "f", "p")
            t = t + t_iL @ g_p @ t_jL.conj().T
        put(R, rb[i], rb[j], t)
        put(tuple(-x for x in R), rb[j], rb[i], t.conj().T)
    basis_map = [idx for atom in re_atoms for idx in blocks[atom]]
    return eff, lambda_cache, zeta_cache, basis_map


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


def prepare_band_terms(Hdict, *, wsvec=None, basis_map=None, zero_tol=0.0):
    dim = next(iter(Hdict.values())).shape[0]
    if basis_map is None:
        basis_map = list(range(dim))
    basis_map = np.asarray(basis_map, dtype=int)
    rows_all, cols_all, R_all, vals_all = [], [], [], []
    for R, mat in Hdict.items():
        rows, cols = np.nonzero(np.abs(mat) > zero_tol)
        for row, col in zip(rows, cols):
            value = mat[row, col]
            row_orig = int(basis_map[row])
            col_orig = int(basis_map[col])
            vecs = None if wsvec is None else wsvec.get((R, row_orig, col_orig))
            if not vecs:
                vecs = ((0, 0, 0),)
            weight = value / len(vecs)
            for vec in vecs:
                rows_all.append(row)
                cols_all.append(col)
                R_all.append(tuple(R[i] + vec[i] for i in range(3)))
                vals_all.append(weight)
    return (
        np.asarray(rows_all, dtype=int),
        np.asarray(cols_all, dtype=int),
        np.asarray(R_all, dtype=float),
        np.asarray(vals_all, dtype=np.complex128),
        dim,
    )


def wsvec_keys_for_model(Hdict, *, basis_map=None, zero_tol=0.0):
    dim = next(iter(Hdict.values())).shape[0]
    if basis_map is None:
        basis_map = list(range(dim))
    basis_map = np.asarray(basis_map, dtype=int)
    keys = set()
    for R, mat in Hdict.items():
        rows, cols = np.nonzero(np.abs(mat) > zero_tol)
        for row, col in zip(rows, cols):
            keys.add((R, int(basis_map[row]), int(basis_map[col])))
    return keys


def bands(Hdict, kfracs, *, wsvec=None, basis_map=None):
    rows, cols, Rs, vals, dim = prepare_band_terms(Hdict, wsvec=wsvec, basis_map=basis_map)
    out = []
    for k in kfracs:
        phase = np.exp(2j * np.pi * (Rs @ k))
        Hk = np.zeros((dim, dim), dtype=np.complex128)
        np.add.at(Hk, (rows, cols), vals * phase)
        Hk = 0.5 * (Hk + Hk.conj().T)
        out.append(np.linalg.eigvalsh(Hk))
    return np.array(out)


def write_bands(path, dist, ev, ticks, *, tb_shift_mode="efermi"):
    """gnuplot format: two columns 'k_dist energy', one band per block, blocks
    separated by a blank line (plot 'file' w l draws each band as one line)."""
    with path.open("w") as f:
        f.write("# ticks: " + "  ".join(f"{lbl}={d:.4f}" for d, lbl in ticks) + "\n")
        f.write(f"# tb_shift_mode: {tb_shift_mode}\n")
        f.write(f"# subtract_efermi_in_plot: {str(tb_shift_mode == 'efermi').lower()}\n")
        f.write("# k_dist(1/Ang)  energy(eV)\n")
        for b in range(ev.shape[1]):
            for d, e in zip(dist, ev[:, b]):
                f.write(f"{d:.6f} {e:.6f}\n")
            f.write("\n")


def main(
    path,
    *,
    out_dir=None,
    lambda_p=None,
    lambda_source="sweep-input",
    zeta_f=None,
    zeta_source="sweep-input",
    onsite_mode="hr",
    tb_plot_shift="fit",
    wsvec_mode="auto",
    degenerate_tol=1.0e-6,
    hermitian_atol=1.0e-8,
    drop_c_axis=False,
    suffix="",
    write_pruned_bondpp=True,
):
    wout, pairs, re_bonds = parse_input(Path(path).read_text())
    hr_path = next(Path(wout).parent.glob("*_hr.dat"))
    win_path = next(Path(wout).parent.glob("*.win"))
    H, num_wann = read_w90_hr(hr_path)
    wsvec_path = None
    wsvec = None
    if wsvec_mode != "off":
        wsvec_path = resolve_wsvec_path(hr_path, wout)
        if wsvec_path is None and wsvec_mode == "required":
            raise FileNotFoundError(f"wsvec not found for {hr_path}")
    lattice, atoms, blocks = atom_blocks(wout, num_wann)

    segs, npts = read_win_kpath(win_path)
    kfracs, dist, ticks = kpath(segs, npts, lattice)
    stem = Path(wout).stem.split(".")[0]
    out_dir = Path(out_dir) if out_dir is not None else REPO / "outputs"
    out_dir.mkdir(exist_ok=True)

    if tb_plot_shift == "fermi":
        tb_plot_shift = "efermi"
    tb_shift_mode = tb_plot_shift
    pruned = build_pruned(
        H,
        atoms,
        blocks,
        pairs,
        num_wann,
        lambda_p=lambda_p,
        lambda_source=lambda_source,
        zeta_f=zeta_f,
        zeta_source=zeta_source,
        onsite_mode=onsite_mode,
        hermitian_atol=hermitian_atol,
        drop_c_axis=drop_c_axis,
    )
    pruned, pruned_meta = pruned
    pruned_path = out_dir / f"{stem}_bands{suffix}.txt"
    pruned_bondpp_path = out_dir / f"{stem}_bands_pruned_bondpp{suffix}.txt"
    downfold_path = out_dir / f"{stem}_bands_downfold{suffix}.txt"
    pruned_bondpp = None
    pruned_bondpp_count = 0
    if write_pruned_bondpp:
        pruned_bondpp, pruned_bondpp_count = build_pruned_bondpp(
            H,
            blocks,
            pruned,
            re_bonds,
            num_wann,
            drop_c_axis=drop_c_axis,
        )
    eff, lambda_cache, zeta_cache, downfold_basis_map = build_downfolded(
        H,
        atoms,
        blocks,
        re_bonds,
        lambda_p=lambda_p,
        lambda_source=lambda_source,
        zeta_f=zeta_f,
        zeta_source=zeta_source,
        onsite_mode=onsite_mode,
        degenerate_tol=degenerate_tol,
        hermitian_atol=hermitian_atol,
        drop_c_axis=drop_c_axis,
    )
    if wsvec_path is not None:
        needed_keys = wsvec_keys_for_model(pruned)
        if pruned_bondpp is not None:
            needed_keys.update(wsvec_keys_for_model(pruned_bondpp))
        needed_keys.update(wsvec_keys_for_model(eff, basis_map=downfold_basis_map))
        wsvec = read_w90_wsvec(wsvec_path, needed_keys=needed_keys)
    write_bands(
        pruned_path,
        dist,
        bands(pruned, kfracs, wsvec=wsvec),
        ticks,
        tb_shift_mode=tb_shift_mode,
    )
    if pruned_bondpp is not None:
        write_bands(
            pruned_bondpp_path,
            dist,
            bands(pruned_bondpp, kfracs, wsvec=wsvec),
            ticks,
            tb_shift_mode=tb_shift_mode,
        )
    write_bands(
        downfold_path,
        dist,
        bands(eff, kfracs, wsvec=wsvec, basis_map=downfold_basis_map),
        ticks,
        tb_shift_mode=tb_shift_mode,
    )

    pp_text = f"  pruned_bondpp bridge p-p pairs={pruned_bondpp_count}" if pruned_bondpp is not None else ""
    print(f"pruned dim={num_wann} (kept pairs={len(pairs)}){pp_text}  "
          f"downfold dim={eff[(0, 0, 0)].shape[0]} (RE-RE bonds={len(re_bonds)})  k-points={len(kfracs)}")
    print(f"onsite_mode={onsite_mode}")
    print(f"tb_plot_shift={tb_shift_mode}")
    if wsvec is not None:
        print(f"wsvec: {wsvec_path}")
    else:
        print(f"wsvec: none ({wsvec_mode})")
    if drop_c_axis:
        print("drop_c_axis: omitted all original hopping blocks with R_z != 0")
    lambda_summary = {**pruned_meta["lambda"], **lambda_cache}
    if lambda_summary:
        print("ligand SOC: " + ", ".join(
            f"{meta['element']}#{lig}: lambda={meta['value']:.6g} eV ({meta['source']})"
            for lig, meta in sorted(lambda_summary.items())
        ))
    zeta_summary = {**pruned_meta["zeta"], **zeta_cache}
    if zeta_summary:
        print("RE SOC: " + ", ".join(
            f"{meta['element']}#{atom}: zeta={meta['value']:.6g} eV ({meta['source']})"
            for atom, meta in sorted(zeta_summary.items())
        ))
    print(f"wrote {pruned_path}")
    if pruned_bondpp is not None:
        print(f"wrote {pruned_bondpp_path}")
    print(f"wrote {downfold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bonds", help="Output file from scripts/bond/enumerate.py")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for band text outputs")
    parser.add_argument("--lambda-p", type=float, default=None, help="Override ligand lambda_p in eV for all bridge ligands")
    parser.add_argument("--lambda-source", choices=("sweep-input", "onsite-fit"), default="sweep-input")
    parser.add_argument("--zeta-f", type=float, default=None, help="Override RE f-shell zeta in eV for all RE sites")
    parser.add_argument("--zeta-source", choices=("sweep-input", "onsite-fit"), default="sweep-input")
    parser.add_argument("--onsite-mode", choices=("hr", "soc-delta"), default="hr")
    parser.add_argument(
        "--tb-plot-shift",
        choices=("fit", "efermi", "fermi", "zero"),
        default="fit",
        help="Metadata for plot alignment; plotting defaults to fit regardless unless --tb-shift-mode header is used.",
    )
    parser.add_argument(
        "--soc-delta-plot-shift",
        choices=("fit", "efermi", "fermi", "zero"),
        default=None,
        help="Deprecated alias for --tb-plot-shift used with --onsite-mode soc-delta.",
    )
    parser.add_argument("--wsvec", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--degenerate-tol", type=float, default=1.0e-6)
    parser.add_argument("--hermitian-atol", type=float, default=1.0e-8)
    parser.add_argument("--drop-c-axis", action="store_true", help="Omit original hopping blocks with R_z != 0")
    parser.add_argument("--suffix", default="", help="Suffix inserted before .txt in output band filenames")
    parser.add_argument(
        "--no-pruned-bondpp",
        action="store_true",
        help="Do not write the pruned model with bridge-ligand p-p hoppings restored",
    )
    args = parser.parse_args()
    main(
        args.bonds,
        out_dir=args.out_dir,
        lambda_p=args.lambda_p,
        lambda_source=args.lambda_source,
        zeta_f=args.zeta_f,
        zeta_source=args.zeta_source,
        onsite_mode=args.onsite_mode,
        tb_plot_shift=args.soc_delta_plot_shift or args.tb_plot_shift,
        wsvec_mode=args.wsvec,
        degenerate_tol=args.degenerate_tol,
        hermitian_atol=args.hermitian_atol,
        drop_c_axis=args.drop_c_axis,
        suffix=args.suffix,
        write_pruned_bondpp=not args.no_pruned_bondpp,
    )
