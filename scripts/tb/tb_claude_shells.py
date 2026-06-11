#!/usr/bin/env python3
"""Distance-shell extension analysis: what minimal additions to the pruned TB
reproduce the DFT near-Fermi bands?

Starting from the enumerate-based pruned model (SOC+Delta onsite), classify all
HR blocks it discards by category and bond distance:
  pp : ligand-ligand    fp : RE-ligand    ff : RE-RE (3rd NN and beyond)
then re-add them shell by shell (cumulative, distance ordered) and score each
model with the near-Fermi level-set metrics of tb_claude_metrics.py.

Per material, writes into <material>/tb-claude/:
  shells.tsv   shell catalogue (category, distance, #blocks, Frobenius norm)
  metrics.tsv  one row per model: baselines, cumulative shells, combos, ceiling
  band txt + near-Fermi plots (plot_tb_bands format) for the baselines, the
  selected minimal model, and the everything-within-dmax model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tb"))
import plot_tb_bands as plot  # noqa: E402
import tb_from_enumerate as tb  # noqa: E402
from tb_claude_metrics import fit_shift, level_set_metrics, sampled_levels, w90_band_dat  # noqa: E402
from enumerate import RARE_EARTH  # noqa: E402

WINDOW = (-0.25, 0.25)
METRIC_COLS = ["shift", "chamfer", "chamfer_ip", "miss", "spur", "env", "envpp", "cover", "nratio"]


def onsite_args(onsite_mode: str) -> dict:
    return dict(lambda_p=None, lambda_source="sweep-input", zeta_f=None,
                zeta_source="sweep-input", onsite_mode=onsite_mode, hermitian_atol=1.0e-8)


def canonical(a: int, b: int, R: tuple) -> tuple:
    mirror = (b, a, tuple(-x for x in R))
    return min((a, b, R), mirror)


def stacking_normal(lattice):
    """Unit normal of the slab plane spanned by a1, a2 (a3 is the stacking
    direction; REX3-R3 cells use the cubic frame where this is (1,1,1))."""
    nhat = np.cross(lattice[0], lattice[1])
    return nhat / np.linalg.norm(nhat)


def slab_ids(lattice, atoms):
    """Slab index per atom. RE projections onto the stacking normal are split
    into n_RE/2 groups at the largest circular gaps (every family stacks slabs
    of 2 RE sheets); each ligand joins the slab of its nearest RE. a1/a2 are
    exactly in-plane (normal = a1 x a2), so the image of atom b displaced by R
    lives in slab s[b] + R[2] * n_slabs; intra-slab iff that equals s[a].
    Note this is purely geometric: REOF-type structures keep covalent RE-F
    bonds across the slab boundary and still count them as inter-slab."""
    nhat = stacking_normal(lattice)
    height = float(lattice[2] @ nhat)
    fracs = [frac for _, frac in atoms]
    re_idx = [i for i, (el, _) in enumerate(atoms) if el in RARE_EARTH]
    n_slabs = max(1, len(re_idx) // 2)
    raw = {i: float((fracs[i] @ lattice) @ nhat) for i in range(len(atoms))}
    # place the origin inside the largest inter-RE gap so no slab straddles it
    re_pos = sorted(raw[i] % height for i in re_idx)
    re_gaps = [(re_pos[(k + 1) % len(re_pos)] - re_pos[k]) % height for k in range(len(re_pos))]
    kmax = max(range(len(re_gaps)), key=lambda k: re_gaps[k])
    z0 = re_pos[kmax] + re_gaps[kmax] / 2.0
    wrap = {i: int(np.floor((raw[i] - z0) / height)) for i in range(len(atoms))}
    pos = {i: raw[i] - z0 - wrap[i] * height for i in range(len(atoms))}
    if n_slabs == 1:
        s = {i: 0 for i in re_idx}
    else:
        order = sorted(re_idx, key=lambda i: pos[i])
        gaps = [(pos[order[(k + 1) % len(order)]] - pos[order[k]]) % height for k in range(len(order))]
        cuts = set(sorted(range(len(order)), key=lambda k: -gaps[k])[:n_slabs])
        if (len(order) - 1) not in cuts:
            raise ValueError("slab straddles the cell boundary; unwrap not implemented")
        s, group = {}, 0
        for k, i in enumerate(order):
            s[i] = group
            if k in cuts:
                group += 1
    G = {i: s[i] + wrap[i] * n_slabs for i in re_idx}
    cells = [np.array([r0, r1, r2]) for r0 in (-1, 0, 1) for r1 in (-1, 0, 1) for r2 in (-1, 0, 1)]
    for i in range(len(atoms)):
        if i in G:
            continue
        _, j, shift = min(
            (np.linalg.norm((fracs[j] - fracs[i] + R) @ lattice), j, tuple(R))
            for j in re_idx for R in cells
        )
        G[i] = G[j] + shift[2] * n_slabs
    return G, n_slabs


def intra_slab(sids, n_slabs: int, a: int, b: int, R) -> bool:
    return sids[a] == sids[b] + R[2] * n_slabs


def find_candidates(H, atoms, blocks, lattice, kept, *, dmax: float, tol: float = 1.0e-10,
                    frames=None):
    """All HR blocks not in the pruned model, keyed canonically.
    Returns {key: (category, distance, frobenius_norm, intra_slab)}."""
    fracs = [frac for _, frac in atoms]
    is_re = [element in RARE_EARTH for element, _ in atoms]
    cands = {}
    for R, HR in H.items():
        for a in range(len(atoms)):
            for b in range(len(atoms)):
                key = canonical(a, b, R)
                if key != (a, b, R) or key in cands:
                    continue
                if (a, b, R) in kept or (R == (0, 0, 0) and a == b):
                    continue
                v = (fracs[b] - fracs[a] + np.array(R)) @ lattice
                d = float(np.linalg.norm(v))
                if d > dmax:
                    continue
                norm = float(np.linalg.norm(HR[np.ix_(blocks[a], blocks[b])]))
                if norm < tol:
                    continue
                cat = "ff" if (is_re[a] and is_re[b]) else "pp" if not (is_re[a] or is_re[b]) else "fp"
                intra = frames is None or intra_slab(frames[0], frames[1], a, b, R)
                cands[key] = (cat, d, norm, intra)
    return cands


def group_shells(cands, *, gap: float = 0.05):
    """Cluster candidate blocks into distance shells per category.
    Returns {category: [(distance, [keys]), ...]} sorted by distance."""
    shells = {"pp": [], "fp": [], "ff": []}
    for cat in shells:
        items = sorted(((d, key) for key, (c, d, *_) in cands.items() if c == cat))
        for d, key in items:
            if shells[cat] and d - shells[cat][-1][0][-1] <= gap:
                shells[cat][-1][0].append(d)
                shells[cat][-1][1].append(key)
            else:
                shells[cat].append(([d], [key]))
    return {cat: [(float(np.mean(ds)), keys) for ds, keys in lst] for cat, lst in shells.items()}


def extend(base, H, blocks, num_wann, keys):
    out = {R: mat.copy() for R, mat in base.items()}
    for a, b, R in keys:
        mat = tb.block(H, R, blocks[a], blocks[b])
        out.setdefault(R, np.zeros((num_wann, num_wann), complex))[np.ix_(blocks[a], blocks[b])] = mat
        Rm = tuple(-x for x in R)
        out.setdefault(Rm, np.zeros((num_wann, num_wann), complex))[np.ix_(blocks[b], blocks[a])] = mat.conj().T
    return out


def run_material(bonds: Path, *, dmax: float, max_shells: dict, window,
                 onsite_mode: str = "soc-delta", tag: str = "socdelta",
                 make_plots: bool = True, in_plane: bool = False) -> list:
    wout, pairs, re_bonds = tb.parse_input(bonds.read_text(encoding="utf-8"))
    hr_path = next(Path(wout).parent.glob("*_hr.dat"))
    win_path = next(Path(wout).parent.glob("*.win"))
    H, num_wann = tb.read_w90_hr(hr_path)
    lattice, atoms, blocks = tb.atom_blocks(wout, num_wann)
    segs, npts = tb.read_win_kpath(win_path)
    kfracs, dist, ticks = tb.kpath(segs, npts, lattice)
    stem = Path(wout).stem.split(".")[0]
    out_dir = bonds.parent / "tb-claude"
    out_dir.mkdir(exist_ok=True)

    pruned, _ = tb.build_pruned(H, atoms, blocks, pairs, num_wann, **onsite_args(onsite_mode))
    bondpp, n_bondpp = tb.build_pruned_bondpp(H, blocks, pruned, re_bonds, num_wann)
    downfold, _, _, downfold_map = tb.build_downfolded(
        H, atoms, blocks, re_bonds, degenerate_tol=1.0e-6, **onsite_args(onsite_mode))

    kept = set()
    for a, b, R in pairs:
        kept.add((a, b, R))
        kept.add((b, a, tuple(-x for x in R)))
    frames = slab_ids(lattice, atoms)
    kept_inter = sum(1 for a, b, R in pairs if not intra_slab(frames[0], frames[1], a, b, R))
    print(f"  slabs={frames[1]}  kept inter-slab pairs={kept_inter}", flush=True)
    cands_all = find_candidates(H, atoms, blocks, lattice, kept, dmax=dmax, frames=frames)
    cands = {k: v for k, v in cands_all.items() if v[3]} if in_plane else cands_all
    shells = group_shells(cands)

    shell_rows = []
    for cat in ("pp", "fp", "ff"):
        for n, (d, keys) in enumerate(shells[cat], start=1):
            frob = float(np.sqrt(sum(cands[k][2] ** 2 for k in keys)))
            shell_rows.append(f"{cat}{n}\t{d:.3f}\t{len(keys)}\t{frob:.4f}")
    (out_dir / "shells.tsv").write_text(
        "shell\tdistance_A\tn_blocks\tfrob_eV\n" + "\n".join(shell_rows) + "\n", encoding="utf-8")

    # model list: baselines, cumulative shells per category, combos
    models = [("pruned", pruned, None, 0), ("bondpp", bondpp, None, n_bondpp),
              ("downfold", downfold, downfold_map, 0)]
    for cat in ("pp", "fp", "ff"):
        keys = []
        for n, (d, shell_keys) in enumerate(shells[cat][: max_shells[cat]], start=1):
            keys = keys + shell_keys
            models.append((f"{cat}{n}_d{d:.2f}", extend(pruned, H, blocks, num_wann, keys), None, len(keys)))
    all_pp = [k for s in shells["pp"] for k in s[1]]
    all_fp = [k for s in shells["fp"] for k in s[1]]
    all_ff = [k for s in shells["ff"] for k in s[1]]
    combos = [("pp+fp", all_pp + all_fp), ("pp+ff", all_pp + all_ff), ("all", all_pp + all_fp + all_ff)]
    if in_plane:  # reference: same pp additions without the intra-slab restriction
        combos.append(("pp_3dref", [k for k, v in cands_all.items() if v[0] == "pp"]))
    for name, keys in combos:
        models.append((name, extend(pruned, H, blocks, num_wann, keys), None, len(keys)))

    wsvec_path = tb.resolve_wsvec_path(hr_path, wout)
    wsvec = None
    if wsvec_path is not None:
        needed = set()
        for _, model, basis_map, _ in models:
            needed.update(tb.wsvec_keys_for_model(model, basis_map=basis_map))
        wsvec = tb.read_w90_wsvec(wsvec_path, needed_keys=needed)

    dft_band = plot.resolve_dft_band(out_dir.parent / "tb", stem)
    efermi = plot.resolve_efermi(out_dir.parent / "tb", dft_band)
    dft_nodes = plot.read_dft_nodes(dft_band)
    pdos_spec = plot.resolve_pdos(dft_band, stem)

    # normalised x intervals of the in-plane (k perpendicular to stacking) segments
    B = 2 * np.pi * np.linalg.inv(lattice).T
    nhat = stacking_normal(lattice)
    seg_ip = [(ticks[s][0] / dist[-1], ticks[s + 1][0] / dist[-1])
              for s, (_, k1, _, k2) in enumerate(segs)
              if abs((k1 @ B) @ nhat) < 1e-4 and abs((k2 @ B) @ nhat) < 1e-4]

    def score(band_path):
        dft_mat, tb_mat, xn = sampled_levels(dft_band, band_path, efermi)
        shift = fit_shift(dft_mat, tb_mat, efermi, window)
        m = level_set_metrics(dft_mat, tb_mat, shift, window)
        mask = np.zeros(len(xn), bool)
        for x0, x1 in seg_ip:
            mask |= (xn >= x0 - 1e-9) & (xn <= x1 + 1e-9)
        m["chamfer_ip"] = (level_set_metrics(dft_mat[mask], tb_mat[mask], shift, window)["chamfer"]
                           if mask.any() else float("nan"))
        return m

    rows, band_files = [], {}
    for name, model, basis_map, n_added in models:
        ev = tb.bands(model, kfracs, wsvec=wsvec, basis_map=basis_map)
        band_path = out_dir / f"{stem}_bands_claude_{name}_{tag}.txt"
        tb.write_bands(band_path, dist, ev, ticks, tb_shift_mode="fit")
        band_files[name] = band_path
        m = score(band_path)
        rows.append([name, n_added] + [m[c] for c in METRIC_COLS])
        print(f"  {name:14s} added={n_added:5d} " +
              " ".join(f"{m[c]:8.4f}" for c in METRIC_COLS), flush=True)
    full_band = w90_band_dat(out_dir.parent / "tb")
    if full_band is not None:
        m = score(full_band)
        rows.append(["w90_full", -1] + [m[c] for c in METRIC_COLS])

    # minimal model: fewest added blocks among extensions whose chamfer is within
    # 10% + 0.5 meV of the best, with a sane level density (guards degenerate fits)
    ext = [r for r in rows if r[0] not in ("pruned", "downfold", "w90_full", "pp_3dref")]
    sane = [r for r in ext if 0.8 <= r[10] <= 1.25 and r[9] >= 0.95]
    pool = sane if sane else ext
    best = min(r[3] for r in pool)
    eligible = [r for r in pool if r[3] <= best * 1.1 + 5.0e-4]
    minimal = min(eligible, key=lambda r: r[1])[0]

    header = "\t".join(["model", "n_added"] + METRIC_COLS)
    body = "\n".join("\t".join([r[0], str(r[1])] + [f"{v:.6f}" for v in r[2:]]) for r in rows)
    (out_dir / f"metrics_{tag}.tsv").write_text(
        f"# window {window[0]} .. {window[1]} eV; onsite={onsite_mode}; dmax={dmax}; "
        f"in_plane={in_plane}; minimal={minimal}\n"
        f"{header}\n{body}\n", encoding="utf-8")

    keep = {"pruned", "bondpp", "downfold", minimal, "all", "pp_3dref"}
    for name, band_path in list(band_files.items()):
        if name not in keep:
            band_path.unlink()
            continue
        row = next(r for r in rows if r[0] == name)
        png = out_dir / f"{stem}_dft_vs_claude_{name}_{tag}.near.-0.25_0.25.png"
        script = png.with_suffix(".gnuplot")
        script.write_text(plot.build_gnuplot(
            dft_band=dft_band, tb_band=band_path, output=png, stem=stem,
            tb_title=f"claude {name} TB ({tag} onsite)", dft_nodes=dft_nodes,
            tb_ticks=plot.read_tb_ticks(band_path), efermi=efermi, tb_shift=-row[2],
            window=window, pdos_spec=pdos_spec), encoding="utf-8")
        if make_plots:
            subprocess.run(["gnuplot", str(script)], check=True)
    print(f"  -> minimal={minimal}  wrote {out_dir}/metrics_{tag}.tsv shells.tsv + plots", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="group/material; default all")
    parser.add_argument("--input-root", type=Path, default=REPO / "data" / "data-DFT-input")
    parser.add_argument("--dmax", type=float, default=8.0)
    parser.add_argument("--max-pp", type=int, default=8)
    parser.add_argument("--max-fp", type=int, default=6)
    parser.add_argument("--max-ff", type=int, default=6)
    parser.add_argument("--window", nargs=2, type=float, default=list(WINDOW))
    parser.add_argument("--onsite-mode", choices=("soc-delta", "hr"), default="soc-delta")
    parser.add_argument("--tag", default=None, help="Output name tag; defaults to socdelta/hr")
    parser.add_argument("--no-plot", action="store_true",
                        help="Write .gnuplot scripts but do not run gnuplot (no-gnuplot hosts)")
    parser.add_argument("--in-plane", action="store_true",
                        help="Only add intra-slab (in-plane) blocks; adds a pp_3dref reference row")
    args = parser.parse_args()
    max_shells = {"pp": args.max_pp, "fp": args.max_fp, "ff": args.max_ff}
    tag = args.tag or (("socdelta2d" if args.in_plane else "socdelta")
                       if args.onsite_mode == "soc-delta" else "hr")

    bonds_files = ([args.input_root / t / "enumerate_nn_bonds.txt" for t in args.targets]
                   if args.targets else sorted(args.input_root.glob("*/*/enumerate_nn_bonds.txt")))
    for idx, bonds in enumerate(bonds_files, start=1):
        print(f"[{idx:02d}/{len(bonds_files):02d}] {bonds.parent.relative_to(args.input_root)}", flush=True)
        run_material(bonds, dmax=args.dmax, max_shells=max_shells, window=tuple(args.window),
                     onsite_mode=args.onsite_mode, tag=tag, make_plots=not args.no_plot,
                     in_plane=args.in_plane)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
