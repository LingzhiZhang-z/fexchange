#!/usr/bin/env python3
"""Strict slab-filtered shell extension for the TB near-Fermi test.

Compared with ``tb_claude_shells.py --in-plane``, this script also removes
inter-slab blocks from the starting enumerate/pruned model before adding any
extra shells.  Results are written under ``<material>/tb-gpt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tb"))
import plot_tb_bands as plot  # noqa: E402
import tb_claude_shells as shells  # noqa: E402
import tb_from_enumerate as tb  # noqa: E402
from tb_claude_metrics import fit_shift, level_set_metrics, sampled_levels, w90_band_dat  # noqa: E402


WINDOW = (-0.25, 0.25)
METRIC_COLS = ["shift", "chamfer", "chamfer_ip", "miss", "spur", "env", "envpp", "cover", "nratio"]


def filter_pairs(pairs: set[tuple[int, int, tuple[int, int, int]]], frames) -> set[tuple[int, int, tuple[int, int, int]]]:
    sids, n_slabs = frames
    return {pair for pair in pairs if shells.intra_slab(sids, n_slabs, *pair)}


def filter_re_bonds(re_bonds: list[dict], frames) -> tuple[list[dict], int, int]:
    """Keep only RE-RE bonds and ligand bridge legs that stay inside one slab."""
    sids, n_slabs = frames
    out = []
    dropped_bonds = 0
    dropped_bridges = 0
    for bond in re_bonds:
        i, j, R = bond["i"], bond["j"], bond["R"]
        if not shells.intra_slab(sids, n_slabs, i, j, R):
            dropped_bonds += 1
            continue
        bridges = []
        for lig, c_lig in bond["bridges"]:
            j_to_lig = tuple(c_lig[idx] - R[idx] for idx in range(3))
            if (
                shells.intra_slab(sids, n_slabs, i, lig, c_lig)
                and shells.intra_slab(sids, n_slabs, j, lig, j_to_lig)
            ):
                bridges.append((lig, c_lig))
            else:
                dropped_bridges += 1
        copied = dict(bond)
        copied["bridges"] = bridges
        out.append(copied)
    return out, dropped_bonds, dropped_bridges


def build_bondpp_intra(H, blocks, pruned, re_bonds, num_wann: int, frames) -> tuple[dict, int]:
    """Add bridge-ligand p-p blocks only when the p-p block is intra-slab."""
    sids, n_slabs = frames
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
                if not shells.intra_slab(sids, n_slabs, lig_a, lig_b, R):
                    continue
                key = (R, lig_a, lig_b)
                mirror_key = (tuple(-x for x in R), lig_b, lig_a)
                if key in seen or mirror_key in seen:
                    continue
                block_mat = tb.block(H, R, blocks[lig_a], blocks[lig_b])
                if not np.any(np.abs(block_mat) > 1.0e-12):
                    continue
                put(R, blocks[lig_a], blocks[lig_b], block_mat)
                put(tuple(-x for x in R), blocks[lig_b], blocks[lig_a], block_mat.conj().T)
                seen.add(key)
                seen.add(mirror_key)
                added += 1
    return out, added


def score_function(lattice, segs, ticks, dist, dft_band: Path, efermi: float):
    B = 2 * np.pi * np.linalg.inv(lattice).T
    nhat = shells.stacking_normal(lattice)
    intervals = [
        (ticks[idx][0] / dist[-1], ticks[idx + 1][0] / dist[-1])
        for idx, (_, k1, _, k2) in enumerate(segs)
        if abs((k1 @ B) @ nhat) < 1.0e-4 and abs((k2 @ B) @ nhat) < 1.0e-4
    ]

    def score(band_path: Path) -> dict[str, float]:
        dft_mat, tb_mat, xn = sampled_levels(dft_band, band_path, efermi)
        shift = fit_shift(dft_mat, tb_mat, efermi, WINDOW)
        metrics = level_set_metrics(dft_mat, tb_mat, shift, WINDOW)
        mask = np.zeros(len(xn), bool)
        for x0, x1 in intervals:
            mask |= (xn >= x0 - 1.0e-9) & (xn <= x1 + 1.0e-9)
        metrics["chamfer_ip"] = (
            level_set_metrics(dft_mat[mask], tb_mat[mask], shift, WINDOW)["chamfer"]
            if mask.any()
            else float("nan")
        )
        return metrics

    return score


def run_material(
    bonds: Path,
    *,
    dmax: float,
    max_shells: dict[str, int],
    onsite_mode: str,
    tag: str,
) -> list[list]:
    wout, pairs, re_bonds = tb.parse_input(bonds.read_text(encoding="utf-8"))
    hr_path = next(Path(wout).parent.glob("*_hr.dat"))
    win_path = next(Path(wout).parent.glob("*.win"))
    H, num_wann = tb.read_w90_hr(hr_path)
    lattice, atoms, blocks = tb.atom_blocks(wout, num_wann)
    segs, npts = tb.read_win_kpath(win_path)
    kfracs, dist, ticks = tb.kpath(segs, npts, lattice)
    stem = Path(wout).stem.split(".")[0]
    out_dir = bonds.parent / "tb-gpt"
    out_dir.mkdir(exist_ok=True)

    frames = shells.slab_ids(lattice, atoms)
    pairs_2d = filter_pairs(pairs, frames)
    re_bonds_2d, dropped_re_bonds, dropped_bridges = filter_re_bonds(re_bonds, frames)
    print(
        f"  slabs={frames[1]}  pairs kept={len(pairs_2d)}/{len(pairs)} "
        f"dropped_re_bonds={dropped_re_bonds} dropped_bridges={dropped_bridges}",
        flush=True,
    )

    pruned, _ = tb.build_pruned(H, atoms, blocks, pairs_2d, num_wann, **shells.onsite_args(onsite_mode))
    bondpp, n_bondpp = build_bondpp_intra(H, blocks, pruned, re_bonds_2d, num_wann, frames)
    downfold, _, _, downfold_map = tb.build_downfolded(
        H, atoms, blocks, re_bonds_2d, degenerate_tol=1.0e-6, **shells.onsite_args(onsite_mode)
    )

    kept = set()
    for a, b, R in pairs_2d:
        kept.add((a, b, R))
        kept.add((b, a, tuple(-x for x in R)))
    cands_all = shells.find_candidates(H, atoms, blocks, lattice, kept, dmax=dmax, frames=frames)
    cands = {key: value for key, value in cands_all.items() if value[3]}
    grouped = shells.group_shells(cands)

    shell_rows = []
    for cat in ("pp", "fp", "ff"):
        for idx, (distance, keys) in enumerate(grouped[cat], start=1):
            frob = float(np.sqrt(sum(cands[key][2] ** 2 for key in keys)))
            shell_rows.append(f"{cat}{idx}\t{distance:.3f}\t{len(keys)}\t{frob:.4f}")
    (out_dir / f"shells_{tag}.tsv").write_text(
        "shell\tdistance_A\tn_blocks\tfrob_eV\n" + "\n".join(shell_rows) + "\n",
        encoding="utf-8",
    )

    models = [("pruned", pruned, None, 0), ("bondpp", bondpp, None, n_bondpp), ("downfold", downfold, downfold_map, 0)]
    for cat in ("pp", "fp", "ff"):
        keys = []
        for idx, (distance, shell_keys) in enumerate(grouped[cat][: max_shells[cat]], start=1):
            keys = keys + shell_keys
            models.append((f"{cat}{idx}_d{distance:.2f}", shells.extend(pruned, H, blocks, num_wann, keys), None, len(keys)))

    all_pp = [key for shell in grouped["pp"] for key in shell[1]]
    all_fp = [key for shell in grouped["fp"] for key in shell[1]]
    all_ff = [key for shell in grouped["ff"] for key in shell[1]]
    combos = [
        ("pp+fp", all_pp + all_fp),
        ("pp+ff", all_pp + all_ff),
        ("all", all_pp + all_fp + all_ff),
        ("pp_3dref", [key for key, value in cands_all.items() if value[0] == "pp"]),
    ]
    for name, keys in combos:
        models.append((name, shells.extend(pruned, H, blocks, num_wann, keys), None, len(keys)))

    wsvec_path = tb.resolve_wsvec_path(hr_path, wout)
    wsvec = None
    if wsvec_path is not None:
        needed = set()
        for _, model, basis_map, _ in models:
            needed.update(tb.wsvec_keys_for_model(model, basis_map=basis_map))
        wsvec = tb.read_w90_wsvec(wsvec_path, needed_keys=needed)

    dft_band = plot.resolve_dft_band(out_dir.parent / "tb", stem)
    efermi = plot.resolve_efermi(out_dir.parent / "tb", dft_band)
    score = score_function(lattice, segs, ticks, dist, dft_band, efermi)

    rows, band_files = [], {}
    for name, model, basis_map, n_added in models:
        ev = tb.bands(model, kfracs, wsvec=wsvec, basis_map=basis_map)
        band_path = out_dir / f"{stem}_bands_gpt_{name}_{tag}.txt"
        tb.write_bands(band_path, dist, ev, ticks, tb_shift_mode="fit")
        band_files[name] = band_path
        metrics = score(band_path)
        rows.append([name, n_added] + [metrics[col] for col in METRIC_COLS])
        print(f"  {name:14s} added={n_added:5d} " + " ".join(f"{metrics[col]:8.4f}" for col in METRIC_COLS), flush=True)

    full_band = w90_band_dat(out_dir.parent / "tb")
    if full_band is not None:
        metrics = score(full_band)
        rows.append(["w90_full", -1] + [metrics[col] for col in METRIC_COLS])

    ext = [row for row in rows if row[0] not in ("pruned", "downfold", "w90_full", "pp_3dref")]
    sane = [row for row in ext if 0.8 <= row[10] <= 1.25 and row[9] >= 0.95]
    pool = sane if sane else ext
    best = min(row[3] for row in pool)
    eligible = [row for row in pool if row[3] <= best * 1.1 + 5.0e-4]
    minimal = min(eligible, key=lambda row: row[1])[0]

    header = "\t".join(["model", "n_added"] + METRIC_COLS)
    body = "\n".join("\t".join([row[0], str(row[1])] + [f"{value:.6f}" for value in row[2:]]) for row in rows)
    (out_dir / f"metrics_{tag}.tsv").write_text(
        f"# window {WINDOW[0]} .. {WINDOW[1]} eV; onsite={onsite_mode}; dmax={dmax}; "
        f"strict_slab=True; minimal={minimal}\n{header}\n{body}\n",
        encoding="utf-8",
    )

    best_single = set()
    for prefix in ("pp", "fp", "ff"):
        candidates = [
            row for row in rows
            if row[0].startswith(prefix) and row[0][len(prefix): len(prefix) + 1].isdigit()
        ]
        if candidates:
            best_single.add(min(candidates, key=lambda row: row[3])[0])
    keep = {"pruned", "bondpp", "downfold", minimal, "all", "pp_3dref", *best_single}
    for name, band_path in list(band_files.items()):
        if name not in keep:
            band_path.unlink()
    print(f"  -> minimal={minimal}  wrote {out_dir}/metrics_{tag}.tsv", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="group/material; default all")
    parser.add_argument("--input-root", type=Path, default=REPO / "data" / "data-DFT-input")
    parser.add_argument("--dmax", type=float, default=9.0)
    parser.add_argument("--max-pp", type=int, default=10)
    parser.add_argument("--max-fp", type=int, default=6)
    parser.add_argument("--max-ff", type=int, default=6)
    parser.add_argument("--onsite-mode", choices=("soc-delta", "hr"), default="soc-delta")
    parser.add_argument("--tag", default="socdelta_strict2d")
    args = parser.parse_args()

    targets = [args.input_root / target / "enumerate_nn_bonds.txt" for target in args.targets]
    if not targets:
        targets = sorted(args.input_root.glob("*/*/enumerate_nn_bonds.txt"))
    max_shells = {"pp": args.max_pp, "fp": args.max_fp, "ff": args.max_ff}
    for idx, bonds in enumerate(targets, start=1):
        print(f"[{idx:02d}/{len(targets):02d}] {bonds.parent.relative_to(args.input_root)}", flush=True)
        run_material(bonds, dmax=args.dmax, max_shells=max_shells, onsite_mode=args.onsite_mode, tag=args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
