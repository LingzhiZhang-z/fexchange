#!/usr/bin/env python3
"""Diagnose which missing HR blocks improve the pruned near-Fermi fit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tb"))

import plot_tb_bands as plot  # noqa: E402
import tb_from_enumerate as tb  # noqa: E402
from enumerate import RARE_EARTH  # noqa: E402


DEFAULT_TARGETS = [
    "REOCl/YbOCl",
    "REX3-R3/DyBr3",
    "REX3-R3/YbI3",
]
WINDOW = (-0.25, 0.25)


@dataclass(frozen=True)
class Variant:
    model: str
    title: str
    categories: tuple[str, ...]


VARIANTS = [
    Variant("pruned_allpp", "pruned + all ligand p-p", ("pp",)),
    Variant("pruned_allpp_extrafp", "pruned + all p-p + extra f-p", ("pp", "fp")),
]
FULLOFF_VARIANT = [
    Variant("pruned_fulloff", "pruned + all missing offdiag HR blocks", ("pp", "fp", "ff")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--input-root", type=Path, default=REPO / "data" / "data-DFT-input")
    parser.add_argument("--threshold", type=float, default=1.0e-8)
    parser.add_argument("--include-fulloff", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    reports = []
    variants = VARIANTS + (FULLOFF_VARIANT if args.include_fulloff else [])
    for target in args.targets:
        bonds = target_to_bonds(input_root, target)
        reports.append(run_one(bonds, variants=variants, threshold=args.threshold, make_plots=not args.no_plot))
    print("\n\n".join(reports))
    return 0


def target_to_bonds(input_root: Path, target: str) -> Path:
    path = Path(target)
    if path.name == "enumerate_nn_bonds.txt":
        return path if path.is_absolute() else REPO / path
    bonds = input_root / target / "enumerate_nn_bonds.txt"
    if not bonds.is_file():
        raise SystemExit(f"target not found: {target}")
    return bonds


def run_one(bonds: Path, *, variants: list[Variant], threshold: float, make_plots: bool) -> str:
    wout, pairs, re_bonds = tb.parse_input(bonds.read_text(encoding="utf-8"))
    hr_path = next(Path(wout).parent.glob("*_hr.dat"))
    win_path = next(Path(wout).parent.glob("*.win"))
    H, num_wann = tb.read_w90_hr(hr_path)
    lattice, atoms, blocks = tb.atom_blocks(wout, num_wann)
    segs, npts = tb.read_win_kpath(win_path)
    kfracs, dist, ticks = tb.kpath(segs, npts, lattice)
    stem = Path(wout).stem.split(".")[0]
    tb_dir = bonds.parent / "tb"
    tb_dir.mkdir(exist_ok=True)

    pruned, _ = tb.build_pruned(
        H,
        atoms,
        blocks,
        pairs,
        num_wann,
        lambda_p=None,
        lambda_source="sweep-input",
        zeta_f=None,
        zeta_source="sweep-input",
        onsite_mode="soc-delta",
        hermitian_atol=1.0e-8,
    )

    wsvec_path = tb.resolve_wsvec_path(hr_path, wout)
    if wsvec_path is None:
        raise FileNotFoundError(f"wsvec not found for {hr_path}")

    model_entries = [("pruned", "pruned", pruned, None)]
    for variant in variants:
        model, counts = add_missing_blocks(
            H,
            atoms,
            blocks,
            pruned,
            categories=set(variant.categories),
            threshold=threshold,
        )
        model_entries.append((variant.model, variant.title, model, counts))

    needed = set()
    for _, _, model, _ in model_entries:
        needed.update(tb.wsvec_keys_for_model(model))
    wsvec = tb.read_w90_wsvec(wsvec_path, needed_keys=needed)

    dft_band = plot.resolve_dft_band(tb_dir, stem)
    efermi = plot.resolve_efermi(tb_dir, dft_band)
    dft_nodes = plot.read_dft_nodes(dft_band)
    pdos_spec = plot.resolve_pdos(dft_band, stem)
    lines = [f"[{bonds.parent.relative_to(REPO / 'data' / 'data-DFT-input')}]"]
    lines.append(f"DFT={dft_band}")
    lines.append(f"E_F={efermi:.6f} eV")
    lines.append("model  edge_rmse(eV)  shift(eV)  added_blocks")
    for model_name, title, model, counts in model_entries:
        band_path = tb_dir / f"{stem}_bands_{model_name}_socdelta.txt"
        tb.write_bands(
            band_path,
            dist,
            tb.bands(model, kfracs, wsvec=wsvec),
            ticks,
            tb_shift_mode="fit",
        )
        shift, rmse = edge_metric(dft_band, band_path, efermi)
        added = "-" if counts is None else format_counts(counts)
        lines.append(f"{model_name:24s} {rmse:12.6f} {shift:10.6f}  {added}")
        if make_plots:
            write_plot(tb_dir, stem, model_name, title, band_path, dft_band, efermi, shift, dft_nodes, pdos_spec)
    report = "\n".join(lines)
    report_path = tb_dir / f"{stem}_pruned_term_diagnostics.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    return report


def add_missing_blocks(H, atoms, blocks, base, *, categories: set[str], threshold: float):
    out = {R: mat.copy() for R, mat in base.items()}
    counts = {"pp": 0, "fp": 0, "ff": 0}
    norms = {"pp": 0.0, "fp": 0.0, "ff": 0.0}
    is_re = [element in RARE_EARTH for element, _ in atoms]

    def category(a: int, b: int) -> str:
        if not is_re[a] and not is_re[b]:
            return "pp"
        if is_re[a] != is_re[b]:
            return "fp"
        return "ff"

    def put(R, rows, cols, mat):
        out.setdefault(R, np.zeros_like(next(iter(out.values()))))[np.ix_(rows, cols)] = mat

    for R, HR in H.items():
        for a in range(len(atoms)):
            for b in range(len(atoms)):
                if R == (0, 0, 0) and a == b:
                    continue
                cat = category(a, b)
                if cat not in categories:
                    continue
                rows, cols = blocks[a], blocks[b]
                mat = HR[np.ix_(rows, cols)]
                norm = float(np.linalg.norm(mat))
                if norm <= threshold:
                    continue
                old = base.get(R)
                if old is None:
                    delta = mat
                else:
                    delta = mat - old[np.ix_(rows, cols)]
                delta_norm = float(np.linalg.norm(delta))
                put(R, rows, cols, mat)
                if delta_norm > threshold:
                    counts[cat] += 1
                    norms[cat] += delta_norm * delta_norm
    return out, {key: (counts[key], norms[key] ** 0.5) for key in counts}


def edge_metric(dft_band: Path, tb_band: Path, efermi: float) -> tuple[float, float]:
    shift = plot.fit_tb_shift(dft_band, tb_band, efermi=efermi, window=WINDOW)
    dft_xs, dft_es = plot.read_band_blocks(dft_band)
    tb_xs, tb_es = plot.read_band_blocks(tb_band)
    tb_x_full = tb_xs[0]
    step = max(1, len(tb_x_full) // 140)
    tb_x = tb_x_full[::step]
    tb_mat = np.vstack([energies[::step] for energies in tb_es]).T - shift
    dft_x = tb_x * (dft_xs[0][-1] / tb_x_full[-1])
    dft_mat = plot.eval_band_blocks(dft_xs, dft_es, dft_x) - efermi

    lo, hi = WINDOW
    errors = []
    for dft_levels, tb_levels in zip(dft_mat, tb_mat):
        near = dft_levels[(dft_levels >= lo) & (dft_levels <= hi)]
        if near.size == 0:
            continue
        dft_edge = float(np.min(near))
        errors.append(float(np.min(np.abs(tb_levels - dft_edge))))
    if not errors:
        return shift, float("nan")
    arr = np.asarray(errors)
    return shift, float(np.sqrt(np.mean(arr * arr)))


def write_plot(
    tb_dir: Path,
    stem: str,
    model_name: str,
    title: str,
    band_path: Path,
    dft_band: Path,
    efermi: float,
    shift: float,
    dft_nodes,
    pdos_spec,
) -> None:
    ticks = plot.read_tb_ticks(band_path)
    script = tb_dir / f"{stem}_dft_vs_{model_name}_socdelta.near.-0.25_0.25.gnuplot"
    png = tb_dir / f"{stem}_dft_vs_{model_name}_socdelta.near.-0.25_0.25.png"
    script.write_text(
        plot.build_gnuplot(
            dft_band=dft_band,
            tb_band=band_path,
            output=png,
            stem=stem,
            tb_title=f"{title} (SOC+Delta onsite)",
            dft_nodes=dft_nodes,
            tb_ticks=ticks,
            efermi=efermi,
            tb_shift=shift,
            window=WINDOW,
            pdos_spec=pdos_spec,
        ),
        encoding="utf-8",
    )
    plot.subprocess.run(["gnuplot", str(script)], check=True)


def format_counts(counts) -> str:
    parts = []
    for key in ("pp", "fp", "ff"):
        count, norm = counts[key]
        if count:
            parts.append(f"{key}:{count}/||={norm:.3f}")
    return ", ".join(parts) if parts else "none"


if __name__ == "__main__":
    raise SystemExit(main())
