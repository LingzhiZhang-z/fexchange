#!/usr/bin/env python3
"""Near-Fermi level-set metrics: DFT vs TB band files (no band-by-band pairing).

For each material under data/data-DFT-input/*/*/tb, score every TB band file
(pruned / pruned_bondpp / downfold; hr and socdelta onsite) and the original
full Wannier90 _band.dat (= ceiling of any pruned model) against DFT bands.

All metrics live in a near-Fermi window (default -0.25..0.25 eV, DFT - E_F),
after one rigid TB shift fitted by minimising the symmetric set distance:
  miss   : mean over DFT levels in window of |nearest TB level| distance
  spur   : mean over TB levels in window of |nearest DFT level| distance
  chamfer: pooled mean of both directions (the fit target)
  env    : RMSE between DFT lower envelope and the nearest TB level
  envpp  : TB / DFT peak-to-peak amplitude of the lower envelope (dispersion scale)
  cover  : fraction of k with DFT levels in window where TB also has one
  nratio : mean TB level count in window / mean DFT level count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tb"))
import plot_tb_bands as plot  # noqa: E402
import tb_from_enumerate as tb  # noqa: E402

WINDOW = (-0.25, 0.25)


def sampled_levels(dft_band: Path, tb_band: Path, efermi: float, nk: int = 140):
    """Return (dft_mat, tb_mat): levels at ~nk shared k samples.
    dft_mat is E - E_F; tb_mat is raw TB energy (shift applied later)."""
    dft_xs, dft_es = plot.read_band_blocks(dft_band)
    tb_xs, tb_es = plot.read_band_blocks(tb_band)
    step = max(1, len(tb_xs[0]) // nk)
    tb_x = tb_xs[0][::step]
    tb_mat = np.vstack([e[::step] for e in tb_es]).T
    dft_x = tb_x * (dft_xs[0][-1] / tb_xs[0][-1])
    dft_mat = plot.eval_band_blocks(dft_xs, dft_es, dft_x) - float(efermi)
    return dft_mat, tb_mat, tb_x / tb_xs[0][-1]


def nearest_dist(sorted_ref: np.ndarray, queries: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_ref, queries)
    lo = np.clip(idx - 1, 0, len(sorted_ref) - 1)
    hi = np.clip(idx, 0, len(sorted_ref) - 1)
    return np.minimum(np.abs(queries - sorted_ref[lo]), np.abs(queries - sorted_ref[hi]))


def chamfer_curve(dft_mat, tb_mat, shifts, window=WINDOW):
    """Pooled chamfer(shift) over all k samples; vectorised over the shift grid."""
    lo, hi = window
    total = np.zeros(len(shifts))
    count = np.zeros(len(shifts))
    for D, T in zip(dft_mat, tb_mat):
        Ts, Ds = np.sort(T), np.sort(D)
        Dw = D[(D >= lo) & (D <= hi)]
        if Dw.size:  # DFT level -> nearest (shifted) TB level
            miss = nearest_dist(Ts, Dw[None, :] - shifts[:, None])
            total += miss.sum(axis=1)
            count += Dw.size
        shifted = T[None, :] + shifts[:, None]  # TB level in window -> nearest DFT level
        in_win = (shifted >= lo) & (shifted <= hi)
        spur = nearest_dist(Ds, shifted)
        total += (spur * in_win).sum(axis=1)
        count += in_win.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.inf)


def fit_shift(dft_mat, tb_mat, efermi, window=WINDOW) -> float:
    coarse = np.unique(np.concatenate([np.arange(-6.0, 6.0, 0.02),
                                       -efermi + np.arange(-6.0, 6.0, 0.02)]))
    best = coarse[int(np.argmin(chamfer_curve(dft_mat, tb_mat, coarse, window)))]
    fine = best + np.arange(-0.03, 0.03, 0.001)
    return float(fine[int(np.argmin(chamfer_curve(dft_mat, tb_mat, fine, window)))])


def level_set_metrics(dft_mat, tb_mat, shift, window=WINDOW) -> dict:
    lo, hi = window
    miss_all, spur_all, env_err, env_d, env_t, cover, n_d, n_t = [], [], [], [], [], [], [], []
    for D, T in zip(dft_mat, tb_mat):
        Tshift = np.sort(T) + shift
        Dw = D[(D >= lo) & (D <= hi)]
        Tw = Tshift[(Tshift >= lo) & (Tshift <= hi)]
        n_d.append(Dw.size)
        n_t.append(Tw.size)
        if Dw.size:
            miss_all.extend(nearest_dist(Tshift, Dw))
            edge = float(Dw.min())
            env_err.append(float(np.min(np.abs(Tshift - edge))))
            cover.append(1.0 if Tw.size else 0.0)
            if Tw.size:
                env_d.append(edge)
                env_t.append(float(Tw.min()))
        if Tw.size:
            spur_all.extend(nearest_dist(np.sort(D), Tw))
    pooled = miss_all + spur_all
    envpp = float("nan")
    if len(env_d) >= 2 and (max(env_d) - min(env_d)) > 1e-6:
        envpp = (max(env_t) - min(env_t)) / (max(env_d) - min(env_d))
    dft_levels = np.concatenate([D[(D >= lo) & (D <= hi)] for D in dft_mat])
    return {
        "dspread": float(np.std(dft_levels)) if dft_levels.size else float("nan"),
        "shift": shift,
        "chamfer": float(np.mean(pooled)) if pooled else float("nan"),
        "miss": float(np.mean(miss_all)) if miss_all else float("nan"),
        "spur": float(np.mean(spur_all)) if spur_all else float("nan"),
        "env": float(np.sqrt(np.mean(np.square(env_err)))) if env_err else float("nan"),
        "envpp": envpp,
        "cover": float(np.mean(cover)) if cover else float("nan"),
        "nratio": float(np.mean(n_t) / np.mean(n_d)) if np.mean(n_d) > 0 else float("nan"),
    }


def score_file(dft_band: Path, tb_band: Path, efermi: float, window=WINDOW) -> dict:
    dft_mat, tb_mat, _ = sampled_levels(dft_band, tb_band, efermi)
    shift = fit_shift(dft_mat, tb_mat, efermi, window)
    return level_set_metrics(dft_mat, tb_mat, shift, window)


def w90_band_dat(tb_dir: Path) -> Path | None:
    bonds = tb_dir.parent / "enumerate_nn_bonds.txt"
    if not bonds.is_file():
        return None
    wout = next(ln.strip() for ln in bonds.read_text().splitlines() if ln.strip().endswith(".wout"))
    hr_path = next(Path(wout).parent.glob("*_hr.dat"))
    wsvec = tb.resolve_wsvec_path(hr_path, wout)
    if wsvec is None:
        return None
    band = Path(str(wsvec).removesuffix("_wsvec.dat") + "_band.dat")
    return band if band.is_file() else None


def material_models(tb_dir: Path, stem: str) -> list[tuple[str, Path]]:
    names = [
        ("pruned_socdelta", f"{stem}_bands_socdelta.txt"),
        ("bondpp_socdelta", f"{stem}_bands_pruned_bondpp_socdelta.txt"),
        ("downfold_socdelta", f"{stem}_bands_downfold_socdelta.txt"),
        ("pruned_hr", f"{stem}_bands.txt"),
        ("bondpp_hr", f"{stem}_bands_pruned_bondpp.txt"),
        ("downfold_hr", f"{stem}_bands_downfold.txt"),
    ]
    out = [(model, tb_dir / name) for model, name in names if (tb_dir / name).is_file()]
    full = w90_band_dat(tb_dir)
    if full is not None:
        out.insert(0, ("w90_full", full))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="group/material; default all")
    parser.add_argument("--input-root", type=Path, default=REPO / "data" / "data-DFT-input")
    parser.add_argument("--window", nargs=2, type=float, default=list(WINDOW))
    parser.add_argument("--output", type=Path,
                        default=REPO / "data" / "data-DFT-input" / "tb_report" / "tb_claude_state.tsv")
    args = parser.parse_args()
    window = tuple(args.window)

    tb_dirs = ([args.input_root / t / "tb" for t in args.targets] if args.targets
               else sorted(args.input_root.glob("*/*/tb")))
    cols = ["dspread", "shift", "chamfer", "miss", "spur", "env", "envpp", "cover", "nratio"]
    rows = []
    for tb_dir in tb_dirs:
        pairs = plot.find_band_pairs(tb_dir)
        if not pairs:
            continue
        stem = pairs[0][0]
        group, material = plot.material_context(tb_dir)
        dft_band = plot.resolve_dft_band(tb_dir, stem)
        efermi = plot.resolve_efermi(tb_dir, dft_band)
        for model, band in material_models(tb_dir, stem):
            m = score_file(dft_band, band, efermi, window)
            rows.append([group, material, model] + [m[c] for c in cols])
            print(f"{group:10s} {material:12s} {model:18s} " +
                  " ".join(f"{m[c]:8.4f}" for c in cols), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("\t".join(["group", "material", "model"] + cols) + "\n")
        for row in rows:
            f.write("\t".join(str(x) if isinstance(x, str) else f"{x:.6f}" for x in row) + "\n")
    print(f"wrote {args.output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
