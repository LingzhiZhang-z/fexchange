#!/usr/bin/env python3
"""Analyze unrotated REChX tensors against z-axis Ising and XXZ models.

Use this for the original organized REChX tree, where crystallographic z is the
C3 axis.  Do not use REChX_cubic here; that tree has been rotated for Kitaev
bond-local analysis.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")
DEFAULT_FAMILIES = ("REOCl", "REOF", "RESI")
DEFAULT_BRANCHES = ("sopt", "sopt_direct", "fopt", "fopt_plus_sopt_direct")
PLOT_BRANCH = "fopt_plus_sopt_direct"
EPS = 1.0e-14

SUMMARY_FIELDS = [
    "family",
    "material",
    "bond",
    "branch",
    "run",
    "U",
    "n",
    "ratio_min",
    "ratio_max",
    "median_z_ising_full_resid",
    "max_z_ising_full_resid",
    "median_z_xxz_full_resid",
    "max_z_xxz_full_resid",
    "median_z_ising_sym_resid",
    "median_z_xxz_sym_resid",
    "median_z_ising_ratio_abs",
    "min_z_axis_alignment",
    "median_z_axis_alignment",
    "max_abs_Szz_mev",
    "max_abs_nonising_sym_mev",
    "max_dm_norm_mev",
    "median_dm_over_sym",
]

AGG_FIELDS = [
    "family",
    "material",
    "branch",
    "run",
    "shell",
    "n",
    "median_z_ising_full_resid",
    "median_z_xxz_full_resid",
    "median_z_ising_sym_resid",
    "median_z_xxz_sym_resid",
    "median_z_ising_ratio_abs",
    "min_z_axis_alignment",
    "median_z_axis_alignment",
    "max_dm_norm_mev",
    "xxz_gain_vs_ising",
]


def ratio(num: float, den: float) -> float:
    if abs(den) < EPS:
        return float("inf") if abs(num) >= EPS else 0.0
    return num / den


def shell_for_bond(bond: str) -> str:
    if bond.startswith("J1-"):
        return "J1"
    if bond.startswith("J2-"):
        return "J2"
    return bond


def iter_u_files(root: Path, families: set[str], branches: set[str], bond_filter: set[str] | None):
    for family in sorted(families):
        family_root = root / family
        if not family_root.exists():
            continue
        for material_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            for bond_root in sorted(path for path in material_root.iterdir() if path.is_dir()):
                if bond_filter is not None and bond_root.name not in bond_filter:
                    continue
                for branch in sorted(branches):
                    branch_root = bond_root / branch
                    if not branch_root.exists():
                        continue
                    for path in sorted(branch_root.glob("*/U_*.txt")):
                        try:
                            u_value = int(path.stem.split("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        yield (
                            family,
                            material_root.name,
                            bond_root.name,
                            branch,
                            path.parts[-2],
                            u_value,
                            path,
                        )


def read_exchange_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        tensor = np.array([float(x) for x in cols[3:12]], dtype=float).reshape(3, 3)
        yield float(cols[0]), tensor


def c3z_metrics(tensor_ev: np.ndarray) -> dict[str, float]:
    m = tensor_ev
    s = 0.5 * (m + m.T)
    full_norm = float(np.linalg.norm(m))
    sym_norm = float(np.linalg.norm(s))

    jperp = 0.5 * (m[0, 0] + m[1, 1])
    ising_full = np.zeros((3, 3), dtype=float)
    ising_full[2, 2] = m[2, 2]
    xxz_full = np.diag([jperp, jperp, m[2, 2]])

    s_jperp = 0.5 * (s[0, 0] + s[1, 1])
    ising_sym = np.zeros((3, 3), dtype=float)
    ising_sym[2, 2] = s[2, 2]
    xxz_sym = np.diag([s_jperp, s_jperp, s[2, 2]])

    evals, evecs = np.linalg.eigh(s)
    imax = int(np.argmax(np.abs(evals)))
    axis_alignment = float(abs(evecs[2, imax]))

    nonising_sym_terms = [
        s[0, 0],
        s[1, 1],
        s[0, 1],
        s[0, 2],
        s[1, 2],
    ]
    max_nonising = float(max(abs(x) for x in nonising_sym_terms))

    d_x = 0.5 * (m[1, 2] - m[2, 1])
    d_y = 0.5 * (m[2, 0] - m[0, 2])
    d_z = 0.5 * (m[0, 1] - m[1, 0])
    dm_norm = float(np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z))

    return {
        "z_ising_full_resid": ratio(float(np.linalg.norm(m - ising_full)), full_norm),
        "z_xxz_full_resid": ratio(float(np.linalg.norm(m - xxz_full)), full_norm),
        "z_ising_sym_resid": ratio(float(np.linalg.norm(s - ising_sym)), sym_norm),
        "z_xxz_sym_resid": ratio(float(np.linalg.norm(s - xxz_sym)), sym_norm),
        "z_ising_ratio_abs": ratio(abs(float(s[2, 2])), max_nonising),
        "z_axis_alignment": axis_alignment,
        "abs_Szz_mev": abs(float(s[2, 2] * 1000.0)),
        "abs_nonising_sym_mev": abs(float(max_nonising * 1000.0)),
        "dm_norm_mev": float(dm_norm * 1000.0),
        "dm_over_sym": ratio(dm_norm, sym_norm),
    }


def summarize(points: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for point in points:
        key = tuple(point[field] for field in ("family", "material", "bond", "branch", "run", "U"))
        grouped[key].append(point)

    out: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        ratios = np.array([float(row["ratio"]) for row in rows], dtype=float)
        ising_full = np.array([float(row["z_ising_full_resid"]) for row in rows], dtype=float)
        xxz_full = np.array([float(row["z_xxz_full_resid"]) for row in rows], dtype=float)
        ising_sym = np.array([float(row["z_ising_sym_resid"]) for row in rows], dtype=float)
        xxz_sym = np.array([float(row["z_xxz_sym_resid"]) for row in rows], dtype=float)
        ising_ratio = np.array([float(row["z_ising_ratio_abs"]) for row in rows], dtype=float)
        axis_align = np.array([float(row["z_axis_alignment"]) for row in rows], dtype=float)
        szz = np.array([float(row["abs_Szz_mev"]) for row in rows], dtype=float)
        nonising = np.array([float(row["abs_nonising_sym_mev"]) for row in rows], dtype=float)
        dm = np.array([float(row["dm_norm_mev"]) for row in rows], dtype=float)
        dm_over_sym = np.array([float(row["dm_over_sym"]) for row in rows], dtype=float)
        finite_ising_ratio = ising_ratio[np.isfinite(ising_ratio)]
        out.append({
            "family": key[0],
            "material": key[1],
            "bond": key[2],
            "branch": key[3],
            "run": key[4],
            "U": key[5],
            "n": len(rows),
            "ratio_min": float(np.min(ratios)),
            "ratio_max": float(np.max(ratios)),
            "median_z_ising_full_resid": float(np.median(ising_full)),
            "max_z_ising_full_resid": float(np.max(ising_full)),
            "median_z_xxz_full_resid": float(np.median(xxz_full)),
            "max_z_xxz_full_resid": float(np.max(xxz_full)),
            "median_z_ising_sym_resid": float(np.median(ising_sym)),
            "median_z_xxz_sym_resid": float(np.median(xxz_sym)),
            "median_z_ising_ratio_abs": float(np.median(finite_ising_ratio)) if finite_ising_ratio.size else float("inf"),
            "min_z_axis_alignment": float(np.min(axis_align)),
            "median_z_axis_alignment": float(np.median(axis_align)),
            "max_abs_Szz_mev": float(np.max(szz)),
            "max_abs_nonising_sym_mev": float(np.max(nonising)),
            "max_dm_norm_mev": float(np.max(dm)),
            "median_dm_over_sym": float(np.median(dm_over_sym)),
        })
    return out


def aggregate(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        grouped[(row["family"], row["material"], row["branch"], row["run"], shell_for_bond(str(row["bond"])))].append(row)

    out: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        ising_full = np.array([float(row["median_z_ising_full_resid"]) for row in rows], dtype=float)
        xxz_full = np.array([float(row["median_z_xxz_full_resid"]) for row in rows], dtype=float)
        med_ising = float(np.median(ising_full))
        med_xxz = float(np.median(xxz_full))
        out.append({
            "family": key[0],
            "material": key[1],
            "branch": key[2],
            "run": key[3],
            "shell": key[4],
            "n": len(rows),
            "median_z_ising_full_resid": med_ising,
            "median_z_xxz_full_resid": med_xxz,
            "median_z_ising_sym_resid": float(np.median([float(row["median_z_ising_sym_resid"]) for row in rows])),
            "median_z_xxz_sym_resid": float(np.median([float(row["median_z_xxz_sym_resid"]) for row in rows])),
            "median_z_ising_ratio_abs": float(np.median([float(row["median_z_ising_ratio_abs"]) for row in rows])),
            "min_z_axis_alignment": float(np.min([float(row["min_z_axis_alignment"]) for row in rows])),
            "median_z_axis_alignment": float(np.median([float(row["median_z_axis_alignment"]) for row in rows])),
            "max_dm_norm_mev": float(np.max([float(row["max_dm_norm_mev"]) for row in rows])),
            "xxz_gain_vs_ising": med_ising - med_xxz,
        })
    return out


def fmt(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf"
        return f"{value:.10g}"
    return str(value)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def short_run(run: str) -> str:
    out = run
    for token in ("_projector", "_baseline", "_exp"):
        out = out.replace(token, "")
    return out


def plot_heatmap(rows: list[dict[str, object]], branch: str, output: Path) -> None:
    selected = [row for row in rows if row["branch"] == branch and row["shell"] in ("J1", "J2")]
    run_keys = sorted({(str(row["family"]), str(row["material"]), str(row["run"])) for row in selected})
    cols = [
        ("J1", "Ising", "median_z_ising_full_resid"),
        ("J1", "XXZ", "median_z_xxz_full_resid"),
        ("J2", "Ising", "median_z_ising_full_resid"),
        ("J2", "XXZ", "median_z_xxz_full_resid"),
    ]
    lookup = {
        (str(row["family"]), str(row["material"]), str(row["run"]), str(row["shell"])): row
        for row in selected
    }
    data = np.full((len(run_keys), len(cols)), np.nan, dtype=float)
    for i, run_key in enumerate(run_keys):
        for j, (shell, _model, metric) in enumerate(cols):
            item = lookup.get((*run_key, shell))
            if item is not None:
                data[i, j] = float(item[metric])

    labels = [f"{family}/{material}\n{short_run(run)}" for family, material, run in run_keys]
    col_labels = [f"{shell}\n{model}" for shell, model, _metric in cols]
    height = max(6.0, 0.34 * len(run_keys) + 1.8)
    fig, ax = plt.subplots(figsize=(7.2, height))
    image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap="viridis_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)), labels=col_labels)
    ax.set_yticks(np.arange(len(run_keys)), labels=labels, fontsize=7)
    ax.set_title(f"Unrotated z-axis C3 model residuals ({branch}); lower is closer")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                colour = "white" if data[i, j] > 0.55 else "black"
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.7, color=colour)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("median full-tensor residual")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--family", action="append", choices=DEFAULT_FAMILIES)
    parser.add_argument("--branch", action="append", default=None)
    parser.add_argument("--bond", action="append")
    parser.add_argument("--plot-branch", default=PLOT_BRANCH)
    parser.add_argument("--summary-output", type=Path,
                        default=Path("outputs/analysis/rechx_c3z_ising_xxz_summary.tsv"))
    parser.add_argument("--aggregate-output", type=Path,
                        default=Path("outputs/analysis/rechx_c3z_ising_xxz_by_run_shell.tsv"))
    parser.add_argument("--figure", type=Path,
                        default=Path("outputs/analysis/figures/rechx_c3z_ising_xxz_residuals_fopt_plus_sopt_direct.png"))
    args = parser.parse_args()

    families = set(args.family or DEFAULT_FAMILIES)
    branches = set(args.branch or DEFAULT_BRANCHES)
    bond_filter = set(args.bond) if args.bond else None

    points: list[dict[str, object]] = []
    files_seen = 0
    for family, material, bond, branch, run, u_value, path in iter_u_files(args.root, families, branches, bond_filter):
        files_seen += 1
        for ratio_value, tensor in read_exchange_rows(path):
            row: dict[str, object] = {
                "family": family,
                "material": material,
                "bond": bond,
                "branch": branch,
                "run": run,
                "U": u_value,
                "ratio": ratio_value,
            }
            row.update(c3z_metrics(tensor))
            points.append(row)

    if not points:
        raise SystemExit("no matching unrotated REChX exchange rows found")

    summary = summarize(points)
    agg = aggregate(summary)
    write_tsv(args.summary_output, SUMMARY_FIELDS, summary)
    write_tsv(args.aggregate_output, AGG_FIELDS, agg)
    plot_heatmap(agg, args.plot_branch, args.figure)
    print(f"wrote {len(summary)} summary rows from {files_seen} files: {args.summary_output}")
    print(f"wrote {len(agg)} aggregate rows: {args.aggregate_output}")
    print(f"wrote figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
