#!/usr/bin/env python3
"""Aggregate and plot [111]-axis Ising/XXZ/C3 model residuals."""
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

DEFAULT_SUMMARY = Path("outputs/analysis/rechx_111_exchange_all_branches_summary.tsv")
DEFAULT_AGGREGATE = Path("outputs/analysis/rechx_111_model_scores_by_run_shell.tsv")
DEFAULT_FIGURE = Path("outputs/analysis/figures/rechx_111_model_residuals_fopt_plus_sopt_direct.png")

FIELDS = [
    "family",
    "material",
    "branch",
    "run",
    "shell",
    "n",
    "median_ising_full_resid",
    "median_xxz_full_resid",
    "median_c3_axial_resid",
    "median_ising_sym_resid",
    "median_xxz_sym_resid",
    "median_ising_ratio_abs",
    "min_axis_alignment111",
    "median_axis_alignment111",
    "max_dm_norm_mev",
    "xxz_gain_vs_ising",
    "c3_gain_vs_xxz",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def shell_for_bond(bond: str) -> str:
    if bond.startswith("J1-"):
        return "J1"
    if bond.startswith("J2-"):
        return "J2"
    return bond


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        shell = shell_for_bond(row["bond"])
        grouped[(row["family"], row["material"], row["branch"], row["run"], shell)].append(row)

    out: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        ising_full = np.array([f(row, "median_ising111_full_resid") for row in items], dtype=float)
        xxz_full = np.array([f(row, "median_xxz111_full_resid") for row in items], dtype=float)
        c3 = np.array([f(row, "median_c3_axial_resid") for row in items], dtype=float)
        ising_sym = np.array([f(row, "median_ising111_resid") for row in items], dtype=float)
        xxz_sym = np.array([f(row, "median_xxz111_resid") for row in items], dtype=float)
        ratio = np.array([f(row, "median_ising_ratio_abs") for row in items], dtype=float)
        min_align = np.array([f(row, "min_axis_alignment111") for row in items], dtype=float)
        med_align = np.array([f(row, "median_axis_alignment111") for row in items], dtype=float)
        dm = np.array([f(row, "max_dm_norm_mev") for row in items], dtype=float)

        med_ising = float(np.median(ising_full))
        med_xxz = float(np.median(xxz_full))
        med_c3 = float(np.median(c3))
        out.append({
            "family": key[0],
            "material": key[1],
            "branch": key[2],
            "run": key[3],
            "shell": key[4],
            "n": len(items),
            "median_ising_full_resid": med_ising,
            "median_xxz_full_resid": med_xxz,
            "median_c3_axial_resid": med_c3,
            "median_ising_sym_resid": float(np.median(ising_sym)),
            "median_xxz_sym_resid": float(np.median(xxz_sym)),
            "median_ising_ratio_abs": float(np.median(ratio)),
            "min_axis_alignment111": float(np.min(min_align)),
            "median_axis_alignment111": float(np.median(med_align)),
            "max_dm_norm_mev": float(np.max(dm)),
            "xxz_gain_vs_ising": med_ising - med_xxz,
            "c3_gain_vs_xxz": med_xxz - med_c3,
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


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in FIELDS})


def short_run(run: str) -> str:
    out = run
    for token in ("_projector", "_baseline", "_exp"):
        out = out.replace(token, "")
    return out


def plot_heatmap(rows: list[dict[str, object]], branch: str, output: Path) -> None:
    selected = [row for row in rows if row["branch"] == branch and row["shell"] in ("J1", "J2")]
    run_keys = sorted({(str(row["family"]), str(row["material"]), str(row["run"])) for row in selected})
    cols = [
        ("J1", "Ising", "median_ising_full_resid"),
        ("J1", "XXZ", "median_xxz_full_resid"),
        ("J1", "C3", "median_c3_axial_resid"),
        ("J2", "Ising", "median_ising_full_resid"),
        ("J2", "XXZ", "median_xxz_full_resid"),
        ("J2", "C3", "median_c3_axial_resid"),
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
    col_labels = [f"{shell}\n{model}" for shell, model, _ in cols]
    height = max(6.0, 0.34 * len(run_keys) + 1.8)
    fig, ax = plt.subplots(figsize=(9.4, height))
    masked = np.ma.masked_invalid(data)
    image = ax.imshow(masked, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(cols)), labels=col_labels)
    ax.set_yticks(np.arange(len(run_keys)), labels=labels, fontsize=7)
    ax.set_title(f"[111] model residuals ({branch}); lower is closer")
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                colour = "white" if data[i, j] > 0.55 else "black"
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=colour)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("median residual")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--branch", default="fopt_plus_sopt_direct")
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    rows = aggregate(read_rows(args.summary))
    write_tsv(args.aggregate_output, rows)
    plot_heatmap(rows, args.branch, args.figure)
    print(f"wrote {len(rows)} aggregate rows: {args.aggregate_output}")
    print(f"wrote figure: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
