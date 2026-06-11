#!/usr/bin/env python3
"""Summarise the current TB shell-extension results in a separate tb-gpt area.

This script does not rerun the band calculations.  It reads the existing
tb-claude metrics, extracts the rows relevant to the pp-only question, and
writes compact TSV/Markdown summaries under tb-gpt paths.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO / "data" / "data-DFT-input"
MEV = 1000.0


def parse_metrics(path: Path) -> tuple[dict[str, str], list[dict[str, float | int | str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"metrics file is too short: {path}")
    meta: dict[str, str] = {}
    if lines[0].startswith("#"):
        for part in lines[0][1:].split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                meta[key.strip()] = value.strip()
    cols = lines[1].split("\t")
    rows: list[dict[str, float | int | str]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        values = line.split("\t")
        row: dict[str, float | int | str] = {"model": values[0], "n_added": int(values[1])}
        for key, value in zip(cols[2:], values[2:]):
            row[key] = float(value)
        rows.append(row)
    return meta, rows


def row_for(rows: list[dict[str, float | int | str]], model: str | None) -> dict[str, float | int | str] | None:
    if model is None:
        return None
    return next((row for row in rows if row["model"] == model), None)


def best_row(rows: list[dict[str, float | int | str]], prefix: str) -> dict[str, float | int | str] | None:
    candidates = [row for row in rows if str(row["model"]).startswith(prefix) and str(row["model"])[len(prefix):1 + len(prefix)].isdigit()]
    return min(candidates, key=lambda row: float(row["chamfer"])) if candidates else None


def mev(row: dict[str, float | int | str] | None, key: str = "chamfer") -> float:
    if row is None or key not in row:
        return math.nan
    return float(row[key]) * MEV


def model_name(row: dict[str, float | int | str] | None) -> str:
    return str(row["model"]) if row is not None else ""


def improvement_fraction(start: float, middle: float, end: float) -> float:
    if math.isnan(start) or math.isnan(middle) or math.isnan(end):
        return math.nan
    denom = start - end
    if abs(denom) < 1.0e-12:
        return math.nan
    return (start - middle) / denom


def verdict(pp_gap_mev: float, recovery: float) -> str:
    if math.isnan(pp_gap_mev):
        return "missing"
    if pp_gap_mev <= 1.0:
        return "pp_close"
    if pp_gap_mev <= 2.5 and (math.isnan(recovery) or recovery >= 0.60):
        return "pp_partial"
    return "needs_fp_ff"


def fmt(value: float, digits: int = 3) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def selected_record(group: str, material: str, metrics3d: Path, metrics2d: Path | None) -> dict[str, str | float]:
    meta3d, rows3d = parse_metrics(metrics3d)
    meta2d, rows2d = parse_metrics(metrics2d) if metrics2d and metrics2d.exists() else ({}, [])

    r3_pruned = row_for(rows3d, "pruned")
    r3_bondpp = row_for(rows3d, "bondpp")
    r3_downfold = row_for(rows3d, "downfold")
    r3_pp = best_row(rows3d, "pp")
    r3_fp = best_row(rows3d, "fp")
    r3_ff = best_row(rows3d, "ff")
    r3_ppfp = row_for(rows3d, "pp+fp")
    r3_ppff = row_for(rows3d, "pp+ff")
    r3_all = row_for(rows3d, "all")
    r3_min = row_for(rows3d, meta3d.get("minimal"))

    r2_pruned = row_for(rows2d, "pruned")
    r2_bondpp = row_for(rows2d, "bondpp")
    r2_downfold = row_for(rows2d, "downfold")
    r2_pp = best_row(rows2d, "pp")
    r2_fp = best_row(rows2d, "fp")
    r2_ff = best_row(rows2d, "ff")
    r2_ppfp = row_for(rows2d, "pp+fp")
    r2_ppff = row_for(rows2d, "pp+ff")
    r2_all = row_for(rows2d, "all")
    r2_3dref = row_for(rows2d, "pp_3dref")
    r2_min = row_for(rows2d, meta2d.get("minimal"))

    pruned2d = mev(r2_pruned)
    pp2d = mev(r2_pp)
    all2d = mev(r2_all)
    pp_gap = pp2d - all2d if not (math.isnan(pp2d) or math.isnan(all2d)) else math.nan
    recovery = improvement_fraction(pruned2d, pp2d, all2d)
    pp3dref = mev(r2_3dref)
    inter_pp_gap = pp2d - pp3dref if not (math.isnan(pp2d) or math.isnan(pp3dref)) else math.nan

    return {
        "group": group,
        "material": material,
        "minimal3d_model": meta3d.get("minimal", ""),
        "minimal2d_model": meta2d.get("minimal", ""),
        "pruned3d_mev": mev(r3_pruned),
        "bondpp3d_mev": mev(r3_bondpp),
        "downfold3d_mev": mev(r3_downfold),
        "pp3d_mev": mev(r3_pp),
        "pp3d_model": model_name(r3_pp),
        "fp3d_mev": mev(r3_fp),
        "fp3d_model": model_name(r3_fp),
        "ff3d_mev": mev(r3_ff),
        "ff3d_model": model_name(r3_ff),
        "ppfp3d_mev": mev(r3_ppfp),
        "ppff3d_mev": mev(r3_ppff),
        "all3d_mev": mev(r3_all),
        "minimal3d_mev": mev(r3_min),
        "pruned2d_mev": pruned2d,
        "bondpp2d_mev": mev(r2_bondpp),
        "downfold2d_mev": mev(r2_downfold),
        "pp2d_mev": pp2d,
        "pp2d_model": model_name(r2_pp),
        "fp2d_mev": mev(r2_fp),
        "fp2d_model": model_name(r2_fp),
        "ff2d_mev": mev(r2_ff),
        "ff2d_model": model_name(r2_ff),
        "ppfp2d_mev": mev(r2_ppfp),
        "ppff2d_mev": mev(r2_ppff),
        "pp_3dref2d_mev": pp3dref,
        "all2d_mev": all2d,
        "minimal2d_mev": mev(r2_min),
        "pp2d_minus_all2d_mev": pp_gap,
        "pp2d_minus_pp3dref_mev": inter_pp_gap,
        "pp2d_recovery_fraction": recovery,
        "verdict": verdict(pp_gap, recovery),
    }


def write_tsv(path: Path, rows: list[dict[str, str | float]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(columns)]
    for row in rows:
        fields = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                fields.append(fmt(value))
            else:
                fields.append(str(value))
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def group_summary(records: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    groups = sorted({str(record["group"]) for record in records})
    rows: list[dict[str, str | float]] = []
    for group in groups:
        subset = [record for record in records if record["group"] == group]
        row: dict[str, str | float] = {"group": group, "n": float(len(subset))}
        for col in (
            "pruned3d_mev",
            "pp3d_mev",
            "all3d_mev",
            "pruned2d_mev",
            "pp2d_mev",
            "pp_3dref2d_mev",
            "all2d_mev",
            "pp2d_minus_all2d_mev",
            "pp2d_minus_pp3dref_mev",
            "pp2d_recovery_fraction",
        ):
            vals = [float(record[col]) for record in subset if not math.isnan(float(record[col]))]
            row[col] = sum(vals) / len(vals) if vals else math.nan
        row["pp_close_n"] = float(sum(1 for record in subset if record["verdict"] == "pp_close"))
        row["pp_partial_n"] = float(sum(1 for record in subset if record["verdict"] == "pp_partial"))
        row["needs_fp_ff_n"] = float(sum(1 for record in subset if record["verdict"] == "needs_fp_ff"))
        rows.append(row)
    return rows


def write_markdown(path: Path, records: list[dict[str, str | float]], groups: list[dict[str, str | float]]) -> None:
    lines = [
        "# tb-gpt summary",
        "",
        "Source: current `tb-claude` metrics.  Values are chamfer errors in meV in the -0.25..0.25 eV window.",
        "",
        "Verdict rule: `pp_close` means in-plane pp-only is within 1 meV of `all`; `pp_partial` means it is within 2.5 meV and recovers most of the pruned-to-all improvement; otherwise it is marked `needs_fp_ff`.",
        "",
        "## Group averages",
        "",
        "| group | n | pruned2d | pp2d | pp_3dref | all2d | pp-all | pp-3dref | recovery | close/partial/need |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            "| {group} | {n:.0f} | {pruned} | {pp} | {pp3dref} | {allv} | {gap} | {igap} | {rec} | {close:.0f}/{partial:.0f}/{need:.0f} |".format(
                group=row["group"],
                n=float(row["n"]),
                pruned=fmt(float(row["pruned2d_mev"]), 2),
                pp=fmt(float(row["pp2d_mev"]), 2),
                pp3dref=fmt(float(row["pp_3dref2d_mev"]), 2),
                allv=fmt(float(row["all2d_mev"]), 2),
                gap=fmt(float(row["pp2d_minus_all2d_mev"]), 2),
                igap=fmt(float(row["pp2d_minus_pp3dref_mev"]), 2),
                rec=fmt(float(row["pp2d_recovery_fraction"]), 2),
                close=float(row["pp_close_n"]),
                partial=float(row["pp_partial_n"]),
                need=float(row["needs_fp_ff_n"]),
            )
        )
    lines += [
        "",
        "## Material-level rows",
        "",
        "| material | pruned2d | pp2d | all2d | pp-all | pp-3dref | best pp shell | minimal2d | verdict |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in sorted(records, key=lambda item: (str(item["group"]), str(item["material"]))):
        lines.append(
            "| {group}/{material} | {pruned} | {pp} | {allv} | {gap} | {igap} | {pp_model} | {minimal} | {verdict} |".format(
                group=row["group"],
                material=row["material"],
                pruned=fmt(float(row["pruned2d_mev"]), 2),
                pp=fmt(float(row["pp2d_mev"]), 2),
                allv=fmt(float(row["all2d_mev"]), 2),
                gap=fmt(float(row["pp2d_minus_all2d_mev"]), 2),
                igap=fmt(float(row["pp2d_minus_pp3dref_mev"]), 2),
                pp_model=row["pp2d_model"],
                minimal=row["minimal2d_model"],
                verdict=row["verdict"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-dir", default="tb-claude")
    parser.add_argument("--output-dir", default="tb-gpt")
    parser.add_argument("--tag", default="socdelta")
    parser.add_argument("--tag-2d", default="socdelta2d")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    records: list[dict[str, str | float]] = []
    for metrics3d in sorted(input_root.glob(f"*/*/{args.source_dir}/metrics_{args.tag}.tsv")):
        source_dir = metrics3d.parent
        group, material = source_dir.relative_to(input_root).parts[:2]
        metrics2d = source_dir / f"metrics_{args.tag_2d}.tsv"
        record = selected_record(group, material, metrics3d, metrics2d if metrics2d.exists() else None)
        records.append(record)

        material_out = input_root / group / material / args.output_dir / "summary_socdelta.tsv"
        material_cols = [
            "group",
            "material",
            "pruned3d_mev",
            "pp3d_mev",
            "all3d_mev",
            "pruned2d_mev",
            "pp2d_mev",
            "pp_3dref2d_mev",
            "all2d_mev",
            "pp2d_minus_all2d_mev",
            "pp2d_minus_pp3dref_mev",
            "pp2d_recovery_fraction",
            "pp2d_model",
            "minimal2d_model",
            "verdict",
        ]
        write_tsv(material_out, [record], material_cols)

    report_root = input_root / "tb_report" / args.output_dir
    all_cols = [
        "group",
        "material",
        "pruned3d_mev",
        "bondpp3d_mev",
        "downfold3d_mev",
        "pp3d_mev",
        "pp3d_model",
        "fp3d_mev",
        "fp3d_model",
        "ff3d_mev",
        "ff3d_model",
        "ppfp3d_mev",
        "ppff3d_mev",
        "all3d_mev",
        "minimal3d_model",
        "minimal3d_mev",
        "pruned2d_mev",
        "bondpp2d_mev",
        "downfold2d_mev",
        "pp2d_mev",
        "pp2d_model",
        "fp2d_mev",
        "fp2d_model",
        "ff2d_mev",
        "ff2d_model",
        "ppfp2d_mev",
        "ppff2d_mev",
        "pp_3dref2d_mev",
        "all2d_mev",
        "minimal2d_model",
        "minimal2d_mev",
        "pp2d_minus_all2d_mev",
        "pp2d_minus_pp3dref_mev",
        "pp2d_recovery_fraction",
        "verdict",
    ]
    groups = group_summary(records)
    group_cols = [
        "group",
        "n",
        "pruned3d_mev",
        "pp3d_mev",
        "all3d_mev",
        "pruned2d_mev",
        "pp2d_mev",
        "pp_3dref2d_mev",
        "all2d_mev",
        "pp2d_minus_all2d_mev",
        "pp2d_minus_pp3dref_mev",
        "pp2d_recovery_fraction",
        "pp_close_n",
        "pp_partial_n",
        "needs_fp_ff_n",
    ]
    write_tsv(report_root / "all_materials.tsv", records, all_cols)
    write_tsv(report_root / "group_summary.tsv", groups, group_cols)
    write_markdown(report_root / "summary.md", records, groups)
    print(f"wrote {report_root} ({len(records)} materials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
