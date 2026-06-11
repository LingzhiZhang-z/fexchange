#!/usr/bin/env python3
"""Plot strict-2D tb-gpt band overlays for manual inspection."""

from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "tb"))
import plot_tb_bands as plot  # noqa: E402


DEFAULT_TARGETS = ["REOF/SmOF", "RESI/SmSI-re", "REOF/CeOF", "RESI/NdSI-re"]
DEFAULT_WINDOW = (-0.25, 0.25)


def read_metrics(path: Path) -> tuple[dict[str, str], list[dict[str, str | int | float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta: dict[str, str] = {}
    for part in lines[0].removeprefix("#").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            meta[key.strip()] = value.strip()
    cols = lines[1].split("\t")
    rows: list[dict[str, str | int | float]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        values = line.split("\t")
        row: dict[str, str | int | float] = {"model": values[0], "n_added": int(values[1])}
        for key, value in zip(cols[2:], values[2:]):
            row[key] = float(value)
        rows.append(row)
    return meta, rows


def row_for(rows: list[dict[str, str | int | float]], model: str) -> dict[str, str | int | float] | None:
    return next((row for row in rows if row["model"] == model), None)


def best_prefix(rows: list[dict[str, str | int | float]], prefix: str) -> dict[str, str | int | float] | None:
    candidates = [
        row for row in rows
        if str(row["model"]).startswith(prefix) and str(row["model"])[len(prefix): len(prefix) + 1].isdigit()
    ]
    return min(candidates, key=lambda row: float(row["chamfer"])) if candidates else None


def stem_from_tb_gpt(tb_gpt: Path, tag: str) -> str:
    band = next(tb_gpt.glob(f"*_bands_gpt_pruned_{tag}.txt"))
    return band.name.split("_bands_gpt_", 1)[0]


def selected_rows(meta: dict[str, str], rows: list[dict[str, str | int | float]]) -> list[tuple[str, dict[str, str | int | float]]]:
    minimal_model = meta.get("minimal", "")
    minimal_label = "minimal/all" if minimal_model == "all" else f"minimal ({minimal_model})"
    specs: list[tuple[str, dict[str, str | int | float] | None]] = [
        ("pruned", row_for(rows, "pruned")),
        ("bondpp", row_for(rows, "bondpp")),
        ("best pp-only", best_prefix(rows, "pp")),
        (minimal_label, row_for(rows, minimal_model)),
        ("pp 3D-ref", row_for(rows, "pp_3dref")),
    ]
    if minimal_model != "all":
        specs.insert(-1, ("all", row_for(rows, "all")))
    out: list[tuple[str, dict[str, str | int | float]]] = []
    seen = set()
    for label, row in specs:
        if row is None:
            continue
        model = str(row["model"])
        key = model
        if key in seen:
            continue
        seen.add(key)
        out.append((label, row))
    return out


def make_plot(
    *,
    tb_gpt: Path,
    stem: str,
    tag: str,
    label: str,
    row: dict[str, str | int | float],
    window: tuple[float, float],
    run: bool,
) -> Path:
    model = str(row["model"])
    band = tb_gpt / f"{stem}_bands_gpt_{model}_{tag}.txt"
    if not band.is_file():
        raise FileNotFoundError(f"Missing band file for {tb_gpt}: {band.name}")
    dft_band = plot.resolve_dft_band(tb_gpt.parent / "tb", stem)
    efermi = plot.resolve_efermi(tb_gpt.parent / "tb", dft_band)
    dft_nodes = plot.read_dft_nodes(dft_band)
    tb_ticks = plot.read_tb_ticks(band)
    pdos_spec = plot.resolve_pdos(dft_band, stem)
    suffix = re.sub(r"[^a-z0-9_.-]+", "_", label.lower().replace("+", "p")).strip("_")
    png = tb_gpt / f"{stem}_dft_vs_gpt_{suffix}_{model}_{tag}.near.-0.25_0.25.png"
    script = png.with_suffix(".gnuplot")
    tb_title = f"tb-gpt {label}: {model} ({float(row['chamfer']) * 1000:.2f} meV)"
    script.write_text(
        plot.build_gnuplot(
            dft_band=dft_band,
            tb_band=band,
            output=png,
            stem=stem,
            tb_title=tb_title,
            dft_nodes=dft_nodes,
            tb_ticks=tb_ticks,
            efermi=efermi,
            tb_shift=-float(row["shift"]),
            window=window,
            pdos_spec=pdos_spec,
        ),
        encoding="utf-8",
    )
    if run:
        subprocess.run(["gnuplot", str(script)], check=True)
    return png


def write_html(output: Path, cards: list[dict[str, str]]) -> None:
    groups = []
    for card in cards:
        rel = os.path.relpath(card["png"], output.parent).replace(os.sep, "/")
        groups.append(
            "<figure>"
            f"<figcaption>{html.escape(card['title'])}<br><span>{html.escape(card['metric'])}</span></figcaption>"
            f"<a href='{html.escape(rel, quote=True)}'><img src='{html.escape(rel, quote=True)}' alt='{html.escape(card['title'], quote=True)}'></a>"
            "</figure>"
        )
    body = "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>tb-gpt strict2d plots</title>",
            "<style>",
            "body{margin:0 auto;max-width:1780px;padding:20px;font:14px/1.45 Arial,sans-serif;background:#f7f7f5;color:#1b1f23}",
            "h1{font-size:24px;margin:0 0 10px}",
            ".grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}",
            "figure{margin:0;background:#fff;border:1px solid #d8dadd;padding:7px}",
            "figcaption{font-weight:700;font-size:12px;margin:0 0 5px}",
            "figcaption span{font-weight:400;color:#555}",
            "img{display:block;width:100%;height:auto}",
            "</style></head><body>",
            "<h1>tb-gpt strict 2D DFT-vs-TB overlays</h1>",
            "<p>Near-Fermi window: -0.25..0.25 eV.  Red dashed curves are TB after the fitted rigid shift; black curves are DFT.</p>",
            "<div class='grid'>",
            "\n".join(groups),
            "</div></body></html>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--input-root", type=Path, default=REPO / "data" / "data-DFT-input")
    parser.add_argument("--tag", default="socdelta_strict2d")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--html", type=Path, default=REPO / "data" / "data-DFT-input" / "tb_report" / "tb-gpt" / "strict2d_plots.html")
    args = parser.parse_args()

    cards: list[dict[str, str]] = []
    for target in args.targets:
        tb_gpt = args.input_root / target / "tb-gpt"
        metrics = tb_gpt / f"metrics_{args.tag}.tsv"
        meta, rows = read_metrics(metrics)
        stem = stem_from_tb_gpt(tb_gpt, args.tag)
        for label, row in selected_rows(meta, rows):
            png = make_plot(
                tb_gpt=tb_gpt,
                stem=stem,
                tag=args.tag,
                label=label,
                row=row,
                window=DEFAULT_WINDOW,
                run=not args.no_run,
            )
            title = f"{target} / {label}"
            metric = f"model={row['model']}; chamfer={float(row['chamfer']) * 1000:.2f} meV; n_added={row['n_added']}"
            cards.append({"png": str(png), "title": title, "metric": metric})
            print(f"wrote {png}")
    write_html(args.html, cards)
    print(f"wrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
