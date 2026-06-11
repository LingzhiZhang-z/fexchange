#!/usr/bin/env python3
"""HTML report for the tb-claude shell-extension analysis (separate from index.html)."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO / "data" / "data-DFT-input"
PLOT_MODELS = ["pruned", "bondpp", "downfold", "minimal", "all"]


def read_metrics(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = {}
    for part in lines[0].removeprefix("#").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            meta[key.strip()] = value.strip()
    cols = lines[1].split("\t")
    rows = []
    for line in lines[2:]:
        values = line.split("\t")
        row = {"model": values[0], "n_added": int(values[1])}
        row.update({c: float(v) for c, v in zip(cols[2:], values[2:])})
        rows.append(row)
    return meta, rows


def row_for(rows: list[dict], model: str) -> dict | None:
    return next((r for r in rows if r["model"] == model), None)


def fmt(row: dict | None, key: str, scale: float = 1.0, digits: int = 1) -> str:
    if row is None:
        return "-"
    return f"{row[key] * scale:.{digits}f}"


def material_cell(meta, rows) -> str:
    minimal = meta.get("minimal", "")
    order = ["pruned", "bondpp", "downfold", minimal, "all", "w90_full"]
    labels = ["pruned", "bondpp", "downfold", f"minimal ({minimal})", "all", "W90 full"]
    cells = []
    for model, label in zip(order, labels):
        row = row_for(rows, model)
        cells.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{fmt(row, 'chamfer', 1000)}</td><td>{fmt(row, 'miss', 1000)}</td>"
            f"<td>{fmt(row, 'env', 1000)}</td><td>{fmt(row, 'envpp', 1, 2)}</td>"
            f"<td>{row['n_added'] if row and row['n_added'] >= 0 else '-'}</td></tr>"
        )
    return (
        "<table class='metrics'><tr><th>model</th><th>chamfer<br>(meV)</th>"
        "<th>miss<br>(meV)</th><th>env<br>(meV)</th><th>envpp</th><th>+blocks</th></tr>"
        + "".join(cells) + "</table>"
    )


def find_plot(tb_claude: Path, stem: str, model: str, tag: str) -> Path | None:
    matches = sorted(tb_claude.glob(f"{stem}_dft_vs_claude_{model}*_{tag}.near.*.png"))
    return matches[0] if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--tag", default="socdelta")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output = (args.output or input_root / "tb_report" / "claude.html").resolve()

    sections, summary_rows = [], []
    for metrics_path in sorted(input_root.glob(f"*/*/tb-claude/metrics_{args.tag}.tsv")):
        tb_claude = metrics_path.parent
        group, material = tb_claude.relative_to(input_root).parts[:2]
        meta, rows = read_metrics(metrics_path)
        minimal = meta.get("minimal", "")
        stem = next(iter(tb_claude.glob(f"*_bands_claude_pruned_{args.tag}.txt"))).name.split("_bands_")[0]

        figures = []
        for key in PLOT_MODELS:
            model = minimal if key == "minimal" else key
            png = find_plot(tb_claude, stem, model, args.tag)
            label = f"minimal: {minimal}" if key == "minimal" else key
            if png is None:
                figures.append(f"<figure><figcaption>{html.escape(label)}</figcaption>"
                               "<div class='missing'>missing</div></figure>")
            else:
                rel = html.escape(os.path.relpath(png, output.parent).replace(os.sep, "/"), quote=True)
                figures.append(f"<figure><figcaption>{html.escape(label)}</figcaption>"
                               f"<a href='{rel}'><img src='{rel}' alt='{html.escape(label)}'></a></figure>")
        sections.append(
            f"<article id='{group}-{material}'><h3>{html.escape(group)} / {html.escape(material)}</h3>"
            + material_cell(meta, rows)
            + "<div class='plots'>" + "".join(figures) + "</div></article>"
        )
        summary_rows.append(
            "<tr>"
            f"<td><a href='#{group}-{material}'>{html.escape(group)}/{html.escape(material)}</a></td>"
            f"<td>{fmt(row_for(rows, 'pruned'), 'chamfer', 1000)}</td>"
            f"<td>{fmt(row_for(rows, 'bondpp'), 'chamfer', 1000)}</td>"
            f"<td>{fmt(row_for(rows, 'downfold'), 'chamfer', 1000)}</td>"
            f"<td>{fmt(row_for(rows, minimal), 'chamfer', 1000)} ({html.escape(minimal)})</td>"
            f"<td>{fmt(row_for(rows, 'all'), 'chamfer', 1000)}</td>"
            f"<td>{fmt(row_for(rows, 'w90_full'), 'chamfer', 1000)}</td>"
            f"<td>{fmt(row_for(rows, 'pruned'), 'envpp', 1, 2)} / {fmt(row_for(rows, minimal), 'envpp', 1, 2)}</td>"
            "</tr>"
        )

    body = "\n".join(
        [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<title>tb-claude shell-extension report</title><style>", CSS, "</style></head><body>",
            "<header><h1>tb-claude: minimal TB additions for the DFT near-Fermi window</h1>",
            "<p>Level-set comparison in the -0.25..0.25 eV window after one rigid TB shift "
            "(chamfer fit; the shift itself is a Fermi-level correction and never penalised). "
            "chamfer = pooled distance between DFT and TB level sets; miss = DFT levels without "
            "a nearby TB level; env = lower-envelope RMSE; envpp = TB/DFT envelope dispersion ratio "
            "(1 = correct dispersion scale). Models: pruned baselines, cumulative distance shells "
            "(pp = ligand-ligand, fp = RE-ligand, ff = RE-RE 3rd NN+), combos, full Wannier90 ceiling. "
            "SOC+Delta onsite throughout.</p>",
            "<table class='summary'><tr><th>material</th><th>pruned</th><th>bondpp</th><th>downfold</th>"
            "<th>minimal</th><th>all&le;dmax</th><th>W90 full</th><th>envpp pruned/min</th></tr>",
            "\n".join(summary_rows),
            "</table><p>chamfer in meV.</p></header>",
            "\n".join(sections),
            "</body></html>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    print(f"wrote {output} ({len(summary_rows)} materials)")
    return 0


CSS = """
body { margin:0 auto; max-width:1780px; padding:20px; font:14px/1.45 Arial,sans-serif;
       color:#1c1f22; background:#f7f7f5; }
h1 { font-size:24px; } h3 { margin:26px 0 8px; border-bottom:1px solid #d8dadd; }
table { border-collapse:collapse; background:#fff; }
td,th { border:1px solid #d8dadd; padding:3px 8px; text-align:right; }
td:first-child,th:first-child { text-align:left; }
.summary { margin-top:12px; } .metrics { font-size:12px; margin-bottom:8px; }
.plots { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }
figure { margin:0; background:#fff; border:1px solid #d8dadd; padding:6px; }
figcaption { font-weight:700; font-size:12px; margin-bottom:4px; }
img { width:100%; height:auto; display:block; }
.missing { display:grid; min-height:120px; place-items:center; color:#5a626b;
           border:1px dashed #d8dadd; }
a { color:#9b1c1c; }
"""


if __name__ == "__main__":
    raise SystemExit(main())
