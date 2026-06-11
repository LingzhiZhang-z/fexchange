#!/usr/bin/env python3
"""Build a compact HTML index for the main TB-vs-DFT plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO / "data" / "data-DFT-input"
DEFAULT_WINDOW_TAG = "-0.25_0.25"


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    note: str


@dataclass(frozen=True)
class MaterialRow:
    group: str
    material: str
    images: dict[str, Path | None]


COLUMNS = [
    Column(
        "pruned",
        "pruned",
        "explicit model: RE f + bridge-ligand p, SOC+Delta onsite",
    ),
    Column(
        "pruned_bondpp",
        "pruned + bond p-p",
        "adds only the p-p hopping between bridge ligands on each RE-RE bond",
    ),
    Column(
        "downfold",
        "downfold",
        "RE-only model after ligand downfolding",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--window-tag", default=DEFAULT_WINDOW_TAG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output = args.output or input_root / "tb_report" / "index.html"
    output = output.resolve()

    rows = find_material_rows(input_root, args.window_tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(rows, input_root, output, args.window_tag), encoding="utf-8")

    missing = sum(1 for row in rows for path in row.images.values() if path is None)
    print(f"wrote {output}")
    print(f"materials={len(rows)} missing_plots={missing}")
    return 0


def find_material_rows(input_root: Path, window_tag: str) -> list[MaterialRow]:
    rows: list[MaterialRow] = []
    for tb_dir in sorted(input_root.glob("*/*/tb")):
        try:
            rel = tb_dir.relative_to(input_root)
        except ValueError:
            continue
        if len(rel.parts) < 3:
            continue
        group, material = rel.parts[0], rel.parts[1]
        rows.append(
            MaterialRow(
                group=group,
                material=material,
                images={column.key: find_image(tb_dir, column.key, window_tag) for column in COLUMNS},
            )
        )
    return rows


def find_image(tb_dir: Path, key: str, window_tag: str) -> Path | None:
    pattern = f"*_dft_vs_{key}_socdelta.near.{window_tag}.png"
    matches = sorted(tb_dir.glob(pattern))
    return matches[0] if matches else None


def build_html(rows: list[MaterialRow], input_root: Path, output: Path, window_tag: str) -> str:
    groups = sorted({row.group for row in rows})
    body: list[str] = []
    body.append("<!doctype html>")
    body.append('<html lang="en">')
    body.append("<head>")
    body.append('<meta charset="utf-8">')
    body.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    body.append("<title>TB fit summary</title>")
    body.append("<style>")
    body.append(CSS)
    body.append("</style>")
    body.append("</head>")
    body.append("<body>")
    body.append("<header>")
    body.append("<h1>TB fit summary</h1>")
    body.append(
        "<p>Main view: SOC+Delta onsite, near-Fermi window "
        f"<code>{html.escape(window_tag.replace('_', ' .. '))} eV</code>. "
        "Black solid lines are DFT; red dashed lines are TB. "
        "Each TB plot uses the near-Fermi lower-envelope shift from the plotting script.</p>"
    )
    body.append('<nav class="groups">')
    for group in groups:
        body.append(f'<a href="#{anchor(group)}">{html.escape(group)}</a>')
    body.append("</nav>")
    body.append("</header>")

    for group in groups:
        group_rows = [row for row in rows if row.group == group]
        body.append(f'<section class="group" id="{anchor(group)}">')
        body.append(f"<h2>{html.escape(group)}</h2>")
        for row in group_rows:
            body.append('<article class="material">')
            body.append(f"<h3>{html.escape(row.material)}</h3>")
            body.append('<div class="plot-grid">')
            for column in COLUMNS:
                body.append('<figure class="plot-card">')
                body.append(f"<figcaption>{html.escape(column.label)}</figcaption>")
                body.append(f'<p class="note">{html.escape(column.note)}</p>')
                image = row.images[column.key]
                if image is None:
                    body.append('<div class="missing">missing</div>')
                else:
                    rel = rel_href(image, output.parent)
                    alt = f"{row.group} {row.material} {column.label}"
                    body.append(f'<a href="{rel}"><img src="{rel}" alt="{html.escape(alt)}"></a>')
                body.append("</figure>")
            body.append("</div>")
            body.append("</article>")
        body.append("</section>")

    body.append("<footer>")
    body.append(
        "<p>Full diagnostic plots, HR-onsite variants, 2D variants, and wider windows remain "
        "inside each material's <code>tb/</code> directory.</p>"
    )
    body.append(f"<p>Input root: <code>{html.escape(str(input_root))}</code></p>")
    body.append("</footer>")
    body.append("</body>")
    body.append("</html>")
    body.append("")
    return "\n".join(body)


def rel_href(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path, base).replace(os.sep, "/"), quote=True)


def anchor(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


CSS = """
:root {
  color-scheme: light;
  --bg: #f7f7f5;
  --panel: #ffffff;
  --text: #1c1f22;
  --muted: #5a626b;
  --line: #d8dadd;
  --accent: #9b1c1c;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 Arial, Helvetica, sans-serif;
}

header,
section,
footer {
  max-width: 1720px;
  margin: 0 auto;
  padding: 20px 24px;
}

header {
  border-bottom: 1px solid var(--line);
}

h1,
h2,
h3,
p {
  margin: 0;
}

h1 {
  font-size: 28px;
  font-weight: 700;
}

header p,
footer p {
  max-width: 980px;
  margin-top: 8px;
  color: var(--muted);
}

code {
  font-family: Consolas, Monaco, monospace;
}

.groups {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
}

.groups a {
  color: var(--accent);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 5px 9px;
  text-decoration: none;
  background: var(--panel);
}

.group {
  padding-top: 26px;
}

.group h2 {
  font-size: 22px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 8px;
}

.material {
  margin-top: 22px;
}

.material h3 {
  font-size: 17px;
  margin-bottom: 10px;
}

.plot-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.plot-card {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
}

figcaption {
  font-size: 15px;
  font-weight: 700;
}

.note {
  min-height: 38px;
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}

img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--line);
  background: #fff;
}

.missing {
  display: grid;
  min-height: 260px;
  place-items: center;
  border: 1px dashed var(--line);
  color: var(--muted);
}

footer {
  border-top: 1px solid var(--line);
  margin-top: 24px;
  padding-bottom: 34px;
}

@media (max-width: 1050px) {
  .plot-grid {
    grid-template-columns: 1fr;
  }
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
