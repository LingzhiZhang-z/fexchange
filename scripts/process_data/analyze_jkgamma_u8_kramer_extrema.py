#!/usr/bin/env python3
"""Summarize U=8 JKGamma extrema by material, Kramers state, and shell.

The definitions are intentionally taken from plot_jkgamma.py so the numbers
match the plotted J/K/Gamma/Gamma' and NNN DM conventions.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_bond_exchange import decompose  # noqa: E402
from plot_jkgamma import BRANCHES, RECHX_FAMS, case_cfg, scale_for_u  # noqa: E402

DEFAULT_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")
DEFAULT_TSV = Path("outputs/analysis/jkgamma_u8_kramer_shell_extrema.tsv")
DEFAULT_MD = Path("outputs/analysis/jkgamma_u8_kramer_shell_extrema.md")
U_VALUE = 8

DISPLAY = {
    "Gamma_prime": "Gp",
    "D_c": "D",
    "D_a": "Dp",
}
ORDER = ["J", "K", "Gamma", "Gp", "D", "Dp"]


@dataclass
class Extremum:
    value: float
    ratio: float
    location: str


@dataclass
class Stat:
    minimum: Extremum | None = None
    maximum: Extremum | None = None

    def update(self, value: float, ratio: float, location: str) -> None:
        item = Extremum(value=value, ratio=ratio, location=location)
        if self.minimum is None or value < self.minimum.value:
            self.minimum = item
        if self.maximum is None or value > self.maximum.value:
            self.maximum = item


def shell_for_case(family: str, bond: str) -> str:
    if family.startswith("REX3"):
        return "NN"
    if bond.startswith("J1-"):
        return "NN"
    if bond.startswith("J2-"):
        return "NNN"
    return "other"


def iter_u8_files(root: Path):
    for family_root in sorted(root.glob("REX3-*")):
        if not family_root.is_dir():
            continue
        family = family_root.name
        for path in sorted(family_root.glob("*/*/*/*/U_08.txt")):
            material, bond, branch, run = path.parts[-5:-1]
            if branch in BRANCHES and case_cfg(family, bond):
                yield family, material, bond, branch, run, path

    cubic_root = root / "REChX_cubic"
    for family in RECHX_FAMS:
        family_root = cubic_root / family
        if not family_root.exists():
            continue
        for path in sorted(family_root.glob("*/*/*/*/U_08.txt")):
            material, bond, branch, run = path.parts[-5:-1]
            if branch in BRANCHES and case_cfg(family, bond):
                yield family, material, bond, branch, run, path


def read_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        tensor = np.array([float(x) for x in cols[3:12]], dtype=float).reshape(3, 3)
        yield float(cols[0]), tensor


def collect(root: Path):
    stats: dict[tuple[str, str, str, str, str], Stat] = defaultdict(Stat)
    files_seen = 0
    for family, material, bond, branch, run, path in iter_u8_files(root):
        rotation, params, axis, _tag, _group = case_cfg(family, bond)
        files_seen += 1
        for ratio, tensor in read_rows(path):
            values = decompose(rotation @ tensor @ rotation.T, axis)
            for source_key, _colour in params:
                exchange = DISPLAY.get(source_key, source_key)
                value = values[source_key] * 1000.0 * scale_for_u(U_VALUE)
                location = f"{bond}/{branch}@jh={ratio:g}"
                stats[(family, material, run, shell_for_case(family, bond), exchange)].update(value, ratio, location)
    return stats, files_seen


def sort_key(key: tuple[str, str, str, str, str]):
    family, material, run, shell, exchange = key
    return (
        family,
        material,
        run,
        {"NN": 0, "NNN": 1}.get(shell, 9),
        ORDER.index(exchange) if exchange in ORDER else 99,
        exchange,
    )


def write_tsv(path: Path, stats) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "family",
                "material",
                "kramer_state",
                "shell",
                "exchange",
                "min_mev",
                "min_at_ratio",
                "min_location",
                "max_mev",
                "max_at_ratio",
                "max_location",
            ]
        )
        for family, material, run, shell, exchange in sorted(stats, key=sort_key):
            stat = stats[(family, material, run, shell, exchange)]
            if stat.minimum is None or stat.maximum is None:
                continue
            writer.writerow(
                [
                    family,
                    material,
                    run,
                    shell,
                    exchange,
                    f"{stat.minimum.value:.6g}",
                    f"{stat.minimum.ratio:.6g}",
                    stat.minimum.location,
                    f"{stat.maximum.value:.6g}",
                    f"{stat.maximum.ratio:.6g}",
                    stat.maximum.location,
                ]
            )
            count += 1
    return count


def cell(stat: Stat | None) -> str:
    if stat is None or stat.minimum is None or stat.maximum is None:
        return ""
    return f"[{stat.minimum.value:.4g},{stat.maximum.value:.4g}]"


def write_markdown(path: Path, stats) -> int:
    groups: dict[tuple[str, str, str, str], dict[str, Stat]] = defaultdict(dict)
    for (family, material, run, shell, exchange), stat in stats.items():
        groups[(family, material, run, shell)][exchange] = stat

    lines = [
        "# JKGamma U=8 Kramers-State/Shell Extrema",
        "",
        "Values are in meV and use the same plotted definitions as `plot_jkgamma.py`.",
        "Each cell is `[min,max]` over matching bonds, both plotted branches, and the full `Jh/U` curve at `U=8`.",
        "",
        "Definitions:",
        "- `Gp = Gamma_prime`",
        "- `D = D_c`",
        "- `Dp = D_a`",
        "",
        "The companion TSV with exact extrema locations is:",
        "`outputs/analysis/jkgamma_u8_kramer_shell_extrema.tsv`",
        "",
    ]

    sections = [
        ("REX3 NN", lambda family, shell: family.startswith("REX3") and shell == "NN", ["J", "K", "Gamma", "Gp"]),
        ("REChX NN", lambda family, shell: family in RECHX_FAMS and shell == "NN", ["J", "K", "Gamma", "Gp"]),
        ("REChX NNN", lambda family, shell: family in RECHX_FAMS and shell == "NNN", ["J", "K", "Gamma", "Gp", "D", "Dp"]),
    ]

    row_count = 0
    for title, include, columns in sections:
        lines.append(f"## {title}")
        lines.append("| family | material | kramer_state | shell | " + " | ".join(columns) + " |")
        lines.append("|---|---|---|---|" + "|".join(["---:"] * len(columns)) + "|")
        for (family, material, run, shell), values in sorted(
            groups.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                {"NN": 0, "NNN": 1}.get(item[0][3], 9),
            ),
        ):
            if not include(family, shell):
                continue
            lines.append(
                f"| {family} | {material} | `{run}` | {shell} | "
                + " | ".join(cell(values.get(column)) for column in columns)
                + " |"
            )
            row_count += 1
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return row_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    stats, files_seen = collect(args.root)
    if files_seen == 0:
        print(f"error: no U_08 files found under {args.root}", file=sys.stderr)
        return 1

    tsv_rows = write_tsv(args.tsv, stats)
    markdown_rows = write_markdown(args.markdown, stats)
    print(f"# files_seen {files_seen}")
    print(f"# extrema_rows {tsv_rows}")
    print(f"# markdown_rows {markdown_rows}")
    print(f"# wrote {args.tsv}")
    print(f"# wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
