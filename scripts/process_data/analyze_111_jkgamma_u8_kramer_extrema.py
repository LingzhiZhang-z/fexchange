#!/usr/bin/env python3
"""Summarize U=8 J/K/Gamma/Gamma' extrema in a fixed 111-axis frame.

The local axes are

    a = [11-2] / sqrt(6)
    b = [-110] / sqrt(2)
    c = [111] / sqrt(3)

REX3 tensors are native cubic-frame tensors and are rotated into this frame.
REChX tensors are read from outputs_organized/REChX_cubic, so they are also
treated as cubic-frame tensors before the fixed 111-frame projection.
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
from plot_jkgamma import BRANCHES, RECHX_FAMS, scale_for_u  # noqa: E402

DEFAULT_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")
DEFAULT_TSV = Path("outputs/analysis/jkgamma_111_from_cubic_u8_kramer_shell_extrema.filtered.tsv")
DEFAULT_MD = Path("outputs/analysis/jkgamma_111_from_cubic_u8_kramer_shell_extrema.filtered.md")
DEFAULT_YAML = Path("outputs/analysis/jkgamma_111_from_cubic_u8_kramer_shell_extrema.filtered.yaml")
U_VALUE = 8
FILTERS = ("YbOCl_baseline", "theta015", "theta045", "theta075")
EXCHANGES = ("J", "K", "Gamma", "Gp", "Da", "Db", "Dc")

A111 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
B111 = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
C111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
Q111 = np.column_stack([A111, B111, C111])


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

    def cell(self) -> str:
        if self.minimum is None or self.maximum is None:
            return ""
        return f"[{self.minimum.value:.6g},{self.maximum.value:.6g}]"


def passes_filter(run: str, filters: tuple[str, ...]) -> bool:
    return not any(item in run for item in filters)


def shell_for_case(family: str, bond: str) -> str:
    if family.startswith("REX3"):
        return "NN"
    if bond.startswith("J1-"):
        return "NN"
    if bond.startswith("J2-"):
        return "NNN"
    return "other"


def iter_u8_files(root: Path, filters: tuple[str, ...]):
    for family_root in sorted(root.glob("REX3-*")):
        if not family_root.is_dir():
            continue
        family = family_root.name
        for path in sorted(family_root.glob("*/*/*/*/U_08.txt")):
            material, bond, branch, run = path.parts[-5:-1]
            if branch in BRANCHES and bond in {"x", "y", "z"} and passes_filter(run, filters):
                yield family, material, bond, branch, run, path

    cubic_root = root / "REChX_cubic"
    for family in RECHX_FAMS:
        family_root = cubic_root / family
        if not family_root.exists():
            continue
        for path in sorted(family_root.glob("*/*/*/*/U_08.txt")):
            material, bond, branch, run = path.parts[-5:-1]
            if branch not in BRANCHES or not passes_filter(run, filters):
                continue
            if bond.startswith("J1-") or bond.startswith("J2-"):
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


def values_111(_family: str, tensor: np.ndarray) -> dict[str, float]:
    tensor = Q111.T @ tensor @ Q111
    decomposed = decompose(tensor, "z")
    return {
        "J": decomposed["J"],
        "K": decomposed["K"],
        "Gamma": decomposed["Gamma"],
        "Gp": decomposed["Gamma_prime"],
        "Da": decomposed["D_a"],
        "Db": decomposed["D_b"],
        "Dc": decomposed["D_c"],
    }


def collect(root: Path, filters: tuple[str, ...]):
    stats: dict[tuple[str, str, str, str, str], Stat] = defaultdict(Stat)
    files_seen = 0
    for family, material, bond, branch, run, path in iter_u8_files(root, filters):
        files_seen += 1
        for ratio, tensor in read_rows(path):
            values = values_111(family, tensor)
            for exchange in EXCHANGES:
                value = values[exchange] * 1000.0 * scale_for_u(U_VALUE)
                location = f"{bond}/{branch}@jh={ratio:g}"
                stats[(family, material, run, shell_for_case(family, bond), exchange)].update(value, ratio, location)
    return stats, files_seen


def sort_group(group: tuple[str, str, str, str]) -> tuple:
    family, material, run, shell = group
    return family, material, run, {"NN": 0, "NNN": 1}.get(shell, 9)


def grouped_stats(stats) -> dict[tuple[str, str, str, str], dict[str, Stat]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Stat]] = defaultdict(dict)
    for (family, material, run, shell, exchange), stat in stats.items():
        grouped[(family, material, run, shell)][exchange] = stat
    return grouped


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
        for key in sorted(
            stats,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                {"NN": 0, "NNN": 1}.get(item[3], 9),
                EXCHANGES.index(item[4]),
            ),
        ):
            family, material, run, shell, exchange = key
            stat = stats[key]
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


def write_markdown(path: Path, stats, filters: tuple[str, ...]) -> int:
    grouped = grouped_stats(stats)
    lines = [
        "# 111-Axis JKGamma U=8 Kramers-State/Shell Extrema",
        "",
        "Values are in meV; each cell is `[min,max]` over matching bonds, both plotted branches, and the full `Jh/U` curve at `U=8`.",
        "Fixed local axes: `a=[11-2]`, `b=[-110]`, `c=[111]`.",
        "For both REX3 and REChX this uses cubic-frame tensors; REChX is read from `outputs_organized/REChX_cubic`.",
        "",
        "Filtered out:",
        *[f"- `{item}`" for item in filters],
        "",
    ]
    sections = [
        ("REX3 NN", lambda family, shell: family.startswith("REX3") and shell == "NN"),
        ("REChX NN", lambda family, shell: family in RECHX_FAMS and shell == "NN"),
        ("REChX NNN", lambda family, shell: family in RECHX_FAMS and shell == "NNN"),
    ]
    row_count = 0
    for title, include in sections:
        lines.append(f"## {title}")
        lines.append("| family | material | kramer_state | " + " | ".join(EXCHANGES) + " |")
        lines.append("|---|---|---|" + "|".join(["---:"] * len(EXCHANGES)) + "|")
        for group, values in sorted(grouped.items(), key=lambda item: sort_group(item[0])):
            family, material, run, shell = group
            if not include(family, shell):
                continue
            lines.append(
                f"| {family} | {material} | `{run}` | "
                + " | ".join(values[exchange].cell() for exchange in EXCHANGES)
                + " |"
            )
            row_count += 1
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return row_count


def write_yaml(path: Path, stats, filters: tuple[str, ...]) -> int:
    grouped = grouped_stats(stats)
    lines = [
        "jkgamma_111_from_cubic_u8_filtered:",
        "  definition:",
        "    unit: meV",
        "    range: '[min,max]'",
        "    axes:",
        "      a: '[11-2]/sqrt(6)'",
        "      b: '[-110]/sqrt(2)'",
        "      c: '[111]/sqrt(3)'",
        "    exchanges: [J, K, Gamma, Gp, Da, Db, Dc]",
        "    filtered_out:",
        *[f"      - {item}" for item in filters],
    ]
    sections = [
        ("REX3_NN", lambda family, shell: family.startswith("REX3") and shell == "NN"),
        ("REChX_NN", lambda family, shell: family in RECHX_FAMS and shell == "NN"),
        ("REChX_NNN", lambda family, shell: family in RECHX_FAMS and shell == "NNN"),
    ]
    row_count = 0
    for section, include in sections:
        lines.append(f"  {section}:")
        for group, values in sorted(grouped.items(), key=lambda item: sort_group(item[0])):
            family, material, run, shell = group
            if not include(family, shell):
                continue
            entries = ", ".join(f"{exchange}: {values[exchange].cell()}" for exchange in EXCHANGES)
            lines.append(f"    - {{family: {family}, material: {material}, state: {run}, {entries}}}")
            row_count += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return row_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--filter", action="append", default=list(FILTERS))
    args = parser.parse_args(argv)

    filters = tuple(args.filter or ())
    stats, files_seen = collect(args.root, filters)
    if files_seen == 0:
        print(f"error: no U_08 files found under {args.root}", file=sys.stderr)
        return 1
    tsv_rows = write_tsv(args.tsv, stats)
    markdown_rows = write_markdown(args.markdown, stats, filters)
    yaml_rows = write_yaml(args.yaml, stats, filters)
    print(f"# files_seen {files_seen}")
    print(f"# extrema_rows {tsv_rows}")
    print(f"# markdown_rows {markdown_rows}")
    print(f"# yaml_rows {yaml_rows}")
    print(f"# wrote {args.tsv}")
    print(f"# wrote {args.markdown}")
    print(f"# wrote {args.yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
