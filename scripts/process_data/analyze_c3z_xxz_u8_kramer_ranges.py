#!/usr/bin/env python3
"""Summarize U=8 z-C3 XXZ J/K and non-XXZ gray ranges by Kramers state.

The result matches the shell-combined gray figures from plot_c3z_xxz.py:
J and K are taken from one representative bond in the shell, while the gray
range spans all non-XXZ components from all bonds in that shell.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_c3z_xxz import (  # noqa: E402
    BRANCHES,
    FAMILIES,
    params_for_bond,
    read_rows,
    representative_bond,
    scale_for_u,
    shell_for_bond,
)

DEFAULT_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")
DEFAULT_TSV = Path("outputs/analysis/c3z_xxz_u8_kramer_shell_ranges.tsv")
DEFAULT_MD = Path("outputs/analysis/c3z_xxz_u8_kramer_shell_ranges.md")
U_VALUE = 8
MODE = "xxz-gray"
KEEP_NN_DM = True


@dataclass
class Extremum:
    value: float
    location: str


@dataclass
class RangeStat:
    minimum: Extremum | None = None
    maximum: Extremum | None = None

    def update(self, value: float, location: str) -> None:
        item = Extremum(value=value, location=location)
        if self.minimum is None or value < self.minimum.value:
            self.minimum = item
        if self.maximum is None or value > self.maximum.value:
            self.maximum = item

    def cell(self) -> str:
        if self.minimum is None or self.maximum is None:
            return ""
        return f"[{self.minimum.value:.4g},{self.maximum.value:.4g}]"

    def absmax(self) -> float:
        if self.minimum is None or self.maximum is None:
            return 0.0
        return max(abs(self.minimum.value), abs(self.maximum.value))


def discover_u8_files(root: Path):
    cases: dict[tuple[str, str, str, str, str], Path] = {}
    for family in FAMILIES:
        family_root = root / family
        if not family_root.exists():
            continue
        for path in sorted(family_root.glob("*/*/*/*/U_08.txt")):
            material, bond, branch, run = path.parts[-5:-1]
            if branch not in BRANCHES:
                continue
            shell = shell_for_bond(bond)
            if shell not in {"J1", "J2"}:
                continue
            cases[(family, material, bond, branch, run)] = path
    return cases


def group_by_shell(cases: dict[tuple[str, str, str, str, str], Path]):
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for (family, material, bond, branch, run), path in cases.items():
        shell = shell_for_bond(bond)
        grouped[(family, material, run, shell)][bond][branch] = path
    return grouped


def collect(root: Path):
    cases = discover_u8_files(root)
    grouped = group_by_shell(cases)
    rows = []
    scale = scale_for_u(U_VALUE)

    for (family, material, run, shell), bond_files in sorted(grouped.items()):
        rep = representative_bond(shell, set(bond_files))
        j_stat = RangeStat()
        k_stat = RangeStat()
        gray_stat = RangeStat()
        gray_components: set[str] = set()

        for bond, branch_files in sorted(bond_files.items()):
            params = params_for_bond(bond, MODE, KEEP_NN_DM)
            for branch, path in sorted(branch_files.items()):
                for ratio, values in read_rows(path):
                    for key, _colour, _lw in params:
                        value = values[key] * 1000.0 * scale
                        location = f"{bond}/{branch}/{key}@jh={ratio:g}"
                        if key == "J" and bond == rep:
                            j_stat.update(value, location)
                        elif key == "K" and bond == rep:
                            k_stat.update(value, location)
                        elif key not in {"J", "K"}:
                            gray_components.add(key)
                            gray_stat.update(value, location)

        rows.append(
            {
                "family": family,
                "material": material,
                "kramer_state": run,
                "shell": "NN" if shell == "J1" else "NNN",
                "jk_source_bond": rep,
                "J": j_stat,
                "K": k_stat,
                "gray": gray_stat,
                "gray_absmax": gray_stat.absmax(),
                "gray_components": ",".join(sorted(gray_components)),
            }
        )
    return rows, len(cases)


def write_tsv(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "material",
        "kramer_state",
        "shell",
        "jk_source_bond",
        "J_min_mev",
        "J_min_location",
        "J_max_mev",
        "J_max_location",
        "K_min_mev",
        "K_min_location",
        "K_max_mev",
        "K_max_location",
        "gray_min_mev",
        "gray_min_location",
        "gray_max_mev",
        "gray_max_location",
        "gray_absmax_mev",
        "gray_components",
    ]
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            j_stat: RangeStat = row["J"]
            k_stat: RangeStat = row["K"]
            gray_stat: RangeStat = row["gray"]
            writer.writerow(
                {
                    "family": row["family"],
                    "material": row["material"],
                    "kramer_state": row["kramer_state"],
                    "shell": row["shell"],
                    "jk_source_bond": row["jk_source_bond"],
                    "J_min_mev": f"{j_stat.minimum.value:.6g}" if j_stat.minimum else "",
                    "J_min_location": j_stat.minimum.location if j_stat.minimum else "",
                    "J_max_mev": f"{j_stat.maximum.value:.6g}" if j_stat.maximum else "",
                    "J_max_location": j_stat.maximum.location if j_stat.maximum else "",
                    "K_min_mev": f"{k_stat.minimum.value:.6g}" if k_stat.minimum else "",
                    "K_min_location": k_stat.minimum.location if k_stat.minimum else "",
                    "K_max_mev": f"{k_stat.maximum.value:.6g}" if k_stat.maximum else "",
                    "K_max_location": k_stat.maximum.location if k_stat.maximum else "",
                    "gray_min_mev": f"{gray_stat.minimum.value:.6g}" if gray_stat.minimum else "",
                    "gray_min_location": gray_stat.minimum.location if gray_stat.minimum else "",
                    "gray_max_mev": f"{gray_stat.maximum.value:.6g}" if gray_stat.maximum else "",
                    "gray_max_location": gray_stat.maximum.location if gray_stat.maximum else "",
                    "gray_absmax_mev": f"{row['gray_absmax']:.6g}",
                    "gray_components": row["gray_components"],
                }
            )
            count += 1
    return count


def write_markdown(path: Path, rows) -> int:
    lines = [
        "# z-C3 XXZ U=8 Kramers-State/Shell Ranges",
        "",
        "Values are in meV and match the shell-combined gray figures.",
        "`J` and `K` are taken from `jk_source_bond`; `gray` spans all non-XXZ gray components from all bonds in the shell.",
        "",
        "Definitions:",
        "- `J = (Jxx + Jyy) / 2`",
        "- `K = Jzz - J`",
        "- `gray = Gxy,Gxz,Gyz,Dx,Dy,Dz`",
        "",
        "The companion TSV with exact extrema locations is:",
        "`outputs/analysis/c3z_xxz_u8_kramer_shell_ranges.tsv`",
        "",
    ]

    for title, shell in [("REChX NN", "NN"), ("REChX NNN", "NNN")]:
        lines.append(f"## {title}")
        lines.append("| family | material | kramer_state | source | J | K | gray | gray_absmax |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for row in rows:
            if row["shell"] != shell:
                continue
            j_stat: RangeStat = row["J"]
            k_stat: RangeStat = row["K"]
            gray_stat: RangeStat = row["gray"]
            lines.append(
                f"| {row['family']} | {row['material']} | `{row['kramer_state']}` | {row['jk_source_bond']} | "
                f"{j_stat.cell()} | {k_stat.cell()} | {gray_stat.cell()} | {row['gray_absmax']:.4g} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    rows, files_seen = collect(args.root)
    if not rows:
        print(f"error: no U_08 files found under {args.root}", file=sys.stderr)
        return 1
    tsv_rows = write_tsv(args.tsv, rows)
    markdown_rows = write_markdown(args.markdown, rows)
    print(f"# files_seen {files_seen}")
    print(f"# rows {tsv_rows}")
    print(f"# markdown_rows {markdown_rows}")
    print(f"# wrote {args.tsv}")
    print(f"# wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
