#!/usr/bin/env python3
"""Create per-bond folders from each material's enumerate_nn_bonds.txt.

Only the first RE block is used. This matches the current workflow convention:
the first RE site is the reference site for each material.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BOND_RE = re.compile(
    r"^\s*(NNN|NN)\s+\[([^\]]+)\]\s+-> RE #(\d+)\s+R=\(([^)]*)\)\s+([0-9.eE+-]+) A"
)


def material_dirs(input_root: Path) -> list[Path]:
    return sorted(
        p
        for p in input_root.glob("*/*")
        if p.is_dir() and p.parent.name != "kramer"
    )


def first_re_block(lines: list[str]) -> tuple[str, list[str]]:
    start = next((i for i, line in enumerate(lines) if line.startswith("RE #")), None)
    if start is None:
        raise ValueError("no RE block found")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("RE #") or lines[i].startswith("-- ligand"):
            end = i
            break
    return lines[start], lines[start + 1:end]


def bond_entries(block: list[str]) -> list[tuple[dict[str, str], list[str]]]:
    entries: list[tuple[dict[str, str], list[str]]] = []
    i = 0
    while i < len(block):
        line = block[i]
        match = BOND_RE.match(line)
        if not match:
            i += 1
            continue

        raw = [line]
        j = i + 1
        while j < len(block) and block[j].startswith("        bridge "):
            raw.append(block[j])
            j += 1

        shell, label, target_re, cell, length = match.groups()
        entries.append(
            (
                {
                    "shell": shell,
                    "bond_label": label,
                    "target_re_atom": target_re,
                    "target_re_cell": cell.replace(" ", ""),
                    "bond_length_A": length,
                },
                raw,
            )
        )
        i = j
    return entries


def write_bond_info(path: Path, header: dict[str, str], raw_lines: list[str]) -> None:
    lines = [f"{key} {value}" for key, value in header.items()]
    lines.extend(["", "# raw scripts/bond/enumerate.py excerpt"])
    lines.extend(raw_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/data-DFT-input"))
    parser.add_argument("--enum-name", default="enumerate_nn_bonds.txt")
    parser.add_argument("--info-name", default="bond_info.txt")
    args = parser.parse_args()

    materials = material_dirs(args.input_root)
    wrote = 0
    missing = 0
    failed = 0

    for mat_dir in materials:
        enum_path = mat_dir / args.enum_name
        if not enum_path.exists():
            print(f"missing_enumerate\t{mat_dir}")
            missing += 1
            continue

        try:
            ref_line, block = first_re_block(enum_path.read_text(encoding="utf-8").splitlines())
            entries = bond_entries(block)
            if not entries:
                raise ValueError("no bonds found in first RE block")
        except Exception as exc:
            print(f"failed\t{mat_dir}\t{exc}")
            failed += 1
            continue

        category = mat_dir.parent.name
        material = mat_dir.name
        for entry, raw in entries:
            bond_dir = mat_dir / entry["bond_label"]
            bond_dir.mkdir(parents=True, exist_ok=True)
            header = {
                "category": category,
                "material": material,
                "source": str(enum_path),
                "reference_re": ref_line,
                **entry,
            }
            write_bond_info(bond_dir / args.info_name, header, raw)
            wrote += 1

    print(f"materials\t{len(materials)}")
    print(f"bond_dirs_written\t{wrote}")
    print(f"missing_enumerate\t{missing}")
    print(f"failed\t{failed}")
    return 1 if missing or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
