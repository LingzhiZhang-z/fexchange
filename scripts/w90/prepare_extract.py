#!/usr/bin/env python3
"""Prepare w90_extract TOML from one bond_info.txt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RARE_EARTH = {
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}


def parse_cell(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",")]


def toml_array(xs: list[int]) -> str:
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def toml_string(path: str | Path) -> str:
    text = str(path)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = ROOT / path
    if candidate.exists():
        return candidate.resolve()
    return path


def read_bond_info(path: Path) -> tuple[dict[str, str], list[tuple[int, list[int]]]]:
    fields: dict[str, str] = {}
    bridges: list[tuple[int, list[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("        bridge "):
            match = re.search(r"#(\d+)\s+R=\(([^)]*)\)", line)
            if match:
                bridges.append((int(match.group(1)), parse_cell(match.group(2))))
            continue
        key, _, value = line.partition(" ")
        fields[key] = value.strip()

    missing = sorted({"source", "reference_re", "target_re_atom", "target_re_cell"} - fields.keys())
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if len(bridges) != 2:
        raise ValueError(f"{path}: expected 2 bridge ligands, got {len(bridges)}")
    return fields, bridges


def wout_from_enumerate(path: Path) -> Path:
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.endswith(".wout"):
            return Path(text)
    raise ValueError(f"{path}: no .wout path found")


def atom_symbols_from_wout(path: Path) -> list[str]:
    atoms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.replace("|", " ").split()
        if len(parts) == 8 and parts[0].isalpha() and parts[1].isdigit():
            atoms.append(parts[0])
    if not atoms:
        raise ValueError(f"{path}: no atom table found")
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bond_info", type=Path)
    parser.add_argument("--f-trace-tol", type=float, default=1.0e-3)
    args = parser.parse_args()

    bond_info = args.bond_info.resolve(strict=True)
    fields, bridges = read_bond_info(bond_info)

    enumerate_path = resolve_input_path(Path(fields["source"]))
    wout = resolve_input_path(wout_from_enumerate(enumerate_path))
    hr_files = sorted(wout.parent.glob("*_hr.dat"))
    if len(hr_files) != 1:
        raise ValueError(f"{wout.parent}: expected one *_hr.dat, got {len(hr_files)}")

    ref = re.search(r"RE #(\d+)", fields["reference_re"])
    if not ref:
        raise ValueError(f"{bond_info}: cannot parse reference_re")

    atoms = atom_symbols_from_wout(wout)
    n_orb = [7 if atom in RARE_EARTH else 3 for atom in atoms]

    out = bond_info.parent / "w90"
    out.mkdir(parents=True, exist_ok=True)

    extract_toml = out / "extract.toml"
    onsite = out / "onsite.txt"
    hopping_fp = out / "hopping_fp.txt"
    hopping_ff_direct = out / "hopping_ff_direct.txt"

    lines = [
        f"hr_path = {toml_string(hr_files[0].resolve())}",
        "spinor = true",
        'energy_unit = "eV"',
        f"f_trace_tol = {float(args.f_trace_tol):.16e}",
        f"n_orb_per_atom = {toml_array(n_orb)}",
        f"onsite_out = {toml_string(onsite.resolve())}",
        f"hopping_fp_out = {toml_string(hopping_fp.resolve())}",
        f"hopping_ff_out = {toml_string(hopping_ff_direct.resolve())}",
        "",
        "[f1_site]",
        f"atom = {int(ref.group(1))}",
        "cell = [0, 0, 0]",
        "",
        "[f2_site]",
        f"atom = {int(fields['target_re_atom'])}",
        f"cell = {toml_array(parse_cell(fields['target_re_cell']))}",
        "",
        "[lig1_site]",
        f"atom = {bridges[0][0]}",
        f"cell = {toml_array(bridges[0][1])}",
        "",
        "[lig2_site]",
        f"atom = {bridges[1][0]}",
        f"cell = {toml_array(bridges[1][1])}",
        "",
    ]
    extract_toml.write_text("\n".join(lines), encoding="utf-8")
    print(extract_toml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
