#!/usr/bin/env python3
"""Extract J/K/Gamma/D channels from spin-exchange dat curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


OUT_COLUMNS = [
    "Jh_over_U",
    "Jh_meV",
    "J",
    "K",
    "Gamma1",
    "Gamma2",
    "Gamma3",
    "D1",
    "D2",
    "D3",
    "delta_xy",
    "mapping_residual",
]

INDEX_COLUMNS = [
    "input_file",
    "output_file",
    "n_points",
    "family",
    "material",
    "case",
    "U_eV",
]


def parse_dat(path: Path) -> tuple[dict[str, str], list[str], list[dict[str, float]]]:
    meta: dict[str, str] = {}
    columns: list[str] = []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                payload = stripped[1:].strip()
                if payload.startswith("columns:"):
                    columns = payload.split(":", 1)[1].strip().split()
                elif ":" in payload:
                    key, value = payload.split(":", 1)
                    meta[key.strip()] = value.strip()
                continue
            if not columns:
                raise ValueError(f"{path}: data row appears before '# columns:'")
            values = [float(x) for x in stripped.split()]
            if len(values) != len(columns):
                raise ValueError(f"{path}: expected {len(columns)} columns, got {len(values)}")
            rows.append(dict(zip(columns, values, strict=True)))
    if not columns:
        raise ValueError(f"{path}: missing '# columns:' header")
    return meta, columns, rows


def get_component(row: dict[str, float], name: str) -> float:
    if name in row:
        return row[name]
    rot_name = f"{name}_rot"
    if rot_name in row:
        return row[rot_name]
    raise KeyError(name)


def channels(row: dict[str, float]) -> dict[str, float]:
    jxx = get_component(row, "Jxx")
    jxy = get_component(row, "Jxy")
    jxz = get_component(row, "Jxz")
    jyx = get_component(row, "Jyx")
    jyy = get_component(row, "Jyy")
    jyz = get_component(row, "Jyz")
    jzx = get_component(row, "Jzx")
    jzy = get_component(row, "Jzy")
    jzz = get_component(row, "Jzz")

    j = 0.5 * (jxx + jyy)
    return {
        "Jh_over_U": row["Jh_over_U"],
        "Jh_meV": row["Jh_meV"],
        "J": j,
        "K": jzz - j,
        "Gamma1": 0.5 * (jxy + jyx),
        "Gamma2": 0.5 * (jxz + jzx),
        "Gamma3": 0.5 * (jyz + jzy),
        "D1": 0.5 * (jyz - jzy),
        "D2": 0.5 * (jzx - jxz),
        "D3": 0.5 * (jxy - jyx),
        "delta_xy": 0.5 * (jxx - jyy),
        "mapping_residual": row.get("mapping_residual", float("nan")),
    }


def fmt(value: Any) -> str:
    try:
        return f"{float(value): .12e}"
    except (TypeError, ValueError):
        return str(value)


def write_curve(out_path: Path, in_path: Path, meta: dict[str, str], rows: list[dict[str, float]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# schema_version: fxe.jk_gamma_d_curve.v1\n")
        f.write(f"# source_file: {in_path}\n")
        for key in ("family", "material", "RE", "n_ele", "case", "irrep", "U_eV", "units"):
            if key in meta:
                f.write(f"# {key}: {meta[key]}\n")
        f.write("# convention: z-bond, c-axis=111\n")
        f.write("# J = (Jxx + Jyy) / 2\n")
        f.write("# K = Jzz - J\n")
        f.write("# Gamma1 = (Jxy + Jyx) / 2\n")
        f.write("# Gamma2 = (Jxz + Jzx) / 2\n")
        f.write("# Gamma3 = (Jyz + Jzy) / 2\n")
        f.write("# D1 = Dx = (Jyz - Jzy) / 2\n")
        f.write("# D2 = Dy = (Jzx - Jxz) / 2\n")
        f.write("# D3 = Dz = (Jxy - Jyx) / 2\n")
        f.write("# delta_xy = (Jxx - Jyy) / 2; diagnostic for non-ideal z-bond form\n")
        f.write("# columns: " + " ".join(OUT_COLUMNS) + "\n")
        for row in rows:
            f.write(" ".join(fmt(row[col]) for col in OUT_COLUMNS) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data/REChX_REX3/results/spin_exchange_dat_111")
    parser.add_argument("--output-root", default="data/REChX_REX3/results/jk_gamma_d_111")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    index_rows: list[dict[str, Any]] = []

    for in_path in sorted(input_root.rglob("U*.dat")):
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        meta, _, in_rows = parse_dat(in_path)
        out_rows = [channels(row) for row in in_rows]
        write_curve(out_path, in_path, meta, out_rows)
        index_rows.append(
            {
                "input_file": str(in_path),
                "output_file": str(out_path),
                "n_points": len(out_rows),
                "family": meta.get("family", ""),
                "material": meta.get("material", ""),
                "case": meta.get("case", ""),
                "U_eV": meta.get("U_eV", ""),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_rows)
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# JK Gamma D Curves",
                "",
                "Generated from `spin_exchange_dat_111`.",
                "",
                "Convention for each 3x3 exchange matrix:",
                "- `J = (Jxx + Jyy) / 2`",
                "- `K = Jzz - J`",
                "- `Gamma1 = (Jxy + Jyx) / 2`",
                "- `Gamma2 = (Jxz + Jzx) / 2`",
                "- `Gamma3 = (Jyz + Jzy) / 2`",
                "- `D1 = Dx = (Jyz - Jzy) / 2`",
                "- `D2 = Dy = (Jzx - Jxz) / 2`",
                "- `D3 = Dz = (Jxy - Jyx) / 2`",
                "- `delta_xy = (Jxx - Jyy) / 2` is a diagnostic, not part of the requested 8-channel model.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"processed={len(index_rows)} output_root={output_root}")


if __name__ == "__main__":
    main()
