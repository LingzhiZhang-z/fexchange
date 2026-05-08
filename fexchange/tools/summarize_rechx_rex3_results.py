#!/usr/bin/env python3
"""Summarize REChX/REX3 exchange point results.

The script treats input manifests as the source of expected U/Jh grid points
and point-result txt files as completed calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT6 = math.sqrt(6.0)

# Active rotation: old 010 -> target 1-10, old 001 -> target 111.
# Columns are the rotated images of old x=100, y=010, z=001 in the same
# Cartesian coordinate convention.
ROT_010_TO_1M10_001_TO_111 = [
    [-1.0 / SQRT6, 1.0 / SQRT2, 1.0 / SQRT3],
    [-1.0 / SQRT6, -1.0 / SQRT2, 1.0 / SQRT3],
    [2.0 / SQRT6, 0.0, 1.0 / SQRT3],
]

RESULT_COLUMNS = [
    "schema_version",
    "family",
    "material",
    "RE",
    "n_ele",
    "case",
    "irrep",
    "status",
    "run_id",
    "U_meV",
    "U_eV",
    "Jh_meV",
    "Jh_over_U",
    "zeta_meV",
    "energy_unit",
    "energy_reference",
    "Jxx",
    "Jxy",
    "Jxz",
    "Jyx",
    "Jyy",
    "Jyz",
    "Jzx",
    "Jzy",
    "Jzz",
    "mapping_residual",
    "Jpm",
    "Jpmpm_y",
    "Jzpm_y_plus",
    "Jzpm_y_minus",
    "Jxy_sym",
    "Jyz_sym",
    "Jxz_asym",
    "dm_x",
    "dm_y",
    "dm_z",
    "input_file",
    "output_file",
    "output_root",
    "parse_error_message",
]

SUMMARY_COLUMNS = [
    "family",
    "material",
    "RE",
    "n_ele",
    "case",
    "irrep",
    "expected",
    "complete",
    "missing",
    "parse_error",
    "duplicate",
    "completion_fraction",
]

DAT_COLUMNS = [
    "Jh_over_U",
    "Jh_meV",
    "Jxx",
    "Jxy",
    "Jxz",
    "Jyx",
    "Jyy",
    "Jyz",
    "Jzx",
    "Jzy",
    "Jzz",
    "mapping_residual",
    "Jpm",
    "Jpmpm_y",
    "Jzpm_y_plus",
    "Jzpm_y_minus",
    "Jxy_sym",
    "Jyz_sym",
    "Jxz_asym",
    "dm_x",
    "dm_y",
    "dm_z",
]

ROTATED_DAT_COLUMNS = [
    "Jh_over_U",
    "Jh_meV",
    "Jxx_rot",
    "Jxy_rot",
    "Jxz_rot",
    "Jyx_rot",
    "Jyy_rot",
    "Jyz_rot",
    "Jzx_rot",
    "Jzy_rot",
    "Jzz_rot",
    "mapping_residual",
]

ROTATED_CSV_COLUMNS = [
    "schema_version",
    "family",
    "material",
    "RE",
    "n_ele",
    "case",
    "run_id",
    "U_meV",
    "U_eV",
    "Jh_meV",
    "Jh_over_U",
    "zeta_meV",
    "Jxx_rot",
    "Jxy_rot",
    "Jxz_rot",
    "Jyx_rot",
    "Jyy_rot",
    "Jyz_rot",
    "Jzx_rot",
    "Jzy_rot",
    "Jzz_rot",
    "mapping_residual",
    "source_output_file",
]


@dataclass(frozen=True)
class OutputPoint:
    path: Path
    meta_path: Path | None
    run_id: str
    values: dict[str, float]
    parse_error: str = ""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def parse_point_txt(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) != 14:
        raise ValueError(f"expected 14 floats, got {len(parts)}")
    vals = [float(x) for x in parts]
    keys = [
        "U_meV",
        "Jh_meV",
        "Jh_over_U",
        "zeta_meV",
        "Jxx",
        "Jxy",
        "Jxz",
        "Jyx",
        "Jyy",
        "Jyz",
        "Jzx",
        "Jzy",
        "Jzz",
        "mapping_residual",
    ]
    return dict(zip(keys, vals, strict=True))


def txt_from_meta_path(meta_path: Path) -> Path:
    suffix = ".meta.json"
    s = str(meta_path)
    if not s.endswith(suffix):
        raise ValueError(f"not a point-result meta path: {meta_path}")
    return Path(s[: -len(suffix)] + ".txt")


def scan_output_root(output_root: Path) -> tuple[dict[str, list[OutputPoint]], list[OutputPoint]]:
    by_run_id: dict[str, list[OutputPoint]] = defaultdict(list)
    all_points: list[OutputPoint] = []
    if not output_root.exists():
        return by_run_id, all_points

    seen_txt: set[Path] = set()
    for meta_path in sorted(output_root.glob("*.meta.json")):
        txt_path = txt_from_meta_path(meta_path)
        seen_txt.add(txt_path)
        run_id = ""
        parse_error = ""
        try:
            meta = load_json(meta_path)
            run_id = str(meta.get("run_id", ""))
            values = parse_point_txt(txt_path)
        except Exception as exc:  # noqa: BLE001 - keep reporting robust.
            values = {}
            parse_error = str(exc)
        point = OutputPoint(txt_path, meta_path, run_id, values, parse_error)
        all_points.append(point)
        if run_id:
            by_run_id[run_id].append(point)

    for txt_path in sorted(output_root.glob("*.txt")):
        if txt_path in seen_txt:
            continue
        try:
            values = parse_point_txt(txt_path)
            parse_error = ""
        except Exception as exc:  # noqa: BLE001
            values = {}
            parse_error = str(exc)
        all_points.append(OutputPoint(txt_path, None, "", values, parse_error))
    return by_run_id, all_points


def close(a: float, b: float, *, atol: float = 1e-6) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def fallback_match(points: list[OutputPoint], *, U: float, Jh: float, zeta: float) -> list[OutputPoint]:
    out: list[OutputPoint] = []
    for point in points:
        vals = point.values
        if not vals:
            continue
        if close(vals["U_meV"], U) and close(vals["Jh_meV"], Jh) and close(vals["zeta_meV"], zeta):
            out.append(point)
    return out


def derived_ybond(vals: dict[str, float]) -> dict[str, float]:
    jxx = vals["Jxx"]
    jxy = vals["Jxy"]
    jxz = vals["Jxz"]
    jyx = vals["Jyx"]
    jyy = vals["Jyy"]
    jyz = vals["Jyz"]
    jzx = vals["Jzx"]
    jzy = vals["Jzy"]
    return {
        "Jpm": 0.25 * (jxx + jyy),
        "Jpmpm_y": 0.25 * (jyy - jxx),
        "Jzpm_y_plus": -0.5 * (jxz + jzx),
        "Jzpm_y_minus": 0.5 * (jxz + jzx),
        "Jxy_sym": 0.5 * (jxy + jyx),
        "Jyz_sym": 0.5 * (jyz + jzy),
        "Jxz_asym": 0.5 * (jxz - jzx),
        "dm_x": 0.5 * (jyz - jzy),
        "dm_y": 0.5 * (jzx - jxz),
        "dm_z": 0.5 * (jxy - jyx),
    }


def exchange_matrix_from_row(row: dict[str, Any]) -> list[list[float]]:
    return [
        [float(row["Jxx"]), float(row["Jxy"]), float(row["Jxz"])],
        [float(row["Jyx"]), float(row["Jyy"]), float(row["Jyz"])],
        [float(row["Jzx"]), float(row["Jzy"]), float(row["Jzz"])],
    ]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def rotate_exchange_active(j: list[list[float]]) -> list[list[float]]:
    r = ROT_010_TO_1M10_001_TO_111
    return matmul(matmul(r, j), transpose(r))


def rotated_rechx_row(row: dict[str, Any]) -> dict[str, Any]:
    j = rotate_exchange_active(exchange_matrix_from_row(row))
    return {
        "schema_version": "fxe.rechx_rotated_exchange.v1",
        "family": row["family"],
        "material": row["material"],
        "RE": row["RE"],
        "n_ele": row["n_ele"],
        "case": row["case"],
        "run_id": row["run_id"],
        "U_meV": row["U_meV"],
        "U_eV": row["U_eV"],
        "Jh_meV": row["Jh_meV"],
        "Jh_over_U": row["Jh_over_U"],
        "zeta_meV": row["zeta_meV"],
        "Jxx_rot": j[0][0],
        "Jxy_rot": j[0][1],
        "Jxz_rot": j[0][2],
        "Jyx_rot": j[1][0],
        "Jyy_rot": j[1][1],
        "Jyz_rot": j[1][2],
        "Jzx_rot": j[2][0],
        "Jzy_rot": j[2][1],
        "Jzz_rot": j[2][2],
        "mapping_residual": row["mapping_residual"],
        "source_output_file": row["output_file"],
    }


def empty_result_fields() -> dict[str, str]:
    return {name: "" for name in RESULT_COLUMNS}


def build_rows(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_cache: dict[Path, tuple[dict[str, list[OutputPoint]], list[OutputPoint]]] = {}
    manifest_paths = sorted(data_root.rglob("input_manifest.json"))

    for manifest_path in manifest_paths:
        manifest = load_json(manifest_path)
        family = str(manifest.get("family", ""))
        material = str(manifest.get("material", ""))
        re_label = str(manifest.get("RE", ""))
        n_ele = manifest.get("n_ele", "")

        for case in manifest.get("cases", []):
            case_name = str(case.get("case", ""))
            irrep = str(case.get("irrep", ""))
            for input_name in case.get("run_inputs", []):
                input_path = Path(input_name)
                cfg = load_toml(input_path)
                sopt = cfg["sopt"]
                paths = cfg["paths"]
                units = cfg.get("units", {})
                output_root = Path(paths["output_root"])
                if output_root not in output_cache:
                    output_cache[output_root] = scan_output_root(output_root)
                by_run_id, all_points = output_cache[output_root]

                run_id = str(cfg.get("run_id", ""))
                U = float(sopt["U"])
                Jh = float(sopt["Jh"])
                zeta = float(sopt["zeta"])
                matches = by_run_id.get(run_id, [])
                if not matches:
                    matches = fallback_match(all_points, U=U, Jh=Jh, zeta=zeta)

                row = empty_result_fields()
                row.update(
                    {
                        "schema_version": "fxe.exchange_results.v1",
                        "family": family,
                        "material": material,
                        "RE": re_label,
                        "n_ele": n_ele,
                        "case": case_name,
                        "irrep": irrep,
                        "run_id": run_id,
                        "U_meV": U,
                        "U_eV": U / 1000.0,
                        "Jh_meV": Jh,
                        "Jh_over_U": float("nan") if U == 0.0 else Jh / U,
                        "zeta_meV": zeta,
                        "energy_unit": str(units.get("energy", "meV")),
                        "energy_reference": str(sopt.get("energy_reference", "")),
                        "input_file": str(input_path),
                        "output_root": str(output_root),
                    }
                )

                if not matches:
                    row["status"] = "missing"
                else:
                    point = matches[0]
                    row["status"] = "complete"
                    if len(matches) > 1:
                        row["status"] = "duplicate"
                    if point.parse_error:
                        row["status"] = "parse_error"
                        row["parse_error_message"] = point.parse_error
                    row["output_file"] = str(point.path)
                    if point.values:
                        row.update(point.values)
                        row.update(derived_ybond(point.values))
                rows.append(row)

    meta = {
        "schema_version": "fxe.exchange_results_manifest.v1",
        "data_root": str(data_root),
        "manifest_count": len(manifest_paths),
        "formulas": {
            "ybond_convention": "For a +y reference bond in the C3v/three-fold convention.",
            "Jpm": "(Jxx + Jyy) / 4",
            "Jpmpm_y": "(Jyy - Jxx) / 4",
            "Jzpm_y_plus": "-(Jxz + Jzx) / 2",
            "Jzpm_y_minus": "+(Jxz + Jzx) / 2",
            "diagnostics": {
                "Jxy_sym": "(Jxy + Jyx) / 2; ideal y-bond C3v value is 0",
                "Jyz_sym": "(Jyz + Jzy) / 2; ideal y-bond C3v value is 0",
                "Jxz_asym": "(Jxz - Jzx) / 2; ideal y-bond C3v value is 0",
                "dm": "antisymmetric exchange components from (J - J^T) / 2",
            },
        },
    }
    return rows, meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["family"],
            row["material"],
            row["RE"],
            row["n_ele"],
            row["case"],
            row["irrep"],
        )
        if key not in grouped:
            grouped[key] = {
                "family": row["family"],
                "material": row["material"],
                "RE": row["RE"],
                "n_ele": row["n_ele"],
                "case": row["case"],
                "irrep": row["irrep"],
                "expected": 0,
                "complete": 0,
                "missing": 0,
                "parse_error": 0,
                "duplicate": 0,
            }
        item = grouped[key]
        item["expected"] += 1
        status = str(row["status"])
        if status in item:
            item[status] += 1
    out = list(grouped.values())
    for item in out:
        expected = int(item["expected"])
        item["completion_fraction"] = 0.0 if expected == 0 else float(item["complete"]) / expected
    return sorted(out, key=lambda x: (str(x["family"]), str(x["material"]), str(x["case"])))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# REChX_REX3 Exchange Results",
                "",
                "Generated by `python -m fexchange.tools.summarize_rechx_rex3_results`.",
                "",
                "Files:",
                "- `exchange_results.csv`: all expected L4 grid points from input manifests, with status.",
                "- `exchange_results_completed.csv`: completed point results only.",
                "- `exchange_3fold_ybond_rechx.csv`: completed REChX points with y-bond three-fold parameters.",
                "- `spin_exchange_dat/`: one `.dat` curve file per material/case/U.",
                "- `spin_exchange_dat_rechx_010_to_1m10/`: REChX-only curves after active tensor rotation.",
                "- `exchange_results_rechx_010_to_1m10.csv`: REChX-only rotated exchange tensors.",
                "- `missing_results.csv`: expected points without a matched point-result file.",
                "- `completion_summary.csv`: counts grouped by material and case.",
                "- `results_manifest.json`: schema and formula metadata.",
                "",
                "REChX rotation:",
                "- Active tensor rotation `J_rot = R J R^T`.",
                "- `R[010] = [1,-1,0]/sqrt(2)`.",
                "- `R[001] = [1,1,1]/sqrt(3)`.",
                "",
                "For the +y reference-bond convention:",
                "- `Jpm = (Jxx + Jyy) / 4`",
                "- `Jpmpm_y = (Jyy - Jxx) / 4`",
                "- `Jzpm_y_plus = -(Jxz + Jzx) / 2`",
                "- `Jzpm_y_minus = +(Jxz + Jzx) / 2`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def safe_path_token(value: Any) -> str:
    token = str(value).strip()
    out = []
    for ch in token:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unknown"


def fmt_dat_value(value: Any) -> str:
    if value == "":
        return "nan"
    try:
        return f"{float(value): .12e}"
    except (TypeError, ValueError):
        return str(value)


def write_spin_exchange_dat(out_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dat_root = out_dir / "spin_exchange_dat"
    index_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row["status"] != "complete":
            continue
        key = (
            row["family"],
            row["material"],
            row["RE"],
            row["n_ele"],
            row["case"],
            row["irrep"],
            float(row["U_eV"]),
        )
        grouped[key].append(row)

    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        family, material, re_label, n_ele, case, irrep, U_eV = key
        group = sorted(group, key=lambda row: float(row["Jh_over_U"]))
        rel_path = (
            Path(safe_path_token(family))
            / safe_path_token(material)
            / safe_path_token(case)
            / f"U{int(round(U_eV)):02d}.dat"
        )
        path = dat_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("# schema_version: fxe.spin_exchange_curve.v1\n")
            f.write(f"# family: {family}\n")
            f.write(f"# material: {material}\n")
            f.write(f"# RE: {re_label}\n")
            f.write(f"# n_ele: {n_ele}\n")
            f.write(f"# case: {case}\n")
            if irrep:
                f.write(f"# irrep: {irrep}\n")
            f.write(f"# U_eV: {U_eV:.6f}\n")
            f.write("# units: meV for exchange entries\n")
            f.write("# columns: " + " ".join(DAT_COLUMNS) + "\n")
            for row in group:
                f.write(" ".join(fmt_dat_value(row.get(col, "")) for col in DAT_COLUMNS) + "\n")

        index_rows.append(
            {
                "family": family,
                "material": material,
                "RE": re_label,
                "n_ele": n_ele,
                "case": case,
                "irrep": irrep,
                "U_eV": U_eV,
                "n_points": len(group),
                "dat_file": str(path),
            }
        )

    write_csv(
        dat_root / "index.csv",
        index_rows,
        ["family", "material", "RE", "n_ele", "case", "irrep", "U_eV", "n_points", "dat_file"],
    )
    return index_rows


def write_rechx_rotated_outputs(
    out_dir: Path,
    completed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rotated_rows = [
        rotated_rechx_row(row)
        for row in completed_rows
        if row["family"] == "REChX"
    ]
    write_csv(out_dir / "exchange_results_rechx_010_to_1m10.csv", rotated_rows, ROTATED_CSV_COLUMNS)

    dat_root = out_dir / "spin_exchange_dat_rechx_010_to_1m10"
    index_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rotated_rows:
        key = (
            row["family"],
            row["material"],
            row["RE"],
            row["n_ele"],
            row["case"],
            float(row["U_eV"]),
        )
        grouped[key].append(row)

    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        family, material, re_label, n_ele, case, U_eV = key
        group = sorted(group, key=lambda row: float(row["Jh_over_U"]))
        rel_path = (
            Path(safe_path_token(family))
            / safe_path_token(material)
            / safe_path_token(case)
            / f"U{int(round(U_eV)):02d}.dat"
        )
        path = dat_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("# schema_version: fxe.rechx_rotated_exchange_curve.v1\n")
            f.write("# rotation: active J_rot = R J R^T\n")
            f.write("# R maps old 010 to [1,-1,0]/sqrt(2)\n")
            f.write("# R maps old 001 to [1,1,1]/sqrt(3)\n")
            f.write("# R_rows:\n")
            for rrow in ROT_010_TO_1M10_001_TO_111:
                f.write("#   " + " ".join(f"{x:.15f}" for x in rrow) + "\n")
            f.write(f"# family: {family}\n")
            f.write(f"# material: {material}\n")
            f.write(f"# RE: {re_label}\n")
            f.write(f"# n_ele: {n_ele}\n")
            f.write(f"# case: {case}\n")
            f.write(f"# U_eV: {U_eV:.6f}\n")
            f.write("# units: meV for exchange entries\n")
            f.write("# columns: " + " ".join(ROTATED_DAT_COLUMNS) + "\n")
            for row in group:
                f.write(" ".join(fmt_dat_value(row.get(col, "")) for col in ROTATED_DAT_COLUMNS) + "\n")

        index_rows.append(
            {
                "family": family,
                "material": material,
                "RE": re_label,
                "n_ele": n_ele,
                "case": case,
                "U_eV": U_eV,
                "n_points": len(group),
                "dat_file": str(path),
            }
        )

    write_csv(
        dat_root / "index.csv",
        index_rows,
        ["family", "material", "RE", "n_ele", "case", "U_eV", "n_points", "dat_file"],
    )
    return rotated_rows, index_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/REChX_REX3")
    parser.add_argument("--out-dir", default="data/REChX_REX3/results")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rows, meta = build_rows(data_root)
    summary_rows = summarize(rows)
    completed_rows = [row for row in rows if row["status"] == "complete"]
    missing_rows = [row for row in rows if row["status"] == "missing"]
    rechx_ybond_rows = [
        row
        for row in completed_rows
        if row["family"] == "REChX" and row.get("Jpm", "") != ""
    ]
    dat_index_rows = write_spin_exchange_dat(out_dir, rows)
    rotated_rechx_rows, rotated_rechx_dat_index = write_rechx_rotated_outputs(out_dir, completed_rows)

    meta.update(
        {
            "expected_points": len(rows),
            "complete_points": len(completed_rows),
            "missing_points": len(missing_rows),
            "duplicate_points": sum(1 for row in rows if row["status"] == "duplicate"),
            "parse_error_points": sum(1 for row in rows if row["status"] == "parse_error"),
            "spin_exchange_dat_files": len(dat_index_rows),
            "rotated_rechx_points": len(rotated_rechx_rows),
            "rotated_rechx_dat_files": len(rotated_rechx_dat_index),
            "rechx_rotation": {
                "type": "active_tensor_rotation",
                "formula": "J_rot = R J R^T",
                "old_010_maps_to": [1.0 / SQRT2, -1.0 / SQRT2, 0.0],
                "old_001_maps_to": [1.0 / SQRT3, 1.0 / SQRT3, 1.0 / SQRT3],
                "R_rows": ROT_010_TO_1M10_001_TO_111,
            },
            "output_dir": str(out_dir),
        }
    )

    write_csv(out_dir / "exchange_results.csv", rows, RESULT_COLUMNS)
    write_csv(out_dir / "exchange_results_completed.csv", completed_rows, RESULT_COLUMNS)
    write_csv(out_dir / "exchange_3fold_ybond_rechx.csv", rechx_ybond_rows, RESULT_COLUMNS)
    write_csv(out_dir / "missing_results.csv", missing_rows, RESULT_COLUMNS)
    write_csv(out_dir / "completion_summary.csv", summary_rows, SUMMARY_COLUMNS)
    (out_dir / "results_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_readme(out_dir / "README.md")

    print(
        "expected={expected_points} complete={complete_points} missing={missing_points} "
        "duplicate={duplicate_points} parse_error={parse_error_points} out={output_dir}".format(**meta)
    )


if __name__ == "__main__":
    main()
