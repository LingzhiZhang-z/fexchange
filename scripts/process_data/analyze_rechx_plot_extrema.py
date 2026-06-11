#!/usr/bin/env python3
"""Report extrema for the REChX cubic jkgamma plots and z-C3 XXZ gray plots.

The cubic table uses the same REChX_cubic tree, J/K/Gamma decomposition, and
U/8 scaling as plot_jkgamma.py.

The gray table uses the unrotated REChX tree and reports only the z-axis XXZ
parameters from plot_c3z_xxz.py:

    J = (Jxx + Jyy) / 2
    K = Jzz - J

Both outputs report scaled meV values, matching the plotted y values. Raw meV
values are included in separate columns.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_bond_exchange import decompose as decompose_jkgamma  # noqa: E402
from plot_jkgamma import BRANCHES, RECHX_FAMS, case_cfg, scale_for_u  # noqa: E402

DEFAULT_ORGANIZED_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")
DEFAULT_OUT_DIR = Path("outputs/analysis")
DEFAULT_CUBIC_OUT = DEFAULT_OUT_DIR / "rechx_cubic_jkgamma_extrema.tsv"
DEFAULT_GRAY_OUT = DEFAULT_OUT_DIR / "rechx_c3z_gray_jk_extrema.tsv"

CUBIC_DISPLAY = {
    "Gamma_prime": "Gammaprime",
    "D_c": "D",
    "D_a": "Dprime",
}

GRAY_EXCHANGES = ("J", "K")


@dataclass
class Extremum:
    value: float
    raw_value: float
    ratio: float


@dataclass
class SeriesStats:
    minimum: Extremum | None = None
    maximum: Extremum | None = None
    points: list[tuple[float, float]] = field(default_factory=list)

    def update(self, ratio: float, raw_value: float, scale: float) -> None:
        value = raw_value * scale
        item = Extremum(value=value, raw_value=raw_value, ratio=ratio)
        if self.minimum is None or value < self.minimum.value:
            self.minimum = item
        if self.maximum is None or value > self.maximum.value:
            self.maximum = item
        self.points.append((ratio, value))

    def trend(self) -> str:
        if len(self.points) < 2:
            return "single_point"
        ordered = sorted(self.points)
        values = [value for _ratio, value in ordered]
        scale = max(1.0, max(abs(value) for value in values))
        tol = 1.0e-8 * scale
        diffs = [b - a for a, b in zip(values, values[1:])]
        if all(abs(diff) <= tol for diff in diffs):
            return "flat"
        if all(diff >= -tol for diff in diffs):
            return "increasing"
        if all(diff <= tol for diff in diffs):
            return "decreasing"
        return "nonmonotonic"


def endpoint_label(ratio: float) -> str:
    if abs(ratio) < 5.0e-10:
        return "jh0"
    if abs(ratio - 0.4) < 5.0e-10:
        return "jh40"
    return "interior"


def read_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        tensor = np.array([float(x) for x in cols[3:12]], dtype=float).reshape(3, 3)
        yield float(cols[0]), tensor


def iter_cubic_files(root: Path, families: set[str], branches: set[str]):
    for family in sorted(families):
        family_root = root / family
        if not family_root.exists():
            continue
        for material_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            for bond_root in sorted(path for path in material_root.iterdir() if path.is_dir()):
                cfg = case_cfg(family, bond_root.name)
                if cfg is None:
                    continue
                for branch in sorted(branches):
                    branch_root = bond_root / branch
                    if not branch_root.exists():
                        continue
                    for path in sorted(branch_root.glob("*/U_*.txt")):
                        try:
                            u_value = int(path.stem.split("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        yield family, material_root.name, bond_root.name, branch, path.parts[-2], u_value, path


def iter_gray_files(root: Path, families: set[str], branches: set[str]):
    for family in sorted(families):
        family_root = root / family
        if not family_root.exists():
            continue
        for material_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            for bond_root in sorted(path for path in material_root.iterdir() if path.is_dir()):
                if not (bond_root.name.startswith("J1-") or bond_root.name.startswith("J2-")):
                    continue
                for branch in sorted(branches):
                    branch_root = bond_root / branch
                    if not branch_root.exists():
                        continue
                    for path in sorted(branch_root.glob("*/U_*.txt")):
                        try:
                            u_value = int(path.stem.split("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        yield family, material_root.name, bond_root.name, branch, path.parts[-2], u_value, path


def gray_values(tensor: np.ndarray) -> dict[str, float]:
    j_value = 0.5 * (tensor[0, 0] + tensor[1, 1])
    return {
        "J": float(j_value),
        "K": float(tensor[2, 2] - j_value),
    }


def collect_cubic(root: Path, families: set[str], branches: set[str]):
    stats: dict[tuple[str, str, str, str, str, int, str, str], SeriesStats] = defaultdict(SeriesStats)
    files_seen = 0
    for family, material, bond, branch, run, u_value, path in iter_cubic_files(root, families, branches):
        cfg = case_cfg(family, bond)
        if cfg is None:
            continue
        rotation, params, axis, _tag, _group = cfg
        scale = scale_for_u(u_value)
        files_seen += 1
        for ratio, tensor in read_rows(path):
            values = decompose_jkgamma(rotation @ tensor @ rotation.T, axis)
            for source_key, _colour in params:
                exchange = CUBIC_DISPLAY.get(source_key, source_key)
                raw_mev = values[source_key] * 1000.0
                stats[(family, material, bond, branch, run, u_value, exchange, source_key)].update(
                    ratio=ratio,
                    raw_value=raw_mev,
                    scale=scale,
                )
    return stats, files_seen


def collect_gray(root: Path, families: set[str], branches: set[str]):
    stats: dict[tuple[str, str, str, str, str, int, str, str], SeriesStats] = defaultdict(SeriesStats)
    files_seen = 0
    for family, material, bond, branch, run, u_value, path in iter_gray_files(root, families, branches):
        scale = scale_for_u(u_value)
        files_seen += 1
        for ratio, tensor in read_rows(path):
            values = gray_values(tensor)
            for exchange in GRAY_EXCHANGES:
                raw_mev = values[exchange] * 1000.0
                stats[(family, material, bond, branch, run, u_value, exchange, exchange)].update(
                    ratio=ratio,
                    raw_value=raw_mev,
                    scale=scale,
                )
    return stats, files_seen


def rows_from_stats(dataset: str, stats):
    for (family, material, bond, branch, run, u_value, exchange, source_key), stat in sorted(stats.items()):
        if stat.minimum is None or stat.maximum is None:
            continue
        yield {
            "dataset": dataset,
            "family": family,
            "material": material,
            "bond": bond,
            "branch": branch,
            "run": run,
            "U": u_value,
            "scale": f"{scale_for_u(u_value):.8g}",
            "exchange": exchange,
            "source_exchange": source_key,
            "min_mev": f"{stat.minimum.value:.10g}",
            "min_at_ratio": f"{stat.minimum.ratio:.10g}",
            "min_endpoint": endpoint_label(stat.minimum.ratio),
            "max_mev": f"{stat.maximum.value:.10g}",
            "max_at_ratio": f"{stat.maximum.ratio:.10g}",
            "max_endpoint": endpoint_label(stat.maximum.ratio),
            "range_mev": f"{stat.maximum.value - stat.minimum.value:.10g}",
            "raw_min_mev": f"{stat.minimum.raw_value:.10g}",
            "raw_max_mev": f"{stat.maximum.raw_value:.10g}",
            "trend": stat.trend(),
        }


def write_tsv(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "family",
        "material",
        "bond",
        "branch",
        "run",
        "U",
        "scale",
        "exchange",
        "source_exchange",
        "min_mev",
        "min_at_ratio",
        "min_endpoint",
        "max_mev",
        "max_at_ratio",
        "max_endpoint",
        "range_mev",
        "raw_min_mev",
        "raw_max_mev",
        "trend",
    ]
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organized-root", type=Path, default=DEFAULT_ORGANIZED_ROOT)
    parser.add_argument("--cubic-root", type=Path, help="default: ORGANIZED_ROOT/REChX_cubic")
    parser.add_argument("--cubic-output", type=Path, default=DEFAULT_CUBIC_OUT)
    parser.add_argument("--gray-output", type=Path, default=DEFAULT_GRAY_OUT)
    parser.add_argument("--family", action="append", choices=RECHX_FAMS)
    parser.add_argument("--branch", action="append", choices=BRANCHES)
    parser.add_argument("--skip-cubic", action="store_true")
    parser.add_argument("--skip-gray", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    families = set(args.family or RECHX_FAMS)
    branches = set(args.branch or BRANCHES)
    cubic_root = args.cubic_root or (args.organized_root / "REChX_cubic")

    if args.skip_cubic and args.skip_gray:
        print("nothing to do: both --skip-cubic and --skip-gray were set", file=sys.stderr)
        return 2

    if not args.skip_cubic:
        cubic_stats, cubic_files = collect_cubic(cubic_root, families, branches)
        if cubic_files == 0:
            print(f"error: no cubic U_*.txt files under {cubic_root}", file=sys.stderr)
            return 1
        cubic_rows = write_tsv(args.cubic_output, rows_from_stats("cubic_jkgamma", cubic_stats))
        print(f"# cubic files: {cubic_files}")
        print(f"# wrote {cubic_rows} rows to {args.cubic_output}")

    if not args.skip_gray:
        gray_stats, gray_files = collect_gray(args.organized_root, families, branches)
        if gray_files == 0:
            print(f"error: no gray U_*.txt files under {args.organized_root}", file=sys.stderr)
            return 1
        gray_rows = write_tsv(args.gray_output, rows_from_stats("c3z_gray_jk", gray_stats))
        print(f"# gray files: {gray_files}")
        print(f"# wrote {gray_rows} rows to {args.gray_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
