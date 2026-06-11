#!/usr/bin/env python3
"""Report plotted exchange min/max values for one material and one bond.

The statistics use the same inputs, decomposition, REChX rotated tree, and U/8
plot scaling as plot_jkgamma.py.  Values are therefore the meV values that appear
in the current J/K/Gamma figures.

Usage:
  python scripts/process_data/analyze_exchange_minmax.py REOF/SmOF/J1-z
  python scripts/process_data/analyze_exchange_minmax.py REX3-R3/YbI3/z --branch sopt --run <run_id>
  python scripts/process_data/analyze_exchange_minmax.py REOF/SmOF/J1-z --aggregate-runs
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

DEFAULT_ORGANIZED_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")


@dataclass
class Extremum:
    value: float
    raw_value: float
    ratio: float
    run: str


@dataclass
class MinMax:
    minimum: Extremum | None = None
    maximum: Extremum | None = None

    def update(self, ratio: float, raw_value: float, scale: float, run: str) -> None:
        value = raw_value * scale
        item = Extremum(value=value, raw_value=raw_value, ratio=ratio, run=run)
        self.absorb(item)

    def absorb(self, item: Extremum) -> None:
        if self.minimum is None or item.value < self.minimum.value:
            self.minimum = item
        if self.maximum is None or item.value > self.maximum.value:
            self.maximum = item


def parse_target(target: str) -> tuple[str, str, str]:
    parts = [part for part in target.strip("/").split("/") if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target must be FAMILY/MATERIAL/BOND, for example REOF/SmOF/J1-z")
    return parts[0], parts[1], parts[2]


def target_root(organized_root: Path, family: str, material: str, bond: str) -> Path:
    if family in RECHX_FAMS:
        return organized_root / "REChX_cubic" / family / material / bond
    return organized_root / family / material / bond


def iter_u_files(root: Path, branches: set[str], run_filter: str | None):
    for path in sorted(root.glob("*/*/U_*.txt")):
        branch, run = path.parts[-3:-1]
        if branch not in branches:
            continue
        if run_filter is not None and run != run_filter:
            continue
        try:
            u = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        yield branch, run, u, path


def read_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        yield float(cols[0]), np.array([float(x) for x in cols[3:12]]).reshape(3, 3)


def collect(family: str, material: str, bond: str, root: Path, branches: set[str], run_filter: str | None):
    cfg = case_cfg(family, bond)
    if cfg is None:
        raise ValueError(f"{family}/{material}/{bond} is not a plottable plot_jkgamma case")
    rotation, params, axis, _, _ = cfg

    stats: dict[tuple[str, str, int, str], MinMax] = defaultdict(MinMax)
    files_seen = 0
    for branch, run, u, path in iter_u_files(root, branches, run_filter):
        files_seen += 1
        scale = scale_for_u(u)
        for ratio, tensor in read_rows(path):
            values = decompose(rotation @ tensor @ rotation.T, axis)
            for key, _ in params:
                raw_mev = values[key] * 1000.0
                stats[(branch, run, u, key)].update(ratio=ratio, raw_value=raw_mev, scale=scale, run=run)

    if files_seen == 0:
        raise FileNotFoundError(f"no matching U_*.txt files under {root}")
    return stats


def write_tsv(rows, output: Path | None) -> None:
    handle = output.open("w", encoding="utf-8", newline="") if output else sys.stdout
    try:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "family",
            "material",
            "bond",
            "branch",
            "run",
            "U",
            "scale",
            "exchange",
            "min_mev",
            "min_at_ratio",
            "min_run",
            "max_mev",
            "max_at_ratio",
            "max_run",
            "range_mev",
            "raw_min_mev",
            "raw_max_mev",
        ])
        for row in rows:
            writer.writerow(row)
    finally:
        if output:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=parse_target, help="FAMILY/MATERIAL/BOND, e.g. REOF/SmOF/J1-z")
    parser.add_argument("--organized-root", type=Path, default=DEFAULT_ORGANIZED_ROOT)
    parser.add_argument("--branch", action="append", choices=BRANCHES, help="restrict branch; repeatable")
    parser.add_argument("--run", help="restrict to one run id")
    parser.add_argument("--aggregate-runs", action="store_true", help="combine all runs per branch/U/exchange")
    parser.add_argument("--output", type=Path, help="write TSV here instead of stdout")
    args = parser.parse_args()

    family, material, bond = args.target
    root = target_root(args.organized_root, family, material, bond)
    branches = set(args.branch or BRANCHES)

    try:
        stats = collect(family, material, bond, root, branches, args.run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.aggregate_runs:
        aggregate: dict[tuple[str, int, str], MinMax] = defaultdict(MinMax)
        for (branch, _run, u, key), stat in stats.items():
            if stat.minimum is not None:
                aggregate[(branch, u, key)].absorb(stat.minimum)
            if stat.maximum is not None:
                aggregate[(branch, u, key)].absorb(stat.maximum)
        row_items = [((branch, "ALL", u, key), stat) for (branch, u, key), stat in aggregate.items()]
    else:
        row_items = list(stats.items())

    rows = []
    for (branch, run, u, key), stat in sorted(row_items):
        if stat.minimum is None or stat.maximum is None:
            continue
        width = stat.maximum.value - stat.minimum.value
        rows.append([
            family,
            material,
            bond,
            branch,
            run,
            u,
            f"{scale_for_u(u):.8g}",
            key,
            f"{stat.minimum.value:.10g}",
            f"{stat.minimum.ratio:.10g}",
            stat.minimum.run,
            f"{stat.maximum.value:.10g}",
            f"{stat.maximum.ratio:.10g}",
            stat.maximum.run,
            f"{width:.10g}",
            f"{stat.minimum.raw_value:.10g}",
            f"{stat.maximum.raw_value:.10g}",
        ])
    write_tsv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
