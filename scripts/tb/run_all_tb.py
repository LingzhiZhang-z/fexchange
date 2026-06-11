#!/usr/bin/env python3
"""Generate and plot all enumerate-based TB variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=REPO / "data" / "data-DFT-input")
    parser.add_argument("--no-hr", action="store_true", help="Skip HR-onsite TB regeneration")
    parser.add_argument("--no-socdelta", action="store_true", help="Skip SOC+Delta onsite TB regeneration")
    parser.add_argument("--no-pruned-bondpp", action="store_true", help="Skip pruned+bridge-ligand-p-p diagnostic bands")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot regeneration")
    parser.add_argument("--wsvec", choices=("auto", "required", "off"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bonds_files = sorted(args.input_root.glob("*/*/enumerate_nn_bonds.txt"))
    if not bonds_files:
        raise SystemExit(f"No enumerate_nn_bonds.txt under {args.input_root}")

    tb_script = REPO / "scripts" / "tb" / "tb_from_enumerate.py"
    plot_script = REPO / "scripts" / "tb" / "plot_tb_bands.py"
    for idx, bonds in enumerate(bonds_files, start=1):
        out_dir = bonds.parent / "tb"
        print(f"[{idx:02d}/{len(bonds_files):02d}] {bonds.parent.relative_to(args.input_root)}", flush=True)
        if not args.no_hr:
            cmd = [
                    sys.executable,
                    str(tb_script),
                    str(bonds),
                    "--out-dir",
                    str(out_dir),
                    "--wsvec",
                    args.wsvec,
                ]
            if args.no_pruned_bondpp:
                cmd.append("--no-pruned-bondpp")
            subprocess.run(cmd, check=True)
        if not args.no_socdelta:
            cmd = [
                    sys.executable,
                    str(tb_script),
                    str(bonds),
                    "--out-dir",
                    str(out_dir),
                    "--wsvec",
                    args.wsvec,
                    "--onsite-mode",
                    "soc-delta",
                    "--suffix",
                    "_socdelta",
                ]
            if args.no_pruned_bondpp:
                cmd.append("--no-pruned-bondpp")
            subprocess.run(cmd, check=True)
    if not args.no_plot:
        subprocess.run([sys.executable, str(plot_script)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
