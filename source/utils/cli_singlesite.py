from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-site LSMS/LSJM pipeline (stub)")
    parser.add_argument("--n", type=int, required=True, help="Electron count")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, help="Optional configuration file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise NotImplementedError(
        "Single-site pipeline is a stub. Implement LSMS/LSJM + CEF steps and write outputs."
    )


if __name__ == "__main__":
    raise SystemExit(main())
