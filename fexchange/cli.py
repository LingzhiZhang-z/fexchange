"""
CLI entry point.

Usage:
    fexchange run INPUT_TOML
    fexchange sweep BASE_TOML
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from fexchange.io.disk import (
    append_error_record,
)
from fexchange.pipeline.runner import run_pipeline
from fexchange.utils.errors import (
    FexchangeError,
    RuntimeError_,
)

logger = logging.getLogger("fexchange")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fexchange")
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="Execute configured level window")
    run_parser.add_argument("toml", type=str, help="Path to a run-input TOML file")
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sweep_parser = sub.add_parser("sweep", help="Run a parameter sweep over a base run-input")
    sweep_parser.add_argument("toml", type=str, help="Path to a base run-input TOML with a [sweep] table")
    sweep_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)
    if args.command not in ("run", "sweep"):
        parser.print_help()
        return 1

    console_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    for handler in logging.getLogger().handlers:
        handler.setLevel(console_level)
    logger.setLevel(logging.DEBUG)

    output_root_hint = "./outputs"
    try:
        if args.command == "sweep":
            from fexchange.pipeline.sweep import run_sweep

            return run_sweep(args.toml, log_level=args.log_level)
        return run_pipeline(args.toml, log_level=args.log_level)
    except FexchangeError as exc:
        print(exc.payload_json(), file=sys.stderr)
        _safe_append_error(output_root_hint, exc.payload())
        return 1
    except Exception as exc:  # pragma: no cover - defensive boundary
        coded = RuntimeError_(
            "FXE-RUNTIME-001",
            f"Unhandled runtime failure: {exc}",
            module="cli",
            stage="main",
            op="run",
        )
        print(coded.payload_json(), file=sys.stderr)
        _safe_append_error(output_root_hint, coded.payload())
        return 1


def _safe_append_error(output_root: str, payload: dict[str, Any]) -> None:
    try:
        append_error_record(output_root, payload)
    except Exception:  # pragma: no cover - best effort only
        pass


if __name__ == "__main__":
    raise SystemExit(main())
