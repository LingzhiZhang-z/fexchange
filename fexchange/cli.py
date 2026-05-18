"""
CLI entry point.

Usage:
    fexchange run INPUT_TOML
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable

from fexchange.io.disk import (
    append_error_record,
)
from fexchange.io.run_input import (
    load_run_input,
    window_includes,
)
from fexchange.pipeline.keys import (
    level_key as _level_key,
)
from fexchange.pipeline.resolve import resolve_core_params as _resolve_core_params
from fexchange.pipeline import stages as _stages
from fexchange.utils.constants import LEVELS, N_ORB
from fexchange.utils.errors import (
    FexchangeError,
    InputError,
    RuntimeError_,
)
from fexchange.utils.numerics import numerics_meta

logger = logging.getLogger("fexchange")
_FULL_KW = ("n_ele", "n_orb", "r42", "r62", "scheme")
_STAGE_DISPATCH: dict[tuple[str, str], tuple[Callable[..., Any], tuple[str, ...]]] = {
    ("sopt", "LMSM"): (_stages.ensure_lsms_all_three, _FULL_KW),
    ("sopt", "LSJM"): (_stages.ensure_lsjm_all_three, _FULL_KW),
    ("sopt", "L0"): (_stages.ensure_l0_sopt, ("n_ele", "n_orb")),
    ("sopt", "L1"): (_stages.ensure_l1_sopt, _FULL_KW),
    ("sopt", "L2"): (_stages.ensure_l2_sopt, _FULL_KW),
    ("sopt", "L3"): (_stages.ensure_l3_sopt, _FULL_KW),
    ("sopt", "L4"): (_stages.ensure_l4_sopt, _FULL_KW),
    ("fopt", "LMSM"): (_stages.ensure_lsms_all_three, _FULL_KW),
    ("fopt", "LSJM"): (_stages.ensure_lsjm_all_three, _FULL_KW),
    ("fopt", "L0"): (_stages.ensure_l0_fopt, ("n_ele", "n_orb")),
    ("fopt", "L1"): (_stages.ensure_l1_fopt, _FULL_KW),
    ("fopt", "L2"): (_stages.ensure_l2_fopt, _FULL_KW),
    ("fopt", "L3"): (_stages.ensure_l3_fopt, _FULL_KW),
}
_PAYLOAD_KEYS = (
    "module",
    "level",
    "stage",
    "op",
    "run_id",
    "key",
    "schema_version",
    "standard_version",
    "expected",
    "actual",
    "paths",
)


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

    args = parser.parse_args(argv)
    if args.command != "run":
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
        return _run_pipeline(args.toml, log_level=args.log_level)
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


def _run_pipeline(toml_path: str, *, log_level: str) -> int:
    t0 = time.time()
    cfg = load_run_input(toml_path)

    output_root = cfg["paths"]["output_root"]
    file_handler = _install_run_log_handler(cfg, log_level=log_level)
    n_ele, r42, r62, scheme = _resolve_core_params(cfg)
    n_orb = N_ORB
    try:
        logger.info("run_log=%s", file_handler.baseFilename if file_handler is not None else "")
        state: dict[str, Any] = {}
        for level in LEVELS:
            if not window_includes(cfg, level):
                continue
            t_level = time.time()
            level_key = _level_key(
                level,
                n_ele=n_ele,
                r42=r42,
                r62=r62,
                cfg=cfg,
            )
            logger.info("=== executing %s ===", level)

            _execute_level(level, cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)

            _emit_stage_summary(level=level, key=level_key, elapsed_s=time.time() - t_level)

        total = time.time() - t0
        logger.info("pipeline complete in %.3fs", total)
        print(f"[fexchange] Done. total={total:.3f}s output_root={output_root}")
        return 0
    except Exception:
        logger.exception("pipeline failed")
        raise
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()


def _execute_level(
    level: str,
    cfg: dict[str, Any],
    state: dict[str, Any],
    **kw: Any,
) -> None:
    branch = str(cfg.get("runtime", {}).get("branch", "sopt"))
    dispatch_key = (branch, level)
    if dispatch_key not in _STAGE_DISPATCH:
        raise InputError("FXE-INPUT-003", f"Unsupported stage for branch: {dispatch_key}", actual={"branch": branch, "level": level})
    fn, keys = _STAGE_DISPATCH[dispatch_key]
    fn(cfg, state, **{k: kw[k] for k in keys})


def _emit_stage_summary(*, level: str, key: str, elapsed_s: float) -> None:
    summary = {
        "level": level,
        "key": key,
        "elapsed_s": elapsed_s,
        "numerics_meta": numerics_meta(),
    }
    logger.info("stage_summary=%s", json.dumps(summary, default=str))


def _install_run_log_handler(cfg: dict[str, Any], *, log_level: str) -> logging.FileHandler | None:
    run_name = cfg.get("runtime", {}).get("run_name")
    if not run_name:
        return None
    path = Path(cfg["paths"]["output_root"]) / str(run_name) / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return handler


def _safe_append_error(output_root: str, payload: dict[str, Any]) -> None:
    try:
        append_error_record(output_root, payload)
    except Exception:  # pragma: no cover - best effort only
        pass


if __name__ == "__main__":
    raise SystemExit(main())
