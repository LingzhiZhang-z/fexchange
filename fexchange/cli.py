"""
CLI entry point.

Usage:
    fexchange run run_input.toml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, Callable

from fexchange.io.disk import (
    append_error_record,
)
from fexchange.io.run_input import (
    load_run_input,
    upstream_for_start_level,
    window_includes,
)
from fexchange.pipeline.keys import (
    level_key as _level_key,
)
from fexchange.pipeline.resolve import resolve_core_params as _resolve_core_params
from fexchange.pipeline import stages as _stages
from fexchange.pipeline.validation import (
    validate_upstream_artifacts as _validate_upstream_artifacts,
)
from fexchange.utils.errors import (
    FexchangeError,
    InputError,
    RuntimeError_,
)
from fexchange.utils.numerics import numerics_meta
from fexchange.utils.parallel import ParallelContext

logger = logging.getLogger("fexchange")

LEVELS = ("LMSM", "LSJM", "L0", "L1", "L2", "L3", "L4")
_FULL_KW = ("n_ele", "n_orb", "r42", "r62", "scheme")
_STAGE_DISPATCH: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {
    "LMSM": (_stages.ensure_lsms_all_three, _FULL_KW),
    "LSJM": (_stages.ensure_lsjm_all_three, _FULL_KW),
    "L0": (_stages.ensure_l0, ("n_ele", "n_orb")),
    "L1": (_stages.ensure_l1, _FULL_KW),
    "L2": (_stages.ensure_l2, _FULL_KW),
    "L3": (_stages.ensure_l3, _FULL_KW),
    "L4": (_stages.ensure_l4, _FULL_KW),
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
    run_parser.add_argument("toml", type=str, help="Path to run_input.toml")
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    output_root_hint = "./outputs"
    try:
        return _run_pipeline(args.toml)
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


def _run_pipeline(toml_path: str) -> int:
    t0 = time.time()
    cfg = load_run_input(toml_path)
    ctx = ParallelContext()

    output_root = cfg["paths"]["output_root"]
    n_ele, r42, r62, scheme = _resolve_core_params(cfg)
    n_orb = 14

    _preflight(cfg, ctx, n_ele=n_ele, r42=r42, r62=r62, scheme=scheme)

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
        _set_level_sharding(ctx, level)
        logger.info("=== executing %s ===", level)

        _execute_level(
            level,
            cfg,
            ctx,
            state,
            n_ele=n_ele,
            n_orb=n_orb,
            r42=r42,
            r62=r62,
            scheme=scheme,
        )

        _emit_stage_summary(
            level=level,
            key=level_key,
            elapsed_s=time.time() - t_level,
            ctx=ctx,
        )

    total = time.time() - t0
    logger.info("pipeline complete in %.3fs", total)
    if ctx.is_root:
        print(f"[fexchange] Done. total={total:.3f}s output_root={output_root}")
    return 0


def _preflight(
    cfg: dict[str, Any],
    ctx: ParallelContext,
    *,
    n_ele: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    """
    Root performs preflight, broadcasts verdict, all ranks obey (00-06 §5).
    """
    plan: dict[str, Any] = {"ok": True, "errors": []}
    if ctx.is_root:
        try:
            start_level = cfg["runtime"]["start_level"]
            required_upstream = upstream_for_start_level(start_level)
            _validate_upstream_artifacts(
                cfg,
                required_upstream,
                n_ele=n_ele,
                r42=r42,
                r62=r62,
                scheme=scheme,
            )
            plan["required_upstream"] = list(required_upstream)
            plan["window"] = (
                cfg["runtime"]["start_level"],
                cfg["runtime"]["end_level"],
            )
        except FexchangeError as exc:
            plan = {"ok": False, "error_payload": exc.payload()}

    plan = ctx.bcast(plan, root=ctx.root_rank)
    if not plan.get("ok", False):
        payload = plan.get("error_payload", {})
        if payload:
            raise RuntimeError_(
                payload.get("code", "FXE-RUNTIME-001"),
                payload.get("message", "Preflight failed"),
                **{k: payload[k] for k in _PAYLOAD_KEYS if k in payload},
            )
        raise RuntimeError_("FXE-RUNTIME-001", "Preflight failed without payload", stage="preflight")
    ctx.barrier()


def _execute_level(
    level: str,
    cfg: dict[str, Any],
    ctx: ParallelContext,
    state: dict[str, Any],
    **kw: Any,
) -> None:
    if level not in _STAGE_DISPATCH:
        raise InputError("FXE-INPUT-003", f"Unknown level: {level}", actual={"level": level})
    fn, keys = _STAGE_DISPATCH[level]

    # Current MPI contract implementation is root-only execution per stage.
    if ctx.parallel_enabled and not ctx.is_root:
        ctx.barrier()
        return

    fn(cfg, ctx, state, **{k: kw[k] for k in keys})
    if ctx.parallel_enabled:
        ctx.barrier()


def _emit_stage_summary(*, level: str, key: str, elapsed_s: float, ctx: ParallelContext) -> None:
    summary = {
        "level": level,
        "key": key,
        "elapsed_s": elapsed_s,
        "numerics_meta": numerics_meta(),
        "parallel_meta": ctx.meta(),
    }
    logger.info("stage_summary=%s", json.dumps(summary, default=str))


def _set_level_sharding(ctx: ParallelContext, level: str) -> None:
    axis = {
        "LMSM": "(alpha,L,S)",
        "LSJM": "(alpha,L,S)",
        "L0": "kappa",
        "L1": "(u,v)",
        "L2": "(u,v)|(r,s)",
        "L3": "denominator_pairs",
        "L4": "root_serial_default",
    }.get(level, "root_serial")
    ctx.set_sharding(
        axis,
        [{"rank": ctx.rank, "scope": "full"}],
    )


def _safe_append_error(output_root: str, payload: dict[str, Any]) -> None:
    try:
        append_error_record(output_root, payload)
    except Exception:  # pragma: no cover - best effort only
        pass


if __name__ == "__main__":
    raise SystemExit(main())
