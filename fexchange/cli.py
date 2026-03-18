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
    point_result_cfg_signature,
    write_point_result_txt,
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

    output_root = cfg["paths"]["output_root"]
    n_ele, r42, r62, scheme = _resolve_core_params(cfg)
    n_orb = N_ORB

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
    _maybe_write_point_result(cfg, state, n_ele=n_ele)
    logger.info("pipeline complete in %.3fs", total)
    print(f"[fexchange] Done. total={total:.3f}s output_root={output_root}")
    return 0


def _execute_level(
    level: str,
    cfg: dict[str, Any],
    state: dict[str, Any],
    **kw: Any,
) -> None:
    if level not in _STAGE_DISPATCH:
        raise InputError("FXE-INPUT-003", f"Unknown level: {level}", actual={"level": level})
    fn, keys = _STAGE_DISPATCH[level]
    fn(cfg, state, **{k: kw[k] for k in keys})


def _emit_stage_summary(*, level: str, key: str, elapsed_s: float) -> None:
    summary = {
        "level": level,
        "key": key,
        "elapsed_s": elapsed_s,
        "numerics_meta": numerics_meta(),
    }
    logger.info("stage_summary=%s", json.dumps(summary, default=str))


def _maybe_write_point_result(cfg: dict[str, Any], state: dict[str, Any], *, n_ele: int) -> None:
    l4 = state.get("l4")
    if not isinstance(l4, dict):
        return
    if "J_mu" not in l4 or "mapping_residual" not in l4:
        return

    physics = cfg.get("physics", {})
    sources = cfg.get("sources", {})
    sopt = cfg["sopt"]
    branches = cfg.get("_branches", {})
    cfg_meta = {
        "cfg_signature": "",
        "run_id": cfg.get("run_id"),
        "title": cfg.get("title"),
        "energy_unit": cfg.get("units", {}).get("energy", "meV"),
        "energy_reference": str(sopt.get("energy_reference", "lsjm_ground")),
        "sources": {
            "hopping_label": str(sources.get("hopping_label", sources.get("hopping_name", "auto"))),
            "projection_label": str(sources.get("projection_label", sources.get("kramer_name", "auto"))),
            "hopping_name": str(sources.get("hopping_name", "auto")),
            "kramer_name": str(sources.get("kramer_name", "auto")),
        },
        "resolved_branches": branches if isinstance(branches, dict) else {},
    }
    cfg_signature = point_result_cfg_signature(cfg_meta)
    cfg_meta["cfg_signature"] = cfg_signature
    path = write_point_result_txt(
        cfg["paths"]["output_root"],
        RE=str(physics.get("RE", "auto")),
        n_ele=n_ele,
        hopping_label=str(sources.get("hopping_label", sources.get("hopping_name", "auto"))),
        projection_label=str(sources.get("projection_label", sources.get("kramer_name", "auto"))),
        U=float(sopt["U"]),
        Jh=float(sopt["Jh"]),
        zeta=float(sopt["zeta"]),
        J_mu=l4["J_mu"],
        mapping_residual=float(l4["mapping_residual"]),
        cfg_signature=cfg_signature,
        cfg_meta=cfg_meta,
    )
    logger.info("point_result=%s", path)


def _safe_append_error(output_root: str, payload: dict[str, Any]) -> None:
    try:
        append_error_record(output_root, payload)
    except Exception:  # pragma: no cover - best effort only
        pass


if __name__ == "__main__":
    raise SystemExit(main())
