"""
TOML run-input loader with strict schema/field validation.

Spec reference: 00-05-RUN_INPUT_SINGLE_FILE.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fexchange.utils.constants import (
    LEVEL_ORDER,
    RE_DEFAULTS_BY_N_ELE,
    RE_TO_N_ELE,
    RUN_SCHEMA_VERSION,
    STANDARD_VERSION,
    UPSTREAM_REQUIREMENTS,
)
from fexchange.utils.errors import InputError, SchemaError

logger = logging.getLogger("fexchange")

_ALLOWED_TOP_LEVEL_SECTIONS = frozenset(
    {
        "schema_version",
        "standard_version",
        "run_id",
        "title",
        "physics",
        "model",
        "sopt",
        "inputs",
        "sources",
        "paths",
        "runtime",
        "checks",
        "_derived",
    }
)


def load_run_input(path: str | Path) -> dict[str, Any]:
    """
    Load and validate a run_input.toml file.

    Returns a parsed configuration dict with ``_derived`` fields.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(
            "FXE-INPUT-001",
            f"Run input file not found: {path}",
            paths={"input_file": str(path)},
        )
    if path.name != "run_input.toml":
        raise InputError(
            "FXE-INPUT-003",
            "Canonical run-input filename must be run_input.toml",
            expected={"filename": "run_input.toml"},
            actual={"filename": path.name},
            paths={"input_file": str(path)},
        )

    cfg = _load_toml(path)

    _validate_top_level(cfg)
    _validate_runtime(cfg)
    _validate_sections(cfg)
    _normalize_sources(cfg)
    _apply_re_defaults(cfg)
    _validate_window_required_sections(cfg)
    _validate_checks(cfg)

    # Section-level validation.  The window gate allows stricter checks on active stages.
    _validate_physics(cfg)
    _validate_model(cfg)
    _validate_sopt(cfg)
    _validate_sources(cfg)
    _validate_inputs(cfg)
    _validate_paths(cfg)

    # Derived ratios (00-05 §4, 05-00 §3)
    if "physics" in cfg:
        p = cfg["physics"]
        cfg["_derived"] = {
            "r42": float(p["F4"]) / float(p["F2"]),
            "r62": float(p["F6"]) / float(p["F2"]),
        }

    logger.info(
        "Run input loaded: run_id=%s, title=%s, window=%s->%s",
        cfg.get("run_id"),
        cfg.get("title"),
        cfg["runtime"]["start_level"],
        cfg["runtime"]["end_level"],
    )
    return cfg


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        loaded = tomllib.load(f)
    if not isinstance(loaded, dict):
        raise SchemaError(
            "FXE-SCHEMA-002",
            "run_input.toml must decode to a top-level object",
            actual={"actual_type": type(loaded).__name__},
        )
    return loaded


def _validate_top_level(cfg: dict[str, Any]) -> None:
    for key in ("schema_version", "standard_version", "run_id", "title"):
        if key not in cfg:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required top-level key: {key}",
                expected={"field_path": key},
            )

    if cfg["schema_version"] != RUN_SCHEMA_VERSION:
        raise SchemaError(
            "FXE-SCHEMA-001",
            "schema_version mismatch",
            expected={"schema_version": RUN_SCHEMA_VERSION},
            actual={"schema_version": cfg["schema_version"]},
        )
    if cfg["standard_version"] != STANDARD_VERSION:
        raise SchemaError(
            "FXE-SCHEMA-001",
            "standard_version mismatch",
            expected={"standard_version": STANDARD_VERSION},
            actual={"standard_version": cfg["standard_version"]},
        )
    _require_non_empty_str(cfg, "run_id")
    _require_non_empty_str(cfg, "title")


def _validate_sections(cfg: dict[str, Any]) -> None:
    for section in ("paths", "runtime", "checks"):
        if section not in cfg:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required section: [{section}]",
                expected={"field_path": section},
            )
    extras = sorted(key for key in cfg if key not in _ALLOWED_TOP_LEVEL_SECTIONS)
    if extras:
        raise InputError(
            "FXE-INPUT-003",
            "Unsupported top-level section(s) in run_input.toml",
            expected={"allowed_sections": sorted(_ALLOWED_TOP_LEVEL_SECTIONS)},
            actual={"unsupported_sections": extras},
        )


def _validate_runtime(cfg: dict[str, Any]) -> None:
    rt = cfg.get("runtime")
    if not isinstance(rt, dict):
        raise InputError(
            "FXE-INPUT-003",
            "runtime section must be a table/object",
            actual={"actual_type": type(rt).__name__},
        )

    for key in ("start_level", "end_level", "on_missing_upstream", "read_first"):
        if key not in rt:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required runtime field: {key}",
                expected={"field_path": f"runtime.{key}"},
            )

    start = rt["start_level"]
    end = rt["end_level"]
    if start not in LEVEL_ORDER:
        raise InputError(
            "FXE-INPUT-003",
            f"Invalid runtime.start_level: {start!r}",
            actual={"start_level": start},
        )
    if end not in LEVEL_ORDER:
        raise InputError(
            "FXE-INPUT-003",
            f"Invalid runtime.end_level: {end!r}",
            actual={"end_level": end},
        )
    if LEVEL_ORDER[start] > LEVEL_ORDER[end]:
        raise InputError(
            "FXE-INPUT-003",
            f"Invalid level window: {start} > {end}",
            actual={"start_level": start, "end_level": end},
        )

    if rt["on_missing_upstream"] != "fail":
        raise InputError(
            "FXE-INPUT-003",
            "runtime.on_missing_upstream must be 'fail'",
            expected={"runtime.on_missing_upstream": "fail"},
            actual={"runtime.on_missing_upstream": rt["on_missing_upstream"]},
        )
    if rt["read_first"] is not True:
        raise InputError(
            "FXE-INPUT-003",
            "runtime.read_first must be true",
            actual={"read_first": rt["read_first"]},
        )


def _validate_window_required_sections(cfg: dict[str, Any]) -> None:
    rt = cfg["runtime"]
    start = rt["start_level"]
    end = rt["end_level"]
    includes = lambda level: LEVEL_ORDER[start] <= LEVEL_ORDER[level] <= LEVEL_ORDER[end]

    required_sections: set[str] = set()
    if any(includes(level) for level in ("LMSM", "L0")):
        required_sections.add("physics")
    if includes("LMSM"):
        required_sections.add("model")
    if any(includes(level) for level in ("L2", "L3", "L4")):
        required_sections.add("inputs")
        required_sections.add("sources")
    if any(includes(level) for level in ("L3", "L4")):
        required_sections.add("sopt")

    for section in sorted(required_sections):
        if section not in cfg:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required section for selected window: [{section}]",
                expected={"field_path": section},
            )


def _validate_checks(cfg: dict[str, Any]) -> None:
    checks = cfg.get("checks")
    if not isinstance(checks, dict):
        raise InputError(
            "FXE-INPUT-003",
            "checks section must be a table/object",
            actual={"actual_type": type(checks).__name__},
        )
    for key in ("strict_mode", "eps_profile"):
        if key not in checks:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required checks field: {key}",
                expected={"field_path": f"checks.{key}"},
            )
    if not isinstance(checks["strict_mode"], bool):
        raise InputError(
            "FXE-INPUT-003",
            "checks.strict_mode must be bool",
            actual={"actual_type": type(checks["strict_mode"]).__name__},
        )
    _require_non_empty_str(checks, "eps_profile", prefix="checks")


def _validate_physics(cfg: dict[str, Any]) -> None:
    if "physics" not in cfg:
        return
    physics = cfg["physics"]
    if not isinstance(physics, dict):
        raise InputError(
            "FXE-INPUT-003",
            "physics section must be a table/object",
            actual={"actual_type": type(physics).__name__},
        )

    for key in ("n_ele", "F2", "F4", "F6"):
        if key not in physics:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required physics field: {key}",
                expected={"field_path": f"physics.{key}"},
            )

    n_ele = physics["n_ele"]
    if isinstance(n_ele, bool) or not isinstance(n_ele, int):
        raise InputError(
            "FXE-INPUT-003",
            "physics.n_ele must be integer",
            actual={"actual_type": type(n_ele).__name__, "value": n_ele},
        )
    if not (1 <= n_ele <= 13):
        raise InputError(
            "FXE-INPUT-003",
            "physics.n_ele must be in [1,13]",
            actual={"n_ele": n_ele},
        )

    for key in ("F2", "F4", "F6"):
        if not _is_number(physics[key]):
            raise InputError(
                "FXE-INPUT-003",
                f"physics.{key} must be numeric",
                actual={key: physics[key]},
            )
    if float(physics["F2"]) == 0.0:
        raise InputError(
            "FXE-INPUT-003",
            "physics.F2 must not be zero",
            actual={"F2": physics["F2"]},
        )


def _validate_model(cfg: dict[str, Any]) -> None:
    if "model" not in cfg:
        return
    model = cfg["model"]
    if not isinstance(model, dict):
        raise InputError(
            "FXE-INPUT-003",
            "model section must be a table/object",
            actual={"actual_type": type(model).__name__},
        )
    if "scheme" not in model:
        raise InputError(
            "FXE-INPUT-002",
            "Missing required model field: scheme",
            expected={"field_path": "model.scheme"},
        )
    if model["scheme"] != "RS":
        raise InputError(
            "FXE-INPUT-003",
            "model.scheme must be 'RS' in this standard version",
            expected={"model.scheme": "RS"},
            actual={"model.scheme": model["scheme"]},
        )


def _validate_sopt(cfg: dict[str, Any]) -> None:
    if "sopt" not in cfg:
        return
    sopt = cfg["sopt"]
    if not isinstance(sopt, dict):
        raise InputError(
            "FXE-INPUT-003",
            "sopt section must be a table/object",
            actual={"actual_type": type(sopt).__name__},
        )
    for key in ("U", "Jh"):
        if key not in sopt:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required sopt field: {key}",
                expected={"field_path": f"sopt.{key}"},
            )
        if not _is_number(sopt[key]):
            raise InputError(
                "FXE-INPUT-003",
                f"sopt.{key} must be numeric",
                actual={key: sopt[key]},
            )
    if "zeta" not in sopt:
        raise InputError(
            "FXE-INPUT-002",
            "Missing required sopt field: zeta",
            expected={"field_path": "sopt.zeta"},
        )
    if not _is_number(sopt["zeta"]):
        raise InputError(
            "FXE-INPUT-003",
            "sopt.zeta must be numeric",
            actual={"zeta": sopt["zeta"]},
        )


def _validate_sources(cfg: dict[str, Any]) -> None:
    if "sources" not in cfg:
        return
    sources = cfg["sources"]
    if not isinstance(sources, dict):
        raise InputError(
            "FXE-INPUT-003",
            "sources section must be a table/object",
            actual={"actual_type": type(sources).__name__},
        )

    rt = cfg["runtime"]
    start = rt["start_level"]
    end = rt["end_level"]
    includes = lambda level: LEVEL_ORDER[start] <= LEVEL_ORDER[level] <= LEVEL_ORDER[end]

    if any(includes(level) for level in ("L2", "L3", "L4")):
        _require_non_empty_str(sources, "hopping_name", prefix="sources")
    if includes("L4"):
        _require_non_empty_str(sources, "kramer_name", prefix="sources")


def _validate_inputs(cfg: dict[str, Any]) -> None:
    if "inputs" not in cfg:
        return
    inputs = cfg["inputs"]
    if not isinstance(inputs, dict):
        raise InputError(
            "FXE-INPUT-003",
            "inputs section must be a table/object",
            actual={"actual_type": type(inputs).__name__},
        )

    rt = cfg["runtime"]
    start = rt["start_level"]
    end = rt["end_level"]
    includes = lambda level: LEVEL_ORDER[start] <= LEVEL_ORDER[level] <= LEVEL_ORDER[end]

    if any(includes(level) for level in ("L2", "L3", "L4")):
        _require_non_empty_str(inputs, "hopping_file", prefix="inputs")
    if includes("L4"):
        _require_non_empty_str(inputs, "projector_file", prefix="inputs")


def _normalize_sources(cfg: dict[str, Any]) -> None:
    if "sources" in cfg and not isinstance(cfg["sources"], dict):
        return
    if "inputs" in cfg and not isinstance(cfg["inputs"], dict):
        return

    inputs = cfg.get("inputs", {})
    if "sources" not in cfg:
        cfg["sources"] = {}
    sources = cfg["sources"]

    hopping_name = _first_non_empty_str(
        sources.get("hopping_name"),
        sources.get("hopping_label"),
        inputs.get("hopping_label"),
        _stem_or_empty(inputs.get("hopping_file")),
    )
    if hopping_name:
        sources["hopping_name"] = hopping_name
        sources.setdefault("hopping_label", hopping_name)

    kramer_name = _first_non_empty_str(
        sources.get("kramer_name"),
        sources.get("projection_label"),
        inputs.get("projection_label"),
        _stem_or_empty(inputs.get("projector_file")),
    )
    if kramer_name:
        sources["kramer_name"] = kramer_name
        sources.setdefault("projection_label", kramer_name)


def _apply_re_defaults(cfg: dict[str, Any]) -> None:
    physics = cfg.get("physics")
    if not isinstance(physics, dict):
        return

    re_name = _normalize_re_name(physics.get("RE", "auto"))
    physics["RE"] = re_name
    if re_name == "auto":
        return

    ref_n_ele = RE_TO_N_ELE[re_name]
    defaults = RE_DEFAULTS_BY_N_ELE[ref_n_ele]
    sopt = cfg.get("sopt")
    if not isinstance(sopt, dict):
        return

    jh = sopt.get("Jh")
    if _is_number(jh):
        jh_value = float(jh)
        physics.setdefault("F2", defaults["F2_per_Jh"] * jh_value)
        physics.setdefault("F4", defaults["F4_per_Jh"] * jh_value)
        physics.setdefault("F6", defaults["F6_per_Jh"] * jh_value)
    sopt.setdefault("zeta", defaults["zeta"])


def _validate_paths(cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise InputError(
            "FXE-INPUT-003",
            "paths section must be a table/object",
            actual={"actual_type": type(paths).__name__},
        )

    if "output_root" not in paths:
        raise InputError(
            "FXE-INPUT-002",
            "Missing required paths field: output_root",
            expected={"field_path": "paths.output_root"},
        )
    if paths["output_root"] != "./outputs":
        raise InputError(
            "FXE-INPUT-003",
            "paths.output_root must be './outputs'",
            expected={"paths.output_root": "./outputs"},
            actual={"paths.output_root": paths["output_root"]},
        )


def window_includes(cfg: dict[str, Any], level: str) -> bool:
    """Return True if the execution window includes `level`."""
    rt = cfg.get("runtime", {})
    start = rt.get("start_level", "")
    end = rt.get("end_level", "")
    return LEVEL_ORDER.get(start, 0) <= LEVEL_ORDER.get(level, 99) <= LEVEL_ORDER.get(end, 0)


def upstream_for_start_level(start_level: str) -> tuple[str, ...]:
    """Upstream artifact requirements for the configured start level."""
    if start_level not in UPSTREAM_REQUIREMENTS:
        raise InputError(
            "FXE-INPUT-003",
            f"Unknown start level: {start_level!r}",
            actual={"start_level": start_level},
        )
    return UPSTREAM_REQUIREMENTS[start_level]

def _is_number(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _stem_or_empty(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return Path(value).stem


def _normalize_re_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "auto"
    if value.strip().lower() == "auto":
        return "auto"
    canonical = value.strip().capitalize()
    if canonical not in RE_TO_N_ELE:
        raise InputError(
            "FXE-INPUT-003",
            "physics.RE must be one of the supported rare-earth presets or 'auto'",
            actual={"physics.RE": value},
        )
    return canonical


def _require_non_empty_str(d: dict[str, Any], key: str, *, prefix: str | None = None) -> None:
    value = d.get(key)
    if not isinstance(value, str) or not value.strip():
        field = f"{prefix}.{key}" if prefix else key
        raise InputError(
            "FXE-INPUT-003",
            f"{field} must be a non-empty string",
            actual={field: value},
        )
