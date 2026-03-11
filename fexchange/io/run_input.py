"""
TOML run-input loader with strict schema/field validation.

Spec reference: 00-05-RUN_INPUT_SINGLE_FILE.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fexchange.utils.errors import InputError, SchemaError
from fexchange.utils.legacy import ensure_non_legacy_path

logger = logging.getLogger("fexchange")

RUN_SCHEMA_VERSION = "fxe.run_input.v1"
STANDARD_VERSION = "2026-02"

# Level ordering (00-05 §4)
LEVEL_ORDER = {
    "LMSM": 1,
    "LSJM": 2,
    "L0": 3,
    "L1": 4,
    "L2": 5,
    "L3": 6,
    "L4": 7,
}

UPSTREAM_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "LMSM": (),
    "LSJM": ("LMSM",),
    "L0": (),
    "L1": ("LMSM", "LSJM", "L0"),
    "L2": ("L1",),
    "L3": ("L2",),
    "L4": ("L3",),
}


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
    legacy_policy = _extract_legacy_policy(cfg)

    _validate_top_level(cfg)
    _validate_runtime(cfg)
    _validate_sections(cfg)
    _validate_window_required_sections(cfg)
    _validate_checks(cfg)

    # Section-level validation.  The window gate allows stricter checks on active stages.
    _validate_physics(cfg)
    _validate_model(cfg)
    _validate_sopt(cfg)
    _validate_sources(cfg)
    _validate_paths(cfg, legacy_policy=legacy_policy)
    _validate_wannier90(cfg, legacy_policy=legacy_policy)

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


def _extract_legacy_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Optional legacy policy block (00-04 §4).

    If absent, legacy access is disabled.
    """
    raw = cfg.get("legacy", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise InputError(
            "FXE-INPUT-003",
            "legacy section must be a table/object",
            actual={"actual_type": type(raw).__name__},
        )
    return {
        "allow_legacy": bool(raw.get("allow_legacy", False)),
        "legacy_scope": list(raw.get("legacy_scope", [])) if raw.get("legacy_scope", None) is not None else [],
        "legacy_reason": raw.get("legacy_reason"),
        "legacy_ticket": raw.get("legacy_ticket"),
    }


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
    # 00-05 §6 window gate: physics is mandatory only when LMSM/L0 is in window.
    if any(includes(level) for level in ("LMSM", "L0")):
        required_sections.add("physics")
    if includes("LMSM"):
        required_sections.add("model")
    if any(includes(level) for level in ("L2", "L3", "L4")):
        required_sections.add("sources")
    if any(includes(level) for level in ("L3", "L4")):
        required_sections.add("sopt")

    sources = cfg.get("sources", {}) if isinstance(cfg.get("sources"), dict) else {}
    if any(includes(level) for level in ("L2", "L3", "L4")) and sources.get("hopping_source") == "wannier90":
        required_sections.add("wannier90")
    if includes("L4") and sources.get("kramer_source") == "file":
        required_sections.add("kramer_input")

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
    for key in ("scheme", "symmetry", "c3v_mode_q3"):
        if key not in model:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required model field: {key}",
                expected={"field_path": f"model.{key}"},
            )
    if model["scheme"] != "RS":
        raise InputError(
            "FXE-INPUT-003",
            "model.scheme must be 'RS' in this standard version",
            expected={"model.scheme": "RS"},
            actual={"model.scheme": model["scheme"]},
        )
    if model["symmetry"] not in {"Oh", "C3v"}:
        raise InputError(
            "FXE-INPUT-003",
            "model.symmetry must be 'Oh' or 'C3v'",
            actual={"model.symmetry": model["symmetry"]},
        )
    if model["c3v_mode_q3"] not in {"cos", "sin"}:
        raise InputError(
            "FXE-INPUT-003",
            "model.c3v_mode_q3 must be 'cos' or 'sin'",
            actual={"model.c3v_mode_q3": model["c3v_mode_q3"]},
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
    for key in ("U", "Jh", "zeta"):
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
    for key in ("hopping_source", "kramer_source"):
        if key not in sources:
            raise InputError(
                "FXE-INPUT-002",
                f"Missing required sources field: {key}",
                expected={"field_path": f"sources.{key}"},
            )

    if sources["hopping_source"] not in {"wannier90", "file"}:
        raise InputError(
            "FXE-INPUT-003",
            "sources.hopping_source must be 'wannier90' or 'file'",
            actual={"sources.hopping_source": sources["hopping_source"]},
        )
    if sources["kramer_source"] not in {"cef", "file"}:
        raise InputError(
            "FXE-INPUT-003",
            "sources.kramer_source must be 'cef' or 'file'",
            actual={"sources.kramer_source": sources["kramer_source"]},
        )

    # Names become mandatory by window.
    rt = cfg["runtime"]
    start = rt["start_level"]
    end = rt["end_level"]
    includes = lambda level: LEVEL_ORDER[start] <= LEVEL_ORDER[level] <= LEVEL_ORDER[end]

    if any(includes(level) for level in ("L2", "L3", "L4")):
        _require_non_empty_str(sources, "hopping_name", prefix="sources")
    if includes("L4"):
        _require_non_empty_str(sources, "kramer_name", prefix="sources")


def _validate_paths(cfg: dict[str, Any], *, legacy_policy: dict[str, Any]) -> None:
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

    sources = cfg.get("sources") if isinstance(cfg.get("sources"), dict) else {}
    if sources.get("hopping_source") == "file":
        _require_non_empty_str(paths, "hopping_file", prefix="paths")
        ensure_non_legacy_path(
            paths["hopping_file"],
            allow_legacy=legacy_policy["allow_legacy"],
            legacy_scope=legacy_policy["legacy_scope"],
            legacy_reason=legacy_policy["legacy_reason"],
            legacy_ticket=legacy_policy["legacy_ticket"],
            for_artifact=True,
        )
    if sources.get("kramer_source") == "file":
        _require_non_empty_str(paths, "kramer_file", prefix="paths")
        ensure_non_legacy_path(
            paths["kramer_file"],
            allow_legacy=legacy_policy["allow_legacy"],
            legacy_scope=legacy_policy["legacy_scope"],
            legacy_reason=legacy_policy["legacy_reason"],
            legacy_ticket=legacy_policy["legacy_ticket"],
            for_artifact=True,
        )


def _validate_wannier90(cfg: dict[str, Any], *, legacy_policy: dict[str, Any]) -> None:
    if "wannier90" not in cfg:
        return
    w90 = cfg["wannier90"]
    if not isinstance(w90, dict):
        raise InputError(
            "FXE-INPUT-003",
            "wannier90 section must be a table/object",
            actual={"actual_type": type(w90).__name__},
        )

    required = (
        "soc_mode",
        "hr_path",
        "win_path",
        "orbital_basis",
        "orbital_order_id",
        "energy_unit",
        "f_site_i",
        "f_site_j",
        "f_site_i_cell",
        "f_site_j_cell",
        "ligand_indices",
        "ligand_cells",
        "all_wannier_atom_indices",
        "delta_mode",
        "delta_reduction",
    )
    for key in required:
        if key not in w90:
            raise InputError(
                "FXE-W90-001",
                f"Missing required wannier90 field: {key}",
                expected={"field_path": f"wannier90.{key}"},
            )

    if w90["soc_mode"] not in {"with_soc", "without_soc"}:
        raise InputError(
            "FXE-W90-002",
            "wannier90.soc_mode must be 'with_soc' or 'without_soc'",
            actual={"wannier90.soc_mode": w90["soc_mode"]},
        )
    if w90["orbital_basis"] != "real_harmonic_default_w90":
        raise InputError(
            "FXE-W90-003",
            "wannier90.orbital_basis must be 'real_harmonic_default_w90'",
            actual={"wannier90.orbital_basis": w90["orbital_basis"]},
        )
    _require_non_empty_str(w90, "orbital_order_id", prefix="wannier90")
    if w90["orbital_order_id"] != "w90_f_default_v1":
        raise InputError(
            "FXE-W90-003",
            "wannier90.orbital_order_id must be 'w90_f_default_v1' in this standard version",
            expected={"wannier90.orbital_order_id": "w90_f_default_v1"},
            actual={"wannier90.orbital_order_id": w90["orbital_order_id"]},
        )
    if w90["energy_unit"] not in {"eV", "meV", "Ha", "Ry"}:
        raise InputError(
            "FXE-W90-003",
            "wannier90.energy_unit must be one of {eV, meV, Ha, Ry}",
            actual={"wannier90.energy_unit": w90["energy_unit"]},
        )
    if w90["soc_mode"] == "without_soc":
        _require_non_empty_str(w90, "spin_completion_rule", prefix="wannier90")
        if w90["spin_completion_rule"] != "up_raw_down_conj_zero_flip_v1":
            raise InputError(
                "FXE-W90-002",
                "wannier90.spin_completion_rule must be 'up_raw_down_conj_zero_flip_v1' for soc_mode='without_soc'",
                expected={"wannier90.spin_completion_rule": "up_raw_down_conj_zero_flip_v1"},
                actual={"wannier90.spin_completion_rule": w90["spin_completion_rule"]},
            )
    if not isinstance(w90["f_site_i"], int) or not isinstance(w90["f_site_j"], int):
        raise InputError(
            "FXE-W90-002",
            "wannier90.f_site_i and wannier90.f_site_j must be integers",
            actual={"f_site_i": w90["f_site_i"], "f_site_j": w90["f_site_j"]},
        )
    if w90["f_site_i"] == w90["f_site_j"]:
        raise InputError(
            "FXE-W90-002",
            "wannier90.f_site_i must be different from wannier90.f_site_j",
            actual={"f_site_i": w90["f_site_i"], "f_site_j": w90["f_site_j"]},
        )

    _require_cell_triplet(w90["f_site_i_cell"], "wannier90.f_site_i_cell")
    _require_cell_triplet(w90["f_site_j_cell"], "wannier90.f_site_j_cell")

    if not isinstance(w90["ligand_indices"], list):
        raise InputError(
            "FXE-W90-002",
            "wannier90.ligand_indices must be an integer array",
            actual={"actual_type": type(w90["ligand_indices"]).__name__},
        )
    if not isinstance(w90["ligand_cells"], list):
        raise InputError(
            "FXE-W90-002",
            "wannier90.ligand_cells must be an array of integer triplets",
            actual={"actual_type": type(w90["ligand_cells"]).__name__},
        )
    if len(w90["ligand_cells"]) != len(w90["ligand_indices"]):
        raise InputError(
            "FXE-W90-002",
            "wannier90.ligand_cells length must match ligand_indices length",
            actual={
                "len_ligand_cells": len(w90["ligand_cells"]),
                "len_ligand_indices": len(w90["ligand_indices"]),
            },
        )
    for i, cell in enumerate(w90["ligand_cells"]):
        _require_cell_triplet(cell, f"wannier90.ligand_cells[{i}]")

    if not isinstance(w90["all_wannier_atom_indices"], list) or len(w90["all_wannier_atom_indices"]) < 1:
        raise InputError(
            "FXE-W90-002",
            "wannier90.all_wannier_atom_indices must be a non-empty integer array",
            actual={"all_wannier_atom_indices": w90["all_wannier_atom_indices"]},
        )

    selected = [w90["f_site_i"], w90["f_site_j"], *w90["ligand_indices"]]
    if len(selected) != len(set(selected)):
        raise InputError(
            "FXE-W90-002",
            "wannier90 selected atom indices must be unique",
            actual={"selected_indices": selected},
        )
    all_indices = set(w90["all_wannier_atom_indices"])
    if not set(selected).issubset(all_indices):
        raise InputError(
            "FXE-W90-002",
            "wannier90 selected indices must be subset of all_wannier_atom_indices",
            actual={"selected_indices": selected, "all_wannier_atom_indices": w90["all_wannier_atom_indices"]},
        )

    if w90["delta_mode"] not in {"manual", "from_onsite"}:
        raise InputError(
            "FXE-W90-002",
            "wannier90.delta_mode must be 'manual' or 'from_onsite'",
            actual={"delta_mode": w90["delta_mode"]},
        )
    if w90["delta_reduction"] not in {"channelwise", "global_mean"}:
        raise InputError(
            "FXE-W90-002",
            "wannier90.delta_reduction must be 'channelwise' or 'global_mean'",
            actual={"delta_reduction": w90["delta_reduction"]},
        )
    if w90["delta_mode"] == "manual":
        if "delta_manual_kind" not in w90:
            raise InputError(
                "FXE-W90-002",
                "wannier90.delta_manual_kind is required when delta_mode='manual'",
                expected={"field_path": "wannier90.delta_manual_kind"},
            )
        if w90["delta_manual_kind"] not in {"channelwise", "global_mean"}:
            raise InputError(
                "FXE-W90-002",
                "wannier90.delta_manual_kind must be 'channelwise' or 'global_mean'",
                actual={"delta_manual_kind": w90["delta_manual_kind"]},
            )
        if w90["delta_manual_kind"] == "global_mean":
            if not _is_number(w90.get("delta_manual_value")):
                raise InputError(
                    "FXE-W90-002",
                    "wannier90.delta_manual_value must be numeric for global_mean",
                    actual={"delta_manual_value": w90.get("delta_manual_value")},
                )
        else:
            _require_non_empty_str(w90, "delta_manual_file", prefix="wannier90")
            ensure_non_legacy_path(
                w90["delta_manual_file"],
                allow_legacy=legacy_policy["allow_legacy"],
                legacy_scope=legacy_policy["legacy_scope"],
                legacy_reason=legacy_policy["legacy_reason"],
                legacy_ticket=legacy_policy["legacy_ticket"],
                for_artifact=True,
            )

    ensure_non_legacy_path(
        w90["hr_path"],
        allow_legacy=legacy_policy["allow_legacy"],
        legacy_scope=legacy_policy["legacy_scope"],
        legacy_reason=legacy_policy["legacy_reason"],
        legacy_ticket=legacy_policy["legacy_ticket"],
        for_artifact=True,
    )
    ensure_non_legacy_path(
        w90["win_path"],
        allow_legacy=legacy_policy["allow_legacy"],
        legacy_scope=legacy_policy["legacy_scope"],
        legacy_reason=legacy_policy["legacy_reason"],
        legacy_ticket=legacy_policy["legacy_ticket"],
        for_artifact=True,
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


def _require_cell_triplet(value: Any, field_path: str) -> None:
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(x, int) for x in value):
        raise InputError(
            "FXE-W90-002",
            f"{field_path} must be an integer triplet [a,b,c]",
            actual={"field_path": field_path, "value": value},
        )


def _is_number(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def _require_non_empty_str(d: dict[str, Any], key: str, *, prefix: str | None = None) -> None:
    value = d.get(key)
    if not isinstance(value, str) or not value.strip():
        field = f"{prefix}.{key}" if prefix else key
        raise InputError(
            "FXE-INPUT-003",
            f"{field} must be a non-empty string",
            actual={field: value},
        )
