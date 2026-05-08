"""TOML run-input loader.  Spec: 05-04-RUN_INPUT."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fexchange.utils.constants import (
    LEVEL_ORDER, RE_DEFAULTS_BY_N_ELE, RE_TO_N_ELE,
    RUN_SCHEMA_VERSION, STANDARD_VERSION,
)
from fexchange.utils.errors import InputError, SchemaError

logger = logging.getLogger("fexchange")

_UNIT_SCALE = {"meV": 1.0, "eV": 1000.0}
_RE_ENERGY_SCALE = 1000.0
_F_KEYS = ("F2_ratio", "F4_ratio", "F6_ratio")
_ALLOWED = frozenset({
    "schema_version", "standard_version", "run_id", "title",
    "units", "physics", "physics_nm1", "physics_np1",
    "model", "sopt", "sopt_nm1", "sopt_np1",
    "inputs", "sources", "paths", "runtime", "checks",
    "_derived", "_branches",
})


# ── Public API ──────────────────────────────────────────────────────────

def load_run_input(path: str | Path) -> dict[str, Any]:
    """Load, validate, normalize, and build branches from a run-input TOML."""
    path = Path(path)
    _chk(path.exists(), "001", f"File not found: {path}")

    cfg = _load_toml(path)
    _validate_structure(cfg)
    _normalize(cfg)
    _validate_fields(cfg)
    _build_branches(cfg)
    _assemble_derived(cfg)

    logger.info("Loaded run_id=%s window=LMSM->%s",
                cfg.get("run_id"), cfg["runtime"]["end_level"])
    return cfg


def window_includes(cfg: dict[str, Any], level: str) -> bool:
    """True if the execution window includes *level*."""
    return LEVEL_ORDER.get(level, 99) <= LEVEL_ORDER.get(
        cfg.get("runtime", {}).get("end_level", ""), 0)


# ── TOML loading ───────────────────────────────────────────────────────

def _load_toml(path: Path) -> dict[str, Any]:
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise SchemaError("FXE-SCHEMA-002", "run-input TOML must be a mapping")
    return data


# ── Validation ─────────────────────────────────────────────────────────

def _validate_structure(cfg: dict[str, Any]) -> None:
    """Required keys, schema version, unknown-section rejection."""
    for k in ("schema_version", "standard_version", "run_id", "title"):
        _chk(k in cfg, "002", f"Missing: {k}")
    if cfg["schema_version"] != RUN_SCHEMA_VERSION:
        raise SchemaError("FXE-SCHEMA-001", "schema_version mismatch",
                          expected={"schema_version": RUN_SCHEMA_VERSION},
                          actual={"schema_version": cfg["schema_version"]})
    if cfg["standard_version"] != STANDARD_VERSION:
        raise SchemaError("FXE-SCHEMA-001", "standard_version mismatch",
                          expected={"standard_version": STANDARD_VERSION},
                          actual={"standard_version": cfg["standard_version"]})
    for k in ("paths", "runtime", "checks"):
        _chk(k in cfg, "002", f"Missing section: [{k}]")
    extras = sorted(k for k in cfg if k not in _ALLOWED)
    _chk(not extras, "003", f"Unknown sections: {extras}")
    _chk("end_level" in cfg["runtime"], "002", "Missing: runtime.end_level")
    _chk(cfg["runtime"]["end_level"] in LEVEL_ORDER, "003",
         f"Invalid end_level: {cfg['runtime']['end_level']!r}")


def _validate_fields(cfg: dict[str, Any]) -> None:
    """Field-level checks — runs AFTER normalization fills defaults."""
    end = cfg["runtime"]["end_level"]
    win = lambda lv: LEVEL_ORDER[lv] <= LEVEL_ORDER[end]

    # window-based section requirements (physics/model always optional)
    if any(win(lv) for lv in ("L2", "L3", "L4")):
        for s in ("inputs", "sources"):
            _chk(s in cfg, "002", f"[{s}] required for window ..{end}")
    if any(win(lv) for lv in ("L3", "L4")):
        _chk("sopt" in cfg, "002", f"[sopt] required for window ..{end}")

    # checks
    c = cfg["checks"]
    _chk(isinstance(c.get("strict_mode"), bool), "003", "checks.strict_mode must be bool")
    _chk(_nonempty(c.get("eps_profile")), "003", "checks.eps_profile must be non-empty string")

    # paths
    output_root = cfg["paths"].get("output_root")
    _chk(_nonempty(output_root), "003", "paths.output_root must be a non-empty string")

    # model (optional; only RS supported)
    if "model" in cfg:
        _chk(cfg["model"].get("scheme") == "RS", "003", "model.scheme must be 'RS'")

    # physics (optional — can be inferred from disk)
    if "physics" in cfg:
        p = cfg["physics"]
        _chk("n_ele" in p, "002", "Missing: physics.n_ele")
        n = p["n_ele"]
        _chk(not isinstance(n, bool) and isinstance(n, int) and 1 <= n <= 13,
             "003", f"physics.n_ele must be int in [1,13], got {n!r}")
        _validate_f_ratios(p, "physics")
    for side in ("physics_nm1", "physics_np1"):
        if side in cfg:
            _validate_f_ratios(cfg[side], side)

    # sopt (optional)
    if "sopt" in cfg:
        s = cfg["sopt"]
        for k in ("U", "Jh", "zeta"):
            _chk(k in s and _num(s[k]), "003", f"sopt.{k}: required, numeric")
        if "offset" in s:
            _chk(_num(s["offset"]), "003", "sopt.offset must be numeric")
        if "energy_reference" in s:
            _chk(s["energy_reference"] in {"lsjm_ground", "zero"}, "003",
                 "sopt.energy_reference must be 'lsjm_ground' or 'zero'")

    # sources / inputs (field presence depends on window)
    if "sources" in cfg:
        src = cfg["sources"]
        if any(win(lv) for lv in ("L2", "L3", "L4")):
            _chk(_nonempty(src.get("hopping_name")), "003", "sources.hopping_name required")
        if win("L4"):
            _chk(_nonempty(src.get("kramer_name")), "003", "sources.kramer_name required")
    if "inputs" in cfg:
        inp = cfg["inputs"]
        if any(win(lv) for lv in ("L2", "L3", "L4")):
            _chk(_nonempty(inp.get("hopping_file")), "003", "inputs.hopping_file required")
        if win("L4"):
            _chk(_nonempty(inp.get("projector_file")), "003", "inputs.projector_file required")


def _validate_f_ratios(section: dict[str, Any], name: str) -> None:
    present = [k for k in _F_KEYS if k in section]
    if not present:
        return
    _chk(len(present) == 3, "003", f"F_ratios must appear together in [{name}]")
    for k in _F_KEYS:
        _chk(_num(section[k]), "003", f"{name}.{k} must be numeric")
    _chk(float(section["F2_ratio"]) != 0.0, "003", f"{name}.F2_ratio must not be zero")


# ── Normalization ──────────────────────────────────────────────────────

def _normalize(cfg: dict[str, Any]) -> None:
    """Units → sources → RE physics defaults → sopt defaults."""
    # units
    if "units" not in cfg:
        cfg["units"] = {"energy": "meV"}
    unit = cfg["units"].get("energy", "meV")
    _chk(unit in _UNIT_SCALE, "003", f"units.energy must be meV|eV, got {unit!r}")
    cfg["units"]["energy"] = unit
    scale = _UNIT_SCALE[unit]
    for sec in ("sopt", "sopt_nm1", "sopt_np1"):
        s = cfg.get(sec)
        if not isinstance(s, dict):
            continue
        for f in ("U", "Jh", "zeta", "offset"):
            if f in s and _num(s[f]):
                s[f] = float(s[f]) * scale

    # sources: canonical names from inputs fallbacks
    inputs = cfg.get("inputs", {})
    src = cfg.setdefault("sources", {})
    hn = _first(src.get("hopping_name"), src.get("hopping_label"),
                inputs.get("hopping_label"), _stem(inputs.get("hopping_file")))
    if hn:
        src["hopping_name"] = hn
        src.setdefault("hopping_label", hn)
    kn = _first(src.get("kramer_name"), src.get("projection_label"),
                inputs.get("projection_label"), _stem(inputs.get("projector_file")))
    if kn:
        src["kramer_name"] = kn
        src.setdefault("projection_label", kn)

    # physics RE preset → F_ratio defaults
    p = cfg.get("physics")
    if isinstance(p, dict):
        re = _re(p.get("RE", "auto"))
        p["RE"] = re
        if re != "auto":
            d = RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re]]
            p.setdefault("F2_ratio", d["F2_per_Jh"])
            p.setdefault("F4_ratio", d["F4_per_Jh"])
            p.setdefault("F6_ratio", d["F6_per_Jh"])

    # sopt defaults: offset, energy_reference, RE-derived zeta
    sopt = cfg.get("sopt")
    if isinstance(sopt, dict):
        sopt.setdefault("offset", 0.0)
        sopt.setdefault("energy_reference", "lsjm_ground")
        if isinstance(p, dict):
            re = p.get("RE", "auto")
            if re != "auto":
                sopt.setdefault("zeta",
                                RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re]]["zeta"] * _RE_ENERGY_SCALE)


# ── Branch construction ───────────────────────────────────────────────

def _build_branches(cfg: dict[str, Any]) -> None:
    physics = cfg.get("physics")
    sopt = cfg.get("sopt")
    if not isinstance(physics, dict) or not isinstance(sopt, dict):
        return
    n = int(physics["n_ele"])
    r42, r62 = _resolve_r42_r62(physics)
    ms = {k: float(sopt[k]) for k in ("U", "Jh", "zeta")}
    ms["offset"] = float(sopt.get("offset", 0.0))
    bn = _make_branch(n, r42, r62, ms, name="n", re=physics.get("RE", "auto"))
    cfg["_branches"] = {
        "nm1": _build_side(cfg, "nm1", n - 1, bn),
        "n":   bn,
        "np1": _build_side(cfg, "np1", n + 1, bn),
    }


def _assemble_derived(cfg: dict[str, Any]) -> None:
    if "_branches" in cfg:
        cfg["_derived"] = dict(cfg["_branches"]["n"]["derived"])
    elif "physics" in cfg:
        f2 = float(cfg["physics"].get("F2_ratio", 0.0))
        if f2 != 0.0:
            p = cfg["physics"]
            cfg["_derived"] = {"r42": float(p["F4_ratio"]) / f2,
                               "r62": float(p["F6_ratio"]) / f2}


def _resolve_r42_r62(sec: dict[str, Any]) -> tuple[float, float]:
    """r42=F4/F2, r62=F6/F2 from F_ratio keys or RE preset."""
    if "F2_ratio" in sec:
        f2 = float(sec["F2_ratio"])
        return float(sec["F4_ratio"]) / f2, float(sec["F6_ratio"]) / f2
    re = _re(sec.get("RE", "auto"))
    _chk(re != "auto", "003", "Cannot resolve ratios: no F_ratios and RE=auto")
    d = RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re]]
    return d["F4_per_Jh"] / d["F2_per_Jh"], d["F6_per_Jh"] / d["F2_per_Jh"]


def _make_branch(n: int, r42: float, r62: float, sopt: dict[str, float],
                 *, name: str, re: str = "auto") -> dict[str, Any]:
    f2, f4, f6 = _slater(r42, r62, sopt["Jh"], name)
    return {
        "physics": {"n_ele": n, "RE": re, "F2": f2, "F4": f4, "F6": f6},
        "sopt": dict(sopt),
        "derived": {"r42": r42, "r62": r62},
    }


def _build_side(cfg: dict[str, Any], side: str, n: int,
                main: dict[str, Any]) -> dict[str, Any]:
    """Build nm1/np1 branch; missing values fall back to main."""
    po = cfg.get(f"physics_{side}") or {}
    so = cfg.get(f"sopt_{side}") or {}

    has_ratios = "F2_ratio" in po
    has_re = "RE" in po and _re(po["RE"]) != "auto"
    r42, r62 = (_resolve_r42_r62(po) if has_ratios or has_re
                else (main["derived"]["r42"], main["derived"]["r62"]))
    re = _re(po["RE"]) if has_re else str(main["physics"].get("RE", "auto"))

    ms = main["sopt"]
    ss = {
        "U": float(so.get("U", ms["U"])),
        "Jh": float(so.get("Jh", ms["Jh"])),
        "zeta": float(so.get("zeta", ms["zeta"])),
        "offset": float(so.get("offset", 0.0)),
    }
    if has_re and "zeta" not in so:
        ss["zeta"] = RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re]]["zeta"] * _RE_ENERGY_SCALE

    return _make_branch(n, r42, r62, ss, name=side, re=re)


def _slater(r42: float, r62: float, jh: float, name: str) -> tuple[float, float, float]:
    """F2, F4, F6 from Jh and derived ratios."""
    denom = 286.0 + 195.0 * r42 + 250.0 * r62
    _chk(denom != 0.0, "003", f"Jh denominator is zero in branch {name}")
    if jh == 0.0:
        return 0.0, 0.0, 0.0
    f2 = 6435.0 * jh / denom
    return f2, r42 * f2, r62 * f2


# ── Helpers ────────────────────────────────────────────────────────────

def _chk(cond: bool, code: str, msg: str, **ctx: Any) -> None:
    if not cond:
        raise InputError(f"FXE-INPUT-{code}", msg, **ctx)


def _num(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def _nonempty(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _first(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _stem(v: Any) -> str:
    return Path(v).stem if isinstance(v, str) and v.strip() else ""


def _re(v: Any) -> str:
    if not isinstance(v, str) or not v.strip() or v.strip().lower() == "auto":
        return "auto"
    c = v.strip().capitalize()
    _chk(c in RE_TO_N_ELE, "003", f"Unknown RE preset: {v!r}")
    return c
