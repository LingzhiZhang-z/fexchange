"""TOML run-input loader.  Spec: 05-04-RUN_INPUT."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fexchange.utils.constants import (
    BRANCH_TERMINAL,
    BRANCHES,
    LEVEL_ORDER,
    RE_DEFAULTS_BY_N_ELE,
    RE_TO_N_ELE,
    RUN_SCHEMA_VERSION,
    STANDARD_VERSION,
)
from fexchange.utils.errors import InputError, SchemaError

logger = logging.getLogger("fexchange")

_F_KEYS = ("F2_ratio", "F4_ratio", "F6_ratio")
_ENERGY_FIELDS = ("U", "Jh", "zeta", "offset", "Uplus", "Uminus")
_LIGAND_FIELDS = ("Delta", "U_p", "lambda_p")
_ALLOWED = frozenset({
    "schema_version",
    "standard_version",
    "run_id",
    "title",
    "units",
    "fsite",
    "fsite_nm1",
    "fsite_np1",
    "model",
    "ligand",
    "inputs",
    "paths",
    "runtime",
    "checks",
    "_derived",
    "_branches",
})


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

    logger.info(
        "Loaded run_id=%s branch=%s window=LMSM->%s",
        cfg.get("run_id"),
        cfg["runtime"]["branch"],
        cfg["runtime"]["end_level"],
    )
    return cfg


def window_includes(cfg: dict[str, Any], level: str) -> bool:
    """True if the execution window includes *level*."""
    return LEVEL_ORDER.get(level, 99) <= LEVEL_ORDER.get(
        cfg.get("runtime", {}).get("end_level", ""), 0
    )


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


def _validate_structure(cfg: dict[str, Any]) -> None:
    for k in ("schema_version", "standard_version", "run_id", "title"):
        _chk(k in cfg, "002", f"Missing: {k}")
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
    for k in ("paths", "runtime", "checks"):
        _chk(k in cfg, "002", f"Missing section: [{k}]")
    extras = sorted(k for k in cfg if k not in _ALLOWED)
    _chk(not extras, "003", f"Unknown sections: {extras}")

    runtime = cfg["runtime"]
    _chk("branch" in runtime, "002", "Missing: runtime.branch")
    _chk(runtime["branch"] in BRANCHES, "003", f"Invalid runtime.branch: {runtime['branch']!r}")
    _chk("end_level" in runtime, "002", "Missing: runtime.end_level")
    _chk(runtime["end_level"] in LEVEL_ORDER, "003", f"Invalid end_level: {runtime['end_level']!r}")
    terminal = BRANCH_TERMINAL[str(runtime["branch"])]
    _chk(
        LEVEL_ORDER[runtime["end_level"]] <= LEVEL_ORDER[terminal],
        "003",
        f"runtime.end_level={runtime['end_level']} exceeds branch terminal {terminal}",
    )


def _validate_fields(cfg: dict[str, Any]) -> None:
    runtime = cfg["runtime"]
    branch = str(runtime["branch"])
    end = str(runtime["end_level"])
    win = lambda lv: LEVEL_ORDER[lv] <= LEVEL_ORDER[end]

    _chk("fsite" in cfg, "002", "[fsite] required")
    _validate_fsite(cfg["fsite"], "fsite", require_core=True)
    for side in ("fsite_nm1", "fsite_np1"):
        if side in cfg:
            _validate_fsite(cfg[side], side, require_core=False)

    c = cfg["checks"]
    _chk(isinstance(c.get("strict_mode"), bool), "003", "checks.strict_mode must be bool")
    _chk(_nonempty(c.get("eps_profile")), "003", "checks.eps_profile must be non-empty string")

    output_root = cfg["paths"].get("output_root")
    _chk(_nonempty(output_root), "003", "paths.output_root must be a non-empty string")

    if "model" not in cfg:
        cfg["model"] = {"scheme": "RS"}
    scheme = str(cfg["model"].get("scheme", "RS")).upper()
    _chk(scheme in {"RS", "ED"}, "003", "model.scheme must be 'RS' or 'ED'")
    cfg["model"]["scheme"] = scheme

    if scheme == "ED" and win("L1"):
        _chk(_nonempty(runtime.get("run_name")), "003", "runtime.run_name required for ED L1+")

    if win("L2"):
        _chk(_nonempty(runtime.get("run_name")), "003", "runtime.run_name required for L2+")
        _chk("inputs" in cfg, "002", f"[inputs] required for window ..{end}")
        _chk(_nonempty(cfg["inputs"].get("hopping_file")), "003", "inputs.hopping_file required")
    if branch == "sopt" and win("L3"):
        _chk(_nonempty(runtime.get("kramer_name")), "003", "runtime.kramer_name required for sopt L3")
        _chk(_nonempty(cfg.get("inputs", {}).get("projector_file")), "003", "inputs.projector_file required")
    if branch == "fopt":
        _validate_ligands(cfg)
        if win("L3"):
            _chk(_nonempty(runtime.get("kramer_name")), "003", "runtime.kramer_name required for fopt L3")
            _chk(_nonempty(cfg.get("inputs", {}).get("projector_file")), "003", "inputs.projector_file required")


def _validate_fsite(section: dict[str, Any], name: str, *, require_core: bool) -> None:
    _chk(isinstance(section, dict), "003", f"[{name}] must be a table")
    if require_core:
        _chk("n_ele" in section, "002", f"Missing: {name}.n_ele")
        n = section["n_ele"]
        _chk(
            not isinstance(n, bool) and isinstance(n, int) and 1 <= n <= 13,
            "003",
            f"{name}.n_ele must be int in [1,13], got {n!r}",
        )
    present = [k for k in _F_KEYS if k in section]
    if require_core:
        has_re = _re(section.get("RE", "auto")) != "auto"
        _chk(len(present) == 3 or has_re, "003", f"[{name}] requires F ratios or RE preset")
    if present:
        for k in present:
            _chk(_num(section[k]), "003", f"{name}.{k} must be numeric")
        if "F2_ratio" in section:
            _chk(float(section["F2_ratio"]) != 0.0, "003", f"{name}.F2_ratio must not be zero")
    for k in _ENERGY_FIELDS:
        if k in section:
            _chk(_num(section[k]), "003", f"{name}.{k} must be numeric")
    if name == "fsite":
        _chk("Uplus" not in section and "Uminus" not in section, "003", "Uplus/Uminus are only valid in side fsite branches")
    elif name == "fsite_np1":
        _chk("Uminus" not in section, "003", "fsite_np1 accepts Uplus, not Uminus")
        _chk(
            not ("Uplus" in section and "offset" in section),
            "003",
            "fsite_np1.Uplus conflicts with fsite_np1.offset",
        )
    elif name == "fsite_nm1":
        _chk("Uplus" not in section, "003", "fsite_nm1 accepts Uminus, not Uplus")
        _chk(
            not ("Uminus" in section and "offset" in section),
            "003",
            "fsite_nm1.Uminus conflicts with fsite_nm1.offset",
        )
    if require_core:
        for k in ("U", "Jh"):
            _chk(k in section and _num(section[k]), "003", f"{name}.{k}: required, numeric")
    if "energy_reference" in section:
        _chk(
            section["energy_reference"] in {"lsjm_ground", "zero"},
            "003",
            f"{name}.energy_reference must be 'lsjm_ground' or 'zero'",
        )


def _validate_ligands(cfg: dict[str, Any]) -> None:
    ligands = cfg.get("ligand")
    _chk(isinstance(ligands, dict), "002", "[ligand.1] and [ligand.2] required for fopt")
    for idx in ("1", "2"):
        lig = ligands.get(idx)
        _chk(isinstance(lig, dict), "002", f"[ligand.{idx}] required for fopt")
        for key in ("Delta", "U_p"):
            _chk(key in lig and _num(lig[key]), "003", f"ligand.{idx}.{key}: required, numeric")
        if "lambda_p" in lig:
            _chk(_num(lig["lambda_p"]), "003", f"ligand.{idx}.lambda_p must be numeric")


def _normalize(cfg: dict[str, Any]) -> None:
    units = cfg.get("units")
    if units is None:
        cfg["units"] = {"energy": "raw"}
    else:
        _chk(isinstance(units, dict), "003", "[units] must be a table")
        unit = units.get("energy", "raw")
        _chk(isinstance(unit, str) and bool(unit.strip()), "003", "units.energy must be a non-empty string")
        units["energy"] = unit

    for sec in ("fsite", "fsite_nm1", "fsite_np1"):
        s = cfg.get(sec)
        if not isinstance(s, dict):
            continue
        for f in _ENERGY_FIELDS:
            if f in s and _num(s[f]):
                s[f] = float(s[f])
    ligands = cfg.get("ligand")
    if isinstance(ligands, dict):
        for lig in ligands.values():
            if not isinstance(lig, dict):
                continue
            for f in _LIGAND_FIELDS:
                if f in lig and _num(lig[f]):
                    lig[f] = float(lig[f])
            lig.setdefault("lambda_p", 0.0)

    fsite = cfg.get("fsite")
    if isinstance(fsite, dict):
        _normalize_fsite_defaults(fsite)
        fsite.setdefault("offset", 0.0)
        fsite.setdefault("energy_reference", "lsjm_ground")


def _normalize_fsite_defaults(section: dict[str, Any]) -> None:
    re = _re(section.get("RE", "auto"))
    section["RE"] = re
    if re != "auto":
        d = RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re]]
        section.setdefault("F2_ratio", d["F2_per_Jh"])
        section.setdefault("F4_ratio", d["F4_per_Jh"])
        section.setdefault("F6_ratio", d["F6_per_Jh"])
        section.setdefault("zeta", d["zeta"])


def _build_branches(cfg: dict[str, Any]) -> None:
    fsite = cfg.get("fsite")
    if not isinstance(fsite, dict):
        return
    n = int(fsite["n_ele"])
    r42, r62 = _resolve_r42_r62(fsite)
    main = _resolve_main_fsite(fsite, r42, r62, name="n")
    cfg["_branches"] = {
        "nm1": _build_side(cfg, "nm1", n - 1, main),
        "n": main,
        "np1": _build_side(cfg, "np1", n + 1, main),
    }


def _assemble_derived(cfg: dict[str, Any]) -> None:
    if "_branches" in cfg:
        cfg["_derived"] = dict(cfg["_branches"]["n"]["derived"])


def _resolve_main_fsite(section: dict[str, Any], r42: float, r62: float, *, name: str) -> dict[str, Any]:
    payload = dict(section)
    payload["offset"] = float(payload.get("offset", 0.0))
    payload["energy_reference"] = str(payload.get("energy_reference", "lsjm_ground"))
    f2, f4, f6 = _slater(r42, r62, float(payload["Jh"]), name)
    payload.update({"F2": f2, "F4": f4, "F6": f6, "r42": r42, "r62": r62})
    return {"fsite": payload, "derived": {"r42": r42, "r62": r62}}


def _build_side(cfg: dict[str, Any], side: str, n: int, main: dict[str, Any]) -> dict[str, Any]:
    override = dict(cfg.get(f"fsite_{side}") or {})
    if "RE" in override:
        _normalize_fsite_defaults(override)
    main_fsite = main["fsite"]
    merged = dict(main_fsite)
    merged.update(override)
    merged["n_ele"] = n
    if "offset" not in override:
        merged["offset"] = 0.0
    r42, r62 = _resolve_r42_r62(merged)
    return _resolve_main_fsite(merged, r42, r62, name=side)


def _resolve_r42_r62(sec: dict[str, Any]) -> tuple[float, float]:
    if all(k in sec for k in _F_KEYS):
        f2 = float(sec["F2_ratio"])
        _chk(f2 != 0.0, "003", "F2_ratio must not be zero")
        return float(sec["F4_ratio"]) / f2, float(sec["F6_ratio"]) / f2
    re = _re(sec.get("RE", "auto"))
    _chk(re != "auto", "003", "Cannot resolve ratios: no F_ratios and RE=auto")
    d = RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re]]
    return d["F4_per_Jh"] / d["F2_per_Jh"], d["F6_per_Jh"] / d["F2_per_Jh"]


def _slater(r42: float, r62: float, jh: float, name: str) -> tuple[float, float, float]:
    denom = 286.0 + 195.0 * r42 + 250.0 * r62
    _chk(denom != 0.0, "003", f"Jh denominator is zero in branch {name}")
    if jh == 0.0:
        return 0.0, 0.0, 0.0
    f2 = 6435.0 * jh / denom
    return f2, r42 * f2, r62 * f2


def _chk(cond: bool, code: str, msg: str, **ctx: Any) -> None:
    if not cond:
        raise InputError(f"FXE-INPUT-{code}", msg, **ctx)


def _num(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float))


def _nonempty(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _re(v: Any) -> str:
    if not isinstance(v, str) or not v.strip() or v.strip().lower() == "auto":
        return "auto"
    c = v.strip().capitalize()
    _chk(c in RE_TO_N_ELE, "003", f"Unknown RE preset: {v!r}")
    return c
