"""
Key and small helper utilities shared by pipeline modules.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from fexchange.io.disk import fmt12
from fexchange.utils.constants import STANDARD_VERSION
from fexchange.utils.errors import InputError

CORE_TOKEN_RE = re.compile(
    r"^n-(?P<n>\d+)_r42-(?P<r42>[-+]?\d+(?:\.\d+)?)_r62-(?P<r62>[-+]?\d+(?:\.\d+)?)_scheme-(?P<scheme>.+)$"
)


class SourceNames(NamedTuple):
    hopping_name: str
    kramer_name: str


def extract_source_names(cfg: dict[str, Any]) -> SourceNames:
    """Extract hopping_name and kramer_name from normalized runtime/input tables."""
    inputs = cfg.get("inputs", {})
    runtime = cfg.get("runtime", {})
    sources = cfg.get("sources", {})
    if not isinstance(inputs, dict):
        inputs = {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(sources, dict):
        sources = {}
    return SourceNames(
        hopping_name=str(inputs.get("hopping_name", sources.get("hopping_name", ""))),
        kramer_name=str(runtime.get("kramer_name", inputs.get("projector_name", sources.get("kramer_name", "")))),
    )


def projector_content_signature(cfg: dict[str, Any]) -> str:
    """Return a short stable digest for the runtime projector file, if present."""
    inputs = cfg.get("inputs", {})
    if not isinstance(inputs, dict):
        return ""
    projector_file = inputs.get("projector_file")
    if not projector_file:
        return ""
    try:
        return hashlib.sha1(Path(str(projector_file)).read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def hopping_content_signature(cfg: dict[str, Any]) -> str:
    """Return a short stable digest for the runtime hopping file, if present."""
    inputs = cfg.get("inputs", {})
    if not isinstance(inputs, dict):
        return ""
    hopping_file = inputs.get("hopping_file")
    if not hopping_file:
        return ""
    try:
        return hashlib.sha1(Path(str(hopping_file)).read_bytes()).hexdigest()[:12]
    except OSError:
        return ""


def energy_unit_signature(cfg: dict[str, Any]) -> str:
    """Normalized energy unit used for scalar parameters and hopping inputs."""
    units = cfg.get("units", {})
    if not isinstance(units, dict):
        return "meV"
    return str(units.get("energy", "meV"))


def level_key(level: str, *, n_ele: int, r42: float, r62: float, cfg: dict[str, Any]) -> str:
    sver = cfg.get("standard_version", STANDARD_VERSION)
    branch = str(cfg.get("runtime", {}).get("branch", "sopt"))
    scheme = _model_scheme(cfg)
    sn = extract_source_names(cfg)
    hopping_name = sn.hopping_name
    kramer_name = sn.kramer_name
    hopping_sig = hopping_content_signature(cfg)
    energy_unit = energy_unit_signature(cfg)
    projector_sig = projector_content_signature(cfg)
    branch_sig = branch_signature(cfg, n_ele=n_ele)
    denom_sig = denominator_signature(cfg, n_ele=n_ele)
    if level == "LMSM":
        return f"LMSM|n={n_ele}|r42={fmt12(r42)}|r62={fmt12(r62)}|sv={sver}"
    if level == "LSJM":
        return f"LSJM|keyLMSM={level_key('LMSM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}|sv={sver}"
    if level == "L0":
        return f"L0|branch={branch}|n={n_ele}|sv={sver}"
    if level == "IONED":
        fsite = _sector_fsite_for_key(cfg, n_ele)
        return (
            f"IONED|n={n_ele}"
            f"|U={fmt12(float(fsite.get('U', 0.0)))}"
            f"|Jh={fmt12(float(fsite.get('Jh', 0.0)))}"
            f"|F2={fmt12(float(fsite.get('F2', 0.0)))}"
            f"|F4={fmt12(float(fsite.get('F4', 0.0)))}"
            f"|F6={fmt12(float(fsite.get('F6', 0.0)))}"
            f"|z={fmt12(float(fsite.get('zeta', 0.0)))}"
            f"|offset={fmt12(float(fsite.get('offset', 0.0)))}"
            f"|scheme=ED|sv={sver}"
        )
    if level == "L1":
        if scheme == "ED":
            key = (
                f"L1|branch={branch}|scheme=ED"
                f"|keyIONED_np1={level_key('IONED', n_ele=n_ele + 1, r42=r42, r62=r62, cfg=cfg)}"
                f"|keyLSJM_n={level_key('LSJM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
                f"|keyIONED_nm1={level_key('IONED', n_ele=n_ele - 1, r42=r42, r62=r62, cfg=cfg)}"
                f"|fn_ground_subspace_id=soc_lowest_hunds_v1|sv={sver}"
            )
        else:
            key = (
                f"L1|branch={branch}|scheme=RS"
                f"|keyLSJM={level_key('LSJM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
                f"|fn_ground_subspace_id=soc_lowest_hunds_v1|sv={sver}"
            )
            if branch_sig:
                key += f"|branchsig={branch_sig}"
        return key
    if level == "L2":
        key = (
            f"L2|branch={branch}|keyL1={level_key('L1', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|hopping_name={hopping_name}"
            f"|energy_unit={energy_unit}"
        )
        if hopping_sig:
            key += f"|hopping_sha1={hopping_sig}"
        return f"{key}|sv={sver}"
    if level == "L3":
        s = cfg["fsite"]
        key = f"L3|branch={branch}|keyL2={level_key('L2', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
        if branch == "fopt":
            ligands = cfg.get("ligand", {})
            lig1 = ligands.get("1", {}) if isinstance(ligands, dict) else {}
            lig2 = ligands.get("2", {}) if isinstance(ligands, dict) else {}
            fopt_key = (
                f"{key}|U={fmt12(float(s['U']))}|Jh={fmt12(float(s['Jh']))}"
                f"|lig1.U_p={fmt12(float(lig1.get('U_p', 0.0)))}"
                f"|lig2.U_p={fmt12(float(lig2.get('U_p', 0.0)))}"
            )
            if kramer_name:
                fopt_key += f"|kramer_name={kramer_name}"
            if projector_sig:
                fopt_key += f"|projector_sha1={projector_sig}"
            return f"{fopt_key}|sv={sver}"
        key = (
            f"{key}"
            f"|U={fmt12(float(s['U']))}|Jh={fmt12(float(s['Jh']))}|z={fmt12(float(s['zeta']))}|sv={sver}"
        )
        if denom_sig:
            key += f"|denomsig={denom_sig}"
        return key
    if level == "L4":
        key = (
            f"L4|branch={branch}|keyL3={level_key('L3', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|kramer_name={kramer_name}"
        )
        if projector_sig:
            key += f"|projector_sha1={projector_sig}"
        return f"{key}|sv={sver}"
    raise InputError("FXE-INPUT-003", f"Unknown level for key generation: {level}")


def three_sectors(n_ele: int) -> list[int]:
    return [n_ele - 1, n_ele, n_ele + 1]


def _model_scheme(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model", {}).get("scheme", "RS")).upper()


def labels_abcd_lex(n_k: int) -> NDArray[np.int64]:
    rows: list[list[int]] = []
    for a in range(n_k):
        for b in range(n_k):
            for c in range(n_k):
                for d in range(n_k):
                    rows.append([a, b, c, d])
    return np.asarray(rows, dtype=np.int64)


def sector_branch(cfg: dict[str, Any], n_ele: int) -> dict[str, Any] | None:
    branches = cfg.get("_branches")
    if not isinstance(branches, dict):
        return None
    for branch in branches.values():
        if not isinstance(branch, dict):
            continue
        fsite = branch.get("fsite")
        if isinstance(fsite, dict) and int(fsite.get("n_ele", -9999)) == n_ele:
            return branch
    return None


def _sector_fsite_for_key(cfg: dict[str, Any], n_ele: int) -> dict[str, Any]:
    branch = sector_branch(cfg, n_ele)
    if isinstance(branch, dict) and isinstance(branch.get("fsite"), dict):
        return branch["fsite"]
    fsite: dict[str, Any] = {}
    physics = cfg.get("physics", {})
    if isinstance(physics, dict):
        fsite.update(physics)
    main = cfg.get("fsite", {})
    if isinstance(main, dict):
        fsite.update(main)
    fsite.setdefault("n_ele", n_ele)
    for key in ("U", "Jh", "F2", "F4", "F6", "zeta", "offset"):
        fsite.setdefault(key, 0.0)
    return fsite


def sector_ratios(cfg: dict[str, Any], n_ele: int, *, default_r42: float, default_r62: float) -> tuple[float, float]:
    branch = sector_branch(cfg, n_ele)
    if not isinstance(branch, dict):
        return default_r42, default_r62
    derived = branch.get("derived")
    if not isinstance(derived, dict):
        return default_r42, default_r62
    return float(derived["r42"]), float(derived["r62"])


def branch_signature(cfg: dict[str, Any], *, n_ele: int) -> str:
    branch_n = sector_branch(cfg, n_ele)
    branch_nm1 = sector_branch(cfg, n_ele - 1)
    branch_np1 = sector_branch(cfg, n_ele + 1)
    if not all(isinstance(branch, dict) for branch in (branch_n, branch_nm1, branch_np1)):
        return ""

    main_derived = branch_n["derived"]
    legacy = {
        "r42": fmt12(float(main_derived["r42"])),
        "r62": fmt12(float(main_derived["r62"])),
    }
    actual = {
        "nm1": _ratio_payload(branch_nm1),
        "np1": _ratio_payload(branch_np1),
    }
    if actual["nm1"] == legacy and actual["np1"] == legacy:
        return ""
    return f"branchsig-{_stable_hash(actual)}"


def denominator_signature(cfg: dict[str, Any], *, n_ele: int) -> str:
    branch_n = sector_branch(cfg, n_ele)
    branch_nm1 = sector_branch(cfg, n_ele - 1)
    branch_np1 = sector_branch(cfg, n_ele + 1)
    if not all(isinstance(branch, dict) for branch in (branch_n, branch_nm1, branch_np1)):
        return ""

    legacy_branch = _denominator_payload(branch_n)
    legacy_branch["offset"] = fmt12(0.0)
    legacy = {
        "energy_reference": "zero",
        "n": legacy_branch,
        "nm1": legacy_branch,
        "np1": legacy_branch,
    }
    actual = {
        "energy_reference": str(cfg.get("fsite", {}).get("energy_reference", "lsjm_ground")),
        "n": _denominator_payload(branch_n),
        "nm1": _denominator_payload(branch_nm1),
        "np1": _denominator_payload(branch_np1),
    }
    if actual == legacy:
        return ""
    return f"denomsig-{_stable_hash(actual)}"


def _ratio_payload(branch: dict[str, Any]) -> dict[str, str]:
    derived = branch["derived"]
    return {
        "r42": fmt12(float(derived["r42"])),
        "r62": fmt12(float(derived["r62"])),
    }


def _denominator_payload(branch: dict[str, Any]) -> dict[str, str]:
    fsite = branch["fsite"]
    return {
        "F2": fmt12(float(fsite["F2"])),
        "F4": fmt12(float(fsite["F4"])),
        "F6": fmt12(float(fsite["F6"])),
        "U": fmt12(float(fsite["U"])),
        "Jh": fmt12(float(fsite["Jh"])),
        "zeta": fmt12(float(fsite["zeta"])),
        "offset": fmt12(float(fsite.get("offset", 0.0))),
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
