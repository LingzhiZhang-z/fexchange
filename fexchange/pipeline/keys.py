"""
Key and small helper utilities shared by pipeline modules.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    """Extract hopping_name and kramer_name from cfg['sources']."""
    sources = cfg.get("sources", {})
    if not isinstance(sources, dict):
        return SourceNames("", "")
    return SourceNames(
        hopping_name=str(sources.get("hopping_name", "")),
        kramer_name=str(sources.get("kramer_name", "")),
    )


def level_key(level: str, *, n_ele: int, r42: float, r62: float, cfg: dict[str, Any]) -> str:
    sver = cfg.get("standard_version", STANDARD_VERSION)
    sn = extract_source_names(cfg)
    hopping_name = sn.hopping_name
    kramer_name = sn.kramer_name
    branch_sig = branch_signature(cfg, n_ele=n_ele)
    denom_sig = denominator_signature(cfg, n_ele=n_ele)
    if level == "LMSM":
        return f"LMSM|n={n_ele}|r42={fmt12(r42)}|r62={fmt12(r62)}|sv={sver}"
    if level == "LSJM":
        return f"LSJM|keyLMSM={level_key('LMSM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}|sv={sver}"
    if level == "L0":
        return f"L0|n={n_ele}|sv={sver}"
    if level == "L1":
        key = (
            f"L1|keyLSJM={level_key('LSJM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|fn_ground_subspace_id=soc_lowest_hunds_v1|sv={sver}"
        )
        if branch_sig:
            key += f"|branchsig={branch_sig}"
        return key
    if level == "L2":
        return (
            f"L2|keyL1={level_key('L1', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|hopping_name={hopping_name}"
            f"|sv={sver}"
        )
    if level == "L3":
        s = cfg["sopt"]
        key = (
            f"L3|keyL2={level_key('L2', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|U={fmt12(float(s['U']))}|Jh={fmt12(float(s['Jh']))}|z={fmt12(float(s['zeta']))}|sv={sver}"
        )
        if denom_sig:
            key += f"|denomsig={denom_sig}"
        return key
    if level == "L4":
        return (
            f"L4|keyL3={level_key('L3', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|kramer_name={kramer_name}"
            f"|sv={sver}"
        )
    raise InputError("FXE-INPUT-003", f"Unknown level for key generation: {level}")


def three_sectors(n_ele: int) -> list[int]:
    return [n_ele - 1, n_ele, n_ele + 1]


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
        physics = branch.get("physics")
        if isinstance(physics, dict) and int(physics.get("n_ele", -9999)) == n_ele:
            return branch
    return None


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
        "energy_reference": str(cfg.get("sopt", {}).get("energy_reference", "lsjm_ground")),
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
    physics = branch["physics"]
    sopt = branch["sopt"]
    return {
        "F2": fmt12(float(physics["F2"])),
        "F4": fmt12(float(physics["F4"])),
        "F6": fmt12(float(physics["F6"])),
        "U": fmt12(float(sopt["U"])),
        "Jh": fmt12(float(sopt["Jh"])),
        "zeta": fmt12(float(sopt["zeta"])),
        "offset": fmt12(float(sopt.get("offset", 0.0))),
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]
