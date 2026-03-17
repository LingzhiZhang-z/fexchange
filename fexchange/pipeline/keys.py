"""
Key and small helper utilities shared by pipeline modules.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.io.disk import fmt12
from fexchange.utils.constants import STANDARD_VERSION
from fexchange.utils.errors import InputError

CORE_TOKEN_RE = re.compile(
    r"^n-(?P<n>\d+)_r42-(?P<r42>[-+]?\d+(?:\.\d+)?)_r62-(?P<r62>[-+]?\d+(?:\.\d+)?)_scheme-(?P<scheme>.+)$"
)


def level_key(level: str, *, n_ele: int, r42: float, r62: float, cfg: dict[str, Any]) -> str:
    sver = cfg.get("standard_version", STANDARD_VERSION)
    sources = cfg.get("sources", {})
    hopping_name = str(sources.get("hopping_name", "")) if isinstance(sources, dict) else ""
    kramer_name = str(sources.get("kramer_name", "")) if isinstance(sources, dict) else ""
    if level == "LMSM":
        return f"LMSM|n={n_ele}|r42={fmt12(r42)}|r62={fmt12(r62)}|sv={sver}"
    if level == "LSJM":
        return f"LSJM|keyLMSM={level_key('LMSM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}|sv={sver}"
    if level == "L0":
        return f"L0|n={n_ele}|sv={sver}"
    if level == "L1":
        return (
            f"L1|keyLSJM={level_key('LSJM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|fn_ground_subspace_id=soc_lowest_hunds_v1|sv={sver}"
        )
    if level == "L2":
        return (
            f"L2|keyL1={level_key('L1', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|hopping_name={hopping_name}"
            f"|sv={sver}"
        )
    if level == "L3":
        s = cfg["sopt"]
        return (
            f"L3|keyL2={level_key('L2', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}"
            f"|U={fmt12(float(s['U']))}|Jh={fmt12(float(s['Jh']))}|z={fmt12(float(s['zeta']))}|sv={sver}"
        )
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
