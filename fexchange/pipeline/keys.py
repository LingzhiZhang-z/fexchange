"""
Key and small helper utilities shared by pipeline modules.

Artifact identity has two regimes (standards/en/05-io/05-00-IO.md §8):

* **Core** levels (``LMSM``/``LSJM``/``L0``) are content-addressed and shared
  across runs under ``output_root/core/``. Each sector's LMSM/LSJM is
  content-addressed by its own ``(n, r42, r62)``, so adjacent sectors
  ``f^(n-1)``/``f^(n+1)`` may use their own Slater ratios (e.g.
  neighboring-element presets).
* **Run-scoped** levels (``IONED``, ``L1``, ``L2``, ``L3``, ``spin12``) keep
  ``runtime.run_name`` in metadata for compatibility. ``IONED`` cache reuse is
  validated by its resolved single-ion input summary, not by this key; the other
  listed stages remain run-scoped.
* **Path-addressed run-scoped** stages (``ligand`` and ``L1/P``) live under the
  run anchor (``output_root/<run_name>/`` alongside ``IONED``/``L1/F``), not in
  the shared ``core/``. They are not keyed by :func:`level_key` (and are absent
  from ``_RUN_SCOPED_LEVELS``): reuse is decided purely by their deterministic
  path (SOC mode + sector for ``ligand``; p-transition token for ``L1/P``).
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

_RUN_SCOPED_LEVELS = frozenset({"IONED", "L1", "L2", "L3", "spin12"})


def level_key(level: str, *, n_ele: int, r42: float, r62: float, cfg: dict[str, Any]) -> str:
    """Artifact identity for *level* (05-00 §8).

    Core levels return a deterministic content tuple; run-scoped levels return
    ``runtime.run_name``. ``r42``/``r62`` are ignored for run-scoped levels.
    """
    sver = cfg.get("standard_version", STANDARD_VERSION)
    if level == "LMSM":
        return f"LMSM|n={n_ele}|r42={fmt12(r42)}|r62={fmt12(r62)}|sv={sver}"
    if level == "LSJM":
        return f"LSJM|keyLMSM={level_key('LMSM', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}|sv={sver}"
    if level == "L0":
        return f"L0|n={n_ele}|sv={sver}"
    if level in _RUN_SCOPED_LEVELS:
        return _run_scoped_key(cfg, level)
    if level == "L4":
        raise InputError("FXE-INPUT-003", "L4 is not a runtime artifact level; use L3")
    raise InputError("FXE-INPUT-003", f"Unknown level for key generation: {level}")


def _run_scoped_key(cfg: dict[str, Any], level: str) -> str:
    run_name = str(cfg.get("runtime", {}).get("run_name", "")).strip()
    if not run_name:
        raise InputError(
            "FXE-INPUT-003",
            f"runtime.run_name is required for run-scoped level {level}",
        )
    return run_name


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
    """Return the resolved per-sector branch payload for ``n_ele`` (denominators)."""
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


def sector_ratios(cfg: dict[str, Any], n_ele: int, *, default_r42: float, default_r62: float) -> tuple[float, float]:
    """Per-sector ``(r42, r62)`` ratios; adjacent sectors may differ from the main sector."""
    branch = sector_branch(cfg, n_ele)
    if not isinstance(branch, dict):
        return default_r42, default_r62
    derived = branch.get("derived")
    if not isinstance(derived, dict):
        return default_r42, default_r62
    return float(derived["r42"]), float(derived["r62"])
