"""
Core parameter resolution for runtime orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fexchange.pipeline.keys import CORE_TOKEN_RE
from fexchange.utils.errors import InputError


def resolve_core_params(cfg: dict[str, Any]) -> tuple[int, float, float, str]:
    """Resolve (n, r42, r62, scheme) from physics or unique upstream core token."""
    fsite = cfg.get("fsite")
    if isinstance(fsite, dict):
        if "_derived" not in cfg:
            raise InputError(
                "FXE-INPUT-003",
                "Missing derived ratio payload for fsite section",
                expected={"field_path": "_derived"},
            )
        scheme = cfg.get("model", {}).get("scheme", "RS")
        return int(fsite["n_ele"]), float(cfg["_derived"]["r42"]), float(cfg["_derived"]["r62"]), str(scheme)

    output_root = Path(cfg["paths"]["output_root"])
    core_root = output_root / "core"
    if not core_root.exists():
        raise InputError(
            "FXE-INPUT-003",
            "fsite section missing and no outputs/core directory to infer core token",
            expected={"field_path": "fsite"},
            paths={"core_root": str(core_root)},
        )

    candidates: list[tuple[int, float, float, str]] = []

    for d in sorted((core_root / "LMSM").iterdir() if (core_root / "LMSM").exists() else []):
        if not d.is_dir():
            continue
        match = CORE_TOKEN_RE.match(d.name)
        if match is None:
            continue
        n_ele = int(match.group("n"))
        r42 = float(match.group("r42"))
        r62 = float(match.group("r62"))
        scheme = match.group("scheme")
        candidates.append((n_ele, r42, r62, scheme))

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise InputError(
            "FXE-INPUT-003",
            "fsite section missing and failed to infer unique core token from upstream artifacts",
            expected={"field_path": "fsite"},
            paths={"core_root": str(core_root)},
        )
    raise InputError(
        "FXE-INPUT-003",
        "fsite section missing and multiple core tokens match required upstream artifacts",
        expected={"field_path": "fsite"},
        actual={
            "candidate_tokens": [
                {
                    "n": n_ele,
                    "r42": r42,
                    "r62": r62,
                    "scheme": scheme,
                }
                for (n_ele, r42, r62, scheme) in candidates
            ]
        },
        paths={"core_root": str(core_root)},
    )
