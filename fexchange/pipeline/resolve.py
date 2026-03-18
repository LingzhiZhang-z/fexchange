"""
Core parameter resolution for runtime orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fexchange.io.disk import build_stage_path
from fexchange.pipeline.keys import CORE_TOKEN_RE, extract_source_names
from fexchange.utils.errors import InputError


def resolve_core_params(cfg: dict[str, Any]) -> tuple[int, float, float, str]:
    """Resolve (n, r42, r62, scheme) from physics or unique upstream core token."""
    physics = cfg.get("physics")
    if isinstance(physics, dict):
        if "_derived" not in cfg:
            raise InputError(
                "FXE-INPUT-003",
                "Missing derived ratio payload for physics section",
                expected={"field_path": "_derived"},
            )
        scheme = cfg.get("model", {}).get("scheme", "RS")
        return int(physics["n_ele"]), float(cfg["_derived"]["r42"]), float(cfg["_derived"]["r62"]), str(scheme)

    output_root = Path(cfg["paths"]["output_root"])
    core_root = output_root / "core"
    sn = extract_source_names(cfg)
    hopping_name = sn.hopping_name
    kramer_name = sn.kramer_name
    if not core_root.exists():
        raise InputError(
            "FXE-INPUT-003",
            "physics section missing and no outputs/core directory to infer core token",
            expected={"field_path": "physics"},
            paths={"core_root": str(core_root)},
        )

    # Always start from LMSM; no upstream preflight needed
    required: tuple[str, ...] = ()
    candidates: list[tuple[int, float, float, str]] = []

    for d in sorted(core_root.iterdir()):
        if not d.is_dir():
            continue
        match = CORE_TOKEN_RE.match(d.name)
        if match is None:
            continue
        n_ele = int(match.group("n"))
        r42 = float(match.group("r42"))
        r62 = float(match.group("r62"))
        scheme = match.group("scheme")
        ok = True
        for level in required:
            if level == "L0":
                continue
            if level == "L2":
                path = build_stage_path(
                    str(output_root),
                    "L2",
                    n=n_ele,
                    r42=r42,
                    r62=r62,
                    scheme=scheme,
                    hopping_name=hopping_name,
                )
            elif level == "L3":
                sopt = cfg["sopt"]
                path = build_stage_path(
                    str(output_root),
                    "L3",
                    n=n_ele,
                    r42=r42,
                    r62=r62,
                    scheme=scheme,
                    hopping_name=hopping_name,
                    U=float(sopt["U"]),
                    Jh=float(sopt["Jh"]),
                    zeta=float(sopt["zeta"]),
                )
            elif level == "L4":
                sopt = cfg["sopt"]
                path = build_stage_path(
                    str(output_root),
                    "L4",
                    n=n_ele,
                    r42=r42,
                    r62=r62,
                    scheme=scheme,
                    hopping_name=hopping_name,
                    U=float(sopt["U"]),
                    Jh=float(sopt["Jh"]),
                    zeta=float(sopt["zeta"]),
                    kramer_name=kramer_name,
                )
            else:
                path = build_stage_path(str(output_root), level, n=n_ele, r42=r42, r62=r62, scheme=scheme)
            if not path.exists():
                ok = False
                break
        if ok:
            candidates.append((n_ele, r42, r62, scheme))

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise InputError(
            "FXE-INPUT-003",
            "physics section missing and failed to infer unique core token from upstream artifacts",
            expected={"field_path": "physics"},
            actual={"required_upstream": list(required)},
            paths={"core_root": str(core_root)},
        )
    raise InputError(
        "FXE-INPUT-003",
        "physics section missing and multiple core tokens match required upstream artifacts",
        expected={"field_path": "physics"},
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
