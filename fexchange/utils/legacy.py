"""
Legacy isolation helpers.

Spec reference: 00-04-LEGACY_ISOLATION.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fexchange.utils.errors import LegacyError


def is_legacy_path(path: str | Path) -> bool:
    """
    Lexically detect whether a path is legacy-tagged (00-04 §2).

    Rules:
    - basename contains ``_LEGACY``
    - basename ends with ``.legacy``
    - any path component equals ``legacy``
    """
    p = Path(path)
    name = p.name
    if "_LEGACY" in name:
        return True
    if name.endswith(".legacy"):
        return True
    for part in p.parts:
        if part.lower() == "legacy":
            return True
    return False


def ensure_non_legacy_path(
    path: str | Path,
    *,
    allow_legacy: bool = False,
    legacy_scope: Iterable[str] | None = None,
    legacy_reason: str | None = None,
    legacy_ticket: str | None = None,
    for_artifact: bool = False,
) -> None:
    """
    Enforce default legacy read block with explicit override (00-04 §3-4).

    Parameters
    ----------
    path:
        Target path that may be loaded/read.
    allow_legacy:
        Explicit legacy override flag.
    legacy_scope:
        Allowlist of paths permitted in legacy mode.
    legacy_reason:
        Human-readable reason for override.
    legacy_ticket:
        Traceability id for override.
    for_artifact:
        If True, emits artifact-load code ``FXE-LEGACY-002``;
        otherwise spec-read code ``FXE-LEGACY-001``.
    """
    target = str(path)
    if not is_legacy_path(target):
        return

    code = "FXE-LEGACY-002" if for_artifact else "FXE-LEGACY-001"
    scope = list(legacy_scope or [])

    if not allow_legacy:
        raise LegacyError(
            code,
            f"Legacy path blocked by default policy: {target}",
            actual={"requested_path": target, "mode_flags": {"allow_legacy": False}},
            paths={"path": target},
        )

    if not scope or not legacy_reason or not legacy_ticket:
        raise LegacyError(
            code,
            f"Legacy override is incomplete for path: {target}",
            expected={
                "allow_legacy": True,
                "legacy_scope": "non-empty allowlist",
                "legacy_reason": "non-empty string",
                "legacy_ticket": "non-empty string",
            },
            actual={
                "allow_legacy": allow_legacy,
                "legacy_scope": scope,
                "legacy_reason": legacy_reason,
                "legacy_ticket": legacy_ticket,
                "requested_path": target,
            },
            paths={"path": target},
        )

    norm_target = str(Path(target))
    norm_scope = {str(Path(s)) for s in scope}
    if norm_target not in norm_scope:
        raise LegacyError(
            code,
            f"Legacy path not in legacy_scope allowlist: {target}",
            expected={"legacy_scope": sorted(norm_scope)},
            actual={"requested_path": norm_target, "mode_flags": {"allow_legacy": True}},
            paths={"path": target},
        )
