"""
Coded exceptions with FXE-* error codes.

Spec reference: 00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.
"""

from __future__ import annotations

import json
import datetime
from typing import Any

from fexchange.utils.constants import RUN_SCHEMA_VERSION, STANDARD_VERSION

# ---------------------------------------------------------------------------
# Domain set (00-03 §3)
# ---------------------------------------------------------------------------
_VALID_DOMAINS = frozenset(
    {"INPUT", "SCHEMA", "BIND", "IO", "NUM", "PHYS", "W90", "RUNTIME"}
)


class FexchangeError(Exception):
    """Base coded exception for fexchange (00-03 §1)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        module: str = "",
        level: str = "",
        stage: str = "",
        op: str = "",
        run_id: str = "",
        key: str = "",
        schema_version: str = RUN_SCHEMA_VERSION,
        standard_version: str = STANDARD_VERSION,
        expected: dict[str, Any] | None = None,
        actual: dict[str, Any] | None = None,
        paths: dict[str, Any] | None = None,
    ) -> None:
        _validate_code(code)
        self.code = code
        self.message = message
        self.module = module
        self.level = level
        self.stage = stage
        self.op = op
        self.run_id = run_id
        self.key = key
        self.schema_version = schema_version
        self.standard_version = standard_version
        self.expected = expected or {}
        self.actual = actual or {}
        self.paths = paths or {}
        self.ts_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        super().__init__(f"[{code}] {message}")

    def payload(self) -> dict[str, Any]:
        """Return the full JSON-serialisable failure payload (00-03 §4)."""
        return {
            "code": self.code,
            "message": self.message,
            "module": self.module,
            "level": self.level,
            "stage": self.stage,
            "op": self.op,
            "run_id": self.run_id,
            "key": self.key,
            "schema_version": self.schema_version,
            "standard_version": self.standard_version,
            "expected": self.expected,
            "actual": self.actual,
            "paths": self.paths,
            "ts_utc": self.ts_utc,
        }

    def payload_json(self) -> str:
        return json.dumps(self.payload(), ensure_ascii=False)


def _validate_code(code: str) -> None:
    parts = code.split("-")
    if len(parts) != 3 or parts[0] != "FXE":
        raise ValueError(f"Invalid error code format: {code!r}")
    domain = parts[1]
    if domain not in _VALID_DOMAINS:
        raise ValueError(f"Unknown domain in error code: {domain!r}")


# ---------------------------------------------------------------------------
# Convenience subclasses for each domain (00-03 §5)
# ---------------------------------------------------------------------------
class InputError(FexchangeError):
    """FXE-INPUT-*: run input file/field/type errors."""


class SchemaError(FexchangeError):
    """FXE-SCHEMA-*: schema/version/payload errors."""


class BindError(FexchangeError):
    """FXE-BIND-*: basis_id / axis / order binding mismatches."""


class IOError_(FexchangeError):
    """FXE-IO-*: file path/hash/read-write failures."""


class NumError(FexchangeError):
    """FXE-NUM-*: numerical stability failures."""


class PhysError(FexchangeError):
    """FXE-PHYS-*: physical constraint violations."""


class W90Error(FexchangeError):
    """FXE-W90-*: Wannier90 parsing/mapping failures."""


class RuntimeError_(FexchangeError):
    """FXE-RUNTIME-*: uncategorised runtime failures."""
