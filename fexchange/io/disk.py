"""
Disk I/O contract: path tokens, read-first cache, atomic write, metadata.

Spec reference: 05-00-IO.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.errors import IOError_, BindError, SchemaError

logger = logging.getLogger("fexchange")

META_REQUIRED_FIELDS = {
    "schema_version",
    "module",
    "level",
    "key",
    "inputs_summary",
    "tensor_name",
    "physical_meaning",
    "basis_id",
    "index_definition",
    "unit",
    "storage_layout",
    "dtype",
    "logical_shape",
    "payload_files",
    "created_at",
}


# ---------------------------------------------------------------------------
# Path-token helpers (05-00 §3)
# ---------------------------------------------------------------------------

def fmt12(x: float) -> str:
    """Fixed-point 12-decimal formatting."""
    return f"{x:.12f}"


def core_dir_token(n: int, r42: float, r62: float, scheme: str = "RS") -> str:
    """Core path token: n-{n}_r42-{r42}_r62-{r62}_scheme-{scheme}."""
    return f"n-{n}_r42-{fmt12(r42)}_r62-{fmt12(r62)}_scheme-{scheme}"


def ujhz_dir_token(U: float, Jh: float, zeta: float) -> str:
    """U-Jh-zeta path token."""
    return f"U-{fmt12(U)}_Jh-{fmt12(Jh)}_z-{fmt12(zeta)}"


def build_stage_path(
    output_root: str,
    stage: str,
    *,
    n: int = 0,
    r42: float = 0.0,
    r62: float = 0.0,
    scheme: str = "RS",
    hopping_name: str = "",
    U: float = 0.0,
    Jh: float = 0.0,
    zeta: float = 0.0,
    kramer_name: str = "",
) -> Path:
    """Build deterministic artifact path for a given stage (05-00 §2-3)."""
    root = Path(output_root)
    core = core_dir_token(n, r42, r62, scheme)

    if stage == "fock":
        return root / "fock"
    elif stage in ("LMSM", "LSJM"):
        return root / "core" / core / stage
    elif stage == "L0":
        return root / "fock"
    elif stage == "L1":
        return root / "core" / core / "L1"
    elif stage == "L2":
        return root / "core" / core / "hopping" / hopping_name / "L2"
    elif stage == "L3":
        return root / "core" / core / "hopping" / hopping_name / ujhz_dir_token(U, Jh, zeta) / "L3"
    elif stage == "L4":
        return root / "core" / core / "hopping" / hopping_name / ujhz_dir_token(U, Jh, zeta) / "kramer" / kramer_name / "L4"
    elif stage == "spin12":
        return root / "core" / core / "hopping" / hopping_name / ujhz_dir_token(U, Jh, zeta) / "kramer" / kramer_name / "spin12"
    else:
        raise IOError_(
            "FXE-IO-001",
            f"Unknown stage: {stage}",
            actual={"stage": stage},
        )


# ---------------------------------------------------------------------------
# Atomic write (05-00 §9)
# ---------------------------------------------------------------------------

def atomic_write_npz(path: Path, **arrays: NDArray) -> str:
    """Atomically write arrays to .npz and return content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise IOError_(
            "FXE-IO-003",
            f"Atomic write failed: {path}",
            paths={"target": str(path)},
        )
    return _file_hash(path)


def atomic_write_json(path: Path, data: dict) -> str:
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise IOError_(
            "FXE-IO-003",
            f"Atomic write failed: {path}",
            paths={"target": str(path)},
        )
    return _file_hash(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Read-first cache (05-00 §9)
# ---------------------------------------------------------------------------

def read_or_compute_npz(
    path: Path,
    compute_fn,
    *,
    expected_keys: list[str] | None = None,
) -> dict[str, NDArray]:
    """
    Read-first cache: if path exists and valid, load; else compute and write.

    compute_fn() must return dict[str, NDArray].
    """
    if path.exists():
        try:
            data = dict(np.load(str(path)))
            if expected_keys and not all(k in data for k in expected_keys):
                logger.warning("Cache miss (missing keys): %s", path)
            else:
                logger.info("Cache hit: %s", path)
                return data
        except Exception:
            logger.warning("Cache invalid: %s", path)

    result = compute_fn()
    atomic_write_npz(path, **result)
    return result


def load_npz_checked(
    path: Path,
    required_keys: list[str],
    *,
    expected_hash: str | None = None,
) -> dict[str, NDArray]:
    """Load NPZ and enforce required-key contract."""
    if not path.exists():
        raise IOError_("FXE-IO-001", f"Required artifact missing: {path}", paths={"path": str(path)})
    if expected_hash is not None:
        actual_hash = _file_hash(path)
        if actual_hash != expected_hash:
            raise IOError_(
                "FXE-IO-002",
                "Artifact hash mismatch",
                expected={"hash_expected": expected_hash},
                actual={"hash_actual": actual_hash},
                paths={"path": str(path)},
            )
    try:
        loaded = dict(np.load(str(path), allow_pickle=False))
    except Exception as exc:
        raise SchemaError("FXE-SCHEMA-002", f"Artifact load failed: {path}", paths={"path": str(path)}) from exc
    missing = [k for k in required_keys if k not in loaded]
    if missing:
        raise SchemaError(
            "FXE-SCHEMA-002",
            f"Artifact missing required keys: {missing}",
            expected={"required_keys": required_keys},
            actual={"present_keys": sorted(loaded.keys())},
            paths={"path": str(path)},
        )
    return loaded


def load_json_checked(path: Path, *, expected_hash: str | None = None) -> dict[str, Any]:
    """Load JSON artifact with IO-coded failures."""
    if not path.exists():
        raise IOError_("FXE-IO-001", f"Required artifact missing: {path}", paths={"path": str(path)})
    if expected_hash is not None:
        actual_hash = _file_hash(path)
        if actual_hash != expected_hash:
            raise IOError_(
                "FXE-IO-002",
                "Artifact hash mismatch",
                expected={"hash_expected": expected_hash},
                actual={"hash_actual": actual_hash},
                paths={"path": str(path)},
            )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise SchemaError("FXE-SCHEMA-002", f"Invalid JSON artifact: {path}", paths={"path": str(path)}) from exc
    if not isinstance(data, dict):
        raise SchemaError(
            "FXE-SCHEMA-002",
            f"Artifact JSON must decode to object: {path}",
            actual={"actual_type": type(data).__name__},
            paths={"path": str(path)},
        )
    return data


def validate_meta(meta: dict[str, Any]) -> None:
    """Validate required metadata fields (05-00 §7)."""
    missing = sorted(field for field in META_REQUIRED_FIELDS if field not in meta)
    if missing:
        raise SchemaError(
            "FXE-SCHEMA-002",
            "Metadata missing required fields",
            expected={"required_fields": sorted(META_REQUIRED_FIELDS)},
            actual={"missing_fields": missing},
        )


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Index record (05-00 §11)
# ---------------------------------------------------------------------------

def append_index_record(
    output_root: str,
    record: dict[str, Any],
) -> None:
    """Append one line to index.jsonl."""
    index_path = Path(output_root) / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    for field in (
        "key",
        "module",
        "level",
        "path",
        "n",
        "r42",
        "r62",
        "U",
        "Jh",
        "zeta",
        "hopping_name",
        "kramer_name",
        "content_hash",
    ):
        record.setdefault(field, None)
    record.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
    with open(index_path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def append_error_record(output_root: str, payload: dict[str, Any]) -> None:
    """Append one-line JSON error payload to outputs/index_errors.jsonl."""
    p = Path(output_root) / "index_errors.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Meta builder (05-00 §7)
# ---------------------------------------------------------------------------

def build_meta(
    *,
    schema_version: str = "fxe.io.v1",
    module: str,
    level: str,
    key: str,
    inputs_summary: dict,
    tensor_name: str,
    physical_meaning: str,
    basis_id: str,
    index_definition: str,
    unit: str = "eV",
    storage_layout: str = "dense",
    dtype: str = "complex128",
    logical_shape: list[int],
    payload_files: list[str],
    extra: dict | None = None,
) -> dict[str, Any]:
    """Build a metadata dict satisfying 05-00 §7."""
    meta: dict[str, Any] = {
        "schema_version": schema_version,
        "module": module,
        "level": level,
        "key": key,
        "inputs_summary": inputs_summary,
        "tensor_name": tensor_name,
        "physical_meaning": physical_meaning,
        "basis_id": basis_id,
        "index_definition": index_definition,
        "unit": unit,
        "storage_layout": storage_layout,
        "dtype": dtype,
        "logical_shape": logical_shape,
        "payload_files": payload_files,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta
