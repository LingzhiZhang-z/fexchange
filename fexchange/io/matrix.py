"""
Runtime matrix input loaders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.errors import IOError_, SchemaError


def load_matrix_file(path: Path, *, preferred_key: str) -> NDArray[np.complexfloating]:
    if not path.exists():
        raise IOError_("FXE-IO-001", f"Required input file missing: {path}", paths={"path": str(path)})
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(path))
        return np.asarray(arr, dtype=np.complex128)
    if suffix == ".npz":
        try:
            data = np.load(str(path))
        except Exception as exc:
            raise SchemaError(
                "FXE-SCHEMA-002",
                f"Invalid NPZ file: {path}",
                paths={"path": str(path)},
            ) from exc
        if preferred_key in data:
            return np.asarray(data[preferred_key], dtype=np.complex128)
        keys = sorted(data.keys())
        if not keys:
            raise SchemaError("FXE-SCHEMA-002", f"Empty NPZ file: {path}", paths={"path": str(path)})
        return np.asarray(data[keys[0]], dtype=np.complex128)
    if suffix in {".txt", ".dat"}:
        try:
            return np.asarray(np.loadtxt(str(path), dtype=np.complex128), dtype=np.complex128)
        except Exception as exc:
            raise SchemaError(
                "FXE-SCHEMA-002",
                f"Invalid text matrix file: {path}",
                paths={"path": str(path)},
            ) from exc
    raise IOError_(
        "FXE-IO-001",
        f"Unsupported matrix input format: {path.suffix}",
        expected={"supported_suffixes": [".npy", ".npz", ".txt", ".dat"]},
        actual={"suffix": path.suffix},
    )


def load_txt_blocks(path: Path, expected_counts: dict[str, int] | None = None) -> dict[str, NDArray[np.complexfloating]]:
    """Load a multi-block text file with ``[key]`` headers and ``real imag`` rows."""
    if not path.exists():
        raise IOError_("FXE-IO-001", f"Required input file missing: {path}", paths={"path": str(path)})

    expected_counts = expected_counts or {}
    blocks: dict[str, list[complex]] = {}
    current_key: str | None = None

    try:
        lines = path.read_text().splitlines()
    except Exception as exc:
        raise IOError_("FXE-IO-001", f"Failed reading text blocks: {path}", paths={"path": str(path)}) from exc

    for lineno, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            key = line[1:-1].strip()
            if not key:
                raise SchemaError("FXE-SCHEMA-002", "Empty block key", paths={"path": str(path)}, actual={"lineno": lineno})
            if key in blocks:
                raise SchemaError(
                    "FXE-SCHEMA-002",
                    f"Duplicate text block key: {key}",
                    paths={"path": str(path)},
                    actual={"key": key, "lineno": lineno},
                )
            blocks[key] = []
            current_key = key
            continue
        if current_key is None:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "Matrix data appeared before any [key] header",
                paths={"path": str(path)},
                actual={"lineno": lineno},
            )
        parts = line.split()
        if len(parts) != 2:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "Text block rows must contain exactly two numbers: real imag",
                paths={"path": str(path)},
                actual={"key": current_key, "lineno": lineno, "row": raw},
            )
        try:
            value = complex(float(parts[0]), float(parts[1]))
        except ValueError as exc:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "Failed parsing text block complex value",
                paths={"path": str(path)},
                actual={"key": current_key, "lineno": lineno, "row": raw},
            ) from exc
        blocks[current_key].append(value)

    if not blocks:
        raise SchemaError("FXE-SCHEMA-002", f"No text blocks found: {path}", paths={"path": str(path)})

    result: dict[str, NDArray[np.complexfloating]] = {}
    for key, values in blocks.items():
        if not values:
            raise SchemaError(
                "FXE-SCHEMA-002",
                f"Text block is empty: {key}",
                paths={"path": str(path)},
                actual={"key": key},
            )
        expected = expected_counts.get(key)
        if expected is not None and len(values) != expected:
            raise SchemaError(
                "FXE-SCHEMA-002",
                f"Wrong row count for text block {key}",
                expected={"rows": expected},
                actual={"key": key, "rows": len(values)},
                paths={"path": str(path)},
            )
        result[key] = np.asarray(values, dtype=np.complex128)
    return result
