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
    if suffix in {".dat", ".txt"}:
        try:
            arr = np.loadtxt(str(path), dtype=np.complex128)
        except Exception as exc:
            raise SchemaError(
                "FXE-SCHEMA-002",
                f"Invalid text matrix file: {path}",
                paths={"path": str(path)},
            ) from exc
        return np.atleast_2d(np.asarray(arr, dtype=np.complex128))
    raise IOError_(
        "FXE-IO-001",
        f"Unsupported matrix input format: {path.suffix}",
        expected={"supported_suffixes": [".npy", ".npz", ".dat", ".txt"]},
        actual={"suffix": path.suffix},
    )
