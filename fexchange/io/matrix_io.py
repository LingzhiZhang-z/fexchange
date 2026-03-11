"""
Runtime matrix input loaders for hopping and projector payloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.io.wannier90 import load_t_mu_from_wannier90
from fexchange.utils.errors import BindError, IOError_, InputError, SchemaError


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
    raise InputError(
        "FXE-INPUT-003",
        f"Unsupported matrix input format: {path.suffix}",
        expected={"supported_suffixes": [".npy", ".npz"]},
        actual={"suffix": path.suffix},
    )


def load_hopping_matrix(cfg: dict[str, Any], *, n_orb: int) -> dict[str, Any]:
    sources = cfg["sources"]
    source_mode = sources["hopping_source"]
    if source_mode == "wannier90":
        hop_payload = load_t_mu_from_wannier90(cfg["wannier90"], n_orb=n_orb, return_payload=True)
        t_mu = hop_payload["t_mu"]
        meta = hop_payload.get("meta", {})
    elif source_mode == "file":
        path = Path(cfg["paths"]["hopping_file"])
        t_mu = load_matrix_file(path, preferred_key="t_mu")
        meta = {"source_mode": "file", "path": str(path)}
    else:
        raise InputError(
            "FXE-INPUT-003",
            f"Unknown hopping_source: {source_mode!r}",
            actual={"hopping_source": source_mode},
        )

    t_mu = np.asarray(t_mu, dtype=np.complex128)
    if t_mu.ndim != 2:
        raise BindError(
            "FXE-BIND-003",
            "Hopping input must be rank-2 matrix (one bond per run)",
            actual={"t_mu_shape": list(t_mu.shape)},
        )
    if t_mu.shape != (n_orb, n_orb):
        raise BindError(
            "FXE-BIND-003",
            f"Hopping shape mismatch: {t_mu.shape} != ({n_orb},{n_orb})",
            expected={"shape": [n_orb, n_orb]},
            actual={"shape": list(t_mu.shape)},
        )
    return {"t_mu": t_mu, "meta": meta}

