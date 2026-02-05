from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse as sp

from ..api.types import FermionOpsCache
from ..core.fermion_ops import build_creation_matrices
from ..core.fock_basis import FockBasis


def _safe_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _cache_dir(cache_dir: Path, n: int, basis_id: str, orbital_order_id: str) -> Path:
    name = f"fermion_ops_n{n}_{_safe_id(basis_id)}_{_safe_id(orbital_order_id)}"
    return cache_dir / name


def build_or_load_ops_cache(
    n: int,
    cache_dir: Path,
    basis_n: FockBasis,
    basis_np1: FockBasis,
    orbital_order_id: str,
) -> FermionOpsCache:
    cache_dir = Path(cache_dir)
    base = _cache_dir(cache_dir, n, basis_n.basis_id, orbital_order_id)
    base.mkdir(parents=True, exist_ok=True)

    meta_path = base / "meta.json"
    d_paths = [base / f"D_p{p}.npz" for p in range(14)]

    if meta_path.exists() and all(p.exists() for p in d_paths):
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("n") == n
            and meta.get("basis_id") == basis_n.basis_id
            and meta.get("orbital_order_id") == orbital_order_id
        ):
            D_create = [sp.load_npz(p) for p in d_paths]
            return FermionOpsCache(
                n=n,
                basis_id=basis_n.basis_id,
                orbital_order_id=orbital_order_id,
                D_create=D_create,
                meta=meta,
            )

    D_create = build_creation_matrices(basis_n, basis_np1)
    for p, mat in enumerate(D_create):
        sp.save_npz(d_paths[p], mat)
    meta = {
        "n": n,
        "basis_id": basis_n.basis_id,
        "orbital_order_id": orbital_order_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return FermionOpsCache(
        n=n,
        basis_id=basis_n.basis_id,
        orbital_order_id=orbital_order_id,
        D_create=D_create,
        meta=meta,
    )
