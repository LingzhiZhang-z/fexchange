from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.sparse as sp

from ..api.io import load_state_set, read_source_paths_txt, write_source_paths_txt
from ..api.types import ProjectedTransitionsCache


def _safe_id(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def _cache_dir(
    cache_dir: Path,
    which: str,
    n: int,
    basis_id: str,
    orbital_order_id: str,
    truncation_id: str,
) -> Path:
    name = f"proj_trans_{which}_n{n}_{_safe_id(basis_id)}_{_safe_id(orbital_order_id)}_{_safe_id(truncation_id)}"
    return cache_dir / name


def _load_cache(base: Path) -> Optional[ProjectedTransitionsCache]:
    params_path = base / "params.json"
    if not params_path.exists():
        return None
    params = json.loads(params_path.read_text())
    src_paths = read_source_paths_txt(base / "source_paths.txt")
    if len(src_paths) != 1:
        return None
    e_mid_path = base / "E_mid.npy"
    if not e_mid_path.exists():
        return None
    T_list = []
    for p in range(14):
        tpath = base / f"T_p{p}.npy"
        if not tpath.exists():
            return None
        T_list.append(np.load(tpath))
    E_mid = np.load(e_mid_path)
    return ProjectedTransitionsCache(
        n=params["n"],
        basis_id=params["basis_id"],
        orbital_order_id=params["orbital_order_id"],
        which=params["which"],
        src_lsjm_path=src_paths[0],
        truncation_id=params["truncation_id"],
        T=T_list,
        E_mid=E_mid,
        meta=params,
    )


def _params_match(params: dict, expected: dict) -> bool:
    for key, val in expected.items():
        if params.get(key) != val:
            return False
    return True


def build_or_load_projected_transitions_cache(
    *,
    n: int,
    cache_dir: Path,
    basis_id: str,
    orbital_order_id: str,
    which: str,
    lsjm_path: Path,
    E_mid: np.ndarray,
    truncation_indices: Optional[np.ndarray],
    truncation_id: str,
    D_create: Optional[List[sp.spmatrix]] = None,
    D_create_nm1: Optional[List[sp.spmatrix]] = None,
    shape_key: Optional[str] = None,
) -> ProjectedTransitionsCache:
    if which not in ("np1", "nm1"):
        raise ValueError("which must be 'np1' or 'nm1'")

    cache_dir = Path(cache_dir)
    base = _cache_dir(cache_dir, which, n, basis_id, orbital_order_id, truncation_id)
    base.mkdir(parents=True, exist_ok=True)

    params_path = base / "params.json"
    src_path = Path(lsjm_path).resolve()
    expected_params = {
        "n": n,
        "basis_id": basis_id,
        "orbital_order_id": orbital_order_id,
        "which": which,
        "truncation_id": truncation_id,
        "shape_key": shape_key,
    }

    if params_path.exists():
        params = json.loads(params_path.read_text())
        src_paths = read_source_paths_txt(base / "source_paths.txt")
        if _params_match(params, expected_params) and src_paths == [str(src_path)]:
            cached = _load_cache(base)
            if cached is not None:
                return cached

    state_set = load_state_set(src_path)
    V_fock = state_set.V_fock
    if truncation_indices is not None:
        V_fock = V_fock[:, truncation_indices]
        E_mid = E_mid[truncation_indices]

    T_list: List[np.ndarray] = []
    if which == "np1":
        if D_create is None:
            raise ValueError("D_create is required for which='np1'")
        Vh = V_fock.conjugate().T
        for p in range(14):
            T_p = Vh @ D_create[p]
            T_list.append(np.asarray(T_p))
    else:
        if D_create_nm1 is None:
            raise ValueError("D_create_nm1 is required for which='nm1'")
        Vh = V_fock.conjugate().T
        for p in range(14):
            C_p = D_create_nm1[p].conjugate().transpose()
            T_p = Vh @ C_p
            T_list.append(np.asarray(T_p))

    for p, T_p in enumerate(T_list):
        np.save(base / f"T_p{p}.npy", T_p)
    np.save(base / "E_mid.npy", E_mid)
    params_path.write_text(
        json.dumps({**expected_params, "created_at": datetime.now().isoformat(timespec="seconds")}, indent=2)
    )
    write_source_paths_txt(base / "source_paths.txt", [str(src_path)])

    return ProjectedTransitionsCache(
        n=n,
        basis_id=basis_id,
        orbital_order_id=orbital_order_id,
        which=which,
        src_lsjm_path=str(src_path),
        truncation_id=truncation_id,
        T=T_list,
        E_mid=E_mid,
        meta=expected_params,
    )
