from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .types import EnergySet, GroundManifold, StateSet


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _json_dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def _json_loads(s: str):
    if not s:
        return None
    return json.loads(s)


def save_state_set(path: Path, state: StateSet) -> None:
    labels_json = _json_dumps(state.labels) if state.labels is not None else ""
    meta_json = _json_dumps(state.meta)
    np.savez(
        path,
        V_fock=state.V_fock,
        labels_json=labels_json,
        meta_json=meta_json,
    )


def load_state_set(path: Path, n: Optional[int] = None, basis_id: Optional[str] = None) -> StateSet:
    data = np.load(path, allow_pickle=False)
    labels_json = data["labels_json"].item() if "labels_json" in data else ""
    meta_json = data["meta_json"].item() if "meta_json" in data else ""
    labels = _json_loads(labels_json)
    meta = _json_loads(meta_json) or {}
    V_fock = data["V_fock"]
    inferred_n = n if n is not None else meta.get("n", -1)
    inferred_basis_id = basis_id if basis_id is not None else meta.get("basis_id", "")
    return StateSet(
        n=inferred_n,
        basis_id=inferred_basis_id,
        V_fock=V_fock,
        labels=labels,
        meta=meta,
    )


def save_energy_set(path: Path, energy: EnergySet) -> None:
    part_map: Dict[str, str] = {}
    payload = {
        "E_total": energy.E_total,
        "E_parts_json": None,
        "meta_json": _json_dumps(energy.meta),
    }
    for name, arr in energy.E_parts.items():
        key = f"part__{name}"
        part_map[name] = key
        payload[key] = arr
    payload["E_parts_json"] = _json_dumps(part_map)
    np.savez(path, **payload)


def load_energy_set(
    path: Path,
    n: Optional[int] = None,
    shape_key: Optional[str] = None,
    energy_key: Optional[str] = None,
) -> EnergySet:
    data = np.load(path, allow_pickle=False)
    meta_json = data["meta_json"].item() if "meta_json" in data else ""
    meta = _json_loads(meta_json) or {}
    E_total = data["E_total"]
    part_map_json = data["E_parts_json"].item() if "E_parts_json" in data else ""
    part_map = _json_loads(part_map_json) or {}
    E_parts = {name: data[key] for name, key in part_map.items() if key in data}
    inferred_n = n if n is not None else meta.get("n", -1)
    inferred_shape = shape_key if shape_key is not None else meta.get("shape_key", "")
    inferred_energy = energy_key if energy_key is not None else meta.get("energy_key", "")
    return EnergySet(
        n=inferred_n,
        shape_key=inferred_shape,
        energy_key=inferred_energy,
        E_total=E_total,
        E_parts=E_parts,
        meta=meta,
    )


def save_ground_manifold(path: Path, ground: GroundManifold) -> None:
    meta_json = _json_dumps(ground.meta)
    np.savez(
        path,
        G_fock=ground.G_fock,
        E0=np.asarray(ground.E0),
        degeneracy=np.asarray(ground.degeneracy),
        tol=np.asarray(ground.tol),
        meta_json=meta_json,
    )


def load_ground_manifold(
    path: Path,
    n: Optional[int] = None,
    basis_id: Optional[str] = None,
) -> GroundManifold:
    data = np.load(path, allow_pickle=False)
    meta_json = data["meta_json"].item() if "meta_json" in data else ""
    meta = _json_loads(meta_json) or {}
    G_fock = data["G_fock"]
    E0 = float(data["E0"]) if "E0" in data else float(meta.get("E0", 0.0))
    degeneracy = int(data["degeneracy"]) if "degeneracy" in data else int(meta.get("degeneracy", 1))
    tol = float(data["tol"]) if "tol" in data else float(meta.get("tol", 0.0))
    inferred_n = n if n is not None else meta.get("n", -1)
    inferred_basis_id = basis_id if basis_id is not None else meta.get("basis_id", "")
    return GroundManifold(
        n=inferred_n,
        basis_id=inferred_basis_id,
        G_fock=G_fock,
        E0=E0,
        degeneracy=degeneracy,
        tol=tol,
        meta=meta,
    )


def write_source_paths_txt(path: Path, paths: List[str]) -> None:
    lines = [str(Path(p).resolve()) for p in paths]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_source_paths_txt(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line]
