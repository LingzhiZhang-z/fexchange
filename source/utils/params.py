"""Parameter file helpers for CLI utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

PATH_KEYS = {
    "t_npy",
    "ground_npz",
    "lsjm_np1_npz",
    "e_np1_npz",
    "lsjm_nm1_npz",
    "e_nm1_npz",
    "cache_dir",
    "out_dir",
}

ParamsDict = Dict[str, Any]


def load_params_file(path: Path) -> ParamsDict:
    """Load parameters from a JSON file and resolve relative paths."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Only JSON parameter files are supported")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Parameter file must contain a JSON object")

    base = path.parent
    for key in PATH_KEYS:
        if key in data and data[key] is not None:
            p = Path(data[key])
            if not p.is_absolute():
                p = (base / p).resolve()
            data[key] = str(p)
    return data
