from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def export_heff(out_dir: Path, Heff: np.ndarray, meta: Optional[Dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heff_path = out_dir / "Heff.npy"
    np.save(heff_path, Heff)
    meta_payload = {
        "shape": Heff.shape,
        "dtype": str(Heff.dtype),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if meta:
        meta_payload.update(meta)
    (out_dir / "meta.json").write_text(json.dumps(meta_payload, indent=2))
    return heff_path
