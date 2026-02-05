from __future__ import annotations

from typing import Tuple

import numpy as np

from ..api.keys import quantize_float


def truncate_by_nstates(E: np.ndarray, n_keep: int | None) -> Tuple[np.ndarray, str]:
    if n_keep is None or n_keep >= len(E):
        return np.arange(len(E)), "all"
    if n_keep <= 0:
        raise ValueError("n_keep must be positive")
    order = np.argsort(E)
    idx = np.array(order[:n_keep], dtype=int)
    return idx, f"nstates_{n_keep}"


def truncate_by_energy_window(E: np.ndarray, emax: float | None) -> Tuple[np.ndarray, str]:
    if emax is None:
        return np.arange(len(E)), "all"
    mask = E <= emax
    idx = np.where(mask)[0]
    return idx.astype(int), f"emax_{quantize_float(emax)}"
