"""
Shared non-numeric constants used across modules.
"""

from __future__ import annotations

import numpy as np

from fexchange.utils.numerics import DTYPE_COMPLEX

N_ORB: int = 14
ELL: int = 3

RUN_SCHEMA_VERSION = "fxe.run_input.v1"
STANDARD_VERSION = "2026-02"

LEVELS = ("LMSM", "LSJM", "L0", "L1", "L2", "L3", "L4")
LEVEL_ORDER = {
    "LMSM": 1,
    "LSJM": 2,
    "L0": 3,
    "L1": 4,
    "L2": 5,
    "L3": 6,
    "L4": 7,
}
UPSTREAM_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "LMSM": (),
    "LSJM": ("LMSM",),
    "L0": (),
    "L1": ("LMSM", "LSJM", "L0"),
    "L2": ("L1",),
    "L3": ("L2",),
    "L4": ("L3",),
}

SIGMA_0 = np.eye(2, dtype=DTYPE_COMPLEX)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=DTYPE_COMPLEX)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE_COMPLEX)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=DTYPE_COMPLEX)

_s2 = np.sqrt(2)

S1_X = np.array([
    [0, 1 / _s2, 0],
    [1 / _s2, 0, 1 / _s2],
    [0, 1 / _s2, 0],
], dtype=DTYPE_COMPLEX)
S1_Y = np.array([
    [0, -1j / _s2, 0],
    [1j / _s2, 0, -1j / _s2],
    [0, 1j / _s2, 0],
], dtype=DTYPE_COMPLEX)
S1_Z = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1],
], dtype=DTYPE_COMPLEX)

Q_X2Y2 = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0],
], dtype=DTYPE_COMPLEX)
Q_XY = np.array([
    [0, 0, -1j],
    [0, 0, 0],
    [1j, 0, 0],
], dtype=DTYPE_COMPLEX)
Q_ZX = np.array([
    [0, 1 / _s2, 0],
    [1 / _s2, 0, -1 / _s2],
    [0, -1 / _s2, 0],
], dtype=DTYPE_COMPLEX)
Q_YZ = np.array([
    [0, -1j / _s2, 0],
    [1j / _s2, 0, 1j / _s2],
    [0, -1j / _s2, 0],
], dtype=DTYPE_COMPLEX)
Q_3Z2R2 = (1 / np.sqrt(3)) * np.array([
    [1, 0, 0],
    [0, -2, 0],
    [0, 0, 1],
], dtype=DTYPE_COMPLEX)
