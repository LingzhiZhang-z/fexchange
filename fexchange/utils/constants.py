"""
Shared small constant matrices used across modules.
"""

from __future__ import annotations

import numpy as np

from fexchange.utils.numerics import DTYPE_COMPLEX

SIGMA_0 = np.eye(2, dtype=DTYPE_COMPLEX)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=DTYPE_COMPLEX)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE_COMPLEX)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=DTYPE_COMPLEX)

