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

RE_DEFAULTS_BY_N_ELE: dict[int, dict[str, float]] = {
    1: {"F2_per_Jh": 12.34209613, "F4_per_Jh": 7.73764967, "F6_per_Jh": 5.58527529, "zeta": 0.099600000},
    2: {"F2_per_Jh": 12.33115193, "F4_per_Jh": 7.75544146, "F6_per_Jh": 5.58391785, "zeta": 0.117600000},
    3: {"F2_per_Jh": 12.31801444, "F4_per_Jh": 7.76678019, "F6_per_Jh": 5.59010294, "zeta": 0.135600000},
    4: {"F2_per_Jh": 12.32117776, "F4_per_Jh": 7.75727499, "F6_per_Jh": 5.59389816, "zeta": 0.155003521},
    5: {"F2_per_Jh": 12.31547934, "F4_per_Jh": 7.75504808, "F6_per_Jh": 5.60215414, "zeta": 0.177600000},
    6: {"F2_per_Jh": 12.31761977, "F4_per_Jh": 7.75986412, "F6_per_Jh": 5.59594898, "zeta": 0.199518400},
    7: {"F2_per_Jh": 12.31380817, "F4_per_Jh": 7.75878886, "F6_per_Jh": 5.60114814, "zeta": 0.224255325},
    8: {"F2_per_Jh": 12.31806858, "F4_per_Jh": 7.76011200, "F6_per_Jh": 5.59524219, "zeta": 0.250645240},
    9: {"F2_per_Jh": 12.32393862, "F4_per_Jh": 7.76261669, "F6_per_Jh": 5.58657320, "zeta": 0.277200000},
    10: {"F2_per_Jh": 12.32192444, "F4_per_Jh": 7.75837829, "F6_per_Jh": 5.59218337, "zeta": 0.308384040},
    11: {"F2_per_Jh": 12.33376168, "F4_per_Jh": 7.74864573, "F6_per_Jh": 5.58623297, "zeta": 0.339600000},
    12: {"F2_per_Jh": 12.32881596, "F4_per_Jh": 7.75488784, "F6_per_Jh": 5.58702202, "zeta": 0.372734800},
    13: {"F2_per_Jh": 12.32692411, "F4_per_Jh": 7.75712981, "F6_per_Jh": 5.58743756, "zeta": 0.408000000},
}

RE_TO_N_ELE: dict[str, int] = {
    "Ce": 1,
    "Pr": 2,
    "Nd": 3,
    "Pm": 4,
    "Sm": 5,
    "Eu": 6,
    "Gd": 7,
    "Tb": 8,
    "Dy": 9,
    "Ho": 10,
    "Er": 11,
    "Tm": 12,
    "Yb": 13,
}

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
BRANCHES = ("sopt", "fopt")
BRANCH_TERMINAL = dict(sopt="L4", fopt="L3")

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
