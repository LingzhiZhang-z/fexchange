from __future__ import annotations

from typing import Dict, Tuple

ORBITAL_ORDER_ID = "site_i_then_site_j_v1"

# Ordering: for m=-3..3, sigma=-1/2 then +1/2
_ORBITALS = [(m, sigma) for m in range(-3, 4) for sigma in (-0.5, 0.5)]
_P_FROM_MS: Dict[Tuple[int, float], int] = {ms: i for i, ms in enumerate(_ORBITALS)}


def p_to_ms(p: int) -> Tuple[int, float]:
    """Map orbital index p in [0,13] to (m, sigma)."""
    return _ORBITALS[p]


def ms_to_p(m: int, sigma: float) -> int:
    """Map (m, sigma) to orbital index p."""
    return _P_FROM_MS[(m, sigma)]
