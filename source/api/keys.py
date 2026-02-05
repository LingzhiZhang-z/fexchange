"""Helpers for stable cache keys and shape normalization."""
from __future__ import annotations

import math


def quantize_float(x: float, precision: float = 1e-12) -> str:
    """Return a stable scientific-notation string for a float value.

    The precision is used both for zeroing tiny values and rounding.
    """
    if not math.isfinite(x):
        return str(x)
    if precision <= 0:
        return f"{x:.12e}"
    if abs(x) < precision:
        x = 0.0
    digits = max(0, int(math.ceil(-math.log10(precision)))) + 2
    xq = round(x, digits)
    return f"{xq:.{digits}e}"


def normalize_shape(F2: float, F4: float, F6: float) -> tuple[float, float, float]:
    """Normalize (F2, F4, F6) by F2.

    Returns (F4/F2, F6/F2, abs(F2)).
    """
    if F2 == 0:
        raise ValueError("F2 cannot be zero for shape normalization")
    r4 = F4 / F2
    r6 = F6 / F2
    F2_abs = abs(F2)
    return r4, r6, F2_abs


def shape_key(
    n: int,
    r4: float,
    r6: float,
    scheme_id: str,
    basis_id: str,
    precision: float = 1e-12,
) -> str:
    """Build a deterministic cache key for a normalized shape."""
    r4s = quantize_float(r4, precision)
    r6s = quantize_float(r6, precision)
    return f"shape|n={n}|r4={r4s}|r6={r6s}|scheme={scheme_id}|basis={basis_id}"


def energy_key(
    shape_key: str,
    F2_abs: float,
    F0_abs: float,
    lambda_abs: float,
    precision: float = 1e-12,
) -> str:
    """Build a deterministic cache key for an energy scale."""
    f2s = quantize_float(F2_abs, precision)
    f0s = quantize_float(F0_abs, precision)
    lams = quantize_float(lambda_abs, precision)
    return f"energy|{shape_key}|F2={f2s}|F0={f0s}|lambda={lams}"
