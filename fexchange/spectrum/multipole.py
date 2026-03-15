"""Simplified multipole analysis and human-readable summary."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.spectrum.doublet import gauge_fix_kramers_pair, project_to_eg_doublet
from fexchange.core.space_j import (
    normalize_J,
    pauli_decompose,
    spin1_decompose,
    project_operators_to_subspace,
)
from fexchange.core.stevens import build_multipole_operators
from fexchange.utils.numerics import DTYPE_COMPLEX
_MULTIPOLE_TYPES = (
    "magnetic_dipole",
    "electric_quadrupole",
    "magnetic_octupole",
)


def _traceless_fro_norm(mat: NDArray[np.complexfloating]) -> float:
    dim = int(mat.shape[0])
    identity = np.eye(dim, dtype=DTYPE_COMPLEX)
    traceless = mat - (np.trace(mat) / dim) * identity
    return float(np.linalg.norm(traceless, ord="fro"))


def _real_pauli(coeffs: dict[str, complex], tol: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("o0", "ox", "oy", "oz"):
        value = complex(coeffs.get(key, 0.0)).real
        if abs(value) <= tol:
            value = 0.0
        out[key] = float(value)
    return out


def _real_spin1(coeffs: dict[str, float], tol: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("s0", "sx", "sy", "sz", "qx2y2", "qxy", "qzx", "qyz", "q3z2r2"):
        value = float(coeffs.get(key, 0.0))
        if abs(value) <= tol:
            value = 0.0
        out[key] = value
    return out


def analyze_multipole_carrying(
    Psi: NDArray[np.complexfloating],
    J: float,
    *,
    tol: float = 1e-10,
    point_group: str | None = None,
) -> dict[str, Any]:
    """Analyze carried multipoles in a projected subspace.

    Optional gauge-fixing for 2D subspaces:
      - Kramers (half-integer J, odd electron): auto-detected via TR check.
      - Non-Kramers: pass point_group so the canonical E_g projection/gauge can
        be reconstructed from (J, point_group).
    """
    J = normalize_J(J, module="multipole")
    subspace_dim = int(Psi.shape[1])

    used_gauge = "none"
    basis = Psi
    if subspace_dim == 2 and int(2 * J) % 2 == 0 and point_group is not None:
        basis = project_to_eg_doublet(J, Psi, point_group, tol=tol)
        used_gauge = f"non_kramers_{point_group.lower()}"
    elif subspace_dim == 2 and int(2 * J) % 2 == 1:
        basis = gauge_fix_kramers_pair(Psi, tol=tol)
        used_gauge = "kramers"
    result: dict[str, Any] = {
        "subspace_dim": subspace_dim,
        "used_gauge": used_gauge,
        "used_kramers_gauge": used_gauge == "kramers",
        "multipoles": {},
    }

    for multipole_type in _MULTIPOLE_TYPES:
        ops = build_multipole_operators(J, multipole_type)
        projected = project_operators_to_subspace(basis, ops, module="multipole")

        family: dict[str, Any] = {}
        for name, mat in projected.items():
            expectation = None
            if subspace_dim == 1:
                expectation = float(np.real(mat[0, 0]))
                norm = abs(expectation)
            else:
                norm = _traceless_fro_norm(mat)
            carried = bool(norm > tol)
            payload: dict[str, Any] = {
                "norm": norm,
                "carried": carried,
            }
            if carried:
                if subspace_dim == 1 and expectation is not None:
                    payload["expectation"] = expectation
                elif subspace_dim == 2:
                    payload["pauli"] = _real_pauli(pauli_decompose(mat), tol)
                elif subspace_dim == 3:
                    payload["spin1"] = _real_spin1(spin1_decompose(mat), tol)
            family[name] = payload

        result["multipoles"][multipole_type] = family

    return result


def format_multipole_summary(analysis: dict[str, Any]) -> str:
    """Format analysis output into a compact human-readable summary."""
    multipoles = analysis.get("multipoles", {})

    lines: list[str] = []
    for multipole_type in _MULTIPOLE_TYPES:
        family = multipoles.get(multipole_type)
        if not isinstance(family, dict):
            continue

        lines.append(f"{multipole_type}:")
        for op_name, payload in family.items():
            if not isinstance(payload, dict):
                continue

            carried = bool(payload.get("carried", False))
            flag = "Yes" if carried else "No"
            expectation = payload.get("expectation")
            pauli = payload.get("pauli")
            spin1 = payload.get("spin1")

            if not carried:
                lines.append(f"  {op_name:<6} {flag}")
            elif isinstance(pauli, dict):
                lines.append(
                    f"  {op_name:<6} {flag:<3} "
                    f"o0={float(pauli.get('o0', 0.0)):.3f} "
                    f"ox={float(pauli.get('ox', 0.0)):.3f} "
                    f"oy={float(pauli.get('oy', 0.0)):.3f} "
                    f"oz={float(pauli.get('oz', 0.0)):.3f}"
                )
            elif isinstance(spin1, dict):
                lines.append(
                    f"  {op_name:<6} {flag:<3} "
                    f"sx={float(spin1.get('sx', 0.0)):.3f} "
                    f"sy={float(spin1.get('sy', 0.0)):.3f} "
                    f"sz={float(spin1.get('sz', 0.0)):.3f}"
                )
                lines.append(
                    "         "
                    f"qx2y2={float(spin1.get('qx2y2', 0.0)):.3f} "
                    f"qxy={float(spin1.get('qxy', 0.0)):.3f} "
                    f"qzx={float(spin1.get('qzx', 0.0)):.3f} "
                    f"qyz={float(spin1.get('qyz', 0.0)):.3f} "
                    f"q3z2r2={float(spin1.get('q3z2r2', 0.0)):.3f}"
                )
            elif expectation is not None:
                lines.append(f"  {op_name:<6} {flag:<3} <O>={float(expectation):.6f}")
            else:
                lines.append(f"  {op_name:<6} {flag}")

    return "\n".join(lines)
