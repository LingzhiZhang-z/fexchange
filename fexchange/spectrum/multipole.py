"""Multipole projection analysis and gauge fixing for CEF subspaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.spectrum.doublet import gauge_fix_kramers_pair, project_to_eg_doublet
from fexchange.core.space_j import (
    _fix_column_phases,
    build_space_j_operator,
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


# ---------------------------------------------------------------------------
# Gauge fixing
# ---------------------------------------------------------------------------

def gauge_fix_subspace(
    Psi: NDArray[np.complexfloating],
    J: float,
    *,
    point_group: str | None = None,
    tol: float = 1e-10,
) -> tuple[NDArray[np.complexfloating], str]:
    """Gauge-fix a degenerate subspace. Returns (basis, gauge_name)."""
    d = int(Psi.shape[1])
    half_int = int(2 * J) % 2 == 1

    if d == 2 and half_int:
        return gauge_fix_kramers_pair(Psi, tol=tol), "kramers"
    if d == 2 and not half_int and point_group is not None:
        return project_to_eg_doublet(J, Psi, point_group, tol=tol), f"non_kramers_{point_group.lower()}"
    if d == 3:
        Jz = build_space_j_operator(J, module="multipole")[0]
        M_Jz = Psi.conj().T @ Jz @ Psi
        evals, vecs = np.linalg.eigh(M_Jz)
        basis = Psi @ vecs[:, np.argsort(evals.real)[::-1]]
        return _fix_column_phases(basis), "triplet_jz"
    return Psi, "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Multipole analysis (assumes Psi is already gauge-fixed)
# ---------------------------------------------------------------------------

def analyze_multipole_carrying(
    Psi: NDArray[np.complexfloating],
    J: float,
    *,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Analyze carried multipoles in a projected subspace.

    Assumes Psi columns are already gauge-fixed.
    """
    J = normalize_J(J, module="multipole")
    subspace_dim = int(Psi.shape[1])

    result: dict[str, Any] = {
        "subspace_dim": subspace_dim,
        "multipoles": {},
    }

    for multipole_type in _MULTIPOLE_TYPES:
        ops = build_multipole_operators(J, multipole_type)
        projected = project_operators_to_subspace(Psi, ops, module="multipole")

        family: dict[str, Any] = {}
        for name, mat in projected.items():
            if subspace_dim == 1:
                expectation = float(np.real(mat[0, 0]))
                norm = abs(expectation)
            else:
                expectation = None
                norm = _traceless_fro_norm(mat)
            carried = bool(norm > tol)
            payload: dict[str, Any] = {"norm": norm, "carried": carried}
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
