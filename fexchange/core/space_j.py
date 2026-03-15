"""
J-space matrices and J quantization helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.checks import check_hermitian
from fexchange.utils.constants import (
    Q_3Z2R2,
    Q_X2Y2,
    Q_XY,
    Q_YZ,
    Q_ZX,
    S1_X,
    S1_Y,
    S1_Z,
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
)
from fexchange.utils.errors import PhysError
from fexchange.utils.numerics import DTYPE_COMPLEX

_EPS_J_QUANT: float = 1e-8


def _fix_column_phases(U: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Fix per-column global phase deterministically by largest-magnitude entry."""
    out = U.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        pivot = int(np.argmax(np.abs(col)))
        amp = col[pivot]
        if abs(amp) > 0.0:
            out[:, j] = col * np.exp(-1j * np.angle(amp))
    return out


def normalize_J(J: float, *, module: str = "space_j") -> float:
    """
    Normalize J to the nearest integer/half-integer when deviation is tiny.

    Hard-fail when deviation from quantized 2J is too large.
    """
    Jf = float(J)
    twoJ = 2.0 * Jf
    twoJ_round = round(twoJ)
    if abs(twoJ - twoJ_round) > _EPS_J_QUANT:
        raise PhysError(
            "FXE-PHYS-001",
            f"J={Jf} is not close to integer/half-integer within tol={_EPS_J_QUANT:.1e}",
            module=module,
            actual={"J": Jf, "2J": twoJ, "nearest_2J": float(twoJ_round), "tol": _EPS_J_QUANT},
        )
    Jq = 0.5 * float(twoJ_round)
    if Jq < 0.0:
        raise PhysError(
            "FXE-PHYS-001",
            f"J must be non-negative, got J={Jq}",
            module=module,
            actual={"J": Jq},
        )
    return Jq


def build_space_j_operator(J: float, *, module: str = "space_j") -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """Build Jz, J+, J-, Jx, Jy operators in |J,M> basis (M = -J, ..., J ascending)."""
    J = normalize_J(J, module=module)
    dim = int(round(2 * J + 1))
    Jz = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
    Jp = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)

    for i in range(dim):
        M = -J + i
        Jz[i, i] = M
        if i + 1 < dim:
            # J+ |J,M> = sqrt(J(J+1) - M(M+1)) |J,M+1>
            Jp[i + 1, i] = np.sqrt(J * (J + 1) - M * (M + 1))

    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    return Jz, Jp, Jm, Jx, Jy


def build_time_reversal_operator(J: float, *, module: str = "space_j") -> NDArray[np.complexfloating]:
    """
    Build the unitary part U_T of time reversal in |J,M> basis.

    Basis order is M = -J, ..., J (ascending).
    We use the convention
        Theta |J,M> = (-1)^(J-M) |J,-M>,
    and Theta(psi) = U_T @ psi.conj().
    """
    J = normalize_J(J, module=module)
    dim = int(round(2 * J + 1))
    U_T = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
    for i in range(dim):
        M = -J + i
        # Column i corresponds to |J,M>. The mapped row is |J,-M>.
        # M_idx = -J + idx, thus M = -J + i, -M = -J + j
        j = int(round(-M + J))
        # Phase factor from Theta |J,M> = (-1)^(J-M) |J,-M>.
        # exp is guaranteed integer for quantized J and M.
        exp = int(round(J - M))
        phase = -1.0 if (exp % 2) else 1.0
        U_T[j, i] = phase
    return U_T


def project_operators_to_subspace(
    Psi: NDArray[np.complexfloating],
    operators: dict[str, NDArray[np.complexfloating]],
    *,
    module: str = "space_j",
) -> dict[str, NDArray[np.complexfloating]]:
    """
    Project operator matrices into the subspace spanned by columns of Psi.

    Returns a dict with the same keys as input `operators`.
    """
    projected: dict[str, NDArray[np.complexfloating]] = {}
    for name, op in operators.items():
        # Internal helper: trust caller-provided dimensions; NumPy will raise on mismatch.
        projected[name] = Psi.conj().T @ op @ Psi

    return projected


def project_J_to_subspace(
    Psi: NDArray[np.complexfloating],
    Jx: NDArray[np.complexfloating],
    Jy: NDArray[np.complexfloating],
    Jz: NDArray[np.complexfloating],
    *,
    module: str = "space_j",
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    """
    Project Jx/Jy/Jz into the subspace spanned by columns of Psi.

    Returns (M_Jx, M_Jy, M_Jz), each checked for Hermiticity.
    """
    projected = project_operators_to_subspace(
        Psi,
        {"Jx": Jx, "Jy": Jy, "Jz": Jz},
        module=module,
    )
    check_hermitian(projected["Jx"], label="M_Jx", module=module)
    check_hermitian(projected["Jy"], label="M_Jy", module=module)
    check_hermitian(projected["Jz"], label="M_Jz", module=module)
    return projected["Jx"], projected["Jy"], projected["Jz"]


def pauli_decompose(M: NDArray[np.complexfloating]) -> dict[str, complex]:
    """
    Decompose a 2x2 matrix on {I, sigma_x, sigma_y, sigma_z}.

    Returns coefficients {o0, ox, oy, oz} such that
    M = o0*I + ox*sigma_x + oy*sigma_y + oz*sigma_z.
    """
    if M.shape != (2, 2):
        raise PhysError(
            "FXE-PHYS-001",
            "pauli_decompose supports 2x2 matrices only",
            module="space_j",
            actual={"shape": list(M.shape)},
        )
    return {
        "o0": 0.5 * np.trace(M),
        "ox": 0.5 * np.trace(SIGMA_X @ M),
        "oy": 0.5 * np.trace(SIGMA_Y @ M),
        "oz": 0.5 * np.trace(SIGMA_Z @ M),
    }


def spin1_decompose(M: NDArray[np.complexfloating]) -> dict[str, float]:
    """
    Decompose a 3x3 Hermitian matrix into S=1 irreducible tensor components.

    Returns coefficients {s0, sx, sy, sz, qx2y2, qxy, qzx, qyz, q3z2r2} such that
    M = s0*I + sx*Sx + sy*Sy + sz*Sz + qx2y2*Qx2y2 + qxy*Qxy + qzx*Qzx + qyz*Qyz + q3z2r2*Q3z2r2.
    """
    if M.shape != (3, 3):
        raise PhysError(
            "FXE-PHYS-001",
            "spin1_decompose supports 3x3 matrices only",
            module="space_j",
            actual={"shape": list(M.shape)},
        )
    trace = np.trace
    return {
        "s0": float(np.real(trace(M) / 3)),
        "sx": float(np.real(trace(S1_X @ M) / 2)),
        "sy": float(np.real(trace(S1_Y @ M) / 2)),
        "sz": float(np.real(trace(S1_Z @ M) / 2)),
        "qx2y2": float(np.real(trace(Q_X2Y2 @ M) / 2)),
        "qxy": float(np.real(trace(Q_XY @ M) / 2)),
        "qzx": float(np.real(trace(Q_ZX @ M) / 2)),
        "qyz": float(np.real(trace(Q_YZ @ M) / 2)),
        "q3z2r2": float(np.real(trace(Q_3Z2R2 @ M) / 2)),
    }
