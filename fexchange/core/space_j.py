"""
J-space matrices and J quantization helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.checks import check_hermitian
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
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=DTYPE_COMPLEX)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=DTYPE_COMPLEX)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE_COMPLEX)
    return {
        "o0": 0.5 * np.trace(M),
        "ox": 0.5 * np.trace(sigma_x @ M),
        "oy": 0.5 * np.trace(sigma_y @ M),
        "oz": 0.5 * np.trace(sigma_z @ M),
    }
