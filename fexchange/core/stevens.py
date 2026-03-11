"""CEF-only Stevens operators in |J,M> basis."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fexchange.core.space_j import build_space_j_operator, normalize_J
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError
from fexchange.utils.numerics import DTYPE_COMPLEX


def _hutchings_diagonal(
    J: float,
    k: int,
    Jz: NDArray[np.complexfloating],
    I: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating] | None:
    X = J * (J + 1)
    Jz2 = Jz @ Jz

    if k == 2:
        return 3 * Jz2 - X * I

    if k == 3:
        Jz3 = Jz2 @ Jz
        return 5 * Jz3 - (3 * X - 1) * Jz

    if k == 4:
        Jz4 = Jz2 @ Jz2
        return 35 * Jz4 - (30 * X - 25) * Jz2 + (3 * X**2 - 6 * X) * I

    if k == 6:
        Jz4 = Jz2 @ Jz2
        Jz6 = Jz4 @ Jz2
        return (
            231 * Jz6
            - (315 * X - 735) * Jz4
            + (105 * X**2 - 525 * X + 294) * Jz2
            - (5 * X**3 - 40 * X**2 + 60 * X) * I
        )

    return None


def _symmetrized_tesseral(
    f: NDArray[np.complexfloating],
    Jp_q: NDArray[np.complexfloating],
    Jm_q: NDArray[np.complexfloating],
    mode: Literal["cos", "sin"],
) -> NDArray[np.complexfloating]:
    if mode == "cos":
        Xq = Jp_q + Jm_q
        return 0.5 * (f @ Xq + Xq @ f)
    elif mode == "sin":
        Xq = Jp_q - Jm_q
        return 0.5 / 1j * (f @ Xq + Xq @ f)
    else:
        raise PhysError(
            "FXE-PHYS-001",
            f"Unknown mode: {mode!r}",
            module="stevens",
            actual={"mode": mode},
        )


def _hutchings_off_diagonal(
    J: float,
    k: int,
    q: int,
    mode: Literal["cos", "sin"],
    Jz: NDArray[np.complexfloating],
    Jp: NDArray[np.complexfloating],
    Jm: NDArray[np.complexfloating],
    I: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating] | None:
    X = J * (J + 1)
    Jz2 = Jz @ Jz
    Jz3 = Jz2 @ Jz

    Jp_q = np.linalg.matrix_power(Jp, q)
    Jm_q = np.linalg.matrix_power(Jm, q)

    f: NDArray[np.complexfloating] | None = None

    if (k, q) == (2, 1):
        f = 0.5 * Jz
    elif (k, q) == (2, 2):
        f = 0.5 * I
    elif (k, q) == (3, 1):
        f = 0.5 * (5 * Jz2 - X * I - 0.5 * I)
    elif (k, q) == (3, 2):
        f = 0.5 * Jz
    elif (k, q) == (3, 3):
        f = 0.5 * I
    elif (k, q) == (4, 3):
        f = 0.5 * Jz
    elif (k, q) == (4, 4):
        f = 0.5 * I
    elif (k, q) == (6, 3):
        f = 0.5 * (11 * Jz3 - (3 * X + 59) * Jz)
    elif (k, q) == (6, 4):
        f = 0.5 * (11 * Jz2 - X * I - 38 * I)
    elif (k, q) == (6, 6):
        f = 0.5 * I

    if f is None:
        return None

    return _symmetrized_tesseral(f, Jp_q, Jm_q, mode)


def build_cef_stevens_operators(
    J: float,
    symmetry: Literal["Oh", "C3v"] = "Oh",
    mode_q3: Literal["cos", "sin"] = "cos",
) -> dict[str, NDArray[np.complexfloating]]:
    """Build the CEF-specific Stevens subset used by hcef.py."""
    J = normalize_J(J, module="stevens")
    if symmetry not in {"Oh", "C3v"}:
        raise PhysError(
            "FXE-PHYS-001",
            f"Unknown symmetry branch: {symmetry!r}",
            module="stevens",
            actual={"symmetry": symmetry},
        )
    if mode_q3 not in {"cos", "sin"}:
        raise PhysError(
            "FXE-PHYS-001",
            f"mode_q3 must be 'cos' or 'sin', got {mode_q3!r}",
            module="stevens",
            actual={"mode_q3": mode_q3},
        )

    Jz, Jp, Jm, _, _ = build_space_j_operator(J, module="stevens")
    dim = int(round(2 * J + 1))
    I = np.eye(dim, dtype=DTYPE_COMPLEX)

    O20 = _hutchings_diagonal(J, 2, Jz, I)
    O40 = _hutchings_diagonal(J, 4, Jz, I)
    O60 = _hutchings_diagonal(J, 6, Jz, I)
    O44c = _hutchings_off_diagonal(J, 4, 4, "cos", Jz, Jp, Jm, I)
    O64c = _hutchings_off_diagonal(J, 6, 4, "cos", Jz, Jp, Jm, I)
    O66 = _hutchings_off_diagonal(J, 6, 6, "cos", Jz, Jp, Jm, I)

    if O20 is None or O40 is None or O60 is None or O44c is None or O64c is None or O66 is None:
        raise PhysError(
            "FXE-PHYS-001",
            "Missing CEF Stevens formula in implementation",
            module="stevens",
            actual={"symmetry": symmetry, "mode_q3": mode_q3, "J": float(J)},
        )

    ops: dict[str, NDArray[np.complexfloating]] = {
        "O20": O20,
        "O40": O40,
        "O60": O60,
        "O44c": O44c,
        "O64c": O64c,
        "O66": O66,
    }

    if symmetry == "C3v":
        O43_eta = _hutchings_off_diagonal(J, 4, 3, mode_q3, Jz, Jp, Jm, I)
        O63_eta = _hutchings_off_diagonal(J, 6, 3, mode_q3, Jz, Jp, Jm, I)

        if O43_eta is None or O63_eta is None or O66 is None:
            raise PhysError(
                "FXE-PHYS-001",
                "Missing C3v Stevens formula in implementation",
                module="stevens",
                actual={"mode_q3": mode_q3, "J": float(J)},
            )

        ops["O43_eta"] = O43_eta
        ops["O63_eta"] = O63_eta

    for name, op in ops.items():
        check_hermitian(op, label=f"Stevens_{name}", module="stevens")

    return ops


def build_multipole_operators(
    J: float,
    multipole_type: str,
) -> dict[str, NDArray[np.complexfloating]]:
    """Build Stevens-form multipole operators in |J,M> basis."""
    J = normalize_J(J, module="stevens")
    Jz, Jp, Jm, Jx, Jy = build_space_j_operator(J, module="stevens")
    dim = int(round(2 * J + 1))
    I = np.eye(dim, dtype=DTYPE_COMPLEX)

    if multipole_type == "magnetic_dipole":
        ops: dict[str, NDArray[np.complexfloating] | None] = {
            "Jx": Jx,
            "Jy": Jy,
            "Jz": Jz,
        }
    elif multipole_type == "electric_quadrupole":
        ops = {
            "O20": _hutchings_diagonal(J, 2, Jz, I),
            "O21c": _hutchings_off_diagonal(J, 2, 1, "cos", Jz, Jp, Jm, I),
            "O21s": _hutchings_off_diagonal(J, 2, 1, "sin", Jz, Jp, Jm, I),
            "O22c": _hutchings_off_diagonal(J, 2, 2, "cos", Jz, Jp, Jm, I),
            "O22s": _hutchings_off_diagonal(J, 2, 2, "sin", Jz, Jp, Jm, I),
        }
    elif multipole_type == "magnetic_octupole":
        ops = {
            "O30": _hutchings_diagonal(J, 3, Jz, I),
            "O31c": _hutchings_off_diagonal(J, 3, 1, "cos", Jz, Jp, Jm, I),
            "O31s": _hutchings_off_diagonal(J, 3, 1, "sin", Jz, Jp, Jm, I),
            "O32c": _hutchings_off_diagonal(J, 3, 2, "cos", Jz, Jp, Jm, I),
            "O32s": _hutchings_off_diagonal(J, 3, 2, "sin", Jz, Jp, Jm, I),
            "O33c": _hutchings_off_diagonal(J, 3, 3, "cos", Jz, Jp, Jm, I),
            "O33s": _hutchings_off_diagonal(J, 3, 3, "sin", Jz, Jp, Jm, I),
        }
    else:
        raise ValueError(
            "multipole_type must be 'magnetic_dipole', 'electric_quadrupole', "
            "or 'magnetic_octupole'"
        )

    if any(val is None for val in ops.values()):
        raise PhysError(
            "FXE-PHYS-001",
            "Missing multipole Stevens formula in implementation",
            module="stevens",
            actual={"J": float(J), "multipole_type": multipole_type},
        )

    out: dict[str, NDArray[np.complexfloating]] = {}
    for name, op in ops.items():
        assert op is not None
        check_hermitian(op, label=f"multipole_{name}", module="stevens")
        out[name] = op
    return out
