"""Canonical basis and gauge helpers for ground-state doublets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from fexchange.core.space_j import (
    build_space_j_operator,
    build_time_reversal_operator,
    normalize_J,
)
from fexchange.core.stevens import build_multipole_operators
from fexchange.utils.numerics import DTYPE_COMPLEX

_SUPPORTED_J = {4, 6, 8}


# Kramers gauge helper
def gauge_fix_kramers_pair(
    Psi: NDArray[np.complexfloating],
    *,
    tol: float,
) -> NDArray[np.complexfloating]:
    """Gauge-fix a Kramers doublet subspace."""
    dim = Psi.shape[0]
    J = 0.5 * (dim - 1)
    Jz, _, _, Jx, Jy = build_space_j_operator(J, module="doublet_basis")
    U_T = build_time_reversal_operator(J, module="doublet_basis")
    U = np.eye(2, dtype=DTYPE_COMPLEX)

    M_Jz = Psi.conj().T @ Jz @ Psi
    evals, vecs_z = np.linalg.eigh(M_Jz)
    U = U @ vecs_z[:, np.argsort(evals.real)[::-1]]

    basis = Psi @ U
    M_Jx = basis.conj().T @ Jx @ basis
    off_diag = M_Jx[0, 1]
    if abs(off_diag) > tol:
        phase = np.angle(off_diag)
        U = U @ np.diag([np.exp(-0.5j * phase), np.exp(0.5j * phase)]).astype(DTYPE_COMPLEX)

    basis = Psi @ U
    w1 = basis[:, 0]
    w2 = U_T @ w1.conj()
    basis_tr = np.column_stack([w1, w2])
    U_tr = basis.conj().T @ basis_tr
    U = U @ U_tr

    basis = Psi @ U
    M_Jx = basis.conj().T @ Jx @ basis
    M_Jy = basis.conj().T @ Jy @ basis
    M_Jz = basis.conj().T @ Jz @ basis

    def satisfies_sign_rule(values: tuple[float, float, float]) -> bool:
        nonzero = [value for value in values if abs(value) > tol]
        if len(nonzero) == 3:
            return len({int(np.sign(value)) for value in nonzero}) == 1
        if len(nonzero) == 2:
            return all(value > tol for value in nonzero)
        if len(nonzero) == 1:
            return nonzero[0] > tol
        return True

    gx = float(np.real(M_Jx[0, 1]))
    gy = float(np.real(1j * M_Jy[0, 1]))
    gz = float(np.real(M_Jz[0, 0]))

    Ux = np.array([[0.0, 1.0j], [1.0j, 0.0]], dtype=DTYPE_COMPLEX)
    Uy = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE_COMPLEX)
    Uz = np.array([[1.0j, 0.0], [0.0, -1.0j]], dtype=DTYPE_COMPLEX)
    candidates = (
        (np.eye(2, dtype=DTYPE_COMPLEX), (gx, gy, gz)),
        (Ux, (gx, -gy, -gz)),
        (Uy, (-gx, gy, -gz)),
        (Uz, (-gx, -gy, gz)),
    )
    for rot, values in candidates:
        if satisfies_sign_rule(values):
            U = U @ rot
            break

    return Psi @ U


# Non-Kramers analytical primitives
def _require_supported_j(J: float) -> float:
    """Restrict the analytical builder to the supported integer J set."""
    J = normalize_J(J, module="nonkramers_basis")
    j_int = int(round(J))
    if j_int not in _SUPPORTED_J:
        raise ValueError(f"Unsupported J={J}; expected one of {sorted(_SUPPORTED_J)}")
    return J


def _require_expected_length(
    coeffs: Sequence[object],
    *,
    expected: int,
    label: str,
) -> None:
    """Reject silently truncated or over-specified coefficient lists."""
    if len(coeffs) != expected:
        raise ValueError(
            f"{label} length mismatch: expected {expected}, got {len(coeffs)}"
        )


def _normalize_or_raise(
    vector: NDArray[np.complexfloating] | NDArray[np.floating],
    *,
    label: str,
) -> NDArray[np.complexfloating] | NDArray[np.floating]:
    """Normalize a state vector and fail hard on zero-norm input."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{label} must not be the zero vector")
    return vector / norm


def _canonical_point_group(point_group: str) -> str:
    """Route equivalent trigonal non-Kramers interfaces through one backend."""
    if point_group == "C3v":
        return "D3d"
    return point_group


def allowed_m_values(J: float, mod: int, residue: int) -> list[int]:
    """Return descending m values in [-J, J] matching the requested congruence."""
    J = normalize_J(J, module="nonkramers_basis")
    j_int = int(round(J))
    return [m for m in range(j_int, -j_int - 1, -1) if m % mod == residue % mod]


def cos_basis_vector(J: float, m: int) -> NDArray[np.float64]:
    """Return C_m = (|m> + |-m>) / sqrt(2), or |0> when m == 0."""
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")

    J = normalize_J(J, module="nonkramers_basis")
    j_int = int(round(J))
    dim = int(round(2 * J + 1))
    vector = np.zeros(dim, dtype=np.float64)

    if m == 0:
        vector[j_int] = 1.0
        return vector

    scale = 1.0 / np.sqrt(2.0)
    vector[j_int - m] = scale
    vector[j_int + m] = scale
    return vector


def _ordered_abs_m_values(m_values: Sequence[int]) -> list[int]:
    """Return distinct absolute m values in ascending order."""
    ordered: list[int] = []
    for m in m_values:
        abs_m = abs(m)
        if abs_m not in ordered:
            ordered.append(abs_m)
    ordered.sort()
    return ordered


def build_oh_eg_doublet(
    J: float,
    coeffs_u: Sequence[float],
    coeffs_v: Sequence[float],
) -> NDArray[np.float64]:
    """Build an Oh E_g doublet from real cosine-sector coefficients."""
    J = _require_supported_j(J)
    dim = int(round(2 * J + 1))

    basis_u = _ordered_abs_m_values(allowed_m_values(J, 4, 0))
    basis_v = _ordered_abs_m_values(allowed_m_values(J, 4, 2))
    _require_expected_length(coeffs_u, expected=len(basis_u), label="coeffs_u")
    _require_expected_length(coeffs_v, expected=len(basis_v), label="coeffs_v")

    partner_u = np.zeros(dim, dtype=np.float64)
    for coeff, abs_m in zip(coeffs_u, basis_u):
        partner_u += float(coeff) * cos_basis_vector(J, abs_m)

    partner_v = np.zeros(dim, dtype=np.float64)
    for coeff, abs_m in zip(coeffs_v, basis_v):
        partner_v += float(coeff) * cos_basis_vector(J, abs_m)

    partner_u = np.asarray(_normalize_or_raise(partner_u, label="Oh partner_u"), dtype=np.float64)
    partner_v = np.asarray(_normalize_or_raise(partner_v, label="Oh partner_v"), dtype=np.float64)

    return np.column_stack([partner_u, partner_v])


def build_d3d_eg_doublet(
    J: float,
    coeffs: Sequence[complex],
) -> NDArray[np.complex128]:
    """Build a D3d E_g doublet from the m ≡ 1 (mod 3) sector."""
    J = _require_supported_j(J)
    j_int = int(round(J))
    dim = int(round(2 * J + 1))
    m_values = allowed_m_values(J, 3, 1)
    _require_expected_length(coeffs, expected=len(m_values), label="coeffs")

    psi_1 = np.zeros(dim, dtype=DTYPE_COMPLEX)
    for coeff, m in zip(coeffs, m_values):
        psi_1[j_int + m] = complex(coeff)

    psi_1 = np.asarray(_normalize_or_raise(psi_1, label="D3d partner_1"), dtype=DTYPE_COMPLEX)

    U_T = build_time_reversal_operator(J)
    psi_2 = (U_T @ psi_1.conj()).astype(DTYPE_COMPLEX)
    return np.column_stack([psi_1, psi_2])


# Non-Kramers gauge/projector helpers
def gauge_fix_non_kramers_oh(
    Psi: NDArray[np.complexfloating],
    tol: float,
) -> NDArray[np.complexfloating]:
    """Gauge-fix a non-Kramers Oh Gamma_3 doublet subspace."""
    dim = Psi.shape[0]
    J = 0.5 * (dim - 1)
    even_anchor = build_multipole_operators(J, "electric_quadrupole")["O20"]
    M_anchor = Psi.conj().T @ even_anchor @ Psi
    evals, vecs = np.linalg.eigh(M_anchor)
    basis = Psi @ vecs[:, np.argsort(evals.real)[::-1]]
    basis = basis.astype(DTYPE_COMPLEX)
    for i in range(2):
        col = basis[:, i].copy()
        pivot = int(np.argmax(np.abs(col)))
        phase = np.angle(col[pivot])
        if abs(phase) > tol:
            basis[:, i] = col * np.exp(-1j * phase)
    return basis


def gauge_fix_non_kramers_d3d(
    Psi: NDArray[np.complexfloating],
    tol: float,
) -> NDArray[np.complexfloating]:
    """Gauge-fix a non-Kramers D3d E_g doublet subspace."""
    dim = Psi.shape[0]
    J = 0.5 * (dim - 1)
    U_T = build_time_reversal_operator(J, module="doublet_basis")
    odd_anchor = build_multipole_operators(J, "magnetic_dipole")["Jz"]
    M_anchor = Psi.conj().T @ odd_anchor @ Psi
    evals, vecs = np.linalg.eigh(M_anchor)
    basis = Psi @ vecs[:, np.argsort(evals.real)[::-1]]
    w1 = basis[:, 0].copy()
    pivot = int(np.argmax(np.abs(w1)))
    phase = np.angle(w1[pivot])
    if abs(phase) > tol:
        w1 = w1 * np.exp(-1j * phase)
    w2 = (U_T @ w1.conj()).astype(DTYPE_COMPLEX)
    return np.column_stack([w1, w2]).astype(DTYPE_COMPLEX)


def _fix_real_vector_sign(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Make the dominant component positive for deterministic real coefficients."""
    if vector.size == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        return -vector
    return vector


def _dominant_real_coeffs(
    projected: NDArray[np.complexfloating],
    tol: float,
) -> NDArray[np.float64]:
    """Recover a real rank-1 sector coefficient vector from a scrambled subspace."""
    gram = projected @ projected.conj().T
    evals, vecs = np.linalg.eigh(gram)
    coeffs = np.real_if_close(vecs[:, int(np.argmax(evals.real))], tol=1000)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if np.linalg.norm(coeffs) <= tol:
        raise ValueError("Projected Oh sector has negligible weight")
    return _fix_real_vector_sign(coeffs)


def _fix_complex_phase(vector: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Make the dominant component real positive for deterministic phase choice."""
    if vector.size == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    amp = vector[pivot]
    if abs(amp) == 0.0:
        return vector
    return vector * np.exp(-1j * np.angle(amp))


def _dominant_complex_coeffs(
    projected: NDArray[np.complexfloating],
    tol: float,
) -> NDArray[np.complexfloating]:
    """Recover the dominant complex sector coefficient vector from a scrambled subspace."""
    gram = projected @ projected.conj().T
    evals, vecs = np.linalg.eigh(gram)
    coeffs = vecs[:, int(np.argmax(evals.real))].astype(DTYPE_COMPLEX)
    if np.linalg.norm(coeffs) <= tol:
        raise ValueError("Projected D3d sector has negligible weight")
    return _fix_complex_phase(coeffs)


def project_to_eg_doublet(
    J: float,
    Psi: NDArray[np.complexfloating],
    point_group: str,
    *,
    tol: float = 1e-10,
) -> NDArray[np.complexfloating]:
    """Project a raw doublet basis onto analytical E_g sectors and gauge-fix it."""
    J = _require_supported_j(J)
    dim = int(round(2 * J + 1))
    if Psi.ndim != 2 or Psi.shape != (dim, 2):
        raise ValueError(
            f"Psi shape mismatch: expected ({dim}, 2), got {tuple(Psi.shape)}"
        )
    point_group = _canonical_point_group(point_group)

    if point_group == "Oh":
        return _project_oh_eg_doublet(J, Psi, tol)
    if point_group == "D3d":
        return _project_d3d_eg_doublet(J, Psi, tol)
    raise ValueError(f"Unsupported point_group: {point_group}")


def _project_oh_eg_doublet(
    J: float,
    Psi: NDArray[np.complexfloating],
    tol: float,
) -> NDArray[np.complexfloating]:
    """Project onto the Oh cosine sectors, rebuild analytically, then gauge-fix."""
    basis_u = np.column_stack(
        [cos_basis_vector(J, abs_m) for abs_m in _ordered_abs_m_values(allowed_m_values(J, 4, 0))]
    )
    basis_v = np.column_stack(
        [cos_basis_vector(J, abs_m) for abs_m in _ordered_abs_m_values(allowed_m_values(J, 4, 2))]
    )

    coeffs_u = _dominant_real_coeffs(basis_u.T @ Psi, tol)
    coeffs_v = _dominant_real_coeffs(basis_v.T @ Psi, tol)
    rebuilt = build_oh_eg_doublet(J, coeffs_u.tolist(), coeffs_v.tolist()).astype(DTYPE_COMPLEX)
    return gauge_fix_non_kramers_oh(rebuilt, tol)


def _project_d3d_eg_doublet(
    J: float,
    Psi: NDArray[np.complexfloating],
    tol: float,
) -> NDArray[np.complexfloating]:
    """Project onto the D3d m ≡ 1 (mod 3) sector, rebuild, then gauge-fix."""
    j_int = int(round(J))
    indices = [j_int + m for m in allowed_m_values(J, 3, 1)]
    coeffs = _dominant_complex_coeffs(Psi[indices, :], tol)
    rebuilt = build_d3d_eg_doublet(J, coeffs.tolist()).astype(DTYPE_COMPLEX)
    return gauge_fix_non_kramers_d3d(rebuilt, tol)
