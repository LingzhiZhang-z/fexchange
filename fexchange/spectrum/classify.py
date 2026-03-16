"""
Point-group irrep classification and multipole compatibility.

Supported point groups: Oh, D3d, C3v.
Inputs: J (total angular momentum) and eigenvectors in |J,M> basis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from fexchange.core.space_j import normalize_J, build_space_j_operator
# Re-export tables for consumer convenience.
from fexchange.spectrum.tables import (  # noqa: F401
    ALIAS_MAP,
    C3V_STAR_CHI,
    C3V_STAR_CLASSES,
    C3V_STAR_IRREPS,
    C3V_STAR_SIZES,
    IRREP_DISPLAY,
    MULTIPOLE_RULES,
    O_STAR_CHI,
    O_STAR_CLASSES,
    O_STAR_IRREPS,
    O_STAR_SIZES,
)
from fexchange.utils.numerics import DTYPE_COMPLEX


def parity_from_J(J: float, n_f: int | None = None) -> int:
    """Return inversion parity from J, optionally checking n_f consistency."""
    J = normalize_J(J, module="symmetry")
    twoJ = int(round(2.0 * float(J)))
    p = +1 if (twoJ % 2 == 0) else -1
    if n_f is not None:
        p_nf = +1 if (int(n_f) % 2 == 0) else -1
        if p_nf != p:
            raise ValueError("n_f parity inconsistent with J parity")
    return p


def build_active_irrep_table(J: float, point_group: str, n_f: int | None = None) -> dict[str, object]:
    """Return single-branch active irrep rows for Oh/D3d/C3v."""
    p = parity_from_J(J, n_f=n_f)

    def _rows_from_arrays(
        classes: tuple[str, ...],
        irreps: tuple[str, ...],
        chi: NDArray[np.complexfloating],
    ) -> dict[str, dict[str, complex]]:
        return {
            irrep: {class_name: complex(chi[i, j]) for j, class_name in enumerate(classes)}
            for i, irrep in enumerate(irreps)
        }

    if point_group == "Oh":
        rows = _rows_from_arrays(O_STAR_CLASSES, O_STAR_IRREPS, O_STAR_CHI)
        return {
            "branch_mode": "single",
            "parity": p,
            "core": "O_star",
            "class_sizes": dict(zip(O_STAR_CLASSES, O_STAR_SIZES, strict=True)),
            "rows": rows,
        }

    if point_group == "C3v":
        rows = _rows_from_arrays(C3V_STAR_CLASSES, C3V_STAR_IRREPS, C3V_STAR_CHI)
        return {
            "branch_mode": "single",
            "parity": None,
            "core": "C3v_star",
            "class_sizes": dict(zip(C3V_STAR_CLASSES, C3V_STAR_SIZES, strict=True)),
            "rows": rows,
        }

    if point_group == "D3d":
        tag = "+" if p > 0 else "-"
        core_rows = _rows_from_arrays(C3V_STAR_CLASSES, C3V_STAR_IRREPS, C3V_STAR_CHI)
        rows = {f"{name}{tag}": dict(chars) for name, chars in core_rows.items()}
        return {
            "branch_mode": "single",
            "parity": p,
            "core": "C3v_star",
            "class_sizes": dict(zip(C3V_STAR_CLASSES, C3V_STAR_SIZES, strict=True)),
            "rows": rows,
        }

    raise ValueError(f"Unsupported point group: {point_group}")


def _rotation_D_matrix(J: float, axis: list[float], angle: float) -> NDArray[np.complexfloating]:
    """Build D^J(R) using axis-angle rotation R = exp(-i * angle * n·J)."""
    Jz, _, _, Jx, Jy = build_space_j_operator(J, module="symmetry")

    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    Jn = n[0] * Jx + n[1] * Jy + n[2] * Jz
    return expm(-1j * angle * Jn).astype(DTYPE_COMPLEX, copy=False)


def _rotation_object_to_D(J: float, rot: Rotation) -> NDArray[np.complexfloating]:
    """Convert a scipy Rotation object to D^J matrix via axis-angle."""
    rv = rot.as_rotvec()
    angle = float(np.linalg.norm(rv))
    if angle < 1e-14:
        dim = int(round(2 * J + 1))
        return np.eye(dim, dtype=DTYPE_COMPLEX)
    axis = (rv / angle).tolist()
    return _rotation_D_matrix(J, axis, angle)


def _classify_oh_proper_rotation(rot: Rotation) -> str:
    """Classify a proper cubic rotation into Oh proper classes."""
    rv = rot.as_rotvec()
    angle = float(np.linalg.norm(rv))
    tol = 1e-7
    if angle < tol:
        return "E"
    if abs(angle - 2.0 * np.pi / 3.0) < tol:
        return "C3"
    if abs(angle - np.pi / 2.0) < tol:
        return "C4"
    if abs(angle - np.pi) < tol:
        axis = np.abs(rv / angle)
        small = int(np.sum(axis < 1e-6))
        if small >= 2:
            return "C2p"
        return "C2"
    raise ValueError("Unable to classify Oh rotation")


def _rotation_R_matrix(J: float, dim: int) -> NDArray[np.complexfloating]:
    """Central 2pi rotation in double-group representation."""
    twoJ = int(round(2.0 * float(J)))
    phase = -1.0 if (twoJ % 2 == 1) else 1.0
    return (phase * np.eye(dim, dtype=DTYPE_COMPLEX)).astype(DTYPE_COMPLEX, copy=False)


def _build_o_star_class_sums(J: float) -> dict[str, NDArray[np.complexfloating]]:
    """Exact class-summed operators Σ_{g in class} D^J(g) for O*."""
    dim = int(round(2 * J + 1))
    class_members: dict[str, list[NDArray[np.complexfloating]]] = {
        "E": [],
        "C3": [],
        "C2": [],
        "C4": [],
        "C2p": [],
    }
    for rot in Rotation.create_group("O"):
        cls = _classify_oh_proper_rotation(rot)
        class_members[cls].append(_rotation_object_to_D(J, rot))

    def _sum_mats(mats: list[NDArray[np.complexfloating]]) -> NDArray[np.complexfloating]:
        S = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
        for mat in mats:
            S += mat
        return S

    I = np.eye(dim, dtype=DTYPE_COMPLEX)
    R = _rotation_R_matrix(J, dim)
    S_C3 = _sum_mats(class_members["C3"])
    S_C4 = _sum_mats(class_members["C4"])
    # _classify_oh_proper_rotation labels 3 axis-2pi rotations as C2p and 6 edge-axis pi rotations as C2.
    # O* rotational core uses split classes 3C2/3RC2 and 6C2p/6RC2p.
    S_C2_axis = _sum_mats(class_members["C2p"])
    S_C2_edge = _sum_mats(class_members["C2"])
    return {
        "E": I,
        "R": R,
        "3C2": S_C2_axis,
        "3RC2": R @ S_C2_axis,
        "C4": S_C4,
        "RC4": R @ S_C4,
        "6C2p": S_C2_edge,
        "6RC2p": R @ S_C2_edge,
        "C3": S_C3,
        "RC3": R @ S_C3,
    }


def _build_c3v_star_class_sums(J: float) -> dict[str, NDArray[np.complexfloating]]:
    """Exact class-summed operators Σ_{g in class} D^J(g) for C3v*."""
    dim = int(round(2 * J + 1))
    I = np.eye(dim, dtype=DTYPE_COMPLEX)
    R = _rotation_R_matrix(J, dim)
    C3p = _rotation_D_matrix(J, [0.0, 0.0, 1.0], 2.0 * np.pi / 3.0)
    C3m = _rotation_D_matrix(J, [0.0, 0.0, 1.0], -2.0 * np.pi / 3.0)
    sigma_axes = (
        [1.0, 0.0, 0.0],
        [-0.5, np.sqrt(3.0) / 2.0, 0.0],
        [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
    )
    S_sigma = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
    for axis in sigma_axes:
        S_sigma += _rotation_D_matrix(J, [float(axis[0]), float(axis[1]), float(axis[2])], np.pi)
    S_2C3 = C3p + C3m
    return {
        "E": I,
        "R": R,
        "2C3": S_2C3,
        "2RC3": R @ S_2C3,
        "3sigma_v": S_sigma,
        "3Rsigma_v": R @ S_sigma,
    }


def _build_rotational_core_class_sums(
    J: float,
    point_group: str,
) -> dict[str, NDArray[np.complexfloating]]:
    """Return exact class-summed operators for the active rotational core."""
    if point_group == "Oh":
        return _build_o_star_class_sums(J)
    if point_group in {"D3d", "C3v"}:
        return _build_c3v_star_class_sums(J)
    raise ValueError(f"Unsupported point group for class sums: {point_group}")


def build_projectors(
    J: float,
    point_group: str = "Oh",
    n_f: int | None = None,
) -> dict[str, NDArray[np.complexfloating]]:
    """Build projection operators for all active irreps of the selected point group."""
    active = build_active_irrep_table(J=J, point_group=point_group, n_f=n_f)
    class_sizes = active["class_sizes"]  # type: ignore[index]
    rows = active["rows"]  # type: ignore[index]
    class_sums = _build_rotational_core_class_sums(J, point_group=point_group)

    order = int(sum(class_sizes.values()))
    dim = int(round(2 * J + 1))
    out: dict[str, NDArray[np.complexfloating]] = {}
    for irrep_name, chars in rows.items():
        d_irrep = int(np.real(chars["E"]))
        P = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
        for class_name in class_sizes:
            chi = chars[class_name]
            P += np.conj(chi) * class_sums[class_name]
        out[irrep_name] = (d_irrep / order) * P
    return out


def _build_oh_rep_matrices(J: float) -> dict[str, NDArray[np.complexfloating]]:
    """One representative matrix per Oh conjugacy class."""
    dim = int(round(2 * J + 1))
    D_inv = parity_from_J(J) * np.eye(dim, dtype=DTYPE_COMPLEX)
    pi = np.pi

    reps: dict[str, NDArray[np.complexfloating]] = {}
    reps["E"] = np.eye(dim, dtype=DTYPE_COMPLEX)
    reps["C3"] = _rotation_D_matrix(J, [1.0, 1.0, 1.0], 2.0 * pi / 3.0)
    reps["C2"] = _rotation_D_matrix(J, [1.0, 1.0, 0.0], pi)
    reps["C4"] = _rotation_D_matrix(J, [0.0, 0.0, 1.0], pi / 2.0)
    reps["C2p"] = _rotation_D_matrix(J, [1.0, 0.0, 0.0], pi)
    reps["i"] = D_inv
    reps["S6"] = D_inv @ np.linalg.inv(reps["C3"])
    reps["sigma_d"] = D_inv @ reps["C2"]
    reps["S4"] = D_inv @ np.linalg.inv(reps["C4"])
    reps["sigma_h"] = D_inv @ reps["C2p"]
    return reps


def build_representation_matrices(
    J: float,
    point_group: str = "Oh",
) -> dict[str, NDArray[np.complexfloating]]:
    """Build class-representative D^J(g) matrices for the selected point group."""
    if point_group == "Oh":
        return _build_oh_rep_matrices(J)
    raise ValueError(f"Unsupported point group for representation matrices: {point_group}")


def classify_irreps(
    J: float,
    evecs: NDArray[np.complexfloating],
    point_group: str = "Oh",
    n_f: int | None = None,
) -> list[str]:
    """
    Classify each eigenvector column into an irrep via character projection operators.
    """
    projectors = build_projectors(J, point_group=point_group, n_f=n_f)

    labels: list[str] = []
    eps_proj = 1e-6
    for i in range(evecs.shape[1]):
        psi = evecs[:, i]
        best = "unknown"
        best_norm = 0.0
        for irrep_name, P in projectors.items():
            proj_norm = float(np.linalg.norm(P @ psi))
            if proj_norm > best_norm:
                best_norm = proj_norm
                best = irrep_name
        labels.append(best if best_norm >= eps_proj else "unknown")
    return labels


def irrep_dimension(irrep: str, point_group: str = "Oh") -> int:
    """Return the dimension of the given irrep (character at identity E)."""
    base = irrep.rstrip("+-")
    if point_group == "Oh":
        if base in O_STAR_IRREPS:
            return int(np.real(O_STAR_CHI[O_STAR_IRREPS.index(base), 0]))
    elif point_group in ("C3v", "D3d"):
        if base in C3V_STAR_IRREPS:
            return int(np.real(C3V_STAR_CHI[C3V_STAR_IRREPS.index(base), 0]))
    raise ValueError(f"Unknown irrep {irrep!r} for point group {point_group!r}")


def allowed_multipoles(irrep: str, point_group: str = "Oh") -> list[str]:
    """Return multipole channels carried by the given irrep."""
    return MULTIPOLE_RULES[point_group][irrep]


def irrep_metadata(
    irrep_primary: str,
    point_group: str,
    J: float,
    n_f: int | None = None,
) -> dict[str, object]:
    """Build output metadata for one irrep label."""
    p = parity_from_J(J, n_f=n_f) if point_group in {"Oh", "D3d"} else 0
    aliases = ALIAS_MAP.get((point_group, p), {}).get(irrep_primary, [])
    display = IRREP_DISPLAY.get(irrep_primary, irrep_primary.replace("Gamma", "Γ"))
    return {
        "irrep_display": display,
        "irrep_primary": irrep_primary,
        "irrep_aliases": aliases,
        "mapping_unverified": len(aliases) == 0,
    }


def classify_with_multipoles(
    J: float,
    evecs: NDArray[np.complexfloating],
    point_group: str = "Oh",
    n_f: int | None = None,
) -> dict[str, object]:
    """Classify all states and return the 02-07 single-branch metadata schema."""
    labels = classify_irreps(J, evecs, point_group=point_group, n_f=n_f)
    ground_irrep = labels[0] if labels else "unknown"
    ground_meta = irrep_metadata(ground_irrep, point_group=point_group, J=J, n_f=n_f)
    excited_irreps: list[dict[str, object]] = []
    for i, irrep in enumerate(labels[1:], start=1):
        meta = irrep_metadata(irrep, point_group=point_group, J=J, n_f=n_f)
        excited_irreps.append({"energy_index": i, **meta})
    return {
        **ground_meta,
        "allowed_multipoles": allowed_multipoles(ground_irrep, point_group=point_group),
        "excited_irreps": excited_irreps,
    }


def build_symmetry_metadata(
    J: float,
    evecs: NDArray[np.complexfloating],
    point_group: str = "Oh",
    symmetry_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify eigenstates and merge optional caller-provided symmetry metadata."""
    out: dict[str, Any] = dict(classify_with_multipoles(J, evecs, point_group=point_group))
    if symmetry_meta is not None:
        out.update(symmetry_meta)
    return out


def analyze_cef_symmetry(
    J: float,
    point_group: str = "Oh",
    *,
    B_params: dict[str, float] | None = None,
    evecs: NDArray[np.complexfloating] | None = None,
    n_f: int | None = None,
    symmetry: str | None = None,
    mode_q3: str = "cos",
) -> dict[str, object]:
    """
    Standalone symmetry analysis from either CEF parameters or precomputed eigenvectors.
    """
    if (B_params is None) == (evecs is None):
        raise ValueError("Require exactly one of B_params and evecs")

    eigenvalues: NDArray[np.float64] | None = None
    if B_params is not None:
        from fexchange.hamiltonian.hcef import build_hcef_matrix_J

        hcef_symmetry = symmetry or ("C3v" if point_group == "D3d" else point_group)
        H = build_hcef_matrix_J(J, B_params, symmetry=hcef_symmetry, mode_q3=mode_q3)
        eigenvalues, evecs = np.linalg.eigh(H)

    assert evecs is not None
    classification = classify_with_multipoles(J, evecs, point_group=point_group, n_f=n_f)
    all_irreps = classify_irreps(J, evecs, point_group=point_group, n_f=n_f)

    out: dict[str, object] = {
        "J": float(J),
        "point_group": point_group,
        **classification,
        "all_irreps": all_irreps,
    }
    if eigenvalues is not None:
        out["eigenvalues"] = eigenvalues
    return out
