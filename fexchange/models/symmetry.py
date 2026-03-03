"""
Point-group irrep tables and multipole compatibility.

Supported point groups are added incrementally.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from fexchange.models.hcef import build_J_matrices
from fexchange.utils.numerics import DTYPE_COMPLEX


# Oh single-valued character table (Bethe notation, gerade sector).
CHARACTER_TABLES: dict[str, dict[str, dict[str, int] | dict[str, int]]] = {
    "Oh": {
        "_class_sizes": {
            "E": 1,
            "C3": 8,
            "C2": 6,
            "C4": 6,
            "C2p": 3,
            "i": 1,
            "S6": 8,
            "sigma_d": 6,
            "S4": 6,
            "sigma_h": 3,
        },
        "Gamma1": {
            "E": 1,
            "C3": 1,
            "C2": 1,
            "C4": 1,
            "C2p": 1,
            "i": 1,
            "S6": 1,
            "sigma_d": 1,
            "S4": 1,
            "sigma_h": 1,
        },
        "Gamma2": {
            "E": 1,
            "C3": 1,
            "C2": -1,
            "C4": -1,
            "C2p": 1,
            "i": 1,
            "S6": 1,
            "sigma_d": -1,
            "S4": -1,
            "sigma_h": 1,
        },
        "Gamma3": {
            "E": 2,
            "C3": -1,
            "C2": 0,
            "C4": 0,
            "C2p": 2,
            "i": 2,
            "S6": -1,
            "sigma_d": 0,
            "S4": 0,
            "sigma_h": 2,
        },
        "Gamma4": {
            "E": 3,
            "C3": 0,
            "C2": -1,
            "C4": 1,
            "C2p": -1,
            "i": 3,
            "S6": 0,
            "sigma_d": -1,
            "S4": 1,
            "sigma_h": -1,
        },
        "Gamma5": {
            "E": 3,
            "C3": 0,
            "C2": 1,
            "C4": -1,
            "C2p": -1,
            "i": 3,
            "S6": 0,
            "sigma_d": 1,
            "S4": -1,
            "sigma_h": -1,
        },
    },
    "D3d": {
        "_class_sizes": {
            "E": 1,
            "2C3": 2,
            "3C2": 3,
            "i": 1,
            "2S6": 2,
            "3sigma_d": 3,
        },
        "A1g": {
            "E": 1,
            "2C3": 1,
            "3C2": 1,
            "i": 1,
            "2S6": 1,
            "3sigma_d": 1,
        },
        "A2g": {
            "E": 1,
            "2C3": 1,
            "3C2": -1,
            "i": 1,
            "2S6": 1,
            "3sigma_d": -1,
        },
        "Eg": {
            "E": 2,
            "2C3": -1,
            "3C2": 0,
            "i": 2,
            "2S6": -1,
            "3sigma_d": 0,
        },
        "A1u": {
            "E": 1,
            "2C3": 1,
            "3C2": 1,
            "i": -1,
            "2S6": -1,
            "3sigma_d": -1,
        },
        "A2u": {
            "E": 1,
            "2C3": 1,
            "3C2": -1,
            "i": -1,
            "2S6": -1,
            "3sigma_d": 1,
        },
        "Eu": {
            "E": 2,
            "2C3": -1,
            "3C2": 0,
            "i": -2,
            "2S6": 1,
            "3sigma_d": 0,
        },
    },
}


MULTIPOLE_RULES: dict[str, dict[str, list[str]]] = {
    "Oh": {
        "Gamma1": ["octupole"],
        "Gamma2": ["octupole"],
        "Gamma3": ["quadrupole"],
        "Gamma4": ["dipole", "octupole"],
        "Gamma5": ["quadrupole", "octupole"],
    },
    "D3d": {
        "A1g": [],
        "A2g": ["dipole"],
        "Eg": ["dipole", "quadrupole"],
        "A1u": [],
        "A2u": ["octupole"],
        "Eu": ["quadrupole", "octupole"],
    },
}


ROTATIONAL_CORE_TABLES: dict[str, dict[str, dict[str, int] | dict[str, complex]]] = {
    "O_star": {
        "class_sizes": {
            "E": 1,
            "R": 1,
            "C2_mix": 6,
            "C4": 6,
            "RC4": 6,
            "C2p_mix": 12,
            "C3": 8,
            "RC3": 8,
        },
        "rows": {
            "Gamma1": {
                "E": 1, "R": 1, "C2_mix": 1, "C4": 1, "RC4": 1, "C2p_mix": 1, "C3": 1, "RC3": 1,
            },
            "Gamma2": {
                "E": 1, "R": 1, "C2_mix": 1, "C4": -1, "RC4": -1, "C2p_mix": -1, "C3": 1, "RC3": 1,
            },
            "Gamma3": {
                "E": 2, "R": 2, "C2_mix": 2, "C4": 0, "RC4": 0, "C2p_mix": 0, "C3": -1, "RC3": -1,
            },
            "Gamma4": {
                "E": 3, "R": 3, "C2_mix": -1, "C4": 1, "RC4": 1, "C2p_mix": -1, "C3": 0, "RC3": 0,
            },
            "Gamma5": {
                "E": 3, "R": 3, "C2_mix": -1, "C4": -1, "RC4": -1, "C2p_mix": 1, "C3": 0, "RC3": 0,
            },
            "Gamma6": {
                "E": 2, "R": -2, "C2_mix": 0, "C4": np.sqrt(2), "RC4": -np.sqrt(2), "C2p_mix": 0, "C3": 1, "RC3": -1,
            },
            "Gamma7": {
                "E": 2, "R": -2, "C2_mix": 0, "C4": -np.sqrt(2), "RC4": np.sqrt(2), "C2p_mix": 0, "C3": 1, "RC3": -1,
            },
            "Gamma8": {
                "E": 4, "R": -4, "C2_mix": 0, "C4": 0, "RC4": 0, "C2p_mix": 0, "C3": -1, "RC3": 1,
            },
        },
    },
    "C3v_star": {
        "class_sizes": {
            "E": 1,
            "R": 1,
            "2C3": 2,
            "2RC3": 2,
            "3sigma_v": 3,
            "3Rsigma_v": 3,
        },
        "rows": {
            "Gamma1": {
                "E": 1, "R": 1, "2C3": 1, "2RC3": 1, "3sigma_v": 1, "3Rsigma_v": 1,
            },
            "Gamma2": {
                "E": 1, "R": 1, "2C3": 1, "2RC3": 1, "3sigma_v": -1, "3Rsigma_v": -1,
            },
            "Gamma3": {
                "E": 2, "R": 2, "2C3": -1, "2RC3": -1, "3sigma_v": 0, "3Rsigma_v": 0,
            },
            "Gamma4": {
                "E": 2, "R": -2, "2C3": 1, "2RC3": -1, "3sigma_v": 0, "3Rsigma_v": 0,
            },
            "Gamma5": {
                "E": 1, "R": -1, "2C3": -1, "2RC3": 1, "3sigma_v": 1j, "3Rsigma_v": -1j,
            },
            "Gamma6": {
                "E": 1, "R": -1, "2C3": -1, "2RC3": 1, "3sigma_v": -1j, "3Rsigma_v": 1j,
            },
        },
    },
}


def parity_from_J(J: float, n_f: int | None = None) -> int:
    """Return inversion parity from J, optionally checking n_f consistency."""
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

    if point_group == "Oh":
        rows = dict(ROTATIONAL_CORE_TABLES["O_star"]["rows"])  # type: ignore[index]
        return {
            "branch_mode": "single",
            "parity": p,
            "core": "O_star",
            "class_sizes": dict(ROTATIONAL_CORE_TABLES["O_star"]["class_sizes"]),  # type: ignore[index]
            "rows": rows,
        }

    if point_group == "C3v":
        rows = dict(ROTATIONAL_CORE_TABLES["C3v_star"]["rows"])  # type: ignore[index]
        return {
            "branch_mode": "single",
            "parity": None,
            "core": "C3v_star",
            "class_sizes": dict(ROTATIONAL_CORE_TABLES["C3v_star"]["class_sizes"]),  # type: ignore[index]
            "rows": rows,
        }

    if point_group == "D3d":
        tag = "+" if p > 0 else "-"
        core_rows = ROTATIONAL_CORE_TABLES["C3v_star"]["rows"]  # type: ignore[index]
        rows = {f"{name}{tag}": dict(chars) for name, chars in core_rows.items()}
        return {
            "branch_mode": "single",
            "parity": p,
            "core": "C3v_star",
            "class_sizes": dict(ROTATIONAL_CORE_TABLES["C3v_star"]["class_sizes"]),  # type: ignore[index]
            "rows": rows,
        }

    raise ValueError(f"Unsupported point group: {point_group}")


def _rotation_D_matrix(J: float, axis: list[float], angle: float) -> NDArray[np.complexfloating]:
    """Build D^J(R) using axis-angle rotation R = exp(-i * angle * n·J)."""
    Jz, Jp, Jm = build_J_matrices(J)
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)

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


def _build_oh_class_operator_sums(J: float) -> dict[str, NDArray[np.complexfloating]]:
    """Build exact class-summed operators Σ_{g in class} D^J(g) for Oh."""
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

    class_sums: dict[str, NDArray[np.complexfloating]] = {}
    for name, mats in class_members.items():
        S = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
        for mat in mats:
            S += mat
        class_sums[name] = S

    D_inv = ((-1) ** int(round(J))) * np.eye(dim, dtype=DTYPE_COMPLEX)
    class_sums["i"] = D_inv
    class_sums["S6"] = D_inv @ class_sums["C3"]
    class_sums["sigma_d"] = D_inv @ class_sums["C2"]
    class_sums["S4"] = D_inv @ class_sums["C4"]
    class_sums["sigma_h"] = D_inv @ class_sums["C2p"]
    return class_sums


def _build_d3d_class_operator_sums(J: float) -> dict[str, NDArray[np.complexfloating]]:
    """Build exact class-summed operators Σ_{g in class} D^J(g) for D3d."""
    dim = int(round(2 * J + 1))
    I = np.eye(dim, dtype=DTYPE_COMPLEX)
    D_c3_p = _rotation_D_matrix(J, [0.0, 0.0, 1.0], 2.0 * np.pi / 3.0)
    D_c3_m = _rotation_D_matrix(J, [0.0, 0.0, 1.0], -2.0 * np.pi / 3.0)
    c2_axes = (
        [1.0, 0.0, 0.0],
        [-0.5, np.sqrt(3.0) / 2.0, 0.0],
        [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
    )
    S_3C2 = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
    for axis in c2_axes:
        S_3C2 += _rotation_D_matrix(J, [float(axis[0]), float(axis[1]), float(axis[2])], np.pi)

    D_inv = ((-1) ** int(round(J))) * I
    return {
        "E": I,
        "2C3": D_c3_p + D_c3_m,
        "3C2": S_3C2,
        "i": D_inv,
        "2S6": D_inv @ (D_c3_p + D_c3_m),
        "3sigma_d": D_inv @ S_3C2,
    }


def _build_class_operator_sums(J: float, point_group: str) -> dict[str, NDArray[np.complexfloating]]:
    """Return exact class-summed operators for supported point groups."""
    if point_group == "Oh":
        return _build_oh_class_operator_sums(J)
    if point_group == "D3d":
        return _build_d3d_class_operator_sums(J)
    raise ValueError(f"Unsupported point group for class sums: {point_group}")


def _build_oh_rep_matrices(J: float) -> dict[str, NDArray[np.complexfloating]]:
    """One representative matrix per Oh conjugacy class."""
    dim = int(round(2 * J + 1))
    D_inv = ((-1) ** int(round(J))) * np.eye(dim, dtype=DTYPE_COMPLEX)
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
) -> list[str]:
    """
    Classify each eigenvector column into an irrep via character projection operators.
    """
    table = CHARACTER_TABLES[point_group]
    class_sizes = table["_class_sizes"]  # type: ignore[index]
    order = int(sum(class_sizes.values()))
    class_sums = _build_class_operator_sums(J, point_group=point_group)

    dim = int(round(2 * J + 1))
    projectors: dict[str, NDArray[np.complexfloating]] = {}
    for irrep_name, chars in table.items():
        if irrep_name == "_class_sizes":
            continue
        d_irrep = int(chars["E"])  # type: ignore[index]
        P = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
        for class_name in class_sizes:
            chi = chars[class_name]  # type: ignore[index]
            P += np.conj(chi) * class_sums[class_name]
        projectors[irrep_name] = (d_irrep / order) * P

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


def allowed_multipoles(irrep: str, point_group: str = "Oh") -> list[str]:
    """Return multipole channels carried by the given irrep."""
    return MULTIPOLE_RULES[point_group][irrep]


def classify_with_multipoles(
    J: float,
    evecs: NDArray[np.complexfloating],
    point_group: str = "Oh",
) -> dict[str, object]:
    """Classify all states and attach ground-state multipole compatibility."""
    labels = classify_irreps(J, evecs, point_group=point_group)
    ground_irrep = labels[0]
    return {
        "ground": {
            "irrep": ground_irrep,
            "multipoles": allowed_multipoles(ground_irrep, point_group=point_group),
        },
        "excited": [{"irrep": labels[i], "energy_index": i} for i in range(1, len(labels))],
    }


def analyze_cef_symmetry(
    J: float,
    point_group: str = "Oh",
    *,
    B_params: dict[str, float] | None = None,
    evecs: NDArray[np.complexfloating] | None = None,
    symmetry: str = "Oh",
    mode_q3: str = "cos",
) -> dict[str, object]:
    """
    Standalone symmetry analysis from either CEF parameters or precomputed eigenvectors.
    """
    if B_params is None and evecs is None:
        raise ValueError("Must provide either B_params or evecs")
    if B_params is not None and evecs is not None:
        raise ValueError("Provide B_params or evecs, not both")

    eigenvalues: NDArray[np.float64] | None = None
    if B_params is not None:
        from fexchange.models.hcef import build_hcef_matrix_J

        H = build_hcef_matrix_J(J, B_params, symmetry=symmetry, mode_q3=mode_q3)
        eigenvalues, evecs = np.linalg.eigh(H)

    assert evecs is not None
    classification = classify_with_multipoles(J, evecs, point_group=point_group)
    all_irreps = classify_irreps(J, evecs, point_group=point_group)

    out: dict[str, object] = {
        "J": float(J),
        "point_group": point_group,
        "ground": classification["ground"],
        "excited": classification["excited"],
        "all_irreps": all_irreps,
    }
    if eigenvalues is not None:
        out["eigenvalues"] = eigenvalues
    return out
