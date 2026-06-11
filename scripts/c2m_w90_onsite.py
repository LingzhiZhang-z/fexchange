#!/usr/bin/env python
"""Rotate a standard-frame C2 CEF Hamiltonian into the code axes.

The input B parameters are assumed to be written in the standard local CEF
frame, i.e. standard 100 is local x and standard 001 is local z.  The
``[axis_in_code]`` section tells the script where those standard local axes live
in the code's Cartesian axes.

By default the B parameters are used as given.  Many papers tabulate B_k^q
*divided by* the Stevens factor theta_k (alpha_J/beta_J/gamma_J); to use those
directly, set ``apply_stevens = true`` and ``ion = "Yb"/"Er"/"Dy"/"Nd"`` (or pass
``--apply-stevens --ion Yb``) and each B_k^q is multiplied back by theta_k.

Outputs the shifted CEF levels, the ground-doublet g-tensor, and -- with
``--projector out.txt`` -- the ground Kramers doublet in the fexchange
``inputs.kramer_file`` (stevens mode) format, ready to feed back into the pipeline.

Input TOML example:

    J = 3.5
    units = "meV"
    ion = "Yb"            # only needed when apply_stevens = true
    apply_stevens = false # if true, multiply every B_k^q by the ion Stevens factor

    [B_params]
    B20 = -2.820
    B22 = -25.956
    B40 = 6.170
    B42 = 42.336
    B44 = -36.335
    B60 = -3.004
    B62 = 10.764
    B64 = 7.482
    B66 = 49.327
    B42i = -0.00642
    B44i = -0.015
    B62i = 0.00856
    B64i = -0.067
    B66i = -0.036

    [axis_in_code]
    x = [1, -2, 1]
    z = [1, 0, -1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fexchange.core.space_j import (
    build_space_j_operator,
    build_time_reversal_operator,
    normalize_J,
    pauli_decompose,
    project_J_to_subspace,
)
from fexchange.spectrum.doublet import gauge_fix_kramers_pair
from fexchange.utils.checks import check_hermitian
from fexchange.utils.errors import PhysError
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_EIG_CLUSTER


EPS_ZERO = 1.0e-14


C2_TEMPLATE_NAMES: tuple[str, ...] = (
    "B20",
    "B22c",
    "B22s",
    "B40",
    "B42c",
    "B42s",
    "B44c",
    "B44s",
    "B60",
    "B62c",
    "B62s",
    "B64c",
    "B64s",
    "B66c",
    "B66s",
)


# RE3+ ground-multiplet Stevens (operator-equivalent) factors theta_k:
#   alpha_J = theta_2 (k=2), beta_J = theta_4 (k=4), gamma_J = theta_6 (k=6).
# Values from the standard table (Hutchings 1964 / Abragam-Bleaney).  A literature
# B_k^q quoted "divided by theta_k" must be multiplied back by these before it can
# multiply the bare Stevens operators built here.  Off by default; see analyze().
STEVENS_FACTORS: dict[str, dict[str, float]] = {
    "Nd": {"J": 4.5, "alpha": -0.6428e-2, "beta": -2.911e-4, "gamma": -37.99e-6},
    "Dy": {"J": 7.5, "alpha": -0.6349e-2, "beta": -0.592e-4, "gamma": 1.035e-6},
    "Er": {"J": 7.5, "alpha": 0.2540e-2, "beta": 0.444e-4, "gamma": 2.070e-6},
    "Yb": {"J": 3.5, "alpha": 3.175e-2, "beta": -17.32e-4, "gamma": 148.0e-6},
}

RANK_TO_FACTOR = {"2": "alpha", "4": "beta", "6": "gamma"}


def resolve_stevens_factors(ion: Any, J: float) -> dict[str, float]:
    if not ion:
        raise ValueError(
            'apply_stevens is on but no ion given; set ion = "Yb"/"Er"/"Dy"/"Nd" or pass --ion'
        )
    key = str(ion).strip().capitalize()
    if key not in STEVENS_FACTORS:
        known = ", ".join(sorted(STEVENS_FACTORS))
        raise ValueError(f"unknown ion {ion!r}; known ions: {known}")
    entry = STEVENS_FACTORS[key]
    if abs(float(entry["J"]) - float(J)) > 1.0e-9:
        raise ValueError(f"ion {key} has ground-multiplet J={entry['J']}, but TOML J={J}")
    return entry


def stevens_factor_for_name(name: str, factors: dict[str, float]) -> float:
    """Pick alpha/beta/gamma by Stevens rank k, read off the name (B2.., B4.., B6..)."""
    return float(factors[RANK_TO_FACTOR[name[1]]])


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    with path.open("rb") as f:
        return tomllib.load(f)


def normalize_vec(v: NDArray[np.floating], *, name: str) -> NDArray[np.floating]:
    norm = float(np.linalg.norm(v))
    if norm < EPS_ZERO:
        raise ValueError(f"{name} vector is zero")
    return v / norm


def make_local_frame(section: dict[str, Any], *, name: str) -> dict[str, list[float]]:
    if "x" not in section or "z" not in section:
        raise ValueError(f"[{name}] must provide x and z vectors")

    z = normalize_vec(np.asarray(section["z"], dtype=float), name=f"{name}.z")
    x0 = normalize_vec(np.asarray(section["x"], dtype=float), name=f"{name}.x")
    x = x0 - float(np.dot(x0, z)) * z
    x = normalize_vec(x, name=f"{name}.x projected perpendicular to z")

    if "y" in section:
        y = normalize_vec(np.asarray(section["y"], dtype=float), name=f"{name}.y")
        if abs(float(np.dot(x, y))) > 1.0e-10 or abs(float(np.dot(y, z))) > 1.0e-10:
            raise ValueError(f"[{name}] y is not orthogonal to x/z")
        handed = float(np.dot(np.cross(x, y), z))
        if abs(handed - 1.0) > 1.0e-10:
            raise ValueError(f"[{name}] is not right-handed: dot(x cross y, z) = {handed:.6e}")
    else:
        y = normalize_vec(np.cross(z, x), name=f"{name}.y")

    return {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()}


def _symmetrized_tesseral(
    f: NDArray[np.complexfloating],
    Jp_q: NDArray[np.complexfloating],
    Jm_q: NDArray[np.complexfloating],
    mode: str,
) -> NDArray[np.complexfloating]:
    if mode == "cos":
        xq = Jp_q + Jm_q
        return 0.5 * (f @ xq + xq @ f)
    if mode == "sin":
        xq = Jp_q - Jm_q
        return 0.5 / 1j * (f @ xq + xq @ f)
    raise PhysError("FXE-PHYS-001", "mode must be cos or sin", actual={"mode": mode})


def _hutchings_diagonal(
    J: float,
    k: int,
    Jz: NDArray[np.complexfloating],
    I: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    X = J * (J + 1)
    Jz2 = Jz @ Jz
    if k == 2:
        return 3 * Jz2 - X * I
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
    raise PhysError("FXE-PHYS-001", "unsupported diagonal Stevens rank", actual={"k": k})


def _hutchings_even_offdiag(
    J: float,
    k: int,
    q: int,
    mode: str,
    Jz: NDArray[np.complexfloating],
    Jp: NDArray[np.complexfloating],
    Jm: NDArray[np.complexfloating],
    I: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    X = J * (J + 1)
    Jz2 = Jz @ Jz
    Jz4 = Jz2 @ Jz2
    Jp_q = np.linalg.matrix_power(Jp, q)
    Jm_q = np.linalg.matrix_power(Jm, q)

    if (k, q) == (2, 2):
        f = 0.5 * I
    elif (k, q) == (4, 2):
        f = 0.5 * (7 * Jz2 - X * I - 5 * I)
    elif (k, q) == (4, 4):
        f = 0.5 * I
    elif (k, q) == (6, 2):
        f = 0.5 * (33 * Jz4 - (18 * X + 123) * Jz2 + (X**2 + 10 * X + 102) * I)
    elif (k, q) == (6, 4):
        f = 0.5 * (11 * Jz2 - X * I - 38 * I)
    elif (k, q) == (6, 6):
        f = 0.5 * I
    else:
        raise PhysError("FXE-PHYS-001", "unsupported C2 Stevens operator", actual={"k": k, "q": q})

    return _symmetrized_tesseral(f, Jp_q, Jm_q, mode)


def build_c2_stevens_templates(
    J: float,
    frame: dict[str, list[float]],
) -> dict[str, NDArray[np.complexfloating]]:
    """Build C2 Stevens operators in the supplied local frame."""
    J = normalize_J(J, module="c2m_w90_onsite")
    Jz0, _, _, Jx0, Jy0 = build_space_j_operator(J, module="c2m_w90_onsite")
    dim = int(round(2 * J + 1))
    I = np.eye(dim, dtype=DTYPE_COMPLEX)

    x = np.asarray(frame["x"], dtype=float)
    y = np.asarray(frame["y"], dtype=float)
    z = np.asarray(frame["z"], dtype=float)

    Jx = x[0] * Jx0 + x[1] * Jy0 + x[2] * Jz0
    Jy = y[0] * Jx0 + y[1] * Jy0 + y[2] * Jz0
    Jz = z[0] * Jx0 + z[1] * Jy0 + z[2] * Jz0
    Jp = Jx + 1j * Jy
    Jm = Jx - 1j * Jy

    ops = {
        "B20": _hutchings_diagonal(J, 2, Jz, I),
        "B22c": _hutchings_even_offdiag(J, 2, 2, "cos", Jz, Jp, Jm, I),
        "B22s": _hutchings_even_offdiag(J, 2, 2, "sin", Jz, Jp, Jm, I),
        "B40": _hutchings_diagonal(J, 4, Jz, I),
        "B42c": _hutchings_even_offdiag(J, 4, 2, "cos", Jz, Jp, Jm, I),
        "B42s": _hutchings_even_offdiag(J, 4, 2, "sin", Jz, Jp, Jm, I),
        "B44c": _hutchings_even_offdiag(J, 4, 4, "cos", Jz, Jp, Jm, I),
        "B44s": _hutchings_even_offdiag(J, 4, 4, "sin", Jz, Jp, Jm, I),
        "B60": _hutchings_diagonal(J, 6, Jz, I),
        "B62c": _hutchings_even_offdiag(J, 6, 2, "cos", Jz, Jp, Jm, I),
        "B62s": _hutchings_even_offdiag(J, 6, 2, "sin", Jz, Jp, Jm, I),
        "B64c": _hutchings_even_offdiag(J, 6, 4, "cos", Jz, Jp, Jm, I),
        "B64s": _hutchings_even_offdiag(J, 6, 4, "sin", Jz, Jp, Jm, I),
        "B66c": _hutchings_even_offdiag(J, 6, 6, "cos", Jz, Jp, Jm, I),
        "B66s": _hutchings_even_offdiag(J, 6, 6, "sin", Jz, Jp, Jm, I),
    }

    for name, op in ops.items():
        check_hermitian(op, label=f"C2_{name}", module="c2m_w90_onsite")
    return ops


def canonical_b_name(name: str) -> str:
    stripped = name.strip().replace("_", "").replace("-", "")
    aliases = {
        "B22": "B22c",
        "B42": "B42c",
        "B44": "B44c",
        "B62": "B62c",
        "B64": "B64c",
        "B66": "B66c",
        "BI22": "B22s",
        "BI42": "B42s",
        "BI44": "B44s",
        "BI62": "B62s",
        "BI64": "B64s",
        "BI66": "B66s",
        "B22I": "B22s",
        "B42I": "B42s",
        "B44I": "B44s",
        "B62I": "B62s",
        "B64I": "B64s",
        "B66I": "B66s",
        "B(I)22": "B22s",
        "B(I)42": "B42s",
        "B(I)44": "B44s",
        "B(I)62": "B62s",
        "B(I)64": "B64s",
        "B(I)66": "B66s",
    }
    upper = stripped.upper()
    if name in C2_TEMPLATE_NAMES:
        return name
    if stripped in C2_TEMPLATE_NAMES:
        return stripped
    if upper in aliases:
        return aliases[upper]
    if upper == "B20" or upper == "B40" or upper == "B60":
        return upper
    raise ValueError(f"unknown C2 B parameter name: {name!r}")


def collect_b_params(section: dict[str, Any]) -> dict[str, float]:
    params = {name: 0.0 for name in C2_TEMPLATE_NAMES}
    for key, value in section.items():
        params[canonical_b_name(str(key))] += float(value)
    return params


def build_hcef(
    J: float,
    B_params: dict[str, float],
    frame: dict[str, list[float]],
) -> NDArray[np.complexfloating]:
    ops = build_c2_stevens_templates(J, frame)
    dim = int(round(2 * float(J) + 1))
    H = np.zeros((dim, dim), dtype=DTYPE_COMPLEX)
    for name in C2_TEMPLATE_NAMES:
        H += float(B_params.get(name, 0.0)) * ops[name]
    check_hermitian(H, label="H_CEF_C2", module="c2m_w90_onsite")
    return H


def ground_state(H: NDArray[np.complexfloating], J: float) -> dict[str, Any]:
    """Diagonalize H_CEF, return shifted levels and the ground doublet projector.

    For half-integer J (Kramers ions: Yb, Er, Dy, Nd, ...) the lowest level is a
    Kramers doublet; its two eigenvectors are gauge-fixed exactly as the main
    pipeline's ``select_kramers_doublet`` does, so the resulting W can be fed back
    to fexchange ``inputs.kramer_file`` (stevens mode).  The g-tensor is the J
    operator projected onto the doublet, expressed in the code Cartesian axes.
    """
    J = normalize_J(J, module="c2m_w90_onsite")
    Jz, _, _, Jx, Jy = build_space_j_operator(J, module="c2m_w90_onsite")
    U_T = build_time_reversal_operator(J, module="c2m_w90_onsite")

    evals, evecs = np.linalg.eigh(H)
    e0 = float(evals[0])
    levels = [float(value - e0) for value in evals]
    degeneracy = int(np.count_nonzero(np.abs(evals - evals[0]) < EPS_EIG_CLUSTER))
    gap = float(evals[degeneracy] - e0) if degeneracy < len(evals) else 0.0

    is_kramers = int(round(2 * J)) % 2 == 1
    if not is_kramers:
        raise PhysError(
            "FXE-PHYS-001",
            "ground doublet projector here supports Kramers (half-integer J) ions only; "
            "integer-J non-Kramers cases need the point-group path in cef_states.py",
            module="c2m_w90_onsite",
            actual={"J": float(J)},
        )

    K = np.column_stack([evecs[:, 0], evecs[:, 1]])
    K = gauge_fix_kramers_pair(K, tol=EPS_ZERO)
    tr_residual = float(np.linalg.norm(U_T @ K[:, 0].conj() - K[:, 1]))

    M_Jx, M_Jy, M_Jz = project_J_to_subspace(K, Jx, Jy, Jz, module="c2m_w90_onsite")
    g_tensor = np.zeros((3, 3), dtype=float)
    for row, M in enumerate((M_Jx, M_Jy, M_Jz)):
        dec = pauli_decompose(M)
        g_tensor[row, :] = [2.0 * dec["ox"].real, 2.0 * dec["oy"].real, 2.0 * dec["oz"].real]
    g_principal = sorted((float(v) for v in np.linalg.svd(g_tensor, compute_uv=False)), reverse=True)

    return {
        "levels": levels,
        "degeneracy": degeneracy,
        "gap_to_first_excited": gap,
        "W": K,
        "g_tensor": g_tensor.tolist(),
        "g_principal": g_principal,
        "tr_residual": tr_residual,
    }


def analyze(
    config: dict[str, Any],
    *,
    apply_stevens_cli: bool | None = None,
    ion_cli: str | None = None,
) -> tuple[dict[str, Any], dict[str, NDArray[np.complexfloating]]]:
    J = float(config["J"])
    units = str(config.get("units", "meV"))
    frame_section = config.get("axis_in_code", config.get("local_frame"))
    if frame_section is None:
        raise ValueError("input TOML must provide [axis_in_code] with x and z vectors")
    axis_in_code = make_local_frame(frame_section, name="axis_in_code")
    B_params_input = collect_b_params(config["B_params"])

    apply_stevens = (
        apply_stevens_cli
        if apply_stevens_cli is not None
        else bool(config.get("apply_stevens", False))
    )
    ion = ion_cli if ion_cli is not None else config.get("ion")
    stevens_factors: dict[str, float] | None = None
    B_params = dict(B_params_input)
    if apply_stevens:
        stevens_factors = resolve_stevens_factors(ion, J)
        B_params = {
            name: value * stevens_factor_for_name(name, stevens_factors)
            for name, value in B_params_input.items()
        }

    H = build_hcef(J, B_params, axis_in_code)
    ground = ground_state(H, J)

    # g-values along the local crystal axes: gx along x ([1-21]), gz along the
    # C2 axis z ([10-1]).  g_n = sqrt(n^T (g g^T) n) with g the code-frame g-tensor.
    G = np.asarray(ground["g_tensor"], dtype=float)
    GGt = G @ G.T
    g_axis = {
        label: float(np.sqrt(max(0.0, n @ GGt @ n)))
        for label in ("x", "y", "z")
        for n in [np.asarray(axis_in_code[label], dtype=float)]
    }

    result: dict[str, Any] = {
        "schema_version": "fxe.c2m_cef_rotate.v2",
        "J": J,
        "units": units,
        "apply_stevens": apply_stevens,
        "ion": (str(ion) if apply_stevens else None),
        "stevens_factors": (
            {k: float(stevens_factors[k]) for k in ("alpha", "beta", "gamma")}
            if stevens_factors is not None
            else None
        ),
        "axis_in_code": axis_in_code,
        "B_params_input": B_params_input,
        "B_params": B_params,
        "levels": ground["levels"],
        "ground_degeneracy": ground["degeneracy"],
        "gap_to_first_excited": ground["gap_to_first_excited"],
        "g_axis": g_axis,
        "g_tensor": ground["g_tensor"],
        "g_principal": ground["g_principal"],
        "tr_residual": ground["tr_residual"],
    }
    arrays: dict[str, NDArray[np.complexfloating]] = {
        "hcef": H,
        "W_ground": ground["W"],
    }

    return result, arrays


def _group_levels(levels: list[float], tol: float = 1.0e-6) -> list[tuple[float, int]]:
    """Collapse a sorted level list into (energy, degeneracy) groups."""
    groups: list[list[float]] = []
    for e in levels:
        if groups and abs(e - groups[-1][0]) < tol:
            groups[-1][1] += 1
        else:
            groups.append([e, 1])
    return [(g[0], int(g[1])) for g in groups]


def summary_lines(result: dict[str, Any]) -> list[str]:
    ax = result["axis_in_code"]
    units = result["units"]
    ion = result["ion"] if result.get("ion") else "-"
    lines = [
        f"# CEF on a C2 site   ion={ion}   J={float(result['J']):g}   units={units}",
        f"#   C2 axis  z = [1,0,-1]  -> code {np.round(ax['z'], 6).tolist()}",
        f"#   in-plane x = [1,-2,1]  -> code {np.round(ax['x'], 6).tolist()}",
    ]
    if result.get("stevens_factors"):
        sf = result["stevens_factors"]
        lines.append(
            f"#   Stevens factors applied: alpha={sf['alpha']:.4e} beta={sf['beta']:.4e} gamma={sf['gamma']:.4e}"
        )
    else:
        lines.append("#   Stevens factors NOT applied (B_params used as given)")

    groups = _group_levels(result["levels"])
    lines += [
        "",
        f"Energy levels (shifted, ground = 0) [{units}]",
        f"  {'level':>5}  {'E':>14}  {'deg':>4}  {'gap_to_prev':>14}",
    ]
    prev = 0.0
    for idx, (energy, deg) in enumerate(groups):
        gap = energy - prev
        lines.append(f"  {idx:>5}  {energy:>14.6f}  {deg:>4}  {gap:>14.6f}")
        prev = energy

    ga = result["g_axis"]
    lines += [
        "",
        f"Ground doublet (degeneracy {int(result['ground_degeneracy'])}), "
        f"gap to first excited = {float(result['gap_to_first_excited']):.6f} {units}",
        "Ground g-values:",
        f"  gx = {ga['x']:.6f}   (along in-plane x [1-21])",
        f"  gy = {ga['y']:.6f}",
        f"  gz = {ga['z']:.6f}   (along C2 axis z [10-1])",
        "  g_principal = " + ", ".join(f"{float(v):.6f}" for v in result["g_principal"]),
        f"  (time-reversal pairing residual {float(result['tr_residual']):.1e})",
    ]
    return lines


def write_projector_txt(
    result: dict[str, Any],
    arrays: dict[str, NDArray[np.complexfloating]],
    path: Path,
    kramer_name: str,
) -> None:
    """Write the ground Kramers doublet as a fexchange ``kramer_file`` (stevens mode).

    Format matches ``pipeline._load_stevens``: one ``[W_state_<idx>]`` block per
    doublet state, each with ``2J+1`` ``real imag`` rows in the |J,M> (M=-J..J)
    basis.  Header comment lines are informational (the loader ignores them).
    """
    W = np.asarray(arrays["W_ground"], dtype=np.complex128)
    ga = result["g_axis"]
    gp = result["g_principal"]
    lines = [
        "# schema_version fxe.cef_projector.v1",
        "# standard_version 2026-02",
        f"# kramer_name {kramer_name}",
        "# point_group C2",
        f"# J {float(result['J']):.12e}",
        f"# gx {ga['x']:.12e}",
        f"# gy {ga['y']:.12e}",
        f"# gz {ga['z']:.12e}",
        "# basis_id complex_spherical_j_v1",
        "# orbital_order_id complex_spherical_jm_-J..J_v1",
        "# ground_irrep -",
        "# doublet_type kramers",
        f"# ion {result['ion'] if result.get('ion') else '-'}",
        "# axis_x_in_code " + " ".join(f"{v:.12e}" for v in result["axis_in_code"]["x"]),
        "# axis_z_in_code " + " ".join(f"{v:.12e}" for v in result["axis_in_code"]["z"]),
        f"# g_principal {gp[0]:.12e} {gp[1]:.12e} {gp[2]:.12e}",
        "",
    ]
    for idx in range(W.shape[1]):
        lines.append(f"[W_state_{idx}]")
        lines.extend(f"{val.real:.12e} {val.imag:.12e}" for val in W[:, idx])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_toml_like(result: dict[str, Any], path: Path) -> None:
    lines = [
        f"J = {float(result['J']):.12e}",
        f'units = "{result["units"]}"',
        "apply_stevens = false",
    ]
    if result.get("apply_stevens"):
        lines.append(f'# B_params below already include Stevens factors for ion "{result["ion"]}"')
    lines.extend([
        "",
        "[axis_in_code]",
        "x = [" + ", ".join(f"{v:.12e}" for v in result["axis_in_code"]["x"]) + "]",
        "y = [" + ", ".join(f"{v:.12e}" for v in result["axis_in_code"]["y"]) + "]",
        "z = [" + ", ".join(f"{v:.12e}" for v in result["axis_in_code"]["z"]) + "]",
        "",
        "[B_params]",
    ])
    for name in C2_TEMPLATE_NAMES:
        lines.append(f"{name} = {float(result['B_params'][name]):.12e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="input TOML with J, B_params, axis_in_code")
    parser.add_argument("--output", "-o", type=Path, help="write text summary")
    parser.add_argument("--json", type=Path, help="write JSON result")
    parser.add_argument("--toml", type=Path, help="write normalized TOML-style C2 input")
    parser.add_argument("--npz", type=Path, help="write H_CEF and ground W matrices")
    parser.add_argument(
        "--projector",
        type=Path,
        help="write ground Kramers doublet as a fexchange kramer_file (stevens mode)",
    )
    parser.add_argument("--kramer-name", default="c2m_cef", help="kramer_name written into the projector header")
    parser.add_argument(
        "--apply-stevens",
        dest="apply_stevens",
        action="store_true",
        default=None,
        help="multiply each B_k^q by the ion Stevens factor (overrides TOML apply_stevens)",
    )
    parser.add_argument(
        "--no-apply-stevens",
        dest="apply_stevens",
        action="store_false",
        help="force Stevens factors off (overrides TOML apply_stevens)",
    )
    parser.add_argument("--ion", help="rare-earth ion for Stevens factors (Yb/Er/Dy/Nd); overrides TOML ion")
    args = parser.parse_args(argv)

    try:
        result, arrays = analyze(
            load_toml(args.config),
            apply_stevens_cli=args.apply_stevens,
            ion_cli=args.ion,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    text = "\n".join(summary_lines(result)) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.toml:
        write_toml_like(result, args.toml)
    if args.projector:
        write_projector_txt(result, arrays, args.projector, args.kramer_name)
    if args.npz:
        np.savez_compressed(args.npz, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
