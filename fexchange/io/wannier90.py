"""
Wannier90 parsing, mapping and reduced hopping construction.

Spec reference:
- 05-02-WANNIER90_INPUT_CONTRACT
- 05-03-WANNIER90_PARSING_RULES
"""

from __future__ import annotations

import logging
import math
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.core.fock_basis import N_ORB
from fexchange.utils.checks import check_unitary, check_hermitian
from fexchange.utils.errors import BindError, W90Error
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_UNITARY, EPS_ZERO, EPS_HERM

logger = logging.getLogger("fexchange")


def parse_hr_dat(path: str | Path) -> dict[str, Any]:
    """
    Parse ``wannier90_hr.dat`` with deterministic line-order interpretation.
    """
    path = Path(path)
    if not path.exists():
        raise W90Error("FXE-W90-001", f"Missing wannier90_hr.dat: {path}", paths={"hr_path": str(path)})

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if len(lines) < 3:
        raise W90Error("FXE-W90-001", f"Malformed hr file (too short): {path}", paths={"hr_path": str(path)})

    try:
        comment = lines[0]
        num_wann = int(lines[1].strip())
        nrpts = int(lines[2].strip())
    except Exception as exc:  # pragma: no cover - defensive parse gate
        raise W90Error(
            "FXE-W90-001",
            f"Cannot parse hr header: {path}",
            paths={"hr_path": str(path)},
        ) from exc

    n_weight_lines = math.ceil(nrpts / 15)
    if len(lines) < 3 + n_weight_lines:
        raise W90Error(
            "FXE-W90-001",
            f"Malformed hr file: missing degeneracy lines ({path})",
            paths={"hr_path": str(path)},
        )

    weights: list[int] = []
    for i in range(n_weight_lines):
        try:
            weights.extend(int(x) for x in lines[3 + i].split())
        except Exception as exc:
            raise W90Error(
                "FXE-W90-001",
                f"Invalid degeneracy line #{i} in {path}",
                paths={"hr_path": str(path)},
            ) from exc
    if len(weights) < nrpts:
        raise W90Error(
            "FXE-W90-001",
            f"Not enough degeneracy weights in {path}",
            actual={"nrpts": nrpts, "n_weights": len(weights)},
            paths={"hr_path": str(path)},
        )
    weights = weights[:nrpts]

    data_start = 3 + n_weight_lines
    expected_rows = nrpts * num_wann * num_wann
    if len(lines) - data_start < expected_rows:
        raise W90Error(
            "FXE-W90-001",
            f"Malformed hr data block in {path}",
            actual={"expected_rows": expected_rows, "actual_rows": len(lines) - data_start},
            paths={"hr_path": str(path)},
        )

    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]] = {}
    line_idx = data_start
    for r_idx in range(nrpts):
        mat = np.zeros((num_wann, num_wann), dtype=DTYPE_COMPLEX)
        R_last = (0, 0, 0)
        w = weights[r_idx]
        if w == 0:
            raise W90Error(
                "FXE-W90-001",
                f"Degeneracy weight cannot be zero in {path}",
                actual={"r_idx": r_idx},
                paths={"hr_path": str(path)},
            )
        for _j in range(num_wann):
            for _i in range(num_wann):
                parts = lines[line_idx].split()
                line_idx += 1
                if len(parts) < 7:
                    raise W90Error(
                        "FXE-W90-001",
                        f"Malformed hr matrix row at line {line_idx}",
                        actual={"line": line_idx, "content": lines[line_idx - 1]},
                        paths={"hr_path": str(path)},
                    )
                R1, R2, R3 = int(parts[0]), int(parts[1]), int(parts[2])
                ii, jj = int(parts[3]) - 1, int(parts[4]) - 1
                re_h, im_h = float(parts[5]), float(parts[6])
                if not (0 <= ii < num_wann and 0 <= jj < num_wann):
                    raise W90Error(
                        "FXE-W90-001",
                        "Wannier matrix index out of range in hr file",
                        actual={"i": ii, "j": jj, "num_wann": num_wann},
                        paths={"hr_path": str(path)},
                    )
                mat[ii, jj] = complex(re_h, im_h) / float(w)
                R_last = (R1, R2, R3)
        H_R[R_last] = mat

    return {
        "comment": comment,
        "num_wann": num_wann,
        "nrpts": nrpts,
        "weights": np.array(weights, dtype=np.int64),
        "H_R": H_R,
    }


def parse_win(path: str | Path) -> dict[str, Any]:
    """
    Parse minimal required fields from ``wannier.win``.
    """
    path = Path(path)
    if not path.exists():
        raise W90Error("FXE-W90-001", f"Missing wannier.win: {path}", paths={"win_path": str(path)})

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    num_wann: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            if key.strip().lower() == "num_wann":
                try:
                    num_wann = int(value.strip())
                except Exception as exc:
                    raise W90Error(
                        "FXE-W90-001",
                        f"Invalid num_wann in {path}",
                        paths={"win_path": str(path)},
                    ) from exc
                break
    if num_wann is None:
        raise W90Error("FXE-W90-001", f"Missing num_wann in {path}", paths={"win_path": str(path)})

    projections = _parse_block(lines, "projections")
    atoms = _parse_block(lines, "atoms_frac") or _parse_block(lines, "atoms_cart")
    unit_cell = _parse_block(lines, "unit_cell_cart")
    if projections is None or atoms is None or unit_cell is None:
        raise W90Error(
            "FXE-W90-001",
            f"wannier.win missing required blocks: projections/atoms/unit_cell ({path})",
            paths={"win_path": str(path)},
        )

    return {
        "num_wann": num_wann,
        "projections": projections,
        "atoms": atoms,
        "unit_cell": unit_cell,
    }


def _parse_block(lines: list[str], block_name: str) -> list[str] | None:
    begin = f"begin {block_name}".lower()
    end = f"end {block_name}".lower()
    in_block = False
    block_lines: list[str] = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped == begin:
            in_block = True
            continue
        if stripped == end and in_block:
            return block_lines
        if in_block:
            content = line.strip()
            if content:
                block_lines.append(content)
    return None


def fetch_H(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
    m: int,
    n: int,
    R: tuple[int, int, int],
) -> complex:
    """
    Fetch ``H[m,n](R)`` with Hermitian completion fallback.
    """
    R_t = (int(R[0]), int(R[1]), int(R[2]))
    if R_t in H_R:
        return complex(H_R[R_t][m, n])
    R_neg = (-R_t[0], -R_t[1], -R_t[2])
    if R_neg in H_R:
        return complex(H_R[R_neg][n, m].conj())
    raise W90Error(
        "FXE-W90-002",
        f"Missing hopping entry and Hermitian fallback at R={R_t}",
        actual={"R": list(R_t), "m": m, "n": n},
    )


def extract_hopping_block(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
    site_i_indices: list[int],
    site_j_indices: list[int],
    R_ij: tuple[int, int, int],
) -> NDArray[np.complexfloating]:
    """
    Extract block ``H_{ij}(R_ij)`` for selected orbital-index lists.
    """
    out = np.zeros((len(site_i_indices), len(site_j_indices)), dtype=DTYPE_COMPLEX)
    for iu, m in enumerate(site_i_indices):
        for jv, n in enumerate(site_j_indices):
            out[iu, jv] = fetch_H(H_R, m, n, R_ij)
    return out


def build_U_r2c(
    orbital_order_id: str = "w90_f_default_v1",
    *,
    spinor: bool = False,
) -> NDArray[np.complexfloating]:
    """
    Build real-harmonic -> complex-harmonic unitary transform.

    Real-basis order is fixed to project convention:
    ``[0, 1, -1, 2, -2, 3, -3]``.
    """
    if orbital_order_id != "w90_f_default_v1":
        raise BindError(
            "FXE-W90-003",
            "Unsupported orbital_order_id for U_r2c",
            expected={"orbital_order_id": "w90_f_default_v1"},
            actual={"orbital_order_id": orbital_order_id},
        )

    sq2 = np.sqrt(2.0)
    tmp = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1 / sq2, 0, -1 / sq2, 0, 0],
            [0, 0, 1j / sq2, 0, 1j / sq2, 0, 0],
            [0, 1 / sq2, 0, 0, 0, 1 / sq2, 0],
            [0, 1j / sq2, 0, 0, 0, -1j / sq2, 0],
            [1 / sq2, 0, 0, 0, 0, 0, -1 / sq2],
            [1j / sq2, 0, 0, 0, 0, 0, 1j / sq2],
        ],
        dtype=DTYPE_COMPLEX,
    )
    U = np.kron(tmp, np.eye(2, dtype=DTYPE_COMPLEX)) if spinor else tmp
    check_unitary(U, label="U_r2c", eps=EPS_UNITARY, module="wannier90")
    return U


def apply_basis_transform(
    h_real: NDArray[np.complexfloating],
    U_r2c: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """
    Apply basis transform ``h_complex = U^T h_real U*``.
    """
    if h_real.ndim != 2 or h_real.shape[0] != h_real.shape[1]:
        raise W90Error(
            "FXE-W90-003",
            "Input matrix must be square for basis transform",
            actual={"shape": list(h_real.shape)},
        )
    if U_r2c.shape != h_real.shape:
        raise W90Error(
            "FXE-W90-003",
            "Transform matrix shape mismatch",
            expected={"shape": list(h_real.shape)},
            actual={"shape": list(U_r2c.shape)},
        )
    check_unitary(U_r2c, label="U_basis", eps=EPS_UNITARY, module="wannier90")
    return U_r2c.T @ h_real @ U_r2c.conj()


def load_t_mu_from_wannier90(
    w90_cfg: dict[str, Any],
    *,
    n_orb: int = N_ORB,
    return_payload: bool = False,
) -> NDArray[np.complexfloating] | dict[str, Any]:
    """
    Build one-bond effective f-f hopping matrix ``t_mu`` from Wannier90 inputs.

    This follows the one-run/one-bond contract and applies:
    - relative-cell fetch rule,
    - direct f-f term,
    - optional ligand-mediated second-order correction with configurable
      denominator policy.
    """
    required = [
        "soc_mode",
        "hr_path",
        "win_path",
        "orbital_order_id",
        "energy_unit",
        "f_site_i",
        "f_site_j",
        "f_site_i_cell",
        "f_site_j_cell",
        "ligand_indices",
        "ligand_cells",
        "all_wannier_atom_indices",
        "delta_mode",
        "delta_reduction",
    ]
    for key in required:
        if key not in w90_cfg:
            raise W90Error("FXE-W90-001", f"Missing Wannier90 config field: {key}")

    soc_mode = str(w90_cfg["soc_mode"])
    if soc_mode not in {"with_soc", "without_soc"}:
        raise W90Error(
            "FXE-W90-002",
            "soc_mode must be 'with_soc' or 'without_soc'",
            actual={"soc_mode": soc_mode},
        )
    if soc_mode == "without_soc":
        rule = str(w90_cfg.get("spin_completion_rule", ""))
        if rule != "up_raw_down_conj_zero_flip_v1":
            raise W90Error(
                "FXE-W90-002",
                "spin_completion_rule must be 'up_raw_down_conj_zero_flip_v1' when soc_mode='without_soc'",
                expected={"spin_completion_rule": "up_raw_down_conj_zero_flip_v1"},
                actual={"spin_completion_rule": rule},
            )

    energy_unit = str(w90_cfg["energy_unit"])
    energy_scale = _energy_scale_to_ev(energy_unit)

    hr = parse_hr_dat(w90_cfg["hr_path"])
    win = parse_win(w90_cfg["win_path"])
    if int(hr["num_wann"]) != int(win["num_wann"]):
        raise W90Error(
            "FXE-W90-003",
            "num_wann mismatch between hr and win",
            actual={"hr_num_wann": hr["num_wann"], "win_num_wann": win["num_wann"]},
        )

    H_R_raw: dict[tuple[int, int, int], NDArray[np.complexfloating]] = hr["H_R"]
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]] = {
        R: energy_scale * np.asarray(mat, dtype=np.complex128)
        for R, mat in H_R_raw.items()
    }
    num_wann = int(hr["num_wann"])
    atom_ids = list(w90_cfg["all_wannier_atom_indices"])
    if len(atom_ids) < 2:
        raise W90Error(
            "FXE-W90-002",
            "all_wannier_atom_indices must include at least two atoms",
            actual={"all_wannier_atom_indices": atom_ids},
        )
    if num_wann % len(atom_ids) != 0:
        raise W90Error(
            "FXE-W90-002",
            "Cannot infer equal per-atom Wannier block from num_wann/all_wannier_atom_indices",
            actual={"num_wann": num_wann, "n_atoms": len(atom_ids)},
        )
    per_atom = num_wann // len(atom_ids)
    if soc_mode == "with_soc":
        f_dim_raw = n_orb
        spin_order_id = "sigma(-1/2,+1/2)_interleaved_v1"
    else:
        if n_orb % 2 != 0:
            raise W90Error(
                "FXE-W90-002",
                "n_orb must be even in soc_mode='without_soc' expansion",
                actual={"n_orb": n_orb},
            )
        f_dim_raw = n_orb // 2
        spin_order_id = "up_raw_down_conj_zero_flip_v1"
    if per_atom < f_dim_raw:
        raise W90Error(
            "FXE-W90-002",
            "Per-atom Wannier block is smaller than required f-manifold size",
            actual={"per_atom": per_atom, "required_n_orb": f_dim_raw},
        )

    atom_to_start = {atom: idx * per_atom for idx, atom in enumerate(atom_ids)}
    f_i = int(w90_cfg["f_site_i"])
    f_j = int(w90_cfg["f_site_j"])
    if f_i == f_j:
        raise W90Error("FXE-W90-002", "f_site_i and f_site_j must be different")
    if f_i not in atom_to_start or f_j not in atom_to_start:
        raise W90Error(
            "FXE-W90-002",
            "Selected f sites are not included in all_wannier_atom_indices",
            actual={"f_site_i": f_i, "f_site_j": f_j, "all_wannier_atom_indices": atom_ids},
        )

    f_i_idx = list(range(atom_to_start[f_i], atom_to_start[f_i] + f_dim_raw))
    f_j_idx = list(range(atom_to_start[f_j], atom_to_start[f_j] + f_dim_raw))
    c_i = _cell_triplet(w90_cfg["f_site_i_cell"], "f_site_i_cell")
    c_j = _cell_triplet(w90_cfg["f_site_j_cell"], "f_site_j_cell")
    R_ij = (c_j[0] - c_i[0], c_j[1] - c_i[1], c_j[2] - c_i[2])

    t_direct = extract_hopping_block(H_R, f_i_idx, f_j_idx, R_ij)
    ligand_indices = list(w90_cfg["ligand_indices"])
    ligand_cells = list(w90_cfg["ligand_cells"])
    if len(ligand_indices) != len(ligand_cells):
        raise W90Error(
            "FXE-W90-002",
            "ligand_indices and ligand_cells length mismatch",
            actual={"len_ligand_indices": len(ligand_indices), "len_ligand_cells": len(ligand_cells)},
        )
    onsite_key = (0, 0, 0)
    if onsite_key not in H_R:
        raise W90Error("FXE-W90-002", "Onsite block R=(0,0,0) is required for denominator policy")
    H0 = np.asarray(H_R[onsite_key], dtype=np.complex128)
    check_hermitian(H0, label="H0_wannier", eps=EPS_HERM, module="wannier90")

    eps_f_i = np.real(np.diag(H0)[f_i_idx])
    eps_f_j = np.real(np.diag(H0)[f_j_idx])

    channel_ti: list[NDArray[np.complexfloating]] = []
    channel_tj: list[NDArray[np.complexfloating]] = []
    channel_delta: list[NDArray[np.floating]] = []
    map_lig: list[dict[str, Any]] = []
    R_io_list: list[tuple[int, int, int]] = []
    R_jo_list: list[tuple[int, int, int]] = []

    for ligand_atom, cell_raw in zip(ligand_indices, ligand_cells):
        if ligand_atom in (f_i, f_j):
            raise W90Error(
                "FXE-W90-002",
                "Ligand list must exclude f_site_i/f_site_j",
                actual={"ligand_atom": ligand_atom, "f_site_i": f_i, "f_site_j": f_j},
            )
        if ligand_atom not in atom_to_start:
            raise W90Error(
                "FXE-W90-002",
                "Ligand index not found in all_wannier_atom_indices",
                actual={"ligand_atom": ligand_atom, "all_wannier_atom_indices": atom_ids},
            )
        c_o = _cell_triplet(cell_raw, "ligand_cells[*]")
        R_io = (c_o[0] - c_i[0], c_o[1] - c_i[1], c_o[2] - c_i[2])
        R_jo = (c_o[0] - c_j[0], c_o[1] - c_j[1], c_o[2] - c_j[2])
        R_io_list.append(R_io)
        R_jo_list.append(R_jo)

        lig_idx = list(range(atom_to_start[ligand_atom], atom_to_start[ligand_atom] + per_atom))
        t_i_lig = extract_hopping_block(H_R, f_i_idx, lig_idx, R_io)  # (f_dim_raw, per_atom)
        t_j_lig = extract_hopping_block(H_R, f_j_idx, lig_idx, R_jo)  # (f_dim_raw, per_atom)

        eps_lig = np.real(np.diag(H0)[lig_idx])
        for p in range(per_atom):
            # Channel tensors for this ligand orbital p.
            channel_ti.append(t_i_lig[:, p].copy())
            channel_tj.append(t_j_lig[:, p].copy())
            map_lig.append(
                {
                    "ligand_atom": int(ligand_atom),
                    "ligand_cell": [int(c_o[0]), int(c_o[1]), int(c_o[2])],
                    "wannier_index": int(lig_idx[p]),
                    "channel_index": int(len(map_lig)),
                }
            )

            delta_uv = np.zeros((f_dim_raw, f_dim_raw), dtype=float)
            for u in range(f_dim_raw):
                for v in range(f_dim_raw):
                    du = eps_f_i[u] - eps_lig[p]
                    dv = eps_f_j[v] - eps_lig[p]
                    denom = du + dv
                    if abs(denom) < EPS_ZERO:
                        delta_uv[u, v] = np.inf
                    else:
                        delta_uv[u, v] = 2.0 * du * dv / denom
            channel_delta.append(delta_uv)

    n_channel = len(channel_ti)
    if n_channel == 0:
        t_eff_raw = t_direct
    else:
        delta_mode = str(w90_cfg["delta_mode"])
        delta_reduction = str(w90_cfg["delta_reduction"])
        if delta_mode not in {"manual", "from_onsite"}:
            raise W90Error("FXE-W90-002", f"Unsupported delta_mode: {delta_mode}")
        if delta_reduction not in {"channelwise", "global_mean"}:
            raise W90Error("FXE-W90-002", f"Unsupported delta_reduction: {delta_reduction}")

        if delta_mode == "manual":
            manual_kind = str(w90_cfg.get("delta_manual_kind", ""))
            if manual_kind not in {"channelwise", "global_mean"}:
                raise W90Error("FXE-W90-002", "delta_manual_kind must be channelwise/global_mean in manual mode")
            if manual_kind != delta_reduction:
                raise W90Error(
                    "FXE-W90-002",
                    "delta_manual_kind must match delta_reduction",
                    actual={"delta_manual_kind": manual_kind, "delta_reduction": delta_reduction},
                )
            if manual_kind == "global_mean":
                if "delta_manual_value" not in w90_cfg:
                    raise W90Error("FXE-W90-002", "delta_manual_value is required for manual global_mean mode")
                delta_global = float(w90_cfg["delta_manual_value"]) * energy_scale
                delta_tensors = [np.full((f_dim_raw, f_dim_raw), delta_global, dtype=float) for _ in range(n_channel)]
            else:
                if "delta_manual_file" not in w90_cfg:
                    raise W90Error("FXE-W90-002", "delta_manual_file is required for manual channelwise mode")
                manual = np.load(str(w90_cfg["delta_manual_file"]))
                if "Delta_puv" not in manual:
                    raise W90Error("FXE-W90-002", "delta_manual_file must contain key Delta_puv")
                delta_raw = np.asarray(manual["Delta_puv"], dtype=float) * energy_scale
                if delta_raw.shape != (n_channel, f_dim_raw, f_dim_raw):
                    raise W90Error(
                        "FXE-W90-002",
                        "Delta_puv shape mismatch",
                        expected={"shape": [n_channel, f_dim_raw, f_dim_raw]},
                        actual={"shape": list(delta_raw.shape)},
                    )
                delta_tensors = [delta_raw[i] for i in range(n_channel)]
        else:
            # from_onsite mode
            delta_tensors = channel_delta
            if delta_reduction == "global_mean":
                all_finite = np.concatenate([d[np.isfinite(d)] for d in delta_tensors])
                if all_finite.size == 0:
                    raise W90Error("FXE-W90-002", "No finite onsite-derived denominators")
                delta_global = float(np.mean(all_finite))
                delta_tensors = [np.full((f_dim_raw, f_dim_raw), delta_global, dtype=float) for _ in range(n_channel)]

        corr = np.zeros((f_dim_raw, f_dim_raw), dtype=DTYPE_COMPLEX)
        for ch in range(n_channel):
            ti = channel_ti[ch]  # (f_dim_raw,)
            tj = channel_tj[ch]  # (f_dim_raw,)
            Delta = delta_tensors[ch]
            for u in range(f_dim_raw):
                for v in range(f_dim_raw):
                    den = float(Delta[u, v])
                    if not np.isfinite(den) or abs(den) < EPS_ZERO:
                        continue
                    corr[u, v] += ti[u] * np.conj(tj[v]) / den
        t_eff_raw = t_direct + corr

    if soc_mode == "without_soc":
        t_spin = _expand_without_soc_to_spinor(t_eff_raw)
    else:
        t_spin = t_eff_raw
    if t_spin.shape != (n_orb, n_orb):
        raise W90Error(
            "FXE-W90-002",
            "Spin-completed hopping has unexpected shape",
            expected={"shape": [n_orb, n_orb]},
            actual={"shape": list(t_spin.shape)},
        )

    orbital_order_id = str(w90_cfg["orbital_order_id"])
    U_r2c = build_U_r2c(orbital_order_id=orbital_order_id, spinor=True)
    t_eff = apply_basis_transform(t_spin, U_r2c)

    map_f_i = [
        {
            "channel_index": int(u),
            "wannier_index": int(f_i_idx[u] if u < len(f_i_idx) else -1),
            "site_atom": int(f_i),
            "site_cell": [int(c_i[0]), int(c_i[1]), int(c_i[2])],
        }
        for u in range(f_dim_raw)
    ]
    map_f_j = [
        {
            "channel_index": int(v),
            "wannier_index": int(f_j_idx[v] if v < len(f_j_idx) else -1),
            "site_atom": int(f_j),
            "site_cell": [int(c_j[0]), int(c_j[1]), int(c_j[2])],
        }
        for v in range(f_dim_raw)
    ]
    source_hashes = {
        "hr_sha256": _sha256_file(Path(w90_cfg["hr_path"])),
        "win_sha256": _sha256_file(Path(w90_cfg["win_path"])),
    }
    if str(w90_cfg.get("delta_mode", "")) == "manual" and "delta_manual_file" in w90_cfg:
        source_hashes["delta_manual_sha256"] = _sha256_file(Path(w90_cfg["delta_manual_file"]))

    meta = {
        "soc_mode": soc_mode,
        "energy_unit_input": energy_unit,
        "energy_scale_to_ev": energy_scale,
        "orbital_order_id": orbital_order_id,
        "spin_completion_rule": w90_cfg.get("spin_completion_rule"),
        "f_site_i_cell": [int(c_i[0]), int(c_i[1]), int(c_i[2])],
        "f_site_j_cell": [int(c_j[0]), int(c_j[1]), int(c_j[2])],
        "ligand_cells": [[int(x) for x in _cell_triplet(c, "ligand_cells[*]")] for c in ligand_cells],
        "R_ij": [int(R_ij[0]), int(R_ij[1]), int(R_ij[2])],
        "R_io": [[int(r[0]), int(r[1]), int(r[2])] for r in R_io_list],
        "R_jo": [[int(r[0]), int(r[1]), int(r[2])] for r in R_jo_list],
        "map_f_i": map_f_i,
        "map_f_j": map_f_j,
        "map_lig": map_lig,
        "order_ids": {
            "atom_order_id": "all_wannier_atom_indices_v1",
            "orbital_order_id": orbital_order_id,
            "spin_order_id": spin_order_id,
            "ligand_order_id": "ligand_indices_input_v1",
        },
        "file_hashes": source_hashes,
        "delta_mode": str(w90_cfg["delta_mode"]),
        "delta_reduction": str(w90_cfg["delta_reduction"]),
    }
    if return_payload:
        return {"t_mu": t_eff, "meta": meta}
    return t_eff


def _energy_scale_to_ev(unit: str) -> float:
    table = {
        "eV": 1.0,
        "meV": 1.0e-3,
        "Ha": 27.211386245988,
        "Ry": 13.605693122994,
    }
    if unit not in table:
        raise W90Error(
            "FXE-W90-003",
            f"Unsupported energy_unit: {unit}",
            expected={"energy_unit": sorted(table.keys())},
            actual={"energy_unit": unit},
        )
    return float(table[unit])


def _expand_without_soc_to_spinor(t_orb: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Expand orbital-only hopping to spinor basis with fixed completion rule."""
    if t_orb.ndim != 2 or t_orb.shape[0] != t_orb.shape[1]:
        raise W90Error(
            "FXE-W90-002",
            "without_soc base hopping must be square matrix",
            actual={"shape": list(t_orb.shape)},
        )
    n = int(t_orb.shape[0])
    t_spin = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    for u in range(n):
        for v in range(n):
            # Project orbital order uses interleaved sigma=(-1/2,+1/2) per m.
            p_dn, p_up = 2 * u, 2 * u + 1
            q_dn, q_up = 2 * v, 2 * v + 1
            t_spin[p_up, q_up] = t_orb[u, v]
            t_spin[p_dn, q_dn] = np.conj(t_orb[u, v])
    return t_spin


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cell_triplet(value: Any, field: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(not isinstance(x, int) for x in value):
        raise W90Error("FXE-W90-002", f"{field} must be an integer triplet")
    return int(value[0]), int(value[1]), int(value[2])
