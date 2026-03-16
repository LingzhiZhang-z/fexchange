"""
Wannier90 parsing, mapping and reduced hopping construction.

Spec reference:
- 05-02-WANNIER90_INPUT_CONTRACT
- 05-03-WANNIER90_PARSING_RULES
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.checks import check_unitary, check_hermitian
from fexchange.utils.constants import N_ORB
from fexchange.utils.errors import BindError, W90Error
from fexchange.utils.numerics import DTYPE_COMPLEX, EPS_UNITARY, EPS_ZERO, EPS_HERM

logger = logging.getLogger("fexchange")


@dataclass(frozen=True)
class _ValidatedConfig:
    soc_mode: str
    energy_unit: str
    energy_scale: float
    orbital_basis: str
    orbital_order_id: str
    delta_mode: str
    delta_reduction: str
    spin_completion_rule: str | None


@dataclass(frozen=True)
class _SiteMap:
    atom_ids: tuple[int, ...]
    per_atom: int
    f_dim_raw: int
    spin_order_id: str
    atom_to_start: dict[int, int]
    f_i: int
    f_j: int
    f_i_idx: tuple[int, ...]
    f_j_idx: tuple[int, ...]
    c_i: tuple[int, int, int]
    c_j: tuple[int, int, int]
    R_ij: tuple[int, int, int]
    ligand_indices: tuple[int, ...]
    ligand_cells: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _LigandChannel:
    t_i: NDArray[np.complexfloating]
    t_j: NDArray[np.complexfloating]
    delta_from_onsite: NDArray[np.floating]
    ligand_atom: int
    ligand_cell: tuple[int, int, int]
    wannier_index: int
    channel_index: int


@dataclass(frozen=True)
class _LigandCorrection:
    matrix: NDArray[np.complexfloating]
    map_lig: list[dict[str, Any]]
    R_io: list[tuple[int, int, int]]
    R_jo: list[tuple[int, int, int]]


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
    U = np.kron(tmp.T, np.eye(2, dtype=DTYPE_COMPLEX)) if spinor else tmp.T
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


def w90_extract(cfg: dict[str, Any], *, n_orb: int = N_ORB) -> dict[str, Any]:
    """
    Extract hopping matrix and local Hamiltonian from Wannier90 outputs.

    Returns a dict with keys ``t_mu``, ``h_local`` and ``meta``.
    """
    logger.info("w90_extract start: soc_mode=%s hr=%s", cfg.get("soc_mode"), cfg.get("hr_path"))
    validated = _validate_config(cfg, n_orb=n_orb)
    H_R, num_wann = _load_wannier_data(cfg, energy_scale=validated.energy_scale)
    site_map = _build_site_map(cfg["all_wannier_atom_indices"], num_wann, validated.soc_mode, n_orb, cfg)
    H0 = _get_onsite_matrix(H_R)
    h_local_raw, eps_f_i, eps_f_j = _extract_onsite(H0, site_map)
    t_direct = extract_hopping_block(H_R, list(site_map.f_i_idx), list(site_map.f_j_idx), site_map.R_ij)
    ligand = _compute_ligand_correction(H_R, H0, site_map, cfg, validated.soc_mode, validated.energy_scale, eps_f_i, eps_f_j)
    t_mu = _finalize_reduced_matrix(t_direct + ligand.matrix, validated, n_orb=n_orb, label="t_mu")
    h_local = _finalize_reduced_matrix(h_local_raw, validated, n_orb=n_orb, label="h_local")
    meta = _build_metadata(cfg, validated, site_map, ligand, num_wann=num_wann)
    logger.info("w90_extract done: t_mu_shape=%s h_local_shape=%s", t_mu.shape, h_local.shape)
    return {"t_mu": t_mu, "h_local": h_local, "meta": meta}


def load_t_mu_from_wannier90(
    w90_cfg: dict[str, Any],
    *,
    n_orb: int = N_ORB,
    return_payload: bool = False,
) -> NDArray[np.complexfloating] | dict[str, Any]:
    """Backward-compatible wrapper around :func:`w90_extract`."""
    result = w90_extract(w90_cfg, n_orb=n_orb)
    if return_payload:
        return {"t_mu": result["t_mu"], "meta": result["meta"]}
    return result["t_mu"]


def _validate_config(cfg: dict[str, Any], *, n_orb: int) -> _ValidatedConfig:
    required = (
        "soc_mode",
        "hr_path",
        "win_path",
        "orbital_basis",
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
    )
    for key in required:
        if key not in cfg:
            raise W90Error("FXE-W90-001", f"Missing Wannier90 config field: {key}")
    if n_orb != N_ORB:
        raise W90Error("FXE-W90-003", "Wannier90 extraction requires n_orb=14", expected={"n_orb": N_ORB}, actual={"n_orb": n_orb})
    soc_mode = str(cfg["soc_mode"])
    if soc_mode not in {"with_soc", "without_soc"}:
        raise W90Error("FXE-W90-002", "soc_mode must be 'with_soc' or 'without_soc'", actual={"soc_mode": soc_mode})
    orbital_basis = str(cfg["orbital_basis"])
    if orbital_basis != "real_harmonic_default_w90":
        raise W90Error("FXE-W90-003", "Unsupported orbital_basis", expected={"orbital_basis": "real_harmonic_default_w90"}, actual={"orbital_basis": orbital_basis})
    delta_mode = str(cfg["delta_mode"])
    delta_reduction = str(cfg["delta_reduction"])
    if delta_mode not in {"manual", "from_onsite"}:
        raise W90Error("FXE-W90-002", "delta_mode must be 'manual' or 'from_onsite'", actual={"delta_mode": delta_mode})
    if delta_reduction not in {"channelwise", "global_mean"}:
        raise W90Error("FXE-W90-002", "delta_reduction must be 'channelwise' or 'global_mean'", actual={"delta_reduction": delta_reduction})
    spin_rule = cfg.get("spin_completion_rule")
    if soc_mode == "without_soc" and spin_rule != "up_raw_down_conj_zero_flip_v1":
        raise W90Error("FXE-W90-002", "spin_completion_rule must be 'up_raw_down_conj_zero_flip_v1' when soc_mode='without_soc'", expected={"spin_completion_rule": "up_raw_down_conj_zero_flip_v1"}, actual={"spin_completion_rule": spin_rule})
    return _ValidatedConfig(
        soc_mode=soc_mode,
        energy_unit=str(cfg["energy_unit"]),
        energy_scale=_energy_scale_to_ev(str(cfg["energy_unit"])),
        orbital_basis=orbital_basis,
        orbital_order_id=str(cfg["orbital_order_id"]),
        delta_mode=delta_mode,
        delta_reduction=delta_reduction,
        spin_completion_rule=None if spin_rule is None else str(spin_rule),
    )


def _load_wannier_data(
    cfg: dict[str, Any],
    *,
    energy_scale: float,
) -> tuple[dict[tuple[int, int, int], NDArray[np.complexfloating]], int]:
    hr = parse_hr_dat(cfg["hr_path"])
    win = parse_win(cfg["win_path"])
    if int(hr["num_wann"]) != int(win["num_wann"]):
        raise W90Error("FXE-W90-003", "num_wann mismatch between hr and win", actual={"hr_num_wann": hr["num_wann"], "win_num_wann": win["num_wann"]})
    H_R = {R: energy_scale * np.asarray(mat, dtype=np.complex128) for R, mat in hr["H_R"].items()}
    return H_R, int(hr["num_wann"])


def _build_site_map(
    atom_ids: list[int],
    num_wann: int,
    soc_mode: str,
    n_orb: int,
    cfg: dict[str, Any],
) -> _SiteMap:
    atoms = tuple(int(atom) for atom in atom_ids)
    _validate_atom_selection(atoms)
    if num_wann % len(atoms) != 0:
        raise W90Error("FXE-W90-002", "Cannot infer equal per-atom Wannier block", actual={"num_wann": num_wann, "n_atoms": len(atoms)})
    per_atom = num_wann // len(atoms)
    f_dim_raw, spin_order_id = _raw_f_dim(soc_mode, n_orb)
    if per_atom < f_dim_raw:
        raise W90Error("FXE-W90-002", "Per-atom Wannier block is smaller than required f-manifold size", actual={"per_atom": per_atom, "required_n_orb": f_dim_raw})
    atom_to_start = {atom: idx * per_atom for idx, atom in enumerate(atoms)}
    f_i = int(cfg["f_site_i"])
    f_j = int(cfg["f_site_j"])
    if f_i == f_j:
        raise W90Error("FXE-W90-002", "f_site_i and f_site_j must be different")
    _require_atom_in_map(atom_to_start, f_i, "f_site_i")
    _require_atom_in_map(atom_to_start, f_j, "f_site_j")
    ligand_indices, ligand_cells = _parse_ligand_sites(cfg, atom_to_start, f_i=f_i, f_j=f_j)
    c_i = _cell_triplet(cfg["f_site_i_cell"], "f_site_i_cell")
    c_j = _cell_triplet(cfg["f_site_j_cell"], "f_site_j_cell")
    return _SiteMap(
        atom_ids=atoms,
        per_atom=per_atom,
        f_dim_raw=f_dim_raw,
        spin_order_id=spin_order_id,
        atom_to_start=atom_to_start,
        f_i=f_i,
        f_j=f_j,
        f_i_idx=_site_indices(atom_to_start, f_i, f_dim_raw),
        f_j_idx=_site_indices(atom_to_start, f_j, f_dim_raw),
        c_i=c_i,
        c_j=c_j,
        R_ij=(c_j[0] - c_i[0], c_j[1] - c_i[1], c_j[2] - c_i[2]),
        ligand_indices=ligand_indices,
        ligand_cells=ligand_cells,
    )


def _validate_atom_selection(atom_ids: tuple[int, ...]) -> None:
    if len(atom_ids) < 2:
        raise W90Error("FXE-W90-002", "all_wannier_atom_indices must include at least two atoms", actual={"all_wannier_atom_indices": list(atom_ids)})
    if len(set(atom_ids)) != len(atom_ids):
        raise W90Error("FXE-W90-002", "Duplicated atom index in all_wannier_atom_indices", actual={"all_wannier_atom_indices": list(atom_ids)})


def _raw_f_dim(soc_mode: str, n_orb: int) -> tuple[int, str]:
    if soc_mode == "with_soc":
        return n_orb, "sigma(-1/2,+1/2)_interleaved_v1"
    if n_orb % 2 != 0:
        raise W90Error("FXE-W90-002", "n_orb must be even in soc_mode='without_soc'", actual={"n_orb": n_orb})
    return n_orb // 2, "up_raw_down_conj_zero_flip_v1"


def _require_atom_in_map(atom_to_start: dict[int, int], atom: int, field: str) -> None:
    if atom not in atom_to_start:
        raise W90Error("FXE-W90-002", f"{field} is not included in all_wannier_atom_indices", actual={field: atom, "all_wannier_atom_indices": sorted(atom_to_start)})


def _parse_ligand_sites(
    cfg: dict[str, Any],
    atom_to_start: dict[int, int],
    *,
    f_i: int,
    f_j: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    ligand_indices = tuple(int(atom) for atom in cfg["ligand_indices"])
    if len(set(ligand_indices)) != len(ligand_indices):
        raise W90Error("FXE-W90-002", "Duplicated atom index in ligand_indices", actual={"ligand_indices": list(ligand_indices)})
    ligand_cells_raw = list(cfg["ligand_cells"])
    if len(ligand_indices) != len(ligand_cells_raw):
        raise W90Error("FXE-W90-002", "ligand_indices and ligand_cells length mismatch", actual={"len_ligand_indices": len(ligand_indices), "len_ligand_cells": len(ligand_cells_raw)})
    for ligand_atom in ligand_indices:
        if ligand_atom in (f_i, f_j):
            raise W90Error("FXE-W90-002", "Ligand list must exclude f_site_i/f_site_j", actual={"ligand_atom": ligand_atom, "f_site_i": f_i, "f_site_j": f_j})
        _require_atom_in_map(atom_to_start, ligand_atom, "ligand_indices[*]")
    ligand_cells = tuple(_cell_triplet(cell, "ligand_cells[*]") for cell in ligand_cells_raw)
    return ligand_indices, ligand_cells


def _site_indices(atom_to_start: dict[int, int], atom: int, f_dim_raw: int) -> tuple[int, ...]:
    start = atom_to_start[atom]
    return tuple(range(start, start + f_dim_raw))


def _get_onsite_matrix(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
) -> NDArray[np.complexfloating]:
    onsite_key = (0, 0, 0)
    if onsite_key not in H_R:
        raise W90Error("FXE-W90-002", "Onsite block R=(0,0,0) is required")
    H0 = np.asarray(H_R[onsite_key], dtype=np.complex128)
    check_hermitian(H0, label="H0_wannier", eps=EPS_HERM, module="wannier90")
    return H0


def _extract_onsite(
    H0: NDArray[np.complexfloating],
    site_map: _SiteMap,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating], NDArray[np.floating]]:
    h_onsite_i = H0[np.ix_(site_map.f_i_idx, site_map.f_i_idx)]
    h_onsite_j = H0[np.ix_(site_map.f_j_idx, site_map.f_j_idx)]
    h_local_raw = 0.5 * (h_onsite_i + h_onsite_j)
    eps_diag = np.real(np.diag(H0))
    return h_local_raw, eps_diag[list(site_map.f_i_idx)], eps_diag[list(site_map.f_j_idx)]


def _compute_ligand_correction(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
    H0: NDArray[np.complexfloating],
    site_map: _SiteMap,
    cfg: dict[str, Any],
    soc_mode: str,
    energy_scale: float,
    eps_f_i: NDArray[np.floating],
    eps_f_j: NDArray[np.floating],
) -> _LigandCorrection:
    del soc_mode
    channels, R_io, R_jo = _collect_ligand_channels(H_R, H0, site_map, eps_f_i, eps_f_j)
    if not channels:
        return _LigandCorrection(matrix=np.zeros((site_map.f_dim_raw, site_map.f_dim_raw), dtype=DTYPE_COMPLEX), map_lig=[], R_io=[], R_jo=[])
    delta_tensors = _resolve_delta_tensors(channels, cfg, energy_scale=energy_scale, f_dim_raw=site_map.f_dim_raw)
    correction = _accumulate_ligand_correction(channels, delta_tensors, f_dim_raw=site_map.f_dim_raw)
    map_lig = [_ligand_meta(channel) for channel in channels]
    logger.debug("Computed ligand correction with %d channels", len(channels))
    return _LigandCorrection(matrix=correction, map_lig=map_lig, R_io=R_io, R_jo=R_jo)


def _collect_ligand_channels(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
    H0: NDArray[np.complexfloating],
    site_map: _SiteMap,
    eps_f_i: NDArray[np.floating],
    eps_f_j: NDArray[np.floating],
) -> tuple[list[_LigandChannel], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    eps_diag = np.real(np.diag(H0))
    channels: list[_LigandChannel] = []
    R_io: list[tuple[int, int, int]] = []
    R_jo: list[tuple[int, int, int]] = []
    for ligand_atom, ligand_cell in zip(site_map.ligand_indices, site_map.ligand_cells):
        R_io_item, R_jo_item = _relative_ligand_vectors(site_map, ligand_cell)
        R_io.append(R_io_item)
        R_jo.append(R_jo_item)
        channels.extend(_collect_ligand_site_channels(H_R, site_map, eps_diag, eps_f_i, eps_f_j, ligand_atom, ligand_cell, R_io_item, R_jo_item, channel_offset=len(channels)))
    return channels, R_io, R_jo


def _relative_ligand_vectors(
    site_map: _SiteMap,
    ligand_cell: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (
        (ligand_cell[0] - site_map.c_i[0], ligand_cell[1] - site_map.c_i[1], ligand_cell[2] - site_map.c_i[2]),
        (ligand_cell[0] - site_map.c_j[0], ligand_cell[1] - site_map.c_j[1], ligand_cell[2] - site_map.c_j[2]),
    )


def _collect_ligand_site_channels(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
    site_map: _SiteMap,
    eps_diag: NDArray[np.floating],
    eps_f_i: NDArray[np.floating],
    eps_f_j: NDArray[np.floating],
    ligand_atom: int,
    ligand_cell: tuple[int, int, int],
    R_io: tuple[int, int, int],
    R_jo: tuple[int, int, int],
    *,
    channel_offset: int,
) -> list[_LigandChannel]:
    lig_idx = _site_indices(site_map.atom_to_start, ligand_atom, site_map.per_atom)
    t_i_lig = extract_hopping_block(H_R, list(site_map.f_i_idx), list(lig_idx), R_io)
    t_j_lig = extract_hopping_block(H_R, list(site_map.f_j_idx), list(lig_idx), R_jo)
    eps_lig = eps_diag[list(lig_idx)]
    return [
        _LigandChannel(
            t_i=t_i_lig[:, p].copy(),
            t_j=t_j_lig[:, p].copy(),
            delta_from_onsite=_onsite_delta(eps_f_i, eps_f_j, float(eps_lig[p])),
            ligand_atom=ligand_atom,
            ligand_cell=ligand_cell,
            wannier_index=int(lig_idx[p]),
            channel_index=channel_offset + p,
        )
        for p in range(site_map.per_atom)
    ]


def _onsite_delta(
    eps_f_i: NDArray[np.floating],
    eps_f_j: NDArray[np.floating],
    eps_lig: float,
) -> NDArray[np.floating]:
    delta = np.zeros((len(eps_f_i), len(eps_f_j)), dtype=float)
    for u, eps_u in enumerate(eps_f_i):
        for v, eps_v in enumerate(eps_f_j):
            du = float(eps_u) - eps_lig
            dv = float(eps_v) - eps_lig
            denom = du + dv
            delta[u, v] = np.inf if abs(denom) < EPS_ZERO else 2.0 * du * dv / denom
    return delta


def _resolve_delta_tensors(
    channels: list[_LigandChannel],
    cfg: dict[str, Any],
    *,
    energy_scale: float,
    f_dim_raw: int,
) -> list[NDArray[np.floating]]:
    delta_mode = str(cfg["delta_mode"])
    delta_reduction = str(cfg["delta_reduction"])
    if delta_mode == "from_onsite":
        return _reduce_onsite_deltas(channels, delta_reduction)
    manual_kind = str(cfg.get("delta_manual_kind", ""))
    if manual_kind not in {"channelwise", "global_mean"}:
        raise W90Error("FXE-W90-002", "delta_manual_kind must be channelwise/global_mean in manual mode")
    if manual_kind != delta_reduction:
        raise W90Error("FXE-W90-002", "delta_manual_kind must match delta_reduction", actual={"delta_manual_kind": manual_kind, "delta_reduction": delta_reduction})
    if manual_kind == "global_mean":
        return _manual_global_delta(cfg, energy_scale=energy_scale, shape=(len(channels), f_dim_raw, f_dim_raw))
    return _manual_channelwise_delta(cfg, energy_scale=energy_scale, shape=(len(channels), f_dim_raw, f_dim_raw))


def _reduce_onsite_deltas(
    channels: list[_LigandChannel],
    delta_reduction: str,
) -> list[NDArray[np.floating]]:
    if delta_reduction == "channelwise":
        return [channel.delta_from_onsite for channel in channels]
    finite = np.concatenate([delta[np.isfinite(delta)] for delta in (channel.delta_from_onsite for channel in channels)])
    if finite.size == 0:
        raise W90Error("FXE-W90-002", "No finite onsite-derived denominators")
    delta_global = float(np.mean(finite))
    return [np.full_like(channel.delta_from_onsite, delta_global, dtype=float) for channel in channels]


def _manual_global_delta(
    cfg: dict[str, Any],
    *,
    energy_scale: float,
    shape: tuple[int, int, int],
) -> list[NDArray[np.floating]]:
    if "delta_manual_value" not in cfg:
        raise W90Error("FXE-W90-002", "delta_manual_value is required for manual global_mean mode")
    delta_global = float(cfg["delta_manual_value"]) * energy_scale
    return [np.full(shape[1:], delta_global, dtype=float) for _ in range(shape[0])]


def _manual_channelwise_delta(
    cfg: dict[str, Any],
    *,
    energy_scale: float,
    shape: tuple[int, int, int],
) -> list[NDArray[np.floating]]:
    if "delta_manual_file" not in cfg:
        raise W90Error("FXE-W90-002", "delta_manual_file is required for manual channelwise mode")
    with np.load(str(cfg["delta_manual_file"])) as manual:
        if "Delta_puv" not in manual:
            raise W90Error("FXE-W90-002", "delta_manual_file must contain key Delta_puv")
        delta_raw = np.asarray(manual["Delta_puv"], dtype=float) * energy_scale
    if delta_raw.shape != shape:
        raise W90Error("FXE-W90-002", "Delta_puv shape mismatch", expected={"shape": list(shape)}, actual={"shape": list(delta_raw.shape)})
    return [delta_raw[idx] for idx in range(shape[0])]


def _accumulate_ligand_correction(
    channels: list[_LigandChannel],
    delta_tensors: list[NDArray[np.floating]],
    *,
    f_dim_raw: int,
) -> NDArray[np.complexfloating]:
    correction = np.zeros((f_dim_raw, f_dim_raw), dtype=DTYPE_COMPLEX)
    for channel, delta in zip(channels, delta_tensors):
        for u in range(f_dim_raw):
            for v in range(f_dim_raw):
                denom = float(delta[u, v])
                if np.isfinite(denom) and abs(denom) >= EPS_ZERO:
                    correction[u, v] += channel.t_i[u] * np.conj(channel.t_j[v]) / denom
    return correction


def _ligand_meta(channel: _LigandChannel) -> dict[str, Any]:
    return {
        "ligand_atom": int(channel.ligand_atom),
        "ligand_cell": [int(x) for x in channel.ligand_cell],
        "wannier_index": int(channel.wannier_index),
        "channel_index": int(channel.channel_index),
    }


def _finalize_reduced_matrix(
    matrix_raw: NDArray[np.complexfloating],
    validated: _ValidatedConfig,
    *,
    n_orb: int,
    label: str,
) -> NDArray[np.complexfloating]:
    matrix_spin = _expand_without_soc_to_spinor(matrix_raw) if validated.soc_mode == "without_soc" else matrix_raw
    if matrix_spin.shape != (n_orb, n_orb):
        raise W90Error("FXE-W90-002", f"{label} has unexpected spinor shape", expected={"shape": [n_orb, n_orb]}, actual={"shape": list(matrix_spin.shape)})
    U_r2c = build_U_r2c(orbital_order_id=validated.orbital_order_id, spinor=True)
    matrix_out = apply_basis_transform(matrix_spin, U_r2c)
    if label == "h_local":
        check_hermitian(matrix_out, label="h_local", eps=EPS_HERM, module="wannier90")
    return matrix_out


def _build_metadata(
    cfg: dict[str, Any],
    validated: _ValidatedConfig,
    site_map: _SiteMap,
    ligand: _LigandCorrection,
    *,
    num_wann: int,
) -> dict[str, Any]:
    return {
        "soc_mode": validated.soc_mode,
        "orbital_basis": validated.orbital_basis,
        "orbital_order_id": validated.orbital_order_id,
        "energy_unit_input": validated.energy_unit,
        "energy_scale_to_ev": validated.energy_scale,
        "spin_completion_rule": validated.spin_completion_rule,
        "num_wann": int(num_wann),
        "all_wannier_atom_indices": [int(atom) for atom in site_map.atom_ids],
        "f_site_i": int(site_map.f_i),
        "f_site_j": int(site_map.f_j),
        "f_site_i_cell": [int(x) for x in site_map.c_i],
        "f_site_j_cell": [int(x) for x in site_map.c_j],
        "ligand_indices": [int(atom) for atom in site_map.ligand_indices],
        "ligand_cells": [[int(x) for x in cell] for cell in site_map.ligand_cells],
        "R_ij": [int(x) for x in site_map.R_ij],
        "R_io": [[int(x) for x in triplet] for triplet in ligand.R_io],
        "R_jo": [[int(x) for x in triplet] for triplet in ligand.R_jo],
        "map_f_i": _build_site_channel_map(site_map.f_i_idx, site_atom=site_map.f_i, site_cell=site_map.c_i),
        "map_f_j": _build_site_channel_map(site_map.f_j_idx, site_atom=site_map.f_j, site_cell=site_map.c_j),
        "map_lig": ligand.map_lig,
        "order_ids": {
            "atom_order_id": "all_wannier_atom_indices_v1",
            "orbital_order_id": validated.orbital_order_id,
            "spin_order_id": site_map.spin_order_id,
            "ligand_order_id": "ligand_indices_input_v1",
        },
        "delta_mode": validated.delta_mode,
        "delta_reduction": validated.delta_reduction,
        "file_hashes": _build_file_hashes(cfg, delta_mode=validated.delta_mode),
    }


def _build_site_channel_map(
    indices: tuple[int, ...],
    *,
    site_atom: int,
    site_cell: tuple[int, int, int],
) -> list[dict[str, Any]]:
    return [
        {
            "channel_index": int(channel),
            "wannier_index": int(wannier_index),
            "site_atom": int(site_atom),
            "site_cell": [int(x) for x in site_cell],
        }
        for channel, wannier_index in enumerate(indices)
    ]


def _build_file_hashes(cfg: dict[str, Any], *, delta_mode: str) -> dict[str, str]:
    hashes = {
        "hr_sha256": _sha256_file(Path(cfg["hr_path"])),
        "win_sha256": _sha256_file(Path(cfg["win_path"])),
    }
    if delta_mode == "manual" and "delta_manual_file" in cfg:
        hashes["delta_manual_sha256"] = _sha256_file(Path(cfg["delta_manual_file"]))
    return hashes


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
