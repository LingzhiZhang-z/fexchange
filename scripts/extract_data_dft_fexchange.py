#!/usr/bin/env python3
"""Batch-extract fexchange inputs from ``data/data-DFT``.

This script is intentionally only an orchestration wrapper.  The actual
Wannier block extraction and f-p-f downfolding are performed by the fopt-branch
tools:

  - ``fexchange.tools.w90_extract.extract_w90_two_bond``
  - ``fexchange.tools.w90_downfold.downfold_to_ff_fixed_ligand``

Output layout:

  data/data-DFT/<family>/<material>/fexchange/<bond>/

Matrix files are plain multi-block text files compatible with
``fexchange.io.matrix.load_txt_blocks``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fexchange.io.matrix import load_txt_blocks
from fexchange.tools.w90_downfold import downfold_to_ff_fixed_ligand
from fexchange.tools.w90_extract import extract_w90_two_bond
from fexchange.utils.constants import RE_DEFAULTS_BY_N_ELE, RE_TO_N_ELE


UP_E_V_DEFAULT = 4.0
LAMBDA_P_E_V_DEFAULT = 0.0
RECHX_CATEGORIES = {"REOCl", "REOX", "RESI"}
COMPACT_FILES = {
    "constants.txt",
    "direct_t_mu.txt",
    "hopping_ff_downfolded.txt",
    "hopping_fp.txt",
    "onsite.txt",
    "projector.txt",
}


@dataclass(frozen=True)
class Site:
    atom: int
    cell: tuple[int, int, int]


@dataclass(frozen=True)
class BondSpec:
    label: str
    shell: str
    f1: Site
    f2: Site
    lig1: Site
    lig2: Site
    direction_rule: str


def site(atom: int, cell: tuple[int, int, int] = (0, 0, 0)) -> Site:
    return Site(atom=int(atom), cell=tuple(int(x) for x in cell))


def site_payload(s: Site) -> dict[str, Any]:
    return {"atom": int(s.atom), "cell": [int(x) for x in s.cell]}


def elem(label: str) -> str:
    match = re.match(r"([A-Z][a-z]?)", label)
    return match.group(1) if match else label


def material_re(material: str) -> str:
    re_name = elem(material)
    if re_name not in RE_TO_N_ELE:
        raise ValueError(f"cannot infer rare-earth element from material name: {material}")
    return re_name


def parse_win(path: Path) -> tuple[np.ndarray, list[tuple[str, np.ndarray]], float | None]:
    cell: list[list[float]] = []
    atoms: list[tuple[str, np.ndarray]] = []
    fermi: float | None = None
    in_cell = False
    in_atoms = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        low = text.lower()
        if not text:
            continue
        if low.startswith("fermi_energy"):
            parts = text.replace("=", " ").split()
            if len(parts) >= 2:
                fermi = float(parts[1])
            continue
        if low.startswith("begin unit_cell_cart"):
            in_cell = True
            continue
        if low.startswith("end unit_cell_cart"):
            in_cell = False
            continue
        if low.startswith("begin atoms_cart"):
            in_atoms = True
            continue
        if low.startswith("end atoms_cart"):
            in_atoms = False
            continue
        if low in {"ang", "bohr"}:
            continue
        if in_cell:
            cell.append([float(x) for x in text.split()[:3]])
        elif in_atoms:
            parts = text.split()
            atoms.append((parts[0], np.array([float(x) for x in parts[1:4]], dtype=float)))

    if len(cell) != 3:
        raise ValueError(f"failed to parse 3 lattice vectors from {path}")
    if not atoms:
        raise ValueError(f"failed to parse atoms_cart from {path}")
    return np.asarray(cell, dtype=float), atoms, fermi


def find_one(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"no *{suffix} under {directory}")
    plain = [p for p in matches if ".pp." not in p.name]
    return plain[0] if plain else matches[0]


def n_orb_per_atom(atoms: list[tuple[str, np.ndarray]]) -> list[int]:
    return [7 if elem(label) in RE_TO_N_ELE else 3 for label, _ in atoms]


def absolute_position(cell: np.ndarray, atoms: list[tuple[str, np.ndarray]], s: Site) -> np.ndarray:
    return atoms[s.atom][1] + np.asarray(s.cell, dtype=int) @ cell


def displacement(cell: np.ndarray, atoms: list[tuple[str, np.ndarray]], s1: Site, s2: Site) -> dict[str, Any]:
    vec = absolute_position(cell, atoms, s2) - absolute_position(cell, atoms, s1)
    length = float(np.linalg.norm(vec))
    unit = vec / length
    frac_vec = np.asarray(s2.cell, dtype=int) - np.asarray(s1.cell, dtype=int)
    return {
        "cart_angstrom": [float(x) for x in vec],
        "length_angstrom": length,
        "unit_cart": [float(x) for x in unit],
        "cell_delta": [int(x) for x in frac_vec],
    }


def rechx_bonds(category: str, material: str) -> list[BondSpec]:
    if category == "REOCl":
        return [
            BondSpec(
                label="J1",
                shell="NN",
                f1=site(2),
                f2=site(5),
                lig1=site(10),
                lig2=site(11),
                direction_rule="REChX NN representative",
            ),
            BondSpec(
                label="J2",
                shell="NNN",
                f1=site(0),
                f2=site(0, (1, 0, 0)),
                lig1=site(6),
                lig2=site(12, (0, -1, 0)),
                direction_rule="REChX in-plane +a NNN representative",
            ),
        ]
    if category == "REOX":
        if material == "NdOF":
            j1 = (site(1), site(4), site(8), site(9))
            j2_ligands = (site(6), site(12, (1, 1, 0)))
        else:
            j1 = (site(2), site(5), site(10), site(11))
            j2_ligands = (site(6, (1, 0, 0)), site(12, (1, 0, 0)))
        return [
            BondSpec("J1", "NN", j1[0], j1[1], j1[2], j1[3], "REChX NN representative"),
            BondSpec(
                "J2",
                "NNN",
                site(0),
                site(0, (1, 0, 0)),
                j2_ligands[0],
                j2_ligands[1],
                "REChX in-plane +a NNN representative",
            ),
        ]
    if category == "RESI":
        if material == "SmSI":
            j1 = (site(1), site(4), site(8), site(9))
            j2_ligands = (site(6, (1, 0, 0)), site(12, (1, 0, 0)))
        else:
            j1 = (site(2), site(5), site(10), site(11))
            j2_ligands = (site(6), site(12, (0, -1, 0)))
        return [
            BondSpec("J1", "NN", j1[0], j1[1], j1[2], j1[3], "REChX NN representative"),
            BondSpec(
                "J2",
                "NNN",
                site(0),
                site(0, (1, 0, 0)),
                j2_ligands[0],
                j2_ligands[1],
                "REChX in-plane +a NNN representative",
            ),
        ]
    raise ValueError(f"unsupported REChX category: {category}")


def rex3_r3_bonds(material: str) -> list[BondSpec]:
    if material == "SmI3":
        lig1, lig2 = site(21), site(20, (-1, 0, 0))
    elif material in {"DyI3", "YbI3"}:
        lig1, lig2 = site(20, (-1, 0, 0)), site(23)
    else:
        lig1, lig2 = site(23), site(20, (-1, 0, 0))
    return [
        BondSpec(
            label="z",
            shell="NN",
            f1=site(4),
            f2=site(5),
            lig1=lig1,
            lig2=lig2,
            direction_rule="REX3 R-3 nearest-neighbor representative",
        )
    ]


def rex3_c2m_bonds() -> list[BondSpec]:
    return [
        BondSpec(
            label="NN_short",
            shell="NN",
            f1=site(0),
            f2=site(1, (1, -1, 0)),
            lig1=site(4, (1, 0, 0)),
            lig2=site(5, (0, -1, -1)),
            direction_rule="REX3 C2/m short nearest-neighbor representative",
        ),
        BondSpec(
            label="NN_long",
            shell="NN",
            f1=site(0),
            f2=site(1, (1, 0, 0)),
            lig1=site(2),
            lig2=site(7, (1, 0, -1)),
            direction_rule="REX3 C2/m long nearest-neighbor representative",
        ),
    ]


def bond_specs(category: str, material: str) -> list[BondSpec]:
    if category in {"REOCl", "REOX", "RESI"}:
        return rechx_bonds(category, material)
    if category == "REX3-R3":
        return rex3_r3_bonds(material)
    if category == "REX3-C2m":
        return rex3_c2m_bonds()
    return []


def q(value: Any) -> str:
    return json.dumps(str(value))


def unquote(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value.strip('"')
    return str(parsed)


def write_extract_toml(
    path: Path,
    *,
    hr_path: Path,
    n_orb: list[int],
    spec: BondSpec,
    onsite_out: Path,
    hopping_fp_out: Path,
    hopping_ff_out: Path,
    fermi_level: float,
    f_trace_tol: float,
) -> None:
    lines = [
        f"hr_path = {q(hr_path)}",
        "spinor = true",
        'energy_unit = "eV"',
        f"fermi_level = {fermi_level:.12g}",
        f"f_trace_tol = {f_trace_tol:.12g}",
        "n_orb_per_atom = [" + ", ".join(str(x) for x in n_orb) + "]",
        f"onsite_out = {q(onsite_out)}",
        f"hopping_fp_out = {q(hopping_fp_out)}",
        f"hopping_ff_out = {q(hopping_ff_out)}",
        "",
    ]
    for name, value in (
        ("f1_site", spec.f1),
        ("f2_site", spec.f2),
        ("lig1_site", spec.lig1),
        ("lig2_site", spec.lig2),
    ):
        lines.extend(
            [
                f"[{name}]",
                f"atom = {value.atom}",
                "cell = [" + ", ".join(str(x) for x in value.cell) + "]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_downfold_toml(
    path: Path,
    *,
    hopping_fp_in: Path,
    direct_t_mu_in: Path,
    output: Path,
    delta_lig1: float,
    delta_lig2: float,
    lambda_p: float,
) -> None:
    lines = [
        f"hopping_fp_in = {q(hopping_fp_in)}",
        f"direct_t_mu_in = {q(direct_t_mu_in)}",
        f"output = {q(output)}",
        f"delta_lig1 = {delta_lig1:.12e}",
        f"delta_lig2 = {delta_lig2:.12e}",
        f"lambda_p = {lambda_p:.12e}",
        "degenerate_tol = 1.0e-6",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_txt(
    path: Path,
    *,
    atoms: list[tuple[str, np.ndarray]],
    spec: BondSpec,
    delta: dict[str, float],
    trace_avg: dict[str, float],
) -> None:
    rows = [
        "# Delta_ligand = E_f_trace_avg - E_p_trace_avg, unit eV",
        "# ligand atom cell element Delta_eV E_f_trace_avg_eV E_p_trace_avg_eV",
    ]
    e_f = 0.5 * (float(trace_avg["f1"]) + float(trace_avg["f2"]))
    for key, lig in (("lig1", spec.lig1), ("lig2", spec.lig2)):
        rows.append(
            f"{key} {lig.atom} "
            f"{lig.cell[0]},{lig.cell[1]},{lig.cell[2]} "
            f"{elem(atoms[lig.atom][0])} "
            f"{float(delta[key]):.12e} {e_f:.12e} {float(trace_avg[key]):.12e}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_lambda_txt(path: Path, *, re_name: str, n_ele: int, zeta_e_v: float) -> None:
    rows = [
        "# lambda_RE/zeta from fexchange.utils.constants.RE_DEFAULTS_BY_N_ELE",
        "# unit eV",
        f"RE {re_name}",
        f"n_ele {n_ele}",
        f"lambda_RE_eV {zeta_e_v:.12e}",
        f"zeta_eV {zeta_e_v:.12e}",
        f"lambda_RE_meV {1000.0 * zeta_e_v:.12e}",
        f"zeta_meV {1000.0 * zeta_e_v:.12e}",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def site_line(prefix: str, payload: dict[str, Any]) -> str:
    cell = payload["cell"]
    return f"{prefix}_atom {payload['atom']}\n{prefix}_cell {cell[0]} {cell[1]} {cell[2]}"


def trace_averages_from_onsite(path: Path) -> dict[str, float]:
    blocks = load_txt_blocks(path, {"h_f1": 196, "h_f2": 196, "h_lig1": 36, "h_lig2": 36})
    matrices = {
        "f1": blocks["h_f1"].reshape(14, 14),
        "f2": blocks["h_f2"].reshape(14, 14),
        "lig1": blocks["h_lig1"].reshape(6, 6),
        "lig2": blocks["h_lig2"].reshape(6, 6),
    }
    return {key: float(np.trace(mat).real / mat.shape[0]) for key, mat in matrices.items()}


def ensure_trace_fields(row: dict[str, Any]) -> dict[str, float]:
    trace = row.get("trace_avg_eV")
    if isinstance(trace, dict) and all(k in trace for k in ("f1", "f2", "lig1", "lig2")):
        return {key: float(trace[key]) for key in ("f1", "f2", "lig1", "lig2")}

    out_dir = row.get("out_dir")
    if out_dir:
        onsite = Path(str(out_dir)) / "onsite.txt"
        if onsite.exists():
            trace = trace_averages_from_onsite(onsite)
            row["trace_avg_eV"] = trace
            row["E_f_trace_avg_eV"] = 0.5 * (trace["f1"] + trace["f2"])
            row["E_lig1_trace_avg_eV"] = trace["lig1"]
            row["E_lig2_trace_avg_eV"] = trace["lig2"]
            row["Delta_lig1_eV"] = row["E_f_trace_avg_eV"] - trace["lig1"]
            row["Delta_lig2_eV"] = row["E_f_trace_avg_eV"] - trace["lig2"]
            return trace

    raise ValueError(f"cannot resolve onsite trace averages for {row.get('out_dir', row.get('material'))}")


def write_constants_from_summary(path: Path, row: dict[str, Any]) -> None:
    trace = ensure_trace_fields(row)
    e_f_trace = 0.5 * (trace["f1"] + trace["f2"])
    lines = [
        "# DFT-extracted scalar parameters and site selection",
        "# Units: eV for energies, Angstrom for bond vectors/lengths.",
        'schema_version "fxe.constants.v1"',
        f"category {row['category']}",
        f"material {row['material']}",
        f"bond {row['bond']}",
        f"shell {row['shell']}",
        f"atom_index_base 0",
        "cell_convention Wannier90_integer_lattice_image",
        "",
        "# Delta_lig = E_f_trace_avg_eV - E_lig_trace_avg_eV",
        f"E_f1_trace_avg_eV {trace['f1']:.12e}",
        f"E_f2_trace_avg_eV {trace['f2']:.12e}",
        f"E_f_trace_avg_eV {e_f_trace:.12e}",
        f"E_lig1_trace_avg_eV {trace['lig1']:.12e}",
        f"E_lig2_trace_avg_eV {trace['lig2']:.12e}",
        f"Delta_definition E_f_trace_avg_eV_minus_E_lig_trace_avg_eV",
        f"Delta_lig1_eV {float(row['Delta_lig1_eV']):.12e}",
        f"Delta_lig2_eV {float(row['Delta_lig2_eV']):.12e}",
        "",
        "# Site selection",
        site_line("f1", row["f1"]),
        site_line("f2", row["f2"]),
        site_line("lig1", row["lig1"]),
        site_line("lig2", row["lig2"]),
        "bond_vector_angstrom " + " ".join(f"{float(x):.12e}" for x in row["bond_vector_angstrom"]),
        f"bond_length_angstrom {float(row['bond_length_angstrom']):.12e}",
        "bond_unit_vector " + " ".join(f"{float(x):.12e}" for x in row["bond_unit_vector"]),
        'onsite_file "onsite.txt"',
        'hopping_fp_file "hopping_fp.txt"',
        'direct_ff_file "direct_t_mu.txt"',
        'folded_ff_file "hopping_ff_downfolded.txt"',
    ]
    onsite_cef = row.get("onsite_cef")
    if onsite_cef:
        lines.extend(constants_onsite_cef_lines(onsite_cef))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def constants_onsite_cef_lines(payload: dict[str, Any]) -> list[str]:
    cef = payload["cef"]
    g_components = payload["g_components"]
    g_tensor_complex = np.asarray(payload["g_tensor"], dtype=np.complex128)
    g_tensor = g_tensor_complex.real
    lines = [
        "",
        "# RE local SOC/CEF fitted from onsite H_f_local = 0.5*(h_f1+h_f2)",
        f"RE {payload['RE']}",
        f"n_ele {int(payload['n_ele'])}",
        f"zeta_eV {float(payload['zeta_eV']):.12e}",
        f"zeta_meV {float(payload['zeta_meV']):.12e}",
        f"cef_point_group {cef['point_group']}",
        f"cef_mode_q3 {cef['mode_q3']}",
        f"cef_J {float(cef['J']):.12e}",
        f'kramer_name "{payload["kramer_name"]}"',
        f'projector_file "{payload["projector_file"]}"',
    ]
    for name, value in cef["B_params_eV"].items():
        lines.append(f"cef_{name}_eV {float(value):.12e}")
        lines.append(f"cef_{name}_meV {float(cef['B_params_meV'][name]):.12e}")
    lines.extend(
        [
            "",
            "# Ground doublet g tensor from cef_states using the fitted CEF parameters",
            f"g_doublet_type {payload['doublet_type']}",
            f"g_ground_irrep {payload['ground_irrep']}",
            f"gx {float(g_components['gx']):.12e}",
            f"gy {float(g_components['gy']):.12e}",
            f"gz {float(g_components['gz']):.12e}",
            "g_tensor_row0 " + " ".join(f"{float(x):.12e}" for x in g_tensor[0]),
            "g_tensor_row1 " + " ".join(f"{float(x):.12e}" for x in g_tensor[1]),
            "g_tensor_row2 " + " ".join(f"{float(x):.12e}" for x in g_tensor[2]),
            f"g_tensor_imag_max_abs {float(payload['g_tensor_imag_max_abs']):.12e}",
        ]
    )
    return lines


def read_constants(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"status": "ok", "out_dir": str(path.parent)}
    sites: dict[str, dict[str, Any]] = {}
    onsite_cef: dict[str, Any] = {"cef": {"B_params_eV": {}, "B_params_meV": {}}}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        values = parts[1:]
        if key in {"category", "material", "bond", "shell"}:
            row[key] = values[0]
        elif key in {"Delta_lig1_eV", "Delta_lig2_eV"}:
            row[key] = float(values[0])
        elif key in {"E_f1_trace_avg_eV", "E_f2_trace_avg_eV", "E_lig1_trace_avg_eV", "E_lig2_trace_avg_eV"}:
            trace = row.setdefault("trace_avg_eV", {})
            trace_key = {
                "E_f1_trace_avg_eV": "f1",
                "E_f2_trace_avg_eV": "f2",
                "E_lig1_trace_avg_eV": "lig1",
                "E_lig2_trace_avg_eV": "lig2",
            }[key]
            trace[trace_key] = float(values[0])
        elif key == "E_f_trace_avg_eV":
            row[key] = float(values[0])
        elif key == "bond_length_angstrom":
            row[key] = float(values[0])
        elif key in {"bond_vector_angstrom", "bond_unit_vector"}:
            row[key] = [float(x) for x in values]
        elif key.endswith("_atom"):
            sites.setdefault(key.removesuffix("_atom"), {})["atom"] = int(values[0])
        elif key.endswith("_cell"):
            sites.setdefault(key.removesuffix("_cell"), {})["cell"] = [int(x) for x in values[:3]]
        elif key == "RE":
            onsite_cef["RE"] = unquote(values[0])
        elif key == "n_ele":
            onsite_cef["n_ele"] = int(values[0])
        elif key in {"zeta_eV", "zeta_meV"}:
            onsite_cef[key] = float(values[0])
        elif key == "cef_point_group":
            onsite_cef["cef"]["point_group"] = unquote(values[0])
        elif key == "cef_mode_q3":
            onsite_cef["cef"]["mode_q3"] = unquote(values[0])
        elif key == "cef_J":
            onsite_cef["cef"]["J"] = float(values[0])
        elif key.startswith("cef_") and key.endswith("_eV"):
            name = key.removeprefix("cef_").removesuffix("_eV")
            onsite_cef["cef"]["B_params_eV"][name] = float(values[0])
        elif key.startswith("cef_") and key.endswith("_meV"):
            name = key.removeprefix("cef_").removesuffix("_meV")
            onsite_cef["cef"]["B_params_meV"][name] = float(values[0])
        elif key == "kramer_name":
            onsite_cef["kramer_name"] = unquote(values[0])
        elif key == "projector_file":
            onsite_cef["projector_file"] = unquote(values[0])
        elif key == "g_doublet_type":
            onsite_cef["doublet_type"] = unquote(values[0])
        elif key == "g_ground_irrep":
            onsite_cef["ground_irrep"] = unquote(values[0])
        elif key in {"gx", "gy", "gz"}:
            onsite_cef.setdefault("g_components", {})[key] = float(values[0])
        elif key.startswith("g_tensor_row"):
            idx = int(key.removeprefix("g_tensor_row"))
            rows = onsite_cef.setdefault("g_tensor", [None, None, None])
            rows[idx] = [float(x) for x in values[:3]]
        elif key == "g_tensor_imag_max_abs":
            onsite_cef["g_tensor_imag_max_abs"] = float(values[0])
    row.update({name: payload for name, payload in sites.items() if name in {"f1", "f2", "lig1", "lig2"}})
    if constants_have_rechx_cef_params(onsite_cef):
        row["onsite_cef"] = onsite_cef
    return row


def is_rechx_category(category: str) -> bool:
    return category in RECHX_CATEGORIES


def constants_have_rechx_cef_params(payload: dict[str, Any]) -> bool:
    cef = payload.get("cef", {})
    return all(
        key in payload
        for key in ("RE", "n_ele", "zeta_eV", "zeta_meV")
    ) and all(
        key in cef
        for key in ("point_group", "mode_q3", "J")
    ) and bool(cef.get("B_params_meV"))


def compute_rechx_cef_state_from_parameters(
    payload: dict[str, Any],
    *,
    material: str,
    bond: str,
    projector_out: Path | None = None,
) -> dict[str, Any]:
    """Run cef_states from already stored CEF parameters and optionally write projector.txt."""
    from fexchange.tools.cef_states import _extract_ground_doublet, _write_projector_txt, compute_cef_states

    cef = payload["cef"]
    kramer_name = str(payload.get("kramer_name") or f"{material}_{bond}_REChX_C3v_sin")
    cef_result = compute_cef_states(
        point_group=cef["point_group"],
        J=float(cef["J"]),
        B_params={name: float(value) for name, value in cef["B_params_meV"].items()},
        mode_q3=cef["mode_q3"],
    )
    doublet = _extract_ground_doublet(cef_result)
    if projector_out is not None:
        _write_projector_txt(
            projector_out,
            doublet,
            result=cef_result,
            kramer_name=kramer_name,
            basis_id="complex_spherical_j_v1",
            orbital_order_id="f7_m-3..3_v1",
        )
    g_tensor = np.asarray(doublet["g_tensor"], dtype=np.complex128)
    updated = {
        **payload,
        "kramer_name": kramer_name,
        "projector_file": "projector.txt",
        "doublet_type": doublet["doublet_type"],
        "ground_irrep": doublet["ground_irrep"],
        "g_components": doublet["g_components"],
        "g_tensor": g_tensor.real.tolist(),
        "g_tensor_imag_max_abs": float(np.max(np.abs(g_tensor.imag))),
    }
    return updated


def compute_rechx_onsite_cef_constants(
    onsite: Path,
    *,
    material: str,
    bond: str,
    projector_out: Path | None = None,
) -> dict[str, Any]:
    """Compute REChX onsite zeta/CEF and cef_states ground-doublet g tensor."""
    from fexchange.tools.w90_onsite import analyze_onsite

    re_name = material_re(material)
    n_ele = RE_TO_N_ELE[re_name]
    kramer_name = f"{material}_{bond}_REChX_C3v_sin"
    onsite_result = analyze_onsite(
        onsite,
        n_ele=n_ele,
        symmetry="C3v",
        mode_q3="sin",
    )
    payload = {
        "RE": re_name,
        "n_ele": n_ele,
        "kramer_name": kramer_name,
        "projector_file": "projector.txt",
        "zeta_eV": float(onsite_result["zeta_eV"]),
        "zeta_meV": float(onsite_result["zeta_meV"]),
        "cef": onsite_result["cef"],
    }
    return compute_rechx_cef_state_from_parameters(
        payload,
        material=material,
        bond=bond,
        projector_out=projector_out,
    )


def write_parameters_toml(
    path: Path,
    *,
    category: str,
    material: str,
    spec: BondSpec,
    re_name: str,
    n_ele: int,
    zeta_e_v: float,
    delta: dict[str, float],
    up_e_v: float,
    lambda_p_e_v: float,
) -> None:
    lines = [
        'schema_version = "fxe.data_dft_fexchange.v2"',
        'energy_unit = "eV"',
        f"category = {q(category)}",
        f"material = {q(material)}",
        f"bond = {q(spec.label)}",
        f"shell = {q(spec.shell)}",
        'atom_index_base = 0',
        'onsite_file = "onsite.txt"',
        'fopt_hopping_file = "hopping_fp.txt"',
        'direct_ff_file = "direct_t_mu.txt"',
        'sopt_hopping_file = "hopping_ff_downfolded.txt"',
        "",
        "[fsite]",
        f"RE = {q(re_name)}",
        f"n_ele = {n_ele}",
        f"zeta = {zeta_e_v:.12e}",
        "",
        "[ligand.1]",
        f"Delta = {float(delta['lig1']):.12e}",
        f"U_p = {up_e_v:.12e}",
        f"lambda_p = {lambda_p_e_v:.12e}",
        "",
        "[ligand.2]",
        f"Delta = {float(delta['lig2']):.12e}",
        f"U_p = {up_e_v:.12e}",
        f"lambda_p = {lambda_p_e_v:.12e}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selection_txt(
    path: Path,
    *,
    category: str,
    material: str,
    spec: BondSpec,
    atoms: list[tuple[str, np.ndarray]],
    bond_vec: dict[str, Any],
    fermi_level: float,
) -> None:
    rows = [
        "# Atom indices are 0-based and match the Wannier90 atoms_cart order.",
        f"category {category}",
        f"material {material}",
        f"bond {spec.label}",
        f"shell {spec.shell}",
        f"direction_rule {spec.direction_rule}",
        f"fermi_level_eV {fermi_level:.12e}",
        f"f1 atom={spec.f1.atom} cell={spec.f1.cell[0]},{spec.f1.cell[1]},{spec.f1.cell[2]} element={elem(atoms[spec.f1.atom][0])}",
        f"f2 atom={spec.f2.atom} cell={spec.f2.cell[0]},{spec.f2.cell[1]},{spec.f2.cell[2]} element={elem(atoms[spec.f2.atom][0])}",
        f"lig1 atom={spec.lig1.atom} cell={spec.lig1.cell[0]},{spec.lig1.cell[1]},{spec.lig1.cell[2]} element={elem(atoms[spec.lig1.atom][0])}",
        f"lig2 atom={spec.lig2.atom} cell={spec.lig2.cell[0]},{spec.lig2.cell[1]},{spec.lig2.cell[2]} element={elem(atoms[spec.lig2.atom][0])}",
        "bond_vector_angstrom " + " ".join(f"{x:.12e}" for x in bond_vec["cart_angstrom"]),
        f"bond_length_angstrom {bond_vec['length_angstrom']:.12e}",
        "bond_unit_vector " + " ".join(f"{x:.12e}" for x in bond_vec["unit_cart"]),
        "bond_cell_delta " + " ".join(str(x) for x in bond_vec["cell_delta"]),
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_outputs(out_dir: Path) -> None:
    load_txt_blocks(out_dir / "onsite.txt", {"h_f1": 196, "h_f2": 196, "h_lig1": 36, "h_lig2": 36})
    load_txt_blocks(
        out_dir / "hopping_fp.txt",
        {"t_f1_lig1": 84, "t_f1_lig2": 84, "t_f2_lig1": 84, "t_f2_lig2": 84},
    )
    load_txt_blocks(out_dir / "direct_t_mu.txt", {"t_mu": 196})
    load_txt_blocks(out_dir / "hopping_ff_downfolded.txt", {"t_mu": 196})
    if not (out_dir / "constants.txt").exists():
        raise FileNotFoundError(out_dir / "constants.txt")


def extract_bond(
    *,
    category: str,
    material_dir: Path,
    material: str,
    win_path: Path,
    hr_path: Path,
    spec: BondSpec,
    overwrite: bool,
    up_e_v: float,
    lambda_p_e_v: float,
    f_trace_tol: float,
) -> dict[str, Any]:
    cell, atoms, fermi = parse_win(win_path)
    fermi_level = 0.0 if fermi is None else float(fermi)
    n_orb = n_orb_per_atom(atoms)
    re_name = material_re(material)
    n_ele = RE_TO_N_ELE[re_name]

    out_dir = material_dir / "fexchange" / spec.label
    constants_file = out_dir / "constants.txt"
    if constants_file.exists() and not overwrite:
        return {"category": category, "material": material, "bond": spec.label, "status": "skipped_existing"}
    out_dir.mkdir(parents=True, exist_ok=True)

    onsite = out_dir / "onsite.txt"
    hopping_fp = out_dir / "hopping_fp.txt"
    direct_ff = out_dir / "direct_t_mu.txt"
    folded_ff = out_dir / "hopping_ff_downfolded.txt"

    extraction = extract_w90_two_bond(
        hr_path=hr_path,
        n_orb_per_atom=n_orb,
        spinor=True,
        f1_site=site_payload(spec.f1),
        f2_site=site_payload(spec.f2),
        lig1_site=site_payload(spec.lig1),
        lig2_site=site_payload(spec.lig2),
        onsite_out=onsite,
        hopping_fp_out=hopping_fp,
        hopping_ff_out=direct_ff,
        energy_unit="eV",
        fermi_level=fermi_level,
        f_trace_tol=f_trace_tol,
    )

    delta_lig1 = float(extraction["delta"]["lig1"])
    delta_lig2 = float(extraction["delta"]["lig2"])
    downfold = downfold_to_ff_fixed_ligand(
        hopping_fp_in=hopping_fp,
        direct_t_mu_in=direct_ff,
        output=folded_ff,
        delta_lig1=delta_lig1,
        delta_lig2=delta_lig2,
        lambda_p=lambda_p_e_v,
    )

    bond_vec = displacement(cell, atoms, spec.f1, spec.f2)
    summary = {
        "status": "ok",
        "category": category,
        "material": material,
        "bond": spec.label,
        "shell": spec.shell,
        "out_dir": str(out_dir),
        "f1": site_payload(spec.f1),
        "f2": site_payload(spec.f2),
        "lig1": site_payload(spec.lig1),
        "lig2": site_payload(spec.lig2),
        "bond_vector_angstrom": bond_vec["cart_angstrom"],
        "bond_length_angstrom": bond_vec["length_angstrom"],
        "bond_unit_vector": bond_vec["unit_cart"],
        "Delta_lig1_eV": delta_lig1,
        "Delta_lig2_eV": delta_lig2,
        "trace_avg_eV": {k: float(v) for k, v in extraction["trace_avg"].items()},
        "E_f_trace_avg_eV": 0.5 * (
            float(extraction["trace_avg"]["f1"]) + float(extraction["trace_avg"]["f2"])
        ),
        "E_lig1_trace_avg_eV": float(extraction["trace_avg"]["lig1"]),
        "E_lig2_trace_avg_eV": float(extraction["trace_avg"]["lig2"]),
        "f_trace_diff_eV": float(extraction["meta"]["f_trace_diff"]),
        "direct_ff_norm_eV": float(downfold["meta"]["direct_norm"]),
        "folding_correction_norm_eV": float(downfold["meta"]["correction_norm"]),
        "files": [
            "constants.txt",
            "onsite.txt",
            "hopping_fp.txt",
            "direct_t_mu.txt",
            "hopping_ff_downfolded.txt",
        ],
    }
    if is_rechx_category(category):
        summary["onsite_cef"] = compute_rechx_onsite_cef_constants(
            onsite,
            material=material,
            bond=spec.label,
            projector_out=out_dir / "projector.txt",
        )
    write_constants_from_summary(constants_file, summary)
    validate_outputs(out_dir)
    return {"status": "ok", **summary}


def material_dirs(root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for material_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            rows.append((category_dir.name, material_dir))
    return rows


def write_root_summary(root: Path, results: list[dict[str, Any]]) -> None:
    def status(row: dict[str, Any]) -> str:
        if row.get("status"):
            return str(row["status"])
        return "failed" if row.get("error") else "ok"

    ok_rows = [r for r in results if status(r) == "ok"]
    lines = [
        "# fexchange extraction summary",
        "# material bond status Delta_lig1_eV Delta_lig2_eV bond_length_A out_dir",
    ]
    for row in results:
        lines.append(
            " ".join(
                [
                    str(row.get("material", "")),
                    str(row.get("bond", "")),
                    status(row),
                    f"{float(row.get('Delta_lig1_eV', np.nan)):.12e}",
                    f"{float(row.get('Delta_lig2_eV', np.nan)):.12e}",
                    f"{float(row.get('bond_length_angstrom', np.nan)):.12e}",
                    str(row.get("out_dir", "")),
                ]
            )
        )
    text = "\n".join(lines) + "\n"
    (root / "fexchange_extraction_summary.txt").write_text(text, encoding="utf-8")
    (root / "fexchange_extraction_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "fxe.data_dft_fexchange.summary.v2",
                "n_results": len(results),
                "n_ok": len(ok_rows),
                "n_failed": sum(1 for row in results if status(row) == "failed"),
                "n_skipped_existing": sum(1 for row in results if status(row) == "skipped_existing"),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def existing_bond_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*/fexchange/*") if p.is_dir())


def load_existing_summary(bond_dir: Path) -> dict[str, Any]:
    constants = bond_dir / "constants.txt"
    if constants.exists():
        return read_constants(constants)
    summary = bond_dir / "summary.txt"
    if summary.exists():
        row = json.loads(summary.read_text(encoding="utf-8"))
        row.setdefault("status", "ok")
        row.setdefault("out_dir", str(bond_dir))
        return row
    return {"status": "failed", "out_dir": str(bond_dir), "error": "missing constants.txt/summary.txt"}


def compact_existing(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bond_dir in existing_bond_dirs(root):
        try:
            row = load_existing_summary(bond_dir)
            if row.get("status") == "ok":
                write_constants_from_summary(bond_dir / "constants.txt", row)
                for child in bond_dir.iterdir():
                    if child.is_file() and child.name not in COMPACT_FILES:
                        child.unlink()
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append({"status": "failed", "out_dir": str(bond_dir), "error": str(exc)})
    return rows


def update_rechx_cef_existing(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bond_dir in existing_bond_dirs(root):
        try:
            row = load_existing_summary(bond_dir)
            if row.get("status") == "ok" and is_rechx_category(str(row.get("category", ""))):
                if constants_have_rechx_cef_params(row.get("onsite_cef", {})):
                    row["onsite_cef"] = compute_rechx_cef_state_from_parameters(
                        row["onsite_cef"],
                        material=str(row["material"]),
                        bond=str(row["bond"]),
                        projector_out=bond_dir / "projector.txt",
                    )
                else:
                    onsite = bond_dir / "onsite.txt"
                    if not onsite.exists():
                        raise FileNotFoundError(onsite)
                    row["onsite_cef"] = compute_rechx_onsite_cef_constants(
                        onsite,
                        material=str(row["material"]),
                        bond=str(row["bond"]),
                        projector_out=bond_dir / "projector.txt",
                    )
                write_constants_from_summary(bond_dir / "constants.txt", row)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append({"status": "failed", "out_dir": str(bond_dir), "error": str(exc)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/data-DFT"))
    parser.add_argument("--category", help="Only process one top-level family, e.g. REX3-R3")
    parser.add_argument("--material", help="Only process one material, e.g. DyBr3")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-root-summary", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Rebuild root summary from existing fexchange constants/summary files and exit.",
    )
    parser.add_argument(
        "--compact-existing",
        action="store_true",
        help="Write constants.txt and delete non-runtime files in existing fexchange bond directories.",
    )
    parser.add_argument(
        "--update-rechx-cef",
        action="store_true",
        help="Update existing REChX projector/g from constants CEF params; fall back to onsite only if params are absent.",
    )
    parser.add_argument("--Up-eV", type=float, default=UP_E_V_DEFAULT)
    parser.add_argument("--lambda-p-eV", type=float, default=LAMBDA_P_E_V_DEFAULT)
    parser.add_argument("--f-trace-tol", type=float, default=1.0e-2)
    args = parser.parse_args()

    if args.compact_existing:
        summaries = compact_existing(args.root)
        write_root_summary(args.root, summaries)
        print(f"compacted={len(summaries)}")
        print(f"wrote {args.root / 'fexchange_extraction_summary.txt'}")
        return 0 if all(row.get("status") == "ok" for row in summaries) else 1

    if args.update_rechx_cef:
        summaries = update_rechx_cef_existing(args.root)
        write_root_summary(args.root, summaries)
        updated = sum(
            1
            for row in summaries
            if row.get("status") == "ok" and is_rechx_category(str(row.get("category", "")))
        )
        print(f"updated_rechx={updated}")
        print(f"wrote {args.root / 'fexchange_extraction_summary.txt'}")
        return 0 if all(row.get("status") == "ok" for row in summaries) else 1

    if args.summarize_existing:
        summaries: list[dict[str, Any]] = []
        for bond_dir in existing_bond_dirs(args.root):
            try:
                summaries.append(load_existing_summary(bond_dir))
            except Exception as exc:  # noqa: BLE001
                summaries.append({"status": "failed", "out_dir": str(bond_dir), "error": str(exc)})
        write_root_summary(args.root, summaries)
        print(f"wrote {args.root / 'fexchange_extraction_summary.txt'}")
        print(f"existing_summaries={len(summaries)}")
        return 0 if all(row.get("status") == "ok" for row in summaries) else 1

    results: list[dict[str, Any]] = []
    for category, material_dir in material_dirs(args.root):
        if args.category and category != args.category:
            continue
        material = material_dir.name
        if args.material and material != args.material:
            continue
        specs = bond_specs(category, material)
        if not specs:
            continue
        try:
            win_path = find_one(material_dir / "wannier", ".win")
            hr_path = find_one(material_dir / "wannier", "_hr.dat")
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "category": category,
                    "material": material,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        for spec in specs:
            try:
                results.append(
                    extract_bond(
                        category=category,
                        material_dir=material_dir,
                        material=material,
                        win_path=win_path,
                        hr_path=hr_path,
                        spec=spec,
                        overwrite=args.overwrite,
                        up_e_v=args.Up_eV,
                        lambda_p_e_v=args.lambda_p_eV,
                        f_trace_tol=args.f_trace_tol,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "category": category,
                        "material": material,
                        "bond": spec.label,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    if not args.no_root_summary:
        write_root_summary(args.root, results)
    n_ok = sum(1 for row in results if row.get("status") == "ok")
    n_failed = sum(1 for row in results if row.get("status") == "failed")
    n_skipped = sum(1 for row in results if row.get("status") == "skipped_existing")
    if not args.no_root_summary:
        print(f"wrote {args.root / 'fexchange_extraction_summary.txt'}")
    print(f"ok={n_ok} failed={n_failed} skipped={n_skipped}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
