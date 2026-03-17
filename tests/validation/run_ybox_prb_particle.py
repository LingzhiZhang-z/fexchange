"""CLI-driven validation runner for YbOX and PRB particle-picture workflows.

This runner intentionally avoids importing fexchange runtime APIs for the
actual calculation path.  It only:
1. parses legacy test inputs,
2. generates tool / CLI input files,
3. invokes the existing CLI entry points,
4. reads saved outputs, and
5. performs plain NumPy-based comparison / plotting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_CANDIDATES = (
    REPO_ROOT / "data",
    Path("/Users/lingzhi/Documents/Code/fexchange/data"),
)
OUTPUT_ROOT = REPO_ROOT / "outputs" / "ybox_prb_particle_20260317"
TOOLS_DIR = REPO_ROOT / "fexchange" / "tools"

YBOX_MATERIALS = ("YbOBr", "YbOCl", "YbOF")
YBOX_TOL_MEW = 1.0e-3
PRB_SITES = ("K", "Rb", "Cs")
PRB_ZETA_MEV = {"K": 120.0, "Rb": 110.0, "Cs": 110.0}
PRB_SLATER_RATIOS = (12.980, 8.163, 5.878)
PRB_U_LIST = (2.0, 3.0, 4.0, 6.0)
PRB_RATIO_LIST = tuple(round(0.01 * idx, 2) for idx in range(21))
PRB_SCALE = {2.0: 1.0, 3.0: 1.5, 4.0: 2.0, 6.0: 3.0}
GAUGE_ROTATIONS = {
    "identity": np.diag([1.0, 1.0, 1.0]),
    "flip_xz": np.diag([-1.0, 1.0, -1.0]),
    "flip_yz": np.diag([1.0, -1.0, -1.0]),
    "flip_xy": np.diag([-1.0, -1.0, 1.0]),
}
SIGMA_0 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMAS = (SIGMA_0, SIGMA_X, SIGMA_Y, SIGMA_Z)


def _resolve_data_root() -> Path:
    for path in DATA_ROOT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate repository data root.")


DATA_ROOT = _resolve_data_root()
TEST_ROOT = DATA_ROOT / "test"


@dataclass(frozen=True)
class LegacyCase:
    material: str
    prefix: str
    hr_relpath: str
    slater_ratios: tuple[float, float, float]
    U_list_eV: tuple[float, ...]
    ratio_list: tuple[float, ...]
    spinor: bool
    ndims: tuple[int, ...]
    atoms0: tuple[int, int, int, int]
    cells: tuple[tuple[int, int, int], ...]
    input_path: Path

    @property
    def reference_dir(self) -> Path:
        return TEST_ROOT / self.material / self.prefix


def _fmt12(value: float) -> str:
    return f"{value:.12f}"


def _core_dir_token(n_ele: int, r42: float, r62: float, scheme: str = "RS") -> str:
    return f"n-{n_ele}_r42-{_fmt12(r42)}_r62-{_fmt12(r62)}_scheme-{scheme}"


def _ujhz_dir_token(U_meV: float, Jh_meV: float, zeta_meV: float) -> str:
    return f"U-{_fmt12(U_meV)}_Jh-{_fmt12(Jh_meV)}_z-{_fmt12(zeta_meV)}"


def _build_l4_path(case_dir: Path, *, n_ele: int, r42: float, r62: float, U_meV: float, Jh_meV: float, zeta_meV: float) -> Path:
    return (
        case_dir
        / "outputs"
        / "core"
        / _core_dir_token(n_ele, r42, r62, "RS")
        / _ujhz_dir_token(U_meV, Jh_meV, zeta_meV)
        / "L4"
        / "data.npz"
    )


def _parse_numeric_list(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split())


def _parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split())


def parse_legacy_input(path: Path) -> LegacyCase:
    raw = path.read_text().splitlines()
    data: dict[str, str] = {}
    rs_lines: list[str] = []
    in_rs = False
    for line in raw:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "<RS":
            in_rs = True
            continue
        if stripped == "RS>":
            in_rs = False
            continue
        if in_rs:
            rs_lines.append(stripped)
            continue
        parts = stripped.split(None, 1)
        key = parts[0]
        value = parts[1] if len(parts) > 1 else ""
        data[key] = value

    if len(rs_lines) != 4:
        raise ValueError(f"{path}: expected 4 RS lines, got {len(rs_lines)}")

    atoms1 = _parse_int_list(data["ATOMS"])
    cells = tuple(tuple(int(x) for x in line.split()) for line in rs_lines)
    return LegacyCase(
        material=path.parent.parent.name,
        prefix=data["PREFIX"],
        hr_relpath=data["FILE_HR"],
        slater_ratios=tuple(_parse_numeric_list(data["SLATER_RATIOS"])),
        U_list_eV=tuple(_parse_numeric_list(data["Us"])),
        ratio_list=tuple(_parse_numeric_list(data["ratios"])),
        spinor=data["SPINOR"].strip().upper() == "TRUE",
        ndims=tuple(_parse_int_list(data["NDIMS"])),
        atoms0=tuple(atom - 1 for atom in atoms1),
        cells=cells,
        input_path=path,
    )


def load_ybox_cases(*, material_filter: set[str] | None = None, case_filter: set[str] | None = None) -> list[LegacyCase]:
    cases: list[LegacyCase] = []
    for material in YBOX_MATERIALS:
        if material_filter and material not in material_filter:
            continue
        input_dir = TEST_ROOT / material / "input"
        for path in sorted(input_dir.glob("*.dat")):
            case = parse_legacy_input(path)
            if case_filter and case.prefix not in case_filter:
                continue
            cases.append(case)
    return cases


def run_cmd(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_complex_matrix_npz(txt_path: Path, npz_path: Path, key: str) -> None:
    arr = np.loadtxt(txt_path, dtype=complex)
    np.savez(npz_path, **{key: arr})


def parse_numeric_kv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def extract_h_local_and_hopping(case: LegacyCase, case_dir: Path) -> tuple[Path, Path]:
    hr_rel = case.hr_relpath
    if hr_rel.startswith("data/"):
        hr_path = DATA_ROOT / hr_rel[len("data/") :]
    else:
        hr_path = DATA_ROOT / hr_rel

    config_path = case_dir / "w90_extract.toml"
    prefix = case_dir / "w90"
    lines = [
        f'hr_path = "{hr_path}"',
        f"n_orb_per_atom = {list(case.ndims)}",
        f"spinor = {str(case.spinor).lower()}",
        'output = "w90"',
        'energy_unit = "eV"',
        "",
        "[f_site_i]",
        f"atom = {case.atoms0[0]}",
        f"cell = {list(case.cells[0])}",
        "",
        "[f_site_j]",
        f"atom = {case.atoms0[1]}",
        f"cell = {list(case.cells[1])}",
        "",
        "[[ligands]]",
        f"atom = {case.atoms0[2]}",
        f"cell = {list(case.cells[2])}",
        "",
        "[[ligands]]",
        f"atom = {case.atoms0[3]}",
        f"cell = {list(case.cells[3])}",
        "",
    ]
    config_path.write_text("\n".join(lines))
    run_cmd([sys.executable, str(TOOLS_DIR / "w90_extract.py"), str(config_path)], cwd=case_dir)

    t_txt = prefix.with_name("w90_t_mu.dat")
    h_txt = prefix.with_name("w90_h_local.dat")
    return h_txt, t_txt


def build_projector_from_h_local(h_local_path: Path, *, n_ele: int, symmetry: str, mode_q3: str, out_dir: Path, kramer_name: str) -> tuple[Path, float]:
    cef_toml = out_dir / "cef_config.toml"
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "w90_decompose_h_local.py"),
        "--h_local_dat",
        str(h_local_path),
        "--rtol",
        "1e-2",
        "--n_ele",
        str(n_ele),
        "--symmetry",
        symmetry,
        "--mode_q3",
        mode_q3,
        "--direct_local_fit",
        "--write_cef_config",
        str(cef_toml),
    ]
    proc = run_cmd(cmd, cwd=out_dir)
    zeta_meV = None
    match = re.search(r"zeta\s*=\s*([0-9.+\-eE]+)\s*meV", proc.stdout)
    if match is not None:
        zeta_meV = float(match.group(1))
    if zeta_meV is None:
        raise RuntimeError(f"Failed to parse zeta from {proc.stdout}")

    projector_npz = out_dir / "projector.npz"
    run_cmd(
        [
            sys.executable,
            str(TOOLS_DIR / "cef_states.py"),
            str(cef_toml),
            "--format",
            "npz",
            "--output",
            str(projector_npz),
            "--kramer-name",
            kramer_name,
            "--silent",
        ],
        cwd=out_dir,
    )
    return projector_npz, zeta_meV


def write_run_input(path: Path, *, n_ele: int, F2: float, F4: float, F6: float, U_meV: float, Jh_meV: float, zeta_meV: float, hopping_file: Path, projector_file: Path) -> None:
    text = f"""schema_version = "fxe.run_input.v1"
standard_version = "2026-02"
run_id = "{path.parent.name}"
title = "{path.parent.name}"

[physics]
n_ele = {n_ele}
F2 = {F2:.12f}
F4 = {F4:.12f}
F6 = {F6:.12f}

[model]
scheme = "RS"

[sopt]
U = {U_meV:.12f}
Jh = {Jh_meV:.12f}
zeta = {zeta_meV:.12f}

[inputs]
hopping_file = "{hopping_file}"
projector_file = "{projector_file}"

[paths]
output_root = "./outputs"

[runtime]
start_level = "LMSM"
end_level = "L4"
on_missing_upstream = "fail"
read_first = true

[checks]
strict_mode = true
eps_profile = "default"
"""
    path.write_text(text)


def heff_to_J_mu(heff: np.ndarray) -> np.ndarray:
    if heff.shape == (2, 2, 2, 2):
        heff_flat = heff.reshape(4, 4)
    elif heff.shape == (4, 4):
        heff_flat = heff.copy()
    else:
        raise ValueError(f"Unexpected Heff shape: {heff.shape}")

    coeff = np.zeros((4, 4), dtype=complex)
    for a in range(4):
        for b in range(4):
            coeff[a, b] = 0.25 * np.trace(np.kron(SIGMAS[a], SIGMAS[b]) @ heff_flat)

    J_mu = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            J_mu[a, b] = float((4.0 * coeff[a + 1, b + 1]).real)
    return J_mu


def load_reference_row(ref_file: Path, ratio: float) -> np.ndarray:
    data = np.loadtxt(ref_file, comments="#")
    idx = np.where(np.isclose(data[:, 2], ratio, atol=1e-12))[0]
    if len(idx) != 1:
        raise ValueError(f"{ref_file}: could not find ratio={ratio}")
    return np.asarray(data[idx[0], 3:12], dtype=float).reshape(3, 3)


def gauge_transform(J_mu: np.ndarray, gauge_label: str) -> np.ndarray:
    R = GAUGE_ROTATIONS[gauge_label]
    return R.T @ J_mu @ R


def best_gauge(J_mu: np.ndarray, ref_J: np.ndarray) -> tuple[str, float]:
    best_label = "identity"
    best_err = float("inf")
    for label in GAUGE_ROTATIONS:
        err = float(np.max(np.abs(gauge_transform(J_mu, label) - ref_J)))
        if err < best_err:
            best_label = label
            best_err = err
    return best_label, best_err


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_particle_point(case_dir: Path, *, n_ele: int, slater_ratios: tuple[float, float, float], zeta_meV: float, hopping_file: Path, projector_file: Path, U_eV: float, ratio: float) -> np.ndarray:
    Jh_eV = U_eV * ratio
    U_meV = float(_fmt12(1000.0 * U_eV))
    Jh_meV = float(_fmt12(1000.0 * Jh_eV))
    if Jh_meV == 0.0:
        tiny = 1.0e-6
        F2 = tiny
        F4 = tiny * (slater_ratios[1] / slater_ratios[0])
        F6 = tiny * (slater_ratios[2] / slater_ratios[0])
    else:
        F2 = slater_ratios[0] * Jh_meV
        F4 = slater_ratios[1] * Jh_meV
        F6 = slater_ratios[2] * Jh_meV
    F2 = float(_fmt12(F2))
    F4 = float(_fmt12(F4))
    F6 = float(_fmt12(F6))
    zeta_meV = float(_fmt12(zeta_meV))

    run_input = case_dir / "run_input.toml"
    write_run_input(
        run_input,
        n_ele=n_ele,
        F2=F2,
        F4=F4,
        F6=F6,
        U_meV=U_meV,
        Jh_meV=Jh_meV,
        zeta_meV=zeta_meV,
        hopping_file=hopping_file,
        projector_file=projector_file,
    )
    with pushd(case_dir):
        run_cmd([sys.executable, "-m", "fexchange.cli", "run", "run_input.toml"], cwd=case_dir)

    r42 = F4 / F2
    r62 = F6 / F2
    l4_file = _build_l4_path(case_dir, n_ele=n_ele, r42=r42, r62=r62, U_meV=U_meV, Jh_meV=Jh_meV, zeta_meV=zeta_meV)
    heff = np.load(l4_file)["Heff_mu_abcd"]
    return heff_to_J_mu(heff)


def write_result_table(path: Path, rows: list[list[float]]) -> None:
    header = "#U JH ratio Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz error"
    with path.open("w") as fh:
        fh.write(header + "\n")
        for row in rows:
            fh.write(" ".join(f"{value:.6f}" for value in row) + "\n")


def case_family(prefix: str) -> str:
    if "_1st_" in prefix:
        return "1st"
    if "_2nd_" in prefix:
        return "2nd"
    return "all"


def material_family_projectors(cases: list[LegacyCase], material_root: Path) -> dict[str, dict[str, Any]]:
    families: dict[str, list[LegacyCase]] = {}
    for case in cases:
        families.setdefault(case_family(case.prefix), []).append(case)

    out: dict[str, dict[str, Any]] = {}
    for family, family_cases in sorted(families.items()):
        extracted_dir = material_root / "extracted" / family
        extracted_dir.mkdir(parents=True, exist_ok=True)
        h_local_paths: list[Path] = []
        h_local_mats: list[np.ndarray] = []

        for case in family_cases:
            case_extract_dir = extracted_dir / case.prefix
            ensure_clean_dir(case_extract_dir)
            h_local_path, _ = extract_h_local_and_hopping(case, case_extract_dir)
            h_local_paths.append(h_local_path)
            h_local_mats.append(np.loadtxt(h_local_path, dtype=complex))

        ref = h_local_mats[0]
        max_diff = max(float(np.max(np.abs(mat - ref))) for mat in h_local_mats)
        projector_dir = material_root / "projector" / family
        projector_dir.mkdir(parents=True, exist_ok=True)
        projector_file, zeta_meV = build_projector_from_h_local(
            h_local_paths[0],
            n_ele=13,
            symmetry="C3v",
            mode_q3="sin",
            out_dir=projector_dir,
            kramer_name=f"{family_cases[0].material}_{family}_particle",
        )
        out[family] = {
            "projector_file": str(projector_file),
            "zeta_meV": zeta_meV,
            "consistency": {"h_local_max_diff": max_diff},
        }
    return out


def material_projector_info(cases: list[LegacyCase], material_root: Path) -> tuple[Path, float, dict[str, float]]:
    info = material_family_projectors(cases, material_root)
    only = next(iter(info.values()))
    return Path(only["projector_file"]), float(only["zeta_meV"]), dict(only["consistency"])


def run_ybox(args: argparse.Namespace) -> dict[str, Any]:
    material_filter = set(args.material) if args.material else None
    case_filter = {args.case} if args.case else None
    cases = load_ybox_cases(material_filter=material_filter, case_filter=case_filter)
    grouped: dict[str, list[LegacyCase]] = {}
    for case in cases:
        grouped.setdefault(case.material, []).append(case)

    summary: dict[str, Any] = {"materials": {}}
    base_root = OUTPUT_ROOT / "ybox"
    base_root.mkdir(parents=True, exist_ok=True)

    for material, mat_cases in grouped.items():
        material_root = base_root / material
        material_root.mkdir(parents=True, exist_ok=True)
        family_projectors = material_family_projectors(mat_cases, material_root)
        family_gauges: dict[str, dict[str, Any]] = {}

        family_cases_map: dict[str, list[LegacyCase]] = {}
        for case in mat_cases:
            family_cases_map.setdefault(case_family(case.prefix), []).append(case)

        for family, family_cases in sorted(family_cases_map.items()):
            projector_meta = family_projectors[family]
            projector_file = Path(projector_meta["projector_file"])
            zeta_meV = float(projector_meta["zeta_meV"])
            rep_case = next(
                case
                for case in sorted(family_cases, key=lambda item: item.prefix)
                if case_filter is None or case.prefix in case_filter
            )
            rep_case_dir = material_root / rep_case.prefix
            ensure_clean_dir(rep_case_dir)
            _, rep_hopping = extract_h_local_and_hopping(rep_case, rep_case_dir)
            rep_J = run_particle_point(
                rep_case_dir,
                n_ele=13,
                slater_ratios=rep_case.slater_ratios,
                zeta_meV=zeta_meV,
                hopping_file=rep_hopping,
                projector_file=projector_file,
                U_eV=args.U if args.U is not None else 6.0,
                ratio=args.ratio if args.ratio is not None else 0.10,
            )
            rep_ref = load_reference_row(
                rep_case.reference_dir / f"result_HOLE_U{(args.U if args.U is not None else 6.0):.2f}.dat",
                args.ratio if args.ratio is not None else 0.10,
            )
            gauge_label, gauge_err = best_gauge(rep_J, rep_ref)
            family_gauges[family] = {
                "gauge_label": gauge_label,
                "gauge_fit_max_abs_err_meV": gauge_err,
            }

        mat_summary: dict[str, Any] = {
            "families": {},
            "cases": {},
        }
        for family, meta in sorted(family_projectors.items()):
            mat_summary["families"][family] = {
                "projector_file": meta["projector_file"],
                "zeta_meV": meta["zeta_meV"],
                "consistency": meta["consistency"],
                **family_gauges[family],
            }

        for case in sorted(mat_cases, key=lambda item: item.prefix):
            family = case_family(case.prefix)
            projector_file = Path(family_projectors[family]["projector_file"])
            zeta_meV = float(family_projectors[family]["zeta_meV"])
            gauge_label = str(family_gauges[family]["gauge_label"])
            case_dir = material_root / case.prefix
            ensure_clean_dir(case_dir)
            _, hopping_file = extract_h_local_and_hopping(case, case_dir)

            U_values = (args.U,) if args.U is not None else case.U_list_eV
            ratio_values = (args.ratio,) if args.ratio is not None else case.ratio_list
            case_summary: dict[str, Any] = {"family": family, "U": {}}

            for U_eV in U_values:
                raw_rows: list[list[float]] = []
                aligned_rows: list[list[float]] = []
                max_raw = 0.0
                max_aligned = 0.0
                ref_file = case.reference_dir / f"result_HOLE_U{U_eV:.2f}.dat"
                for ratio in ratio_values:
                    J_mu = run_particle_point(
                        case_dir,
                        n_ele=13,
                        slater_ratios=case.slater_ratios,
                        zeta_meV=zeta_meV,
                        hopping_file=hopping_file,
                        projector_file=projector_file,
                        U_eV=U_eV,
                        ratio=ratio,
                    )
                    ref_J = load_reference_row(ref_file, ratio)
                    raw_err = float(np.max(np.abs(J_mu - ref_J)))
                    aligned = gauge_transform(J_mu, gauge_label)
                    aligned_err = float(np.max(np.abs(aligned - ref_J)))
                    max_raw = max(max_raw, raw_err)
                    max_aligned = max(max_aligned, aligned_err)

                    Jh_eV = U_eV * ratio
                    raw_rows.append([U_eV, Jh_eV, ratio, *J_mu.reshape(-1), 0.0])
                    aligned_rows.append([U_eV, Jh_eV, ratio, *aligned.reshape(-1), 0.0])

                write_result_table(case_dir / f"result_PARTICLE_RAW_U{U_eV:.2f}.dat", raw_rows)
                write_result_table(case_dir / f"result_PARTICLE_GAUGEALIGNED_U{U_eV:.2f}.dat", aligned_rows)
                case_summary["U"][f"{U_eV:.2f}"] = {
                    "max_raw_abs_err_meV": max_raw,
                    "max_aligned_abs_err_meV": max_aligned,
                    "pass": bool(max_aligned <= YBOX_TOL_MEW),
                }
            mat_summary["cases"][case.prefix] = case_summary

        summary["materials"][material] = mat_summary

    summary_path = base_root / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _deprecated_material_projector_info_impl(cases: list[LegacyCase], material_root: Path) -> tuple[Path, float, dict[str, float]]:
    extracted_dir = material_root / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    h_local_paths: list[Path] = []
    h_local_mats: list[np.ndarray] = []

    for case in cases:
        case_extract_dir = extracted_dir / case.prefix
        ensure_clean_dir(case_extract_dir)
        h_local_path, _ = extract_h_local_and_hopping(case, case_extract_dir)
        h_local_paths.append(h_local_path)
        h_local_mats.append(np.loadtxt(h_local_path, dtype=complex))

    ref = h_local_mats[0]
    max_diff = max(float(np.max(np.abs(mat - ref))) for mat in h_local_mats)
    projector_dir = material_root / "projector"
    projector_dir.mkdir(parents=True, exist_ok=True)
    projector_file, zeta_meV = build_projector_from_h_local(
        h_local_paths[0],
        n_ele=13,
        symmetry="C3v",
        mode_q3="sin",
        out_dir=projector_dir,
        kramer_name=f"{cases[0].material}_particle",
    )
    return projector_file, zeta_meV, {"h_local_max_diff": max_diff}


def build_prb_projector(site_dir: Path) -> Path:
    projector = site_dir / "projector.npz"
    run_cmd(
        [
            sys.executable,
            str(TOOLS_DIR / "cef_states.py"),
            "--point-group",
            "Oh",
            "--J",
            "2.5",
            "--B4",
            "0.01",
            "--B6",
            "0.0",
            "--format",
            "npz",
            "--output",
            str(projector),
            "--kramer-name",
            "prb_gamma7",
            "--silent",
        ],
        cwd=site_dir,
    )
    return projector


def build_prb_projector_literature(site_dir: Path) -> Path:
    projector = site_dir / "projector.npz"
    w = np.array(
        [
            [0.0, -1.0 / np.sqrt(6.0)],
            [np.sqrt(5.0 / 6.0), 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, np.sqrt(5.0 / 6.0)],
            [-1.0 / np.sqrt(6.0), 0.0],
        ],
        dtype=complex,
    )
    np.savez(
        projector,
        W=w,
        kramer_name="prb_gamma7_literature",
        kramer_labels=np.array([0, 1]),
        ground_irrep="Gamma7",
        ground_energy=0.0,
        doublet_type="kramers",
        n_j=6,
        schema_version="fxe.cef_states.v1",
        standard_version="2026-02",
        basis_id="complex_spherical_j_v1",
        orbital_order_id="f7_m-3..3_v1",
        unit="meV",
        J=2.5,
        point_group="Oh",
    )
    return projector


def build_prb_hopping(site_dir: Path, a_site: str) -> Path:
    base = site_dir / "prb_t_mu.dat"
    hopping = site_dir / f"prb_t_mu_{a_site}.dat"
    run_cmd(
        [
            sys.executable,
            str(TOOLS_DIR / "slater_koster_pf.py"),
            "literature",
            "--output",
            str(base),
        ],
        cwd=site_dir,
    )
    return hopping


def extract_prb_channels(J_mu: np.ndarray) -> dict[str, float]:
    Jxx, Jxy, Jxz = J_mu[0]
    Jyx, Jyy, Jyz = J_mu[1]
    Jzx, Jzy, Jzz = J_mu[2]
    J = 0.5 * (Jxx + Jyy)
    K = Jzz - J
    Gamma = 0.5 * (Jxy + Jyx)
    Gamma_p = 0.25 * (Jxz + Jzx + Jyz + Jzy)
    leakage = max(
        abs(Jxx - Jyy),
        abs(Jxy - Jyx),
        abs(Jxz - Jyz),
        abs(Jzx - Jzy),
    )
    return {
        "J": float(J),
        "K": float(K),
        "Gamma": float(Gamma),
        "Gamma_p": float(Gamma_p),
        "symmetry_leakage": float(leakage),
    }


def canonicalize_prb_exchange(J_mu: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    J_sym = 0.5 * (np.asarray(J_mu, dtype=float) + np.asarray(J_mu, dtype=float).T)
    evals, evecs = np.linalg.eigh(J_sym)

    # For the cubic z-bond convention, the antisymmetric in-plane axis should
    # stay as close as possible to the current xy plane.
    axis_idx = min(range(3), key=lambda idx: abs(float(evecs[2, idx])))
    u = evecs[:, axis_idx].astype(float)
    if float(u[0] - u[1]) < 0.0:
        u = -u

    z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    z = z0 - u * float(np.dot(u, z0))
    if np.linalg.norm(z) < 1.0e-12:
        x0 = np.array([1.0, 0.0, 0.0], dtype=float)
        z = x0 - u * float(np.dot(u, x0))
    z = z / np.linalg.norm(z)
    v = np.cross(z, u)
    v = v / np.linalg.norm(v)

    x = (u + v) / np.sqrt(2.0)
    y = (v - u) / np.sqrt(2.0)
    R = np.column_stack([x, y, z])
    if np.linalg.det(R) < 0.0:
        z = -z
        v = np.cross(z, u)
        v = v / np.linalg.norm(v)
        x = (u + v) / np.sqrt(2.0)
        y = (v - u) / np.sqrt(2.0)
        R = np.column_stack([x, y, z])

    return R.T @ J_sym @ R, {
        "axis_rule": "min_abs_z_eigenvector",
        "axis_index": int(axis_idx),
        "eigenvalue": float(evals[axis_idx]),
        "rotation": R.tolist(),
    }


def prb_plot_styles() -> dict[str, dict[str, Any]]:
    return {
        "colors": {
            "J": "#4e8b3d",
            "K": "#e04b3f",
            "Gamma": "#f08a24",
            "Gamma_p": "#5566cc",
        },
        "labels": {
            "J": "Heisenberg J",
            "K": "Kitaev K",
            "Gamma": r"$\Gamma$",
            "Gamma_p": r"$\Gamma'$",
        },
    }


def plot_prb(summary: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 10.5), sharex=True)
    styles = prb_plot_styles()
    colors = styles["colors"]
    linestyles = {
        2.0: "-",
        3.0: (0, (4, 2)),
        4.0: (0, (2, 2)),
        6.0: (0, (1, 2)),
    }
    titles = {"K": "K$_2$PrO$_3$", "Rb": "Rb$_2$PrO$_3$", "Cs": "Cs$_2$PrO$_3$"}
    labels = styles["labels"]
    legend_text = {
        2.0: "U = 2 eV",
        3.0: "(x1.5) U = 3 eV",
        4.0: "(x2) U = 4 eV",
        6.0: "(x3) U = 6 eV",
    }

    for ax, site in zip(axes, PRB_SITES, strict=True):
        site_data = summary["sites"][site]["curves"]
        for U_eV in PRB_U_LIST:
            rows = site_data[f"{U_eV:.2f}"]
            ratios = [row["ratio"] for row in rows]
            scale = PRB_SCALE[U_eV]
            for channel in ("J", "K", "Gamma", "Gamma_p"):
                values = [scale * row[channel] for row in rows]
                ax.plot(ratios, values, color=colors[channel], linestyle=linestyles[U_eV], linewidth=1.5)
        ax.set_title(titles[site], loc="left", fontsize=12)
        ax.set_ylabel("COUPLING CONST. (meV)")
        ax.set_xlim(0.0, 0.20)
        ax.set_ylim(-0.5, 3.0)
        ax.grid(alpha=0.15, linewidth=0.6)
        x_txt = 0.02
        ax.text(x_txt, 1.05, labels["J"], color=colors["J"], fontsize=11)
        ax.text(0.12 if site != "Cs" else 0.02, 1.85 if site == "K" else (0.62 if site == "Rb" else -0.25), labels["K"], color=colors["K"], fontsize=11)
        ax.text(0.175, 0.38 if site == "K" else (0.22 if site == "Rb" else 0.18), labels["Gamma"], color=colors["Gamma"], fontsize=11)
        ax.text(0.175, 0.02 if site == "K" else (-0.18 if site == "Rb" else 0.02), labels["Gamma_p"], color=colors["Gamma_p"], fontsize=11)
        y0 = 2.55
        for idx, U_eV in enumerate(PRB_U_LIST):
            ax.plot([0.02, 0.06], [y0 - 0.18 * idx, y0 - 0.18 * idx], color="black", linestyle=linestyles[U_eV], linewidth=1.5)
            ax.text(0.065, y0 - 0.18 * idx, legend_text[U_eV], va="center", fontsize=9)

    axes[-1].set_xlabel(r"$J_H/U$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_prb(args: argparse.Namespace) -> dict[str, Any]:
    site_filter = set(args.material) if args.material else None
    base_root = OUTPUT_ROOT / "prb"
    base_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"sites": {}}

    for site in PRB_SITES:
        if site_filter and site not in site_filter:
            continue
        site_dir = base_root / site
        ensure_clean_dir(site_dir)
        if args.prb_projector == "literature":
            projector = build_prb_projector_literature(site_dir)
        else:
            projector = build_prb_projector(site_dir)
        hopping = build_prb_hopping(site_dir, site)
        site_summary: dict[str, Any] = {
            "projector_file": str(projector),
            "hopping_file": str(hopping),
            "zeta_meV": PRB_ZETA_MEV[site],
            "canonical_gauge": None,
            "projector_mode": args.prb_projector,
            "use_canonical_gauge": bool(args.prb_canonical_gauge),
            "curves": {},
        }
        U_values = (args.U,) if args.U is not None else PRB_U_LIST
        ratio_values = (args.ratio,) if args.ratio is not None else PRB_RATIO_LIST
        for U_eV in U_values:
            rows: list[dict[str, float]] = []
            for ratio in ratio_values:
                J_mu = run_particle_point(
                    site_dir,
                    n_ele=1,
                    slater_ratios=PRB_SLATER_RATIOS,
                    zeta_meV=PRB_ZETA_MEV[site],
                    hopping_file=hopping,
                    projector_file=projector,
                    U_eV=U_eV,
                    ratio=ratio,
                )
                if args.prb_canonical_gauge:
                    J_eff, gauge_meta = canonicalize_prb_exchange(J_mu)
                    if site_summary["canonical_gauge"] is None:
                        site_summary["canonical_gauge"] = gauge_meta
                else:
                    J_eff = J_mu
                channels = extract_prb_channels(J_eff)
                rows.append({"ratio": ratio, **channels})
            site_summary["curves"][f"{U_eV:.2f}"] = rows
            rows_txt = [
                [
                    row["ratio"],
                    row["J"],
                    row["K"],
                    row["Gamma"],
                    row["Gamma_p"],
                    row["symmetry_leakage"],
                ]
                for row in rows
            ]
            header = "#ratio J K Gamma GammaPrime symmetry_leakage"
            out_file = site_dir / f"prb_curves_U{U_eV:.2f}.dat"
            with out_file.open("w") as fh:
                fh.write(header + "\n")
                for vals in rows_txt:
                    fh.write(" ".join(f"{val:.6f}" for val in vals) + "\n")
        summary["sites"][site] = site_summary

    summary_path = base_root / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.ratio is None:
        plot_prb(summary, base_root / "prb_particle_plot.png")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLI-driven YbOX/PRB particle validation runner")
    p.add_argument("--task", choices=("ybox", "prb", "all"), default="ybox")
    p.add_argument("--material", action="append", help="Restrict YbOX materials")
    p.add_argument("--case", help="Restrict to one legacy PREFIX")
    p.add_argument("--U", type=float, help="Restrict to one U in eV")
    p.add_argument("--ratio", type=float, help="Restrict to one Jh/U ratio")
    p.add_argument("--prb-projector", choices=("cef", "literature"), default="cef")
    p.add_argument(
        "--prb-canonical-gauge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the PRB continuous canonical pseudospin rotation before channel extraction.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.task in {"ybox", "all"}:
        summary = run_ybox(args)
        print(json.dumps(summary, indent=2))
    if args.task in {"prb", "all"}:
        summary = run_prb(args)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
