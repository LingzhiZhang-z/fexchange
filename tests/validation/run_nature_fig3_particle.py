from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fexchange.tools.slater_koster_pf import (  # noqa: E402
    _expand_to_spinor_interleaved,
    fpf_cubic_7x7,
    to_complex_basis_spinor,
)
from fexchange.utils.constants import RE_DEFAULTS_BY_N_ELE, RE_TO_N_ELE, RUN_SCHEMA_VERSION, STANDARD_VERSION  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "test" / "nature_fig3_particle"
HOPPING_ROOT = DATA_ROOT / "hopping"
COMPLEX_HOPPING_ROOT = DATA_ROOT / "complex_hopping"
REFERENCE_ROOT = DATA_ROOT / "reference"
STATIC_ROOT = DATA_ROOT / "static"
FIG3_ASSET_ROOT = DATA_ROOT / "fig3"
TABLE_ASSET_ROOT = DATA_ROOT / "table_estimates"
TABLE_STRICT_ASSET_ROOT = DATA_ROOT / "table_strict"
TABLE2_ASSET_ROOT = DATA_ROOT / "table2"
TABLE2_LITERATURE_FILE = REFERENCE_ROOT / "table2_literature.tsv"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "nature_fig3_particle"
FIG3_OUTPUT_ROOT = OUTPUT_ROOT / "fig3"
TABLE_OUTPUT_ROOT = OUTPUT_ROOT / "table_estimates"
TABLE_STRICT_OUTPUT_ROOT = OUTPUT_ROOT / "table_strict"
TABLE2_OUTPUT_ROOT = OUTPUT_ROOT / "table2"
WORK_ROOT = OUTPUT_ROOT / "workdirs"
FIG3_WORK_ROOT = WORK_ROOT / "fig3"
TABLE_WORK_ROOT = WORK_ROOT / "table_estimates"
TABLE_STRICT_WORK_ROOT = WORK_ROOT / "table_strict"
TABLE2_WORK_ROOT = WORK_ROOT / "table2"
DEFAULT_STATIC_RATIO = 0.7
RATIO_VALUES = tuple(float(v) for v in np.linspace(0.1, 1.0, 10))
TABLE_RATIO = 0.7
TABLE_TPFSIGMA_EV = 0.35
TABLE_DELTA_PF_EV = 1.0
STRICT_TABLE_RUNTIME_HOPPING_SCALE = math.sqrt(1000.0)
POINT_RESULT_RE = re.compile(r"point_result=(.+\.txt)")
NATURE_SHELL_RUNNER = REPO_ROOT / "tests" / "validation" / "run_nature_fig3_table2.sh"
NATURE_GNUPLOT_SCRIPT = REPO_ROOT / "tests" / "validation" / "plot_nature_fig3_abef.gp"


@dataclass(frozen=True)
class BranchSpec:
    re_name: str | None
    Jh_eV: float | None = None
    zeta_eV: float | None = None
    offset_eV: float = 0.0
    U_eV: float = 0.0


@dataclass(frozen=True)
class Fig3Case:
    key: str
    panel: str
    title: str
    n_ele: int
    re_name: str
    irrep: str
    j2: int
    main_Jh_eV: float
    nm1: BranchSpec
    np1: BranchSpec
    projector_terms: tuple[tuple[tuple[int, complex], ...], tuple[tuple[int, complex], ...]]
    reference_file: str | None
    asset_only: bool = False


_CASES = (
    Fig3Case(
        key="f1_ce_g7",
        panel="a",
        title="4f^1 Gamma7",
        n_ele=1,
        re_name="Ce",
        irrep="Gamma7",
        j2=5,
        main_Jh_eV=0.1,
        nm1=BranchSpec(re_name=None, Jh_eV=0.0, zeta_eV=0.0, offset_eV=1.8),
        np1=BranchSpec(re_name="Pr", Jh_eV=0.6, offset_eV=4.59871471159652),
        projector_terms=(
            ((-3, -0.912870929175277j), (5, 0.408248290463863j)),
            ((3, -0.912870929175277j), (-5, 0.408248290463863j)),
        ),
        reference_file="f1_ce_g7.tsv",
    ),
    Fig3Case(
        key="f3_nd_g6",
        panel="b",
        title="4f^3 Gamma6",
        n_ele=3,
        re_name="Nd",
        irrep="Gamma6",
        j2=9,
        main_Jh_eV=0.7,
        nm1=BranchSpec(re_name="Pr", Jh_eV=0.6, offset_eV=6.59871471159652),
        np1=BranchSpec(re_name="Pm", Jh_eV=0.8, offset_eV=8.22792700703035),
        projector_terms=(
            ((-7, 0.204124145231931j), (1, 0.763762615825973j), (9, 0.612372435695794j)),
            ((7, 0.204124145231931j), (-1, 0.763762615825973j), (-9, 0.612372435695794j)),
        ),
        reference_file="f3_nd_g6.tsv",
    ),
    Fig3Case(
        key="f5_sm_g7",
        panel="c",
        title="4f^5 Gamma7",
        n_ele=5,
        re_name="Sm",
        irrep="Gamma7",
        j2=5,
        main_Jh_eV=0.7,
        nm1=BranchSpec(re_name="Pm", Jh_eV=0.8, offset_eV=12.12792700703040),
        np1=BranchSpec(re_name="Eu", Jh_eV=0.9, offset_eV=14.19903680549650),
        projector_terms=(
            ((-3, -0.912870929175277j), (5, 0.408248290463863j)),
            ((3, -0.912870929175277j), (-5, 0.408248290463863j)),
        ),
        reference_file=None,
        asset_only=True,
    ),
    Fig3Case(
        key="f9_dy_g6",
        panel="d",
        title="4f^9 Gamma6",
        n_ele=9,
        re_name="Dy",
        irrep="Gamma6",
        j2=15,
        main_Jh_eV=1.1,
        nm1=BranchSpec(re_name="Tb", Jh_eV=1.0, offset_eV=26.27596786450340),
        np1=BranchSpec(re_name="Ho", Jh_eV=0.8, offset_eV=23.01081246392870),
        projector_terms=(
            ((-15, 0.581843335157039j), (-7, 0.330718913883074j), (1, 0.718070330817254j), (9, 0.190940653956493j)),
            ((15, -0.581843335157039j), (7, -0.330718913883074j), (-1, -0.718070330817254j), (-9, -0.190940653956493j)),
        ),
        reference_file=None,
        asset_only=True,
    ),
    Fig3Case(
        key="f9_dy_g7",
        panel="d2",
        title="4f^9 Gamma7",
        n_ele=9,
        re_name="Dy",
        irrep="Gamma7",
        j2=15,
        main_Jh_eV=1.1,
        nm1=BranchSpec(re_name="Tb", Jh_eV=1.0, offset_eV=26.27596786450340),
        np1=BranchSpec(re_name="Ho", Jh_eV=0.8, offset_eV=23.01081246392870),
        projector_terms=(
            ((-11, 0.239356776939085j), (-3, 0.450693909432999j), (5, -0.581843335157039j), (13, -0.633278506398778j)),
            ((11, -0.239356776939085j), (3, -0.450693909432999j), (-5, 0.581843335157039j), (-13, 0.633278506398778j)),
        ),
        reference_file=None,
        asset_only=True,
    ),
    Fig3Case(
        key="f11_er_g7",
        panel="e",
        title="4f^11 Gamma7",
        n_ele=11,
        re_name="Er",
        irrep="Gamma7",
        j2=15,
        main_Jh_eV=0.9,
        nm1=BranchSpec(re_name="Ho", Jh_eV=0.8, offset_eV=27.41081246392870),
        np1=BranchSpec(re_name="Tm", Jh_eV=0.5, offset_eV=18.60329259035310),
        projector_terms=(
            ((-11, 0.239356776939085j), (-3, 0.450693909432999j), (5, -0.581843335157039j), (13, -0.633278506398778j)),
            ((11, -0.239356776939085j), (3, -0.450693909432999j), (-5, 0.581843335157039j), (-13, 0.633278506398778j)),
        ),
        reference_file="f11_er_g7.tsv",
    ),
    Fig3Case(
        key="f13_yb_g6",
        panel="f",
        title="4f^13 Gamma6",
        n_ele=13,
        re_name="Yb",
        irrep="Gamma6",
        j2=7,
        main_Jh_eV=0.1,
        nm1=BranchSpec(re_name="Tm", Jh_eV=0.5, offset_eV=24.20329259035310),
        np1=BranchSpec(re_name=None, Jh_eV=0.0, zeta_eV=0.0, offset_eV=6.4),
        projector_terms=(
            ((-7, -0.645497224367903j), (1, -0.763762615825973j)),
            ((7, 0.645497224367903j), (-1, 0.763762615825973j)),
        ),
        reference_file="f13_yb_g6.tsv",
    ),
)
_CASE_BY_KEY = {case.key: case for case in _CASES}


def all_cases() -> tuple[Fig3Case, ...]:
    return _CASES


def active_cases() -> tuple[Fig3Case, ...]:
    return tuple(case for case in _CASES if not case.asset_only)


def asset_only_cases() -> tuple[Fig3Case, ...]:
    return tuple(case for case in _CASES if case.asset_only)


def default_run_cases() -> tuple[Fig3Case, ...]:
    return active_cases()


def case_by_key(key: str) -> Fig3Case:
    return _CASE_BY_KEY[key]


def table_estimate_cases() -> tuple[Fig3Case, ...]:
    return (
        case_by_key("f1_ce_g7"),
        case_by_key("f3_nd_g6"),
        case_by_key("f5_sm_g7"),
        case_by_key("f9_dy_g6"),
        case_by_key("f9_dy_g7"),
        case_by_key("f11_er_g7"),
        case_by_key("f13_yb_g6"),
    )


def table_strict_cases() -> tuple[Fig3Case, ...]:
    return active_cases()


def table2_cases() -> tuple[Fig3Case, ...]:
    return (
        case_by_key("f1_ce_g7"),
        case_by_key("f3_nd_g6"),
        case_by_key("f5_sm_g7"),
        case_by_key("f9_dy_g6"),
        case_by_key("f9_dy_g7"),
        case_by_key("f11_er_g7"),
        case_by_key("f13_yb_g6"),
    )


def table2_default_run_cases() -> tuple[Fig3Case, ...]:
    return active_cases()


def build_projector(case: Fig3Case) -> np.ndarray:
    W = np.zeros((case.j2 + 1, 2), dtype=np.complex128)
    for col, terms in enumerate(case.projector_terms):
        for m2, coeff in terms:
            idx = _row_index(case.j2, m2)
            W[idx, col] = coeff
    gram = W.conj().T @ W
    if not np.allclose(gram, np.eye(2), atol=1.0e-12):
        raise ValueError(f"Projector for {case.key} is not orthonormal")
    return W


def canonicalize_exchange(J_mu: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    J = np.asarray(J_mu, dtype=float)
    Jsym = 0.5 * (J + J.T)
    evals, evecs = np.linalg.eigh(Jsym)
    axis_idx = int(np.argmin(np.abs(evecs[2, :])))
    u = np.asarray(evecs[:, axis_idx], dtype=float)
    if (u[0] - u[1]) < 0.0:
        u = -u

    global_z = np.array([0.0, 0.0, 1.0], dtype=float)
    z = global_z - u * float(np.dot(u, global_z))
    if np.linalg.norm(z) < 1.0e-14:
        global_x = np.array([1.0, 0.0, 0.0], dtype=float)
        z = global_x - u * float(np.dot(u, global_x))
    z = _normalize(z)
    v = _normalize(np.cross(z, u))

    sq2 = math.sqrt(2.0)
    x = (u + v) / sq2
    y = (v - u) / sq2
    R = np.column_stack((x, y, z))

    if np.linalg.det(R) < 0.0:
        z = -z
        v = _normalize(np.cross(z, u))
        x = (u + v) / sq2
        y = (v - u) / sq2
        R = np.column_stack((x, y, z))

    Jcan = R.T @ Jsym @ R
    Jcan[np.abs(Jcan) < 1.0e-14] = 0.0
    meta = {
        "axis_rule": "min_abs_z_eigenvector",
        "axis_index": axis_idx,
        "eigenvalues": [float(v) for v in evals],
        "rotation": R.tolist(),
    }
    return Jcan, meta


def extract_channels(J_mu: np.ndarray) -> dict[str, float]:
    J = 0.5 * (np.asarray(J_mu, dtype=float) + np.asarray(J_mu, dtype=float).T)
    Jiso = 0.5 * (float(J[0, 0]) + float(J[1, 1]))
    leakage = max(
        abs(float(J[0, 0] - J[1, 1])),
        abs(float(J[0, 1] - J[1, 0])),
        abs(float(J[0, 2] - J[1, 2])),
        abs(float(J[2, 0] - J[2, 1])),
    )
    return {
        "J": Jiso,
        "K": float(J[2, 2]) - Jiso,
        "Gamma": 0.5 * float(J[0, 1] + J[1, 0]),
        "GammaPrime": 0.25 * float(J[0, 2] + J[2, 0] + J[1, 2] + J[2, 1]),
        "symmetry_leakage": leakage,
    }


def load_reference_curve(case: Fig3Case) -> np.ndarray:
    if not case.reference_file:
        raise ValueError(f"{case.key} has no vendored reference curve")
    path = REFERENCE_ROOT / case.reference_file
    data = np.loadtxt(path, comments="#", dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] != 5:
        raise ValueError(f"Reference curve must have 5 columns, got {data.shape} for {path}")
    return data


def static_asset_paths(case: Fig3Case) -> dict[str, Path]:
    static_dir = STATIC_ROOT / case.key
    projector = static_dir / "projector.npz"
    run_input = static_dir / "run_input.toml"
    write_projector(case, projector)
    write_run_input(
        case,
        ratio=DEFAULT_STATIC_RATIO,
        hopping_file=build_complex_hopping(DEFAULT_STATIC_RATIO).resolve(),
        projector_file=projector.resolve(),
        output_path=run_input,
    )
    return {"root": static_dir, "projector": projector, "run_input": run_input}


def prepare_static_assets(cases: Sequence[Fig3Case] | None = None) -> list[dict[str, Path]]:
    selected = tuple(cases) if cases is not None else all_cases()
    return [static_asset_paths(case) for case in selected]


def fig3_asset_paths(case: Fig3Case, *, ratios: Sequence[float] = RATIO_VALUES) -> dict[str, Any]:
    asset_root = FIG3_ASSET_ROOT / case.key
    projector = asset_root / "projector.npz"
    write_projector(case, projector)
    run_inputs: dict[float, Path] = {}
    for ratio in ratios:
        run_input = asset_root / f"run_input_ratio_{ratio:.1f}.toml"
        write_run_input(
            case,
            ratio=ratio,
            hopping_file=build_complex_hopping(ratio).resolve(),
            projector_file=projector.resolve(),
            output_path=run_input,
        )
        run_inputs[float(ratio)] = run_input
    return {"root": asset_root, "projector": projector, "run_inputs": run_inputs}


def prepare_fig3_assets(
    cases: Sequence[Fig3Case] | None = None,
    *,
    ratios: Sequence[float] = RATIO_VALUES,
) -> list[dict[str, Any]]:
    selected = tuple(cases) if cases is not None else all_cases()
    return [fig3_asset_paths(case, ratios=ratios) for case in selected]


def table_asset_paths(case: Fig3Case) -> dict[str, Path]:
    asset_root = TABLE_ASSET_ROOT / case.key
    projector = asset_root / "projector.npz"
    run_input = asset_root / "run_input.toml"
    write_projector(case, projector)
    write_run_input(
        case,
        ratio=TABLE_RATIO,
        hopping_file=build_complex_hopping(TABLE_RATIO).resolve(),
        projector_file=projector.resolve(),
        output_path=run_input,
    )
    return {"root": asset_root, "projector": projector, "run_input": run_input}


def prepare_table_assets(cases: Sequence[Fig3Case] | None = None) -> list[dict[str, Path]]:
    selected = tuple(cases) if cases is not None else table_estimate_cases()
    return [table_asset_paths(case) for case in selected]


def table_strict_asset_paths(
    case: Fig3Case,
    *,
    ratio: float = TABLE_RATIO,
    pf_sigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> dict[str, Path]:
    asset_root = TABLE_STRICT_ASSET_ROOT / case.key
    projector = asset_root / "projector.npz"
    run_input = asset_root / "run_input.toml"
    hopping = build_complex_hopping(
        ratio,
        pf_sigma_eV=pf_sigma_eV,
        delta_pf_eV=delta_pf_eV,
        output_scale=STRICT_TABLE_RUNTIME_HOPPING_SCALE,
    )
    write_projector(case, projector)
    write_run_input(
        case,
        ratio=ratio,
        hopping_file=hopping.resolve(),
        projector_file=projector.resolve(),
        output_path=run_input,
        hopping_label_suffix=f"pf_sigma_{pf_sigma_eV:.3f}_delta_{delta_pf_eV:.3f}",
    )
    return {"root": asset_root, "projector": projector, "run_input": run_input, "hopping": hopping}


def prepare_table_strict_assets(
    cases: Sequence[Fig3Case] | None = None,
    *,
    ratio: float = TABLE_RATIO,
    pf_sigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> list[dict[str, Path]]:
    selected = tuple(cases) if cases is not None else table_strict_cases()
    return [
        table_strict_asset_paths(case, ratio=ratio, pf_sigma_eV=pf_sigma_eV, delta_pf_eV=delta_pf_eV)
        for case in selected
    ]


def table2_asset_paths(
    case: Fig3Case,
    *,
    ratio: float = TABLE_RATIO,
    pf_sigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> dict[str, Path]:
    asset_root = TABLE2_ASSET_ROOT / case.key
    projector = asset_root / "projector.npz"
    run_input = asset_root / "run_input.toml"
    hopping = build_complex_hopping(
        ratio,
        pf_sigma_eV=pf_sigma_eV,
        delta_pf_eV=delta_pf_eV,
        output_scale=STRICT_TABLE_RUNTIME_HOPPING_SCALE,
    )
    write_projector(case, projector)
    write_run_input(
        case,
        ratio=ratio,
        hopping_file=hopping.resolve(),
        projector_file=projector.resolve(),
        output_path=run_input,
        hopping_label_suffix=f"pf_sigma_{pf_sigma_eV:.3f}_delta_{delta_pf_eV:.3f}",
    )
    return {"root": asset_root, "projector": projector, "run_input": run_input, "hopping": hopping}


def prepare_table2_assets(
    cases: Sequence[Fig3Case] | None = None,
    *,
    ratio: float = TABLE_RATIO,
    pf_sigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> list[dict[str, Path]]:
    selected = tuple(cases) if cases is not None else table2_cases()
    return [
        table2_asset_paths(case, ratio=ratio, pf_sigma_eV=pf_sigma_eV, delta_pf_eV=delta_pf_eV)
        for case in selected
    ]


def write_projector(case: Fig3Case, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    W = build_projector(case)
    np.savez(
        output_path,
        W=W,
        kramer_name=np.array(f"{case.key}_literature"),
        ground_irrep=np.array(case.irrep),
        title=np.array(case.title),
        n_ele=np.array(case.n_ele),
        panel=np.array(case.panel),
        m2_values=np.arange(-case.j2, case.j2 + 1, 2, dtype=int),
    )
    return output_path


def write_run_input(
    case: Fig3Case,
    *,
    ratio: float,
    hopping_file: str | Path,
    projector_file: str | Path,
    output_path: Path,
    hopping_label_suffix: str = "complex_raw",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hopping_label = f"{case.key}_pfpi_{ratio:.1f}_{hopping_label_suffix}"
    projection_label = f"{case.key}_{case.irrep.lower()}_literature"
    lines = [
        f'schema_version = "{RUN_SCHEMA_VERSION}"',
        f'standard_version = "{STANDARD_VERSION}"',
        f'run_id = "{case.key}_r{ratio:.1f}"',
        f'title = "{case.title}"',
        "",
        "[units]",
        'energy = "eV"',
        "",
        "[physics]",
        f"n_ele = {case.n_ele}",
        f'RE = "{case.re_name}"',
        "",
    ]
    lines.extend(_physics_branch_lines("physics_nm1", case.nm1))
    lines.extend(_physics_branch_lines("physics_np1", case.np1))
    lines.extend(
        [
            "[model]",
            'scheme = "RS"',
            "",
            "[sopt]",
            f"U = {_fmt_float(0.0)}",
            f"Jh = {_fmt_float(case.main_Jh_eV)}",
            f"zeta = {_fmt_float(_zeta_eV(case.re_name))}",
            f"offset = {_fmt_float(0.0)}",
            'energy_reference = "zero"',
            "",
        ]
    )
    lines.extend(_sopt_branch_lines("sopt_nm1", case.nm1))
    lines.extend(_sopt_branch_lines("sopt_np1", case.np1))
    lines.extend(
        [
            "[sources]",
            f'hopping_label = "{hopping_label}"',
            f'projection_label = "{projection_label}"',
            "",
            "[inputs]",
            f'hopping_file = "{_path_text(hopping_file)}"',
            f'projector_file = "{_path_text(projector_file)}"',
            "",
            "[paths]",
            'output_root = "./outputs"',
            "",
            "[runtime]",
            'end_level = "L4"',
            "",
            "[checks]",
            "strict_mode = true",
            'eps_profile = "default"',
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def hopping_path_for_ratio(ratio: float) -> Path:
    path = HOPPING_ROOT / f"HtSH_Oh.pfpi.-{ratio:.1f}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_complex_hopping(
    ratio: float,
    output_path: Path | None = None,
    *,
    pf_sigma_eV: float = 1.0,
    delta_pf_eV: float = 1.0,
    output_scale: float = 1.0,
) -> Path:
    target = output_path if output_path is not None else _default_complex_hopping_path(
        ratio,
        pf_sigma_eV=pf_sigma_eV,
        delta_pf_eV=delta_pf_eV,
    )
    cubic_path = target.with_name(target.stem + "_cubic_7x7.dat")
    target.parent.mkdir(parents=True, exist_ok=True)

    cubic_7x7 = fpf_cubic_7x7(
        float(pf_sigma_eV),
        -float(ratio) * float(pf_sigma_eV),
        float(delta_pf_eV),
    )
    # Runtime matrix files are consumed as raw internal energies (meV).
    cubic_7x7 = float(output_scale) * cubic_7x7
    complex_14x14 = to_complex_basis_spinor(_expand_to_spinor_interleaved(cubic_7x7))
    np.savetxt(cubic_path, cubic_7x7, fmt="%.12f")
    np.savetxt(target, complex_14x14, fmt="%.12f")
    return target


def run_cmd(cmd: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_case(case: Fig3Case, *, ratios: Sequence[float]) -> dict[str, Any]:
    static = static_asset_paths(case)
    case_root = OUTPUT_ROOT / case.key
    run_inputs_root = case_root / "run_inputs"
    logs_root = case_root / "logs"
    case_root.mkdir(parents=True, exist_ok=True)
    run_inputs_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    rows: list[list[float]] = []
    per_point: list[dict[str, Any]] = []
    for ratio in ratios:
        run_input = case_root / "run_input.toml"
        write_run_input(
            case,
            ratio=ratio,
            hopping_file=build_complex_hopping(ratio).resolve(),
            projector_file=static["projector"].resolve(),
            output_path=run_input,
        )
        shutil.copy2(run_input, run_inputs_root / f"run_input_ratio_{ratio:.1f}.toml")

        proc = run_cmd(
            [sys.executable, "-m", "fexchange.cli", "run", "run_input.toml"],
            cwd=case_root,
        )
        log_path = logs_root / f"run_ratio_{ratio:.1f}.log"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"fexchange CLI failed for {case.key} ratio={ratio:.1f}; see {log_path}"
            )

        point_result = _point_result_from_logs(case_root, proc)
        J_mu, mapping_residual = read_point_result(point_result)
        J_can, canonical_meta = canonicalize_exchange(J_mu)
        channels = scale_channels_to_mev(extract_channels(J_can))
        rows.append(
            [
                ratio,
                channels["J"],
                channels["K"],
                channels["Gamma"],
                channels["GammaPrime"],
                channels["symmetry_leakage"],
                mapping_residual,
            ]
        )
        per_point.append(
            {
                "ratio": ratio,
                "point_result": str(point_result),
                "mapping_residual": mapping_residual,
                "raw_J_mu": np.asarray(J_mu, dtype=float).tolist(),
                "canonical_J_mu": np.asarray(J_can, dtype=float).tolist(),
                "channels": channels,
                "canonical_meta": canonical_meta,
            }
        )

    curves = np.asarray(rows, dtype=float)
    curve_path = case_root / "curve.tsv"
    _write_curve(curve_path, curves)

    reference = None
    compare = None
    if case.reference_file:
        reference = load_reference_curve(case)
        compare = compare_curves(curves[:, :5], reference)
        (case_root / "compare.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")

    payload = {
        "case": case.key,
        "panel": case.panel,
        "curve_path": str(curve_path),
        "reference_path": str(REFERENCE_ROOT / case.reference_file) if case.reference_file else "",
        "compare": compare,
        "points": per_point,
    }
    (case_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "case": case,
        "curves": curves,
        "reference": reference,
        "curve_path": curve_path,
        "compare": compare,
        "points": per_point,
    }


def read_point_result(path: Path) -> tuple[np.ndarray, float]:
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    if values.shape != (14,):
        raise ValueError(f"Point result must contain 14 floats, got {values.shape} for {path}")
    return values[4:13].reshape(3, 3), float(values[13])


def compare_curves(curves: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    curves = np.asarray(curves, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ref_by_ratio = {round(float(row[0]), 12): row for row in reference}
    aligned_rows = []
    for row in curves:
        key = round(float(row[0]), 12)
        if key not in ref_by_ratio:
            raise ValueError(f"Reference ratio {row[0]:.12f} not found")
        aligned_rows.append(ref_by_ratio[key])
    aligned = np.asarray(aligned_rows, dtype=float)
    if curves.shape != aligned.shape:
        raise ValueError(f"Curve shape mismatch after alignment: {curves.shape} != {aligned.shape}")
    delta = curves - aligned
    names = ("ratio", "J", "K", "Gamma", "GammaPrime")
    return {
        "max_abs": {name: float(np.max(np.abs(delta[:, idx]))) for idx, name in enumerate(names[1:], start=1)},
        "rmse": {name: float(np.sqrt(np.mean(np.square(delta[:, idx])))) for idx, name in enumerate(names[1:], start=1)},
    }


def scale_channels_to_mev(channels: dict[str, float]) -> dict[str, float]:
    scale = 1000.0
    return {
        "J": scale * float(channels["J"]),
        "K": scale * float(channels["K"]),
        "Gamma": scale * float(channels["Gamma"]),
        "GammaPrime": scale * float(channels["GammaPrime"]),
        "symmetry_leakage": scale * float(channels["symmetry_leakage"]),
    }


def table_estimate_scale_mev(
    *,
    tpfsigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> float:
    return 1000.0 * float(tpfsigma_eV) ** 4 / float(delta_pf_eV) ** 2


def summarize_point_result(path: Path) -> dict[str, Any]:
    J_mu, mapping_residual = read_point_result(path)
    J_can, canonical_meta = canonicalize_exchange(J_mu)
    channels_raw = extract_channels(J_can)
    return {
        "raw_J_mu": np.asarray(J_mu, dtype=float),
        "canonical_J_mu": np.asarray(J_can, dtype=float),
        "channels_raw": channels_raw,
        "channels_fig3": scale_channels_to_mev(channels_raw),
        "mapping_residual": float(mapping_residual),
        "canonical_meta": canonical_meta,
    }


def fig3_row_from_point_result(*, ratio: float, path: Path) -> list[float]:
    summary = summarize_point_result(path)
    channels = summary["channels_fig3"]
    return [
        float(ratio),
        float(channels["J"]),
        float(channels["K"]),
        float(channels["Gamma"]),
        float(channels["GammaPrime"]),
        float(channels["symmetry_leakage"]),
        float(summary["mapping_residual"]),
    ]


def table_estimate_from_point_result(
    case: Fig3Case,
    *,
    path: Path,
    tpfsigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> dict[str, Any]:
    summary = summarize_point_result(path)
    scale = table_estimate_scale_mev(tpfsigma_eV=tpfsigma_eV, delta_pf_eV=delta_pf_eV)
    channels = summary["channels_fig3"]
    J = scale * float(channels["J"])
    K = scale * float(channels["K"])
    ratio = float("inf") if abs(J) < 1.0e-15 else abs(K / J)
    return {
        "case": case.key,
        "panel": case.panel,
        "ratio_abs_pfpi_pf_sigma": TABLE_RATIO,
        "tpfsigma_eV": float(tpfsigma_eV),
        "delta_pf_eV": float(delta_pf_eV),
        "scale_meV": float(scale),
        "J_meV": J,
        "K_meV": K,
        "Gamma_meV": scale * float(channels["Gamma"]),
        "GammaPrime_meV": scale * float(channels["GammaPrime"]),
        "abs_K_over_J": ratio,
        "mapping_residual": float(summary["mapping_residual"]),
        "point_result": str(path),
    }


def table_strict_from_point_result(
    case: Fig3Case,
    *,
    path: Path,
    tpfsigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> dict[str, Any]:
    summary = summarize_point_result(path)
    channels = summary["channels_fig3"]
    J = float(channels["J"])
    K = float(channels["K"])
    ratio = float("inf") if abs(J) < 1.0e-15 else abs(K / J)
    return {
        "case": case.key,
        "panel": case.panel,
        "ratio_abs_pfpi_pf_sigma": TABLE_RATIO,
        "tpfsigma_eV": float(tpfsigma_eV),
        "delta_pf_eV": float(delta_pf_eV),
        "scale_meV": 1.0,
        "J_meV": J,
        "K_meV": K,
        "Gamma_meV": float(channels["Gamma"]),
        "GammaPrime_meV": float(channels["GammaPrime"]),
        "abs_K_over_J": ratio,
        "mapping_residual": float(summary["mapping_residual"]),
        "point_result": str(path),
    }


def load_table2_literature() -> dict[str, dict[str, float | str]]:
    if not TABLE2_LITERATURE_FILE.exists():
        raise FileNotFoundError(TABLE2_LITERATURE_FILE)
    data = np.genfromtxt(TABLE2_LITERATURE_FILE, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    rows = np.atleast_1d(data)
    table: dict[str, dict[str, float | str]] = {}
    for row in rows:
        key = str(row["case"])
        table[key] = {
            "case": key,
            "panel": str(row["panel"]),
            "title": str(row["title"]),
            "J_meV": float(row["J_meV"]),
            "K_meV": float(row["K_meV"]),
            "abs_K_over_J": float(row["abs_K_over_J"]),
        }
    return table


def table2_from_point_result(
    case: Fig3Case,
    *,
    path: Path,
    tpfsigma_eV: float = TABLE_TPFSIGMA_EV,
    delta_pf_eV: float = TABLE_DELTA_PF_EV,
) -> dict[str, Any]:
    entry = table_strict_from_point_result(
        case,
        path=path,
        tpfsigma_eV=tpfsigma_eV,
        delta_pf_eV=delta_pf_eV,
    )
    literature = load_table2_literature().get(case.key)
    if literature is None:
        return entry
    entry["literature_J_meV"] = float(literature["J_meV"])
    entry["literature_K_meV"] = float(literature["K_meV"])
    entry["literature_abs_K_over_J"] = float(literature["abs_K_over_J"])
    entry["delta_J_meV"] = float(entry["J_meV"]) - float(literature["J_meV"])
    entry["delta_K_meV"] = float(entry["K_meV"]) - float(literature["K_meV"])
    entry["delta_abs_K_over_J"] = float(entry["abs_K_over_J"]) - float(literature["abs_K_over_J"])
    return entry


def nature_plot_styles() -> dict[str, dict[str, str]]:
    return {
        "colors": {
            "J": "#111111",
            "K": "#cf3c2f",
            "Gamma": "#f08a24",
            "GammaPrime": "#5566cc",
        },
        "labels": {
            "J": r"$J$",
            "K": r"$K$",
            "Gamma": r"$\Gamma$",
            "GammaPrime": r"$\Gamma'$",
        },
    }


def plot_results(results: Sequence[dict[str, Any]], *, figure_path: Path) -> Path:
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    styles = nature_plot_styles()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True)
    order = [case_by_key("f1_ce_g7"), case_by_key("f3_nd_g6"), case_by_key("f11_er_g7"), case_by_key("f13_yb_g6")]
    by_key = {result["case"].key: result for result in results}

    for ax, case in zip(axes.flat, order, strict=True):
        result = by_key[case.key]
        curves = result["curves"]
        reference = result["reference"]
        for idx, name in enumerate(("J", "K", "Gamma", "GammaPrime"), start=1):
            color = styles["colors"][name]
            label = styles["labels"][name]
            ax.plot(curves[:, 0], curves[:, idx], color=color, linewidth=2.0, label=label)
            if reference is not None:
                ax.scatter(
                    reference[:, 0],
                    reference[:, idx],
                    s=32,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.1,
                )
        ax.axhline(0.0, color="#999999", linewidth=0.8, alpha=0.6)
        ax.set_title(f"({case.panel}) {case.title}")
        ax.set_xlim(0.1, 1.0)
        ax.set_xticks(np.linspace(0.1, 1.0, 10))
        ax.grid(alpha=0.18, linewidth=0.6)

    for ax in axes[:, 0]:
        ax.set_ylabel("meV")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$|pf\pi / pf\sigma|$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(figure_path, dpi=220)
    fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)
    return figure_path


def run_suite(cases: Sequence[Fig3Case], *, ratios: Sequence[float], make_plot: bool = True) -> dict[str, Any]:
    prepare_static_assets(cases)
    results = [run_case(case, ratios=ratios) for case in cases]
    figure_path = OUTPUT_ROOT / "nature_fig3_abef.png"
    if make_plot and all(result["reference"] is not None for result in results):
        plot_results(results, figure_path=figure_path)
    summary = {
        "cases": [result["case"].key for result in results],
        "ratios": [float(r) for r in ratios],
        "figure_path": str(figure_path),
        "curve_paths": {result["case"].key: str(result["curve_path"]) for result in results},
        "compare": {result["case"].key: result["compare"] for result in results},
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce Nature Fig.3 particle panels with fexchange.")
    parser.add_argument("--case", action="append", dest="cases", help="Case key to run/prepare")
    parser.add_argument("--smoke", action="store_true", help="Run active cases at ratio=0.1 only")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare static projectors/run_input files")
    args = parser.parse_args(argv)

    if args.cases:
        selected = tuple(case_by_key(key) for key in args.cases)
    elif args.prepare_only:
        selected = all_cases()
    else:
        selected = default_run_cases()

    if args.prepare_only:
        prepare_static_assets(selected)
        return 0

    ratios = (0.1,) if args.smoke else RATIO_VALUES
    run_suite(selected, ratios=ratios, make_plot=not args.smoke)
    return 0


def _fmt_float(value: float) -> str:
    return f"{value:.14f}"


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-14:
        raise ValueError("Cannot normalize near-zero vector")
    return np.asarray(vec, dtype=float) / norm


def _row_index(j2: int, m2: int) -> int:
    if (m2 + j2) % 2 != 0:
        raise ValueError(f"Incompatible m2={m2} for j2={j2}")
    idx = (m2 + j2) // 2
    if idx < 0 or idx > j2:
        raise ValueError(f"m2={m2} out of range for j2={j2}")
    return idx


def _zeta_eV(re_name: str) -> float:
    return RE_DEFAULTS_BY_N_ELE[RE_TO_N_ELE[re_name]]["zeta"]


def _physics_branch_lines(header: str, branch: BranchSpec) -> list[str]:
    if branch.re_name is None:
        return []
    return [f"[{header}]", f'RE = "{branch.re_name}"', ""]


def _sopt_branch_lines(header: str, branch: BranchSpec) -> list[str]:
    if (
        branch.re_name is None
        and branch.Jh_eV is None
        and branch.zeta_eV is None
        and branch.offset_eV == 0.0
        and branch.U_eV == 0.0
    ):
        return []
    lines = [f"[{header}]"]
    lines.append(f"U = {_fmt_float(branch.U_eV)}")
    if branch.Jh_eV is not None:
        lines.append(f"Jh = {_fmt_float(branch.Jh_eV)}")
    if branch.zeta_eV is not None:
        lines.append(f"zeta = {_fmt_float(branch.zeta_eV)}")
    if branch.offset_eV != 0.0:
        lines.append(f"offset = {_fmt_float(branch.offset_eV)}")
    lines.append("")
    return lines


def _path_text(path_like: str | Path) -> str:
    return str(path_like).replace("\\", "/")


def _default_complex_hopping_path(
    ratio: float,
    *,
    pf_sigma_eV: float,
    delta_pf_eV: float,
) -> Path:
    if math.isclose(float(pf_sigma_eV), 1.0, abs_tol=1.0e-15) and math.isclose(float(delta_pf_eV), 1.0, abs_tol=1.0e-15):
        return COMPLEX_HOPPING_ROOT / f"t_mu_ratio_{ratio:.1f}.dat"
    return COMPLEX_HOPPING_ROOT / (
        f"t_mu_ratio_{ratio:.1f}_pf_sigma_{pf_sigma_eV:.3f}_delta_{delta_pf_eV:.3f}.dat"
    )


def _repo_relative(path: Path) -> Path:
    return path.resolve().relative_to(REPO_ROOT)


def _point_result_from_logs(case_root: Path, proc: subprocess.CompletedProcess[str]) -> Path:
    merged = proc.stdout + "\n" + proc.stderr
    matches = POINT_RESULT_RE.findall(merged)
    if matches:
        return (case_root / matches[-1].strip()).resolve()
    txt_files = sorted((case_root / "outputs").glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No point_result txt found under {case_root / 'outputs'}")
    return txt_files[-1].resolve()


def _write_curve(path: Path, curves: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "#ratio J K Gamma GammaPrime symmetry_leakage mapping_residual"
    np.savetxt(path, curves, fmt="%.12f", header=header, comments="")


if __name__ == "__main__":
    raise SystemExit(main())
