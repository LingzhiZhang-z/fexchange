"""Stage ensure functions for LMSM/LSJM and SOPT/FOPT branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fexchange.core.fock import det_index, dim_sector, enumerate_dets, make_basis_id
from fexchange.io.disk import build_stage_path
from fexchange.io.matrix import load_matrix_file, load_txt_blocks
from fexchange.pipeline.artifacts import (
    persist_l0,
    persist_l0_fopt,
    persist_ion_ed,
    persist_l1,
    persist_l1_fopt,
    persist_l2,
    persist_l2_fopt,
    persist_l3_sopt,
    persist_l3_fopt,
    persist_ligand,
    persist_stateset,
    try_load_l0,
    try_load_l0_fopt,
    try_load_ion_ed,
    try_load_l1,
    try_load_l1_fopt,
    try_load_l2,
    try_load_l2_fopt,
    try_load_l3_sopt,
    try_load_l3_fopt,
    try_load_ligand,
    try_load_stateset,
)
from fexchange.pipeline.keys import (
    labels_abcd_lex,
    level_key,
    sector_branch,
    sector_ratios,
    three_sectors,
)
from fexchange.pipeline.source_writer import write_core_source_txt, write_run_source_txt
from fexchange.pipeline.validation import validate_labels_abcd
from fexchange.spectrum.energy import compute_intermediate_energies, compute_ligand_energies
from fexchange.spectrum.ion_ed import build_ion_ed
from fexchange.spectrum.ligand import build_ligand_spectrum
from fexchange.spectrum.lsjm import build_lsjm, select_soc_lowest_subspace
from fexchange.spectrum.lsms import build_lsms
from fexchange.sopt.spin12 import spin12_map
from fexchange.utils.checks import check_hermitian, check_orthonormal
from fexchange.utils.errors import BindError, IOError_, InputError, SchemaError


def _reference_scheme(scheme: str) -> str:
    return "RS" if str(scheme).upper() == "ED" else scheme


def _uses_ion_ed(scheme: str) -> bool:
    return str(scheme).upper() == "ED"


def _kramer_source(cfg: dict[str, Any]) -> str:
    return str(cfg.get("runtime", {}).get("kramer_source", "stevens"))


def _uses_manual_kramer(cfg: dict[str, Any]) -> bool:
    return _kramer_source(cfg) == "manual"


def _ls_reference_sectors(n_ele: int, scheme: str) -> tuple[int, ...]:
    if _uses_ion_ed(scheme):
        return (n_ele,)
    return tuple(three_sectors(n_ele))


def _sector_fsite(cfg: dict[str, Any], n_ele: int) -> dict[str, Any]:
    branch = sector_branch(cfg, n_ele)
    if isinstance(branch, dict) and isinstance(branch.get("fsite"), dict):
        return branch["fsite"]
    fsite = cfg.get("fsite", {})
    if isinstance(fsite, dict):
        return fsite
    raise InputError(
        "FXE-INPUT-003",
        "Missing fsite parameters for IONED sector",
        actual={"n_ele": n_ele},
    )


def _load_hcef_1b(cfg: dict[str, Any], *, n_orb: int) -> np.ndarray | None:
    inputs = cfg.get("inputs", {})
    if not isinstance(inputs, dict) or not inputs.get("hcef_file"):
        return None
    path = Path(str(inputs["hcef_file"]))
    if path.suffix.lower() in {".txt", ".dat"}:
        try:
            blocks = load_txt_blocks(path, {"hcef": n_orb * n_orb})
            h = np.asarray(blocks["hcef"], dtype=np.complex128).reshape(n_orb, n_orb)
            check_hermitian(h, label="hcef_file", module="pipeline")
            return h
        except SchemaError:
            pass
    h = np.asarray(load_matrix_file(path, preferred_key="hcef"), dtype=np.complex128)
    if h.ndim == 1 and h.size == n_orb * n_orb:
        h = h.reshape(n_orb, n_orb)
    if h.ndim != 2 or h.shape != (n_orb, n_orb):
        raise BindError(
            "FXE-BIND-003",
            f"hcef_file shape mismatch: {h.shape} != ({n_orb},{n_orb})",
            expected={"shape": [n_orb, n_orb]},
            actual={"shape": list(h.shape)},
            paths={"path": str(path)},
        )
    check_hermitian(h, label="hcef_file", module="pipeline")
    return h


def _load_manual_kramer(path: Path, *, n_ele: int, n_orb: int) -> dict[str, Any]:
    if not path.exists():
        raise IOError_("FXE-IO-001", f"Required input file missing: {path}", paths={"path": str(path)})
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise IOError_("FXE-IO-001", f"Failed reading kramer_file: {path}", paths={"path": str(path)}) from exc

    logical_lines: list[tuple[int, str]] = []
    for lineno, raw in enumerate(raw_lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if line:
            logical_lines.append((lineno, line))
    if not logical_lines:
        raise SchemaError("FXE-SCHEMA-002", "kramer_file is empty", paths={"path": str(path)})

    first_lineno, first = logical_lines[0]
    first_tokens = first.replace("=", " ").split()
    if len(first_tokens) != 2 or first_tokens[0].lower() != "fn":
        raise SchemaError(
            "FXE-SCHEMA-002",
            "kramer_file first non-comment line must be 'fn <n>'",
            actual={"lineno": first_lineno, "line": first},
            paths={"path": str(path)},
        )
    try:
        file_n = int(first_tokens[1])
    except ValueError as exc:
        raise SchemaError(
            "FXE-SCHEMA-002",
            "kramer_file fn value must be an integer",
            actual={"lineno": first_lineno, "line": first},
            paths={"path": str(path)},
        ) from exc
    if file_n != n_ele:
        raise BindError(
            "FXE-BIND-003",
            "kramer_file fn does not match fsite.n_ele",
            expected={"n_ele": n_ele},
            actual={"fn": file_n},
            paths={"path": str(path)},
        )

    blocks: dict[int, dict[int, complex]] = {}
    current: int | None = None
    dets = enumerate_dets(n_ele, n_orb)
    for lineno, line in logical_lines[1:]:
        if line.startswith("[") and line.endswith("]"):
            key = line[1:-1].strip()
            if not key.startswith("K_state_"):
                raise SchemaError(
                    "FXE-SCHEMA-002",
                    "kramer_file block keys must be [K_state_<idx>]",
                    actual={"lineno": lineno, "key": key},
                    paths={"path": str(path)},
                )
            try:
                idx = int(key.removeprefix("K_state_"))
            except ValueError as exc:
                raise SchemaError(
                    "FXE-SCHEMA-002",
                    "kramer_file K_state index must be an integer",
                    actual={"lineno": lineno, "key": key},
                    paths={"path": str(path)},
                ) from exc
            if idx in blocks:
                raise SchemaError(
                    "FXE-SCHEMA-002",
                    "Duplicate K_state block",
                    actual={"lineno": lineno, "key": key},
                    paths={"path": str(path)},
                )
            blocks[idx] = {}
            current = idx
            continue

        if current is None:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "kramer_file data appeared before any [K_state_<idx>] block",
                actual={"lineno": lineno, "line": line},
                paths={"path": str(path)},
            )
        parts = line.split()
        if len(parts) != n_orb + 2:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "kramer_file data rows must be: real imag occ_0 ... occ_13",
                expected={"fields": n_orb + 2},
                actual={"lineno": lineno, "fields": len(parts)},
                paths={"path": str(path)},
            )
        try:
            coeff = complex(float(parts[0]), float(parts[1]))
            occ = [int(v) for v in parts[2:]]
        except ValueError as exc:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "Failed parsing kramer_file row",
                actual={"lineno": lineno, "line": line},
                paths={"path": str(path)},
            ) from exc
        if any(v not in (0, 1) for v in occ):
            raise SchemaError(
                "FXE-SCHEMA-002",
                "kramer_file occupation entries must be 0 or 1",
                actual={"lineno": lineno, "occupation": occ},
                paths={"path": str(path)},
            )
        if sum(occ) != n_ele:
            raise BindError(
                "FXE-BIND-003",
                "kramer_file determinant occupation count does not match fn",
                expected={"n_ele": n_ele},
                actual={"lineno": lineno, "occupation_count": int(sum(occ))},
                paths={"path": str(path)},
            )
        det = sum(int(v) << p for p, v in enumerate(occ))
        if det in blocks[current]:
            raise SchemaError(
                "FXE-SCHEMA-002",
                "Duplicate determinant in K_state block",
                actual={"lineno": lineno, "det": det, "state": current},
                paths={"path": str(path)},
            )
        det_index(det, dets)
        blocks[current][det] = coeff

    indices = sorted(blocks)
    if indices != [0, 1]:
        raise BindError(
            "FXE-BIND-003",
            "manual kramer_file must contain exactly [K_state_0] and [K_state_1]",
            expected={"indices": [0, 1]},
            actual={"indices": indices},
            paths={"path": str(path)},
        )
    V = np.zeros((dim_sector(n_ele, n_orb), len(indices)), dtype=np.complex128)
    for col, state_idx in enumerate(indices):
        for det, coeff in blocks[state_idx].items():
            V[det_index(det, dets), col] = coeff
    check_orthonormal(V, label="K_fock_manual", module="projection")
    return {
        "U_n_soc0": V,
        "n_j": int(V.shape[1]),
        "subspace_id": "direct_kramer_fock_v1",
        "meta": {
            "n_j": int(V.shape[1]),
            "basis_id": make_basis_id(n_ele, n_orb),
            "source_format": "manual_fn_occupancy_v1",
            "path": str(path),
        },
    }


def _select_main_subspace(cfg: dict[str, Any], state: dict[str, Any], *, n_ele: int, n_orb: int) -> dict[str, Any]:
    if _uses_manual_kramer(cfg):
        return _load_manual_kramer(Path(str(cfg["inputs"]["kramer_file"])), n_ele=n_ele, n_orb=n_orb)
    soc0 = select_soc_lowest_subspace(state[f"lsjm_{n_ele}"])
    soc0["subspace_id"] = "soc_lowest_hunds_v1"
    return soc0


def ensure_lsms_all_three(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    output_root = cfg["paths"]["output_root"]
    core_scheme = _reference_scheme(scheme)
    for sec in _ls_reference_sectors(n_ele, scheme):
        key = f"lsms_{sec}"
        if key in state:
            continue
        sec_r42, sec_r62 = sector_ratios(cfg, sec, default_r42=r42, default_r62=r62)
        stage_dir = build_stage_path(output_root, "LMSM", n=sec, r42=sec_r42, r62=sec_r62, scheme=core_scheme)
        loaded = try_load_stateset(stage_dir, level="LMSM", n_ele=sec)
        if loaded is not None:
            state[key] = loaded
            continue

        result = build_lsms(sec, n_orb, r42=sec_r42, r62=sec_r62)
        state[key] = result
        branch = sector_branch(cfg, sec)
        fsite = branch["fsite"] if isinstance(branch, dict) else cfg.get("fsite", {})
        persist_stateset(
            stage_dir,
            result,
            level="LMSM",
            n_ele=sec,
            cfg=cfg,
            physics=fsite,
            r42=sec_r42,
            r62=sec_r62,
        )
        write_core_source_txt(stage_dir, {"level": "LMSM", "n_ele": sec, "r42": sec_r42, "r62": sec_r62})


def ensure_lsjm_all_three(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    output_root = cfg["paths"]["output_root"]
    ensure_lsms_all_three(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    core_scheme = _reference_scheme(scheme)

    for sec in _ls_reference_sectors(n_ele, scheme):
        key = f"lsjm_{sec}"
        if key in state:
            continue
        sec_r42, sec_r62 = sector_ratios(cfg, sec, default_r42=r42, default_r62=r62)
        stage_dir = build_stage_path(output_root, "LSJM", n=sec, r42=sec_r42, r62=sec_r62, scheme=core_scheme)
        loaded = try_load_stateset(stage_dir, level="LSJM", n_ele=sec)
        if loaded is not None:
            state[key] = loaded
            continue

        result = build_lsjm(state[f"lsms_{sec}"], n_orb)
        state[key] = result
        branch = sector_branch(cfg, sec)
        fsite = branch["fsite"] if isinstance(branch, dict) else cfg.get("fsite", {})
        persist_stateset(
            stage_dir,
            result,
            level="LSJM",
            n_ele=sec,
            cfg=cfg,
            physics=fsite,
            r42=sec_r42,
            r62=sec_r62,
        )
        write_core_source_txt(stage_dir, {"level": "LSJM", "n_ele": sec, "r42": sec_r42, "r62": sec_r62})


def ensure_ion_ed_adjacent(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
) -> None:
    output_root = cfg["paths"]["output_root"]
    run_name = str(cfg.get("runtime", {}).get("run_name", ""))
    hcef_1b = _load_hcef_1b(cfg, n_orb=n_orb)
    for sec in (n_ele - 1, n_ele + 1):
        key = f"ioned_{sec}"
        if key in state:
            continue
        sec_r42, sec_r62 = sector_ratios(cfg, sec, default_r42=r42, default_r62=r62)
        stage_dir = build_stage_path(output_root, "IONED", n=sec, run_name=run_name)
        expected_key = level_key("IONED", n_ele=sec, r42=sec_r42, r62=sec_r62, cfg=cfg)
        loaded = try_load_ion_ed(stage_dir, expected_key=expected_key)
        if loaded is not None:
            state[key] = loaded
            continue

        fsite = _sector_fsite(cfg, sec)
        result = build_ion_ed(
            sec,
            F0=float(fsite.get("U", 0.0)),
            F2=float(fsite.get("F2", 0.0)),
            F4=float(fsite.get("F4", 0.0)),
            F6=float(fsite.get("F6", 0.0)),
            zeta=float(fsite["zeta"]),
            offset=float(fsite.get("offset", 0.0)),
            n_orb=n_orb,
            lsjm_reference=state.get(f"lsjm_{sec}"),
            hcef_1b=hcef_1b,
        )
        state[key] = result
        persist_ion_ed(stage_dir, cfg, result, n_ele=sec, r42=sec_r42, r62=sec_r62)
        write_run_source_txt(Path(output_root) / run_name, cfg)


def ensure_l0_sopt(cfg: dict[str, Any], state: dict[str, Any], *, n_ele: int, n_orb: int) -> None:
    from fexchange.sopt.precompute import build_L0

    if "l0" in state:
        return
    stage_dir = build_stage_path(cfg["paths"]["output_root"], "L0")
    loaded = try_load_l0(stage_dir, n_ele=n_ele)
    if loaded is not None:
        state["l0"] = loaded
        return

    result = build_L0(n_ele, n_orb)
    state["l0"] = result
    persist_l0(stage_dir, cfg, result, n_ele=n_ele)
    write_core_source_txt(stage_dir, {"level": "L0", "branch": "sopt", "n_ele": n_ele})


def ensure_l1_sopt(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.sopt.precompute import build_L1

    if "l1" in state:
        return
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L1",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        run_name=str(cfg.get("runtime", {}).get("run_name", "")),
    )
    loaded = try_load_l1(stage_dir, expected_key=level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg))
    if loaded is not None:
        state["l1"] = loaded
        return

    ensure_l0_sopt(cfg, state, n_ele=n_ele, n_orb=n_orb)
    ensure_lsjm_all_three(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    if _uses_ion_ed(scheme):
        ensure_ion_ed_adjacent(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62)
    soc0 = _select_main_subspace(cfg, state, n_ele=n_ele, n_orb=n_orb)
    state["soc0"] = soc0
    U_np1 = state[f"ioned_{n_ele + 1}"]["V_fock"] if _uses_ion_ed(scheme) else state[f"lsjm_{n_ele + 1}"]["V_fock"]
    U_nm1 = state[f"ioned_{n_ele - 1}"]["V_fock"] if _uses_ion_ed(scheme) else state[f"lsjm_{n_ele - 1}"]["V_fock"]
    result = build_L1(
        state["l0"],
        U_np1,
        soc0["U_n_soc0"],
        U_nm1,
    )
    state["l1"] = result
    persist_l1(stage_dir, cfg, result, n_ele=n_ele, r42=r42, r62=r62, soc0=soc0)
    write_run_source_txt(Path(cfg["paths"]["output_root"]) / str(cfg["runtime"]["run_name"]), cfg)


def ensure_l2_sopt(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.sopt.contraction import build_L2

    if "l2" in state:
        return
    run_dir = Path(cfg["paths"]["output_root"]) / cfg["runtime"]["run_name"]
    write_run_source_txt(run_dir, cfg)
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L2",
        run_name=cfg["runtime"]["run_name"],
    )
    loaded = try_load_l2(stage_dir, expected_key=level_key("L2", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg))
    if loaded is not None:
        state["l2"] = loaded
        return

    ensure_l1_sopt(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    if _uses_manual_kramer(cfg):
        W = np.eye(int(state["l1"]["n_j"]), dtype=np.complex128)
    else:
        W = _load_projector(Path(cfg["inputs"]["projector_file"]), n_j=state["l1"]["n_j"])
    check_orthonormal(W, label="W_input", module="projection")
    t_mu = _load_hopping_sopt(Path(cfg["inputs"]["hopping_file"]), n_orb=n_orb)
    state["t_mu"] = t_mu
    state["W"] = W
    result = build_L2(state["l1"], t_mu, W)
    state["l2"] = result
    persist_l2(stage_dir, cfg, result, n_ele=n_ele, r42=r42, r62=r62)


def ensure_l3_sopt(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.sopt.contraction import build_L3

    if "l3" in state:
        return
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L3",
        run_name=cfg["runtime"]["run_name"],
    )
    loaded = try_load_l3_sopt(stage_dir, expected_key=level_key("L3", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg))
    if loaded is not None:
        state["l3"] = loaded
        return

    ensure_l2_sopt(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    ensure_lsjm_all_three(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    if _uses_ion_ed(scheme):
        ensure_ion_ed_adjacent(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62)
    E_u_np1, E_u_nm1 = compute_intermediate_energies(cfg, state, n_ele=n_ele)
    n_k = int(state["l2"]["n_k"])
    labels = labels_abcd_lex(n_k)
    validate_labels_abcd(labels, n_k=n_k)
    result = build_L3(state["l2"], E_u_np1, E_u_nm1, n_ele=n_ele)
    if n_k == 2:
        result.update(spin12_map(result["Heff_mu_abcd"]))
    state["l3"] = result
    state["labels_abcd"] = labels
    persist_l3_sopt(stage_dir, cfg, result, n_ele=n_ele, r42=r42, r62=r62, labels=labels)


def ensure_l0_fopt(cfg: dict[str, Any], state: dict[str, Any], *, n_ele: int, n_orb: int) -> None:
    from fexchange.fopt.precompute import compute_L0_f, compute_L0_p

    if "l0" in state:
        return
    stage_dir = build_stage_path(cfg["paths"]["output_root"], "L0")
    loaded = try_load_l0_fopt(stage_dir, n_ele=n_ele)
    if loaded is not None:
        state["l0"] = loaded
        return
    result = {"f": compute_L0_f(n_ele, n_orb), "p": compute_L0_p()}
    state["l0"] = result
    persist_l0_fopt(stage_dir, cfg, result, n_ele=n_ele)
    write_core_source_txt(stage_dir, {"level": "L0", "branch": "fopt", "n_ele": n_ele})


def ensure_ligand(cfg: dict[str, Any], state: dict[str, Any]) -> None:
    if "ligand_spectra" not in state:
        state["ligand_spectra"] = {}
    spectra = state["ligand_spectra"]
    for ligand in (1, 2):
        soc_token = _ligand_soc_token(cfg, ligand)
        soc = soc_token == "soc"
        for n_p in (4, 5, 6):
            key = (ligand, n_p)
            if key in spectra:
                continue
            stage_dir = build_stage_path(
                cfg["paths"]["output_root"],
                "ligand",
                ligand_soc=soc_token,
                ligand_n=n_p,
            )
            loaded = try_load_ligand(stage_dir)
            if loaded is not None:
                spectra[key] = loaded
                continue
            result = build_ligand_spectrum(n_p, soc=soc)
            spectra[key] = result
            persist_ligand(stage_dir, cfg, result, soc=soc)
            write_core_source_txt(stage_dir, {"level": "ligand", "n_ele": n_p, "soc": soc})


def ensure_l1_fopt(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.fopt.precompute import compute_L1_f, compute_L1_p

    if "l1" in state:
        return
    f_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L1_F",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        run_name=str(cfg.get("runtime", {}).get("run_name", "")),
    )
    p_dirs = _p_l1_dirs(cfg)
    loaded = try_load_l1_fopt(
        f_dir,
        p_dirs,
        expected_key=level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
    )
    if loaded is not None:
        state["l1"] = loaded
        return

    ensure_l0_fopt(cfg, state, n_ele=n_ele, n_orb=n_orb)
    ensure_lsjm_all_three(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    if _uses_ion_ed(scheme):
        ensure_ion_ed_adjacent(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62)
    ensure_ligand(cfg, state)
    soc0 = _select_main_subspace(cfg, state, n_ele=n_ele, n_orb=n_orb)
    state["soc0"] = soc0
    U_np1 = state[f"ioned_{n_ele + 1}"]["V_fock"] if _uses_ion_ed(scheme) else state[f"lsjm_{n_ele + 1}"]["V_fock"]
    U_nm1 = state[f"ioned_{n_ele - 1}"]["V_fock"] if _uses_ion_ed(scheme) else state[f"lsjm_{n_ele - 1}"]["V_fock"]
    f_vertex = compute_L1_f(
        state["l0"]["f"],
        U_np1,
        soc0["U_n_soc0"],
        U_nm1,
    )
    p_vertices = {
        ligand: compute_L1_p(
            state["l0"]["p"],
            state["ligand_spectra"][(ligand, 6)]["V_fock"],
            state["ligand_spectra"][(ligand, 5)]["V_fock"],
            state["ligand_spectra"][(ligand, 4)]["V_fock"],
        )
        for ligand in (1, 2)
    }
    result = {"f": {1: f_vertex, 2: f_vertex}, "p": p_vertices}
    state["l1"] = result
    persist_l1_fopt(f_dir, p_dirs, cfg, result, n_ele=n_ele, r42=r42, r62=r62, main_subspace=soc0)
    write_run_source_txt(Path(cfg["paths"]["output_root"]) / str(cfg["runtime"]["run_name"]), cfg)


def ensure_l2_fopt(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.fopt.contraction import build_L2

    if "l2" in state:
        return
    run_dir = Path(cfg["paths"]["output_root"]) / cfg["runtime"]["run_name"]
    write_run_source_txt(run_dir, cfg)
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L2",
        run_name=cfg["runtime"]["run_name"],
    )
    loaded = try_load_l2_fopt(stage_dir, expected_key=level_key("L2", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg))
    if loaded is not None:
        state["l2"] = loaded
        return
    ensure_l1_fopt(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    n_j = state["l1"]["f"][1]["F_n_np1"].shape[2]
    if _uses_manual_kramer(cfg):
        W = np.eye(int(n_j), dtype=np.complex128)
    else:
        W = _load_projector(Path(cfg["inputs"]["projector_file"]), n_j=n_j)
    check_orthonormal(W, label="W_input", module="projection")
    t = _load_hopping_fopt(Path(cfg["inputs"]["hopping_file"]), n_orb_f=n_orb, n_orb_p=6)
    state["t_per_pair"] = t
    state["W"] = W
    result = build_L2(state["l1"], t, W)
    state["l2"] = result
    persist_l2_fopt(stage_dir, cfg, result, n_ele=n_ele, r42=r42, r62=r62)


def ensure_l3_fopt(
    cfg: dict[str, Any],
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.fopt.contraction import FOPT_PROCESS_LABELS, build_L3

    if "l3" in state:
        return
    ligands = cfg["ligand"]
    stage_dir = build_stage_path(
        cfg["paths"]["output_root"],
        "L3",
        run_name=cfg["runtime"]["run_name"],
    )
    loaded = try_load_l3_fopt(stage_dir, expected_key=level_key("L3", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg))
    if loaded is not None:
        state["l3"] = loaded
        return
    ensure_l2_fopt(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    ensure_lsjm_all_three(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    if _uses_ion_ed(scheme):
        ensure_ion_ed_adjacent(cfg, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62)
    ensure_ligand(cfg, state)
    E_u_np1, E_u_nm1 = compute_intermediate_energies(cfg, state, n_ele=n_ele)
    energies: dict[str, Any] = {"f_np1": E_u_np1, "f_nm1": E_u_nm1, "E_0": 0.0}
    for ligand in (1, 2):
        params = ligands[str(ligand)]
        for n_p, key in ((5, "p_5"), (4, "p_4")):
            energies[f"{key}_lig{ligand}"] = compute_ligand_energies(
                state["ligand_spectra"][(ligand, n_p)],
                Delta=float(params["Delta"]),
                U_p=float(params["U_p"]),
                lambda_p=float(params.get("lambda_p", 0.0)),
            )
    l3_raw = build_L3(state["l2"], energies, n_ele=n_ele, return_per_path=True)
    result_arr = np.asarray(l3_raw["H_total"], dtype=np.complex128)
    process_heff = np.stack(
        [np.asarray(l3_raw["per_process"][proc], dtype=np.complex128) for proc in FOPT_PROCESS_LABELS],
        axis=0,
    )
    result = {
        "h_eff_4": result_arr,
        "h_eff_4_processes": process_heff,
        "process_labels": np.asarray(FOPT_PROCESS_LABELS),
        "n_k": int(result_arr.shape[0]),
    }
    spin12_total = spin12_map(result_arr)
    spin12_processes = [spin12_map(process_heff[idx]) for idx in range(process_heff.shape[0])]
    result.update(
        {
            "J_mu": spin12_total["J_mu"],
            "mapping_residual": spin12_total["mapping_residual"],
            "J_mu_processes": np.stack([item["J_mu"] for item in spin12_processes], axis=0),
            "mapping_residual_processes": np.asarray(
                [item["mapping_residual"] for item in spin12_processes],
                dtype=float,
            ),
        }
    )
    state["l3"] = result
    persist_l3_fopt(stage_dir, cfg, result, n_ele=n_ele, r42=r42, r62=r62)


def _load_hopping_sopt(path: Path, *, n_orb: int) -> np.ndarray:
    if path.suffix.lower() in {".txt", ".dat"}:
        try:
            blocks = load_txt_blocks(path, {"t_mu": n_orb * n_orb})
            return blocks["t_mu"].reshape(n_orb, n_orb).astype(np.complex128)
        except SchemaError:
            pass
    t_mu = np.asarray(load_matrix_file(path, preferred_key="t_mu"), dtype=np.complex128)
    if t_mu.ndim != 2 or t_mu.shape != (n_orb, n_orb):
        raise BindError(
            "FXE-BIND-003",
            f"Hopping shape mismatch: {t_mu.shape} != ({n_orb},{n_orb})",
            expected={"shape": [n_orb, n_orb]},
            actual={"shape": list(t_mu.shape)},
        )
    return t_mu


def _load_hopping_fopt(path: Path, *, n_orb_f: int, n_orb_p: int) -> dict[tuple[int, int], np.ndarray]:
    expected_keys = {
        f"t_f{fs}_lig{lig}": n_orb_f * n_orb_p
        for fs in (1, 2)
        for lig in (1, 2)
    }
    if path.suffix.lower() in {".txt", ".dat"}:
        blocks = load_txt_blocks(path, expected_keys)
        return {
            (fs, lig): blocks[f"t_f{fs}_lig{lig}"].reshape(n_orb_f, n_orb_p).astype(np.complex128)
            for fs in (1, 2)
            for lig in (1, 2)
        }
    raise InputError(
        "FXE-INPUT-003",
        "fopt hopping_file must be multi-block txt",
        actual={"path": str(path), "suffix": path.suffix},
    )


def _load_projector(path: Path, *, n_j: int) -> np.ndarray:
    """Load projector W.

    Text format (.txt / .dat): one block per doublet state, named
    ``[W_state_0]``, ``[W_state_1]``, ... ``[W_state_{n_k-1}]``. Each block
    contains exactly ``n_j`` rows of ``real imag`` lines (one column of W).
    State indices must be contiguous starting at 0; columns are stacked in
    numeric order to form W with shape ``(n_j, n_k)``.

    Binary format (.npy / .npz): a rank-2 array (or flat array reshapeable to
    rank 2) with ``shape[0] == n_j``.
    """
    if path.suffix.lower() in {".txt", ".dat"}:
        blocks = load_txt_blocks(path)
        state_keys = [k for k in blocks if k.startswith("W_state_")]
        if not state_keys:
            raise BindError(
                "FXE-BIND-003",
                "Projector txt must contain [W_state_<idx>] blocks (one per doublet state)",
                actual={"keys": sorted(blocks)},
                paths={"path": str(path)},
            )
        try:
            state_keys.sort(key=lambda k: int(k.removeprefix("W_state_")))
        except ValueError as exc:
            raise BindError(
                "FXE-BIND-003",
                "Projector block key must be of form W_state_<integer>",
                actual={"keys": sorted(state_keys)},
                paths={"path": str(path)},
            ) from exc
        indices = [int(k.removeprefix("W_state_")) for k in state_keys]
        expected = list(range(len(indices)))
        if indices != expected:
            raise BindError(
                "FXE-BIND-003",
                "Projector W_state_<idx> blocks must be contiguous starting at 0",
                expected={"indices": expected},
                actual={"indices": indices},
                paths={"path": str(path)},
            )
        for k in state_keys:
            if blocks[k].size != n_j:
                raise BindError(
                    "FXE-BIND-003",
                    f"Projector block {k} has {blocks[k].size} rows, expected dim_soc_lowest(n)={n_j}",
                    expected={"rows_per_state": n_j},
                    actual={"key": k, "rows": int(blocks[k].size)},
                    paths={"path": str(path)},
                )
        W = np.stack(
            [np.asarray(blocks[k], dtype=np.complex128) for k in state_keys],
            axis=1,
        )
    else:
        W = np.asarray(load_matrix_file(path, preferred_key="W"), dtype=np.complex128)
        if W.ndim == 1 and W.size % n_j == 0:
            W = W.reshape(n_j, W.size // n_j)
    if W.ndim != 2:
        raise BindError("FXE-BIND-003", "Projector W must be rank-2", actual={"W_shape": list(W.shape)})
    if W.shape[0] != n_j:
        raise BindError(
            "FXE-BIND-003",
            f"W.shape[0]={W.shape[0]} != dim_soc_lowest(n)={n_j}",
            expected={"W_shape_0": n_j},
            actual={"W_shape": list(W.shape)},
        )
    return W


def _ligand_soc_token(cfg: dict[str, Any], ligand: int) -> str:
    lambda_p = float(cfg["ligand"][str(ligand)].get("lambda_p", 0.0))
    return "soc" if lambda_p != 0.0 else "nsoc"


def _p_l1_dirs(cfg: dict[str, Any]) -> dict[int, dict[str, Path]]:
    output_root = cfg["paths"]["output_root"]
    result: dict[int, dict[str, Path]] = {}
    for ligand in (1, 2):
        soc_token = _ligand_soc_token(cfg, ligand)
        result[ligand] = {
            "P_5_6": build_stage_path(
                output_root,
                "L1_P",
                p_transition=f"n-5_to_6_{soc_token}",
            ),
            "P_4_5": build_stage_path(
                output_root,
                "L1_P",
                p_transition=f"n-4_to_5_{soc_token}",
            ),
        }
    return result
