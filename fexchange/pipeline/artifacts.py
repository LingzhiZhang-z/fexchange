"""
Artifact loading and persistence helpers for pipeline stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from fexchange.io.disk import (
    append_index_record,
    atomic_write_json,
    atomic_write_npz,
    build_meta,
    load_json_checked,
    load_npz_checked,
    validate_meta,
    write_exchange_matrix_txt,
)
from fexchange.pipeline.keys import (
    level_key,
    sector_branch,
)
from fexchange.pipeline.source_writer import build_resolved_inputs_summary
from fexchange.utils.numerics import numerics_meta

logger = logging.getLogger("fexchange")


def _fsite_for_index(cfg: dict[str, Any], n_ele: int) -> dict[str, Any]:
    branch = sector_branch(cfg, n_ele)
    if isinstance(branch, dict) and isinstance(branch.get("fsite"), dict):
        return branch["fsite"]
    fsite = cfg.get("fsite", {})
    return fsite if isinstance(fsite, dict) else {}

# Artifact file spec: maps level -> list of (filename, minimum_required_npz_keys_or_None_for_json).
# These are the minimum keys needed for loading/validation, not the full set of stored keys.
# Used by both try_load_* and validate_upstream_artifacts to maintain one source of truth.
ARTIFACT_FILE_SPEC: dict[str, list[tuple[str, list[str] | None]]] = {
    "LMSM": [
        ("LMSM.npz", ["V_fock", "coef_F0", "coef_F2", "coef_F4", "coef_F6"]),
        ("meta.json", None),
    ],
    "LSJM": [
        ("LSJM.npz", ["V_fock", "coef_F0", "coef_F2", "coef_F4", "coef_F6", "coef_zeta"]),
        ("meta.json", None),
    ],
    "IONED": [
        ("states.npz", ["V_fock_ed", "energies"]),
        ("meta.json", None),
    ],
    "L0": [
        # L0 uses dynamic filenames (n{n_ele}{n_ele+1}.npz etc.), handled specially.
        # This entry covers the meta files only.
    ],
    "L1": [
        ("data.npz", ["A", "B"]),
        ("meta.json", None),
    ],
    "L2": [
        ("data.npz", ["M_A", "M_B"]),
        ("meta.json", None),
    ],
    "L3": [
        ("data.npz", ["h_mu_abcd", "Heff_mu_abcd"]),
        ("meta.json", None),
    ],
    "L3_FOPT": [
        (
            "data.npz",
            [
                "h_eff_4",
                "h_eff_4_processes",
                "process_labels",
                "J_mu",
                "mapping_residual",
                "J_mu_processes",
                "mapping_residual_processes",
                "n_k",
            ],
        ),
        ("meta.json", None),
    ],
}


def try_load_stateset(stage_dir: Path, *, level: str, n_ele: int) -> dict[str, Any] | None:
    if not stage_dir.exists():
        return None
    try:
        payload = load_npz_checked(
            stage_dir / f"{level}.npz",
            ["V_fock", "coef_F0", "coef_F2", "coef_F4", "coef_F6"],
        )
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        result = {
            "V_fock": np.asarray(payload["V_fock"], dtype=np.complex128),
            "labels": meta.get("labels", []),
            "basis_id": meta.get("basis_id", f"fock14_n{n_ele}_lex_v1"),
            "n_ele": n_ele,
            "coef_F0": np.asarray(payload["coef_F0"], dtype=float),
            "coef_F2": np.asarray(payload["coef_F2"], dtype=float),
            "coef_F4": np.asarray(payload["coef_F4"], dtype=float),
            "coef_F6": np.asarray(payload["coef_F6"], dtype=float),
            "state_order_id": meta.get("state_order_id", f"{level.lower()}_canonical_v1"),
            "orbital_order_id": meta.get("orbital_order_id"),
            "j_order_id": meta.get("j_order_id"),
        }
        if "coef_zeta" in payload:
            result["coef_zeta"] = np.asarray(payload["coef_zeta"], dtype=float)

        # Recover LSJM/LSMS physics payload from metadata if present.
        inputs_summary = meta.get("inputs_summary", {})
        if isinstance(inputs_summary, dict):
            physics: dict[str, float] = {}
            for key in ("F2", "F4", "F6"):
                value = inputs_summary.get(key)
                if value is not None:
                    physics[key] = float(value)
            if physics:
                result["physics"] = physics
        return result
    except Exception as exc:
        logger.warning("Invalid cached %s artifact at %s (%s), recomputing", level, stage_dir, exc)
        return None


def try_load_ion_ed(stage_dir: Path, *, expected_key: str | None = None) -> dict[str, Any] | None:
    try:
        payload = load_npz_checked(stage_dir / "states.npz", ["V_fock_ed", "energies"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("IONED key mismatch")
        V = np.asarray(payload["V_fock_ed"], dtype=np.complex128)
        result: dict[str, Any] = {
            "V_fock": V,
            "V_fock_ed": V,
            "energies": np.asarray(payload["energies"], dtype=float),
            "labels": meta.get("labels", []),
            "basis_id": meta.get("basis_id", ""),
            "n_ele": int(meta.get("n_ele", 0)),
            "n_orb": int(meta.get("n_orb", 14)),
            "state_order_id": meta.get("state_order_id", "ioned_energy_J_M_v1"),
            "j_order_id": meta.get("j_order_id", "ioned_J_M_v1"),
            "orbital_order_id": meta.get("orbital_order_id"),
            "physics": meta.get("physics", {}),
        }
        for key, dtype in (("J", float), ("M", float), ("energy_group", np.int64)):
            if key in payload:
                result[key] = np.asarray(payload[key], dtype=dtype)
        return result
    except Exception as exc:
        logger.warning("Invalid cached IONED artifact at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l0(stage_dir: Path, *, n_ele: int) -> dict[str, Any] | None:
    try:
        x = load_npz_checked(stage_dir / f"f_create_{n_ele}_to_{n_ele + 1}.npz", ["data"])["data"]
        y = load_npz_checked(stage_dir / f"f_create_{n_ele - 1}_to_{n_ele}.npz", ["data"])["data"]
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        return {
            "X": np.asarray(x, dtype=np.complex128),
            "Y": np.asarray(y, dtype=np.complex128),
            "n_ele": n_ele,
            "n_orb": 14,
        }
    except Exception as exc:
        logger.warning("Invalid cached L0 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l1(stage_dir: Path, *, expected_key: str | None = None) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["A", "B"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("L1 key mismatch")
        A = np.asarray(d["A"], dtype=np.complex128)
        B = np.asarray(d["B"], dtype=np.complex128)
        return {
            "A": A,
            "B": B,
            "n_orb": int(A.shape[0]),
            "n_u": int(A.shape[1]),
            "n_j": int(A.shape[2]),
            "n_v": int(B.shape[2]),
        }
    except Exception as exc:
        logger.warning("Invalid cached L1 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l2(stage_dir: Path, *, expected_key: str | None = None) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["M_A", "M_B"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("L2 key mismatch")
        M_A = np.asarray(d["M_A"], dtype=np.complex128)
        M_B = np.asarray(d["M_B"], dtype=np.complex128)
        return {
            "M_A": M_A,
            "M_B": M_B,
            "n_u": int(M_A.shape[0]),
            "n_v": int(M_A.shape[1]),
            "n_j": int(meta.get("n_j", M_A.shape[2])),
            "n_k": int(M_A.shape[2]),
        }
    except Exception as exc:
        logger.warning("Invalid cached L2 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l3_sopt(stage_dir: Path, *, expected_key: str | None = None) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["h_mu_abcd", "Heff_mu_abcd"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("SOPT L3 key mismatch")
        h = np.asarray(d["h_mu_abcd"], dtype=np.complex128)
        heff = np.asarray(d["Heff_mu_abcd"], dtype=np.complex128)
        result = {"h_mu_abcd": h, "Heff_mu_abcd": heff, "n_k": int(h.shape[0])}
        if "J_mu" in d:
            result["J_mu"] = np.asarray(d["J_mu"], dtype=float)
        if "mapping_residual" in d:
            result["mapping_residual"] = float(np.asarray(d["mapping_residual"]).item())
        return result
    except Exception as exc:
        logger.warning("Invalid cached SOPT L3 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l0_fopt(stage_dir: Path, *, n_ele: int) -> dict[str, Any] | None:
    try:
        f = try_load_l0(stage_dir, n_ele=n_ele)
        if f is None:
            return None
        p56 = load_npz_checked(stage_dir / "p_create_5_to_6.npz", ["data"])["data"]
        p45 = load_npz_checked(stage_dir / "p_create_4_to_5.npz", ["data"])["data"]
        return {
            "f": f,
            "p": {
                "P_raw_5_6": np.asarray(p56, dtype=np.complex128),
                "P_raw_4_5": np.asarray(p45, dtype=np.complex128),
                "n_orb_p": 6,
                "basis_id_p_6": "fock6_n6_lex_v1",
                "basis_id_p_5": "fock6_n5_lex_v1",
                "basis_id_p_4": "fock6_n4_lex_v1",
            },
        }
    except Exception as exc:
        logger.warning("Invalid cached FOPT L0 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l1_fopt(
    f_dir: Path,
    p_dirs: dict[int, dict[str, Path]],
    *,
    expected_key: str | None = None,
) -> dict[str, Any] | None:
    try:
        f_payload = load_npz_checked(f_dir / "data.npz", ["F_n_np1", "F_nm1_n"])
        meta = load_json_checked(f_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("FOPT L1 key mismatch")
        f_vertex = {
            "F_n_np1": np.asarray(f_payload["F_n_np1"], dtype=np.complex128),
            "F_nm1_n": np.asarray(f_payload["F_nm1_n"], dtype=np.complex128),
            "n_orb_f": int(np.asarray(f_payload["F_n_np1"]).shape[0]),
        }
        p_vertices: dict[int, dict[str, Any]] = {}
        for ligand, dirs in p_dirs.items():
            p56 = load_npz_checked(dirs["P_5_6"] / "data.npz", ["data"])["data"]
            p45 = load_npz_checked(dirs["P_4_5"] / "data.npz", ["data"])["data"]
            p_vertices[ligand] = {
                "P_5_6": np.asarray(p56, dtype=np.complex128),
                "P_4_5": np.asarray(p45, dtype=np.complex128),
                "n_orb_p": int(np.asarray(p56).shape[0]),
            }
        return {"f": {1: f_vertex, 2: f_vertex}, "p": p_vertices}
    except Exception as exc:
        logger.warning("Invalid cached FOPT L1 at %s (%s), recomputing", f_dir, exc)
        return None


def try_load_ligand(stage_dir: Path) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["V_fock", "coef_Delta", "coef_U_p", "coef_lambda_p"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        return {
            "V_fock": np.asarray(d["V_fock"], dtype=np.complex128),
            "coef_Delta": np.asarray(d["coef_Delta"], dtype=float),
            "coef_U_p": np.asarray(d["coef_U_p"], dtype=float),
            "coef_lambda_p": np.asarray(d["coef_lambda_p"], dtype=float),
            "n_ele": int(meta.get("n_ele", 0)),
            "soc": bool(meta.get("soc", False)),
            "dim": int(np.asarray(d["V_fock"]).shape[1]),
            "basis_id": meta.get("basis_id", ""),
        }
    except Exception as exc:
        logger.warning("Invalid cached ligand artifact at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l2_fopt(stage_dir: Path, *, expected_key: str | None = None) -> dict[tuple[int, int, str], dict[str, Any]] | None:
    try:
        result: dict[tuple[int, int, str], dict[str, Any]] = {}
        for fs in (1, 2):
            for lig in (1, 2):
                d = load_npz_checked(
                    stage_dir / f"V_plus_f{fs}_lig{lig}.npz",
                    ["fn_p6", "fn_p5", "fnm1_p6"],
                )
                for sector in ("fn_p6", "fn_p5", "fnm1_p6"):
                    arr = np.asarray(d[sector], dtype=np.complex128)
                    result[(fs, lig, sector)] = {
                        "V_plus": arr,
                        "dim_high_f": int(arr.shape[0]),
                        "dim_high_p": int(arr.shape[1]),
                        "dim_low_f": int(arr.shape[2]),
                        "dim_low_p": int(arr.shape[3]),
                    }
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("FOPT L2 key mismatch")
        return result
    except Exception as exc:
        logger.warning("Invalid cached FOPT L2 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l3_fopt(stage_dir: Path, *, expected_key: str | None = None) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(
            stage_dir / "data.npz",
            [
                "h_eff_4",
                "h_eff_4_processes",
                "process_labels",
                "J_mu",
                "mapping_residual",
                "J_mu_processes",
                "mapping_residual_processes",
                "n_k",
            ],
        )
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        if expected_key is not None and meta.get("key") != expected_key:
            raise ValueError("FOPT L3 key mismatch")
        h = np.asarray(d["h_eff_4"], dtype=np.complex128)
        return {
            "h_eff_4": h,
            "h_eff_4_processes": np.asarray(d["h_eff_4_processes"], dtype=np.complex128),
            "process_labels": np.asarray(d["process_labels"]).astype(str),
            "J_mu": np.asarray(d["J_mu"], dtype=float),
            "mapping_residual": float(np.asarray(d["mapping_residual"]).item()),
            "J_mu_processes": np.asarray(d["J_mu_processes"], dtype=float),
            "mapping_residual_processes": np.asarray(d["mapping_residual_processes"], dtype=float),
            "n_k": int(np.asarray(d["n_k"]).item()),
        }
    except Exception as exc:
        logger.warning("Invalid cached FOPT L3 at %s (%s), recomputing", stage_dir, exc)
        return None


def persist_stateset(
    stage_dir: Path,
    result: dict[str, Any],
    *,
    level: str,
    n_ele: int,
    cfg: dict[str, Any],
    physics: dict[str, Any],
    r42: float,
    r62: float,
) -> None:
    output_root = cfg["paths"]["output_root"]
    payload: dict[str, Any] = {
        "V_fock": result["V_fock"],
        "coef_F0": result["coef_F0"],
        "coef_F2": result["coef_F2"],
        "coef_F4": result["coef_F4"],
        "coef_F6": result["coef_F6"],
    }
    if "coef_zeta" in result:
        payload["coef_zeta"] = result["coef_zeta"]
    content_hash = atomic_write_npz(stage_dir / f"{level}.npz", **payload)

    module_name = "representations.lsms" if level == "LMSM" else "representations.lsjm"
    meta = build_meta(
        module=module_name,
        level=level,
        key=level_key(level, n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={
            "n": n_ele,
            "F2": physics.get("F2"),
            "F4": physics.get("F4"),
            "F6": physics.get("F6"),
        },
        tensor_name="V_fock",
        physical_meaning=f"{level} basis states in Fock representation",
        basis_id=result["basis_id"],
        index_definition="(alpha_fock,state)",
        logical_shape=[*result["V_fock"].shape],
        payload_files=[f"{level}.npz"],
        extra={
            "labels": result["labels"],
            "state_order_id": result.get("state_order_id", ""),
            "orbital_order_id": result.get(
                "orbital_order_id",
                "f14_m-3..3_sigma(-1/2,+1/2)_interleaved_v1",
            ),
            "j_order_id": result.get("j_order_id"),
            "n_orb": result.get("n_orb"),
            "n_ele": result.get("n_ele"),
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": level_key(level, n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
            "module": module_name,
            "level": level,
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "content_hash": content_hash,
        },
    )


def persist_ion_ed(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
) -> None:
    output_root = cfg["paths"]["output_root"]
    physics = result.get("physics", {})
    payload = {
        "V_fock_ed": result["V_fock_ed"],
        "energies": np.asarray(result["energies"], dtype=float),
        "J": np.asarray(result.get("J", []), dtype=float),
        "M": np.asarray(result.get("M", []), dtype=float),
        "energy_group": np.asarray(result.get("energy_group", []), dtype=np.int64),
    }
    content_hash = atomic_write_npz(stage_dir / "states.npz", **payload)
    meta = build_meta(
        module="spectrum.ion_ed",
        level="IONED",
        key=level_key("IONED", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={
            "n": n_ele,
            "U": float(physics.get("F0", 0.0)),
            "F2": float(physics.get("F2", 0.0)),
            "F4": float(physics.get("F4", 0.0)),
            "F6": float(physics.get("F6", 0.0)),
            "zeta": float(physics.get("zeta", 0.0)),
            "offset": float(physics.get("offset", 0.0)),
            "includes_hcef": bool(physics.get("includes_hcef", False)),
        },
        tensor_name="V_fock_ed, energies",
        physical_meaning="Full single-ion ED eigenstates of H_int + H_soc (+ H_cef when supplied)",
        basis_id=str(result.get("basis_id", f"fock14_n{n_ele}_lex_v1")),
        index_definition="V_fock_ed(alpha_fock,state), energies(state)",
        logical_shape=[*np.asarray(result["V_fock_ed"]).shape],
        payload_files=["states.npz"],
        extra={
            "labels": result.get("labels", []),
            "state_order_id": result.get("state_order_id", "ioned_energy_J_M_v1"),
            "j_order_id": result.get("j_order_id", "ioned_J_M_v1"),
            "orbital_order_id": result.get(
                "orbital_order_id",
                "f14_m-3..3_sigma(-1/2,+1/2)_interleaved_v1",
            ),
            "n_ele": int(result.get("n_ele", n_ele)),
            "n_orb": int(result.get("n_orb", 14)),
            "physics": physics,
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": meta["key"],
            "module": "spectrum.ion_ed",
            "level": "IONED",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "U": float(physics.get("F0", 0.0)),
            "Jh": float(_fsite_for_index(cfg, n_ele).get("Jh", 0.0)),
            "zeta": float(physics.get("zeta", 0.0)),
            "content_hash": content_hash,
        },
    )


def persist_l0(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
) -> None:
    output_root = cfg["paths"]["output_root"]
    hash_x = atomic_write_npz(stage_dir / f"f_create_{n_ele}_to_{n_ele + 1}.npz", data=result["X"])
    hash_y = atomic_write_npz(stage_dir / f"f_create_{n_ele - 1}_to_{n_ele}.npz", data=result["Y"])
    meta = build_meta(
        module="sopt.precompute",
        level="L0",
        key=level_key("L0", n_ele=n_ele, r42=0.0, r62=0.0, cfg=cfg),
        inputs_summary={"n": n_ele},
        tensor_name="f_create transitions",
        physical_meaning="Fock basis f-shell transition tensors",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="data(kappa,high,low)",
        logical_shape=[*result["X"].shape],
        payload_files=[f"f_create_{n_ele}_to_{n_ele + 1}.npz", f"f_create_{n_ele - 1}_to_{n_ele}.npz"],
        extra={"numerics_meta": numerics_meta()},
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": level_key("L0", n_ele=n_ele, r42=0.0, r62=0.0, cfg=cfg),
            "module": "sopt.precompute",
            "level": "L0",
            "path": str(stage_dir),
            "n": n_ele,
            "content_hash": f"{hash_x}:{hash_y}",
        },
    )


def persist_l0_fopt(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
) -> None:
    persist_l0(stage_dir, cfg, result["f"], n_ele=n_ele)
    hash_p56 = atomic_write_npz(stage_dir / "p_create_5_to_6.npz", data=result["p"]["P_raw_5_6"])
    hash_p45 = atomic_write_npz(stage_dir / "p_create_4_to_5.npz", data=result["p"]["P_raw_4_5"])
    append_index_record(
        cfg["paths"]["output_root"],
        {
            "key": level_key("L0", n_ele=n_ele, r42=0.0, r62=0.0, cfg=cfg),
            "module": "fopt.precompute",
            "level": "L0",
            "path": str(stage_dir),
            "n": n_ele,
            "content_hash": f"{hash_p56}:{hash_p45}",
        },
    )


def persist_ligand(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    soc: bool,
) -> None:
    content_hash = atomic_write_npz(
        stage_dir / "data.npz",
        V_fock=result["V_fock"],
        coef_Delta=result["coef_Delta"],
        coef_U_p=result["coef_U_p"],
        coef_lambda_p=result["coef_lambda_p"],
    )
    meta = build_meta(
        module="spectrum.ligand",
        level="ligand",
        key=f"ligand|n={int(result['n_ele'])}|soc={bool(soc)}|sv={cfg.get('standard_version')}",
        inputs_summary={"n_ele": int(result["n_ele"]), "soc": bool(soc)},
        tensor_name="V_fock, ligand coefficients",
        physical_meaning="Ligand p-shell spectrum coefficients",
        basis_id=str(result.get("basis_id", "")),
        index_definition="V_fock(fock,state)",
        logical_shape=[*result["V_fock"].shape],
        payload_files=["data.npz"],
        extra={
            "n_ele": int(result["n_ele"]),
            "soc": bool(soc),
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        cfg["paths"]["output_root"],
        {
            "key": meta["key"],
            "module": "spectrum.ligand",
            "level": "ligand",
            "path": str(stage_dir),
            "n": int(result["n_ele"]),
            "content_hash": content_hash,
        },
    )


def persist_l1_fopt(
    f_dir: Path,
    p_dirs: dict[int, dict[str, Path]],
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
    main_subspace: dict[str, Any] | None = None,
) -> None:
    f_vertex = result["f"][1]
    subspace_id = "soc_lowest_hunds_v1"
    subspace_meta: dict[str, Any] = {}
    if isinstance(main_subspace, dict):
        subspace_id = str(main_subspace.get("subspace_id", subspace_id))
        subspace_meta = dict(main_subspace.get("meta", {}))
        if subspace_id == "soc_lowest_hunds_v1" and not subspace_meta:
            subspace_meta = {
                "alpha0": main_subspace.get("alpha0"),
                "L0": main_subspace.get("L0"),
                "S0": main_subspace.get("S0"),
                "J0": main_subspace.get("J0"),
                "n_j": main_subspace.get("n_j"),
            }
    hash_f = atomic_write_npz(
        f_dir / "data.npz",
        F_n_np1=f_vertex["F_n_np1"],
        F_nm1_n=f_vertex["F_nm1_n"],
    )
    meta = build_meta(
        module="fopt.precompute",
        level="L1",
        key=level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={"n": n_ele, "branch": "fopt"},
        tensor_name="F_n_np1/F_nm1_n vertices",
        physical_meaning="FOPT f-shell local transition vertices",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="F_n_np1(kappa,u,j), F_nm1_n(kappa,j,v)",
        logical_shape=[*f_vertex["F_n_np1"].shape],
        payload_files=["data.npz"],
        extra={
            "fn_ground_subspace_id": subspace_id,
            "main_subspace_meta": subspace_meta,
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(f_dir / "meta.json", meta)

    p_hashes: list[str] = [hash_f]
    for ligand, p_vertex in result["p"].items():
        for key, filename_key in (("P_5_6", "P_5_6"), ("P_4_5", "P_4_5")):
            stage_dir = p_dirs[ligand][filename_key]
            p_hashes.append(atomic_write_npz(stage_dir / "data.npz", data=p_vertex[key]))
            p_meta = build_meta(
                module="fopt.precompute",
                level="L1",
                key=f"{level_key('L1', n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)}|ligand={ligand}|{key}",
                inputs_summary={"ligand": ligand, "tensor": key},
                tensor_name=key,
                physical_meaning="FOPT ligand p-shell local transition vertex",
                basis_id="fock6_p_shell_v1",
                index_definition="data(kappa,high,low)",
                logical_shape=[*p_vertex[key].shape],
                payload_files=["data.npz"],
                extra={"numerics_meta": numerics_meta()},
            )
            atomic_write_json(stage_dir / "meta.json", p_meta)
    append_index_record(
        cfg["paths"]["output_root"],
        {
            "key": level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
            "module": "fopt.precompute",
            "level": "L1",
            "path": str(f_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "content_hash": ":".join(p_hashes),
        },
    )


def persist_l2_fopt(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[tuple[int, int, str], dict[str, Any]],
    *,
    n_ele: int,
    r42: float,
    r62: float,
) -> None:
    hashes: list[str] = []
    kramer_name = str(cfg.get("runtime", {}).get("kramer_name", ""))
    for fs in (1, 2):
        for lig in (1, 2):
            hashes.append(
                atomic_write_npz(
                    stage_dir / f"V_plus_f{fs}_lig{lig}.npz",
                    fn_p6=result[(fs, lig, "fn_p6")]["V_plus"],
                    fn_p5=result[(fs, lig, "fn_p5")]["V_plus"],
                    fnm1_p6=result[(fs, lig, "fnm1_p6")]["V_plus"],
                )
            )
    meta = build_meta(
        module="fopt.contraction",
        level="L2",
        key=level_key("L2", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary=build_resolved_inputs_summary(
            cfg,
            n_ele=n_ele,
            r42=r42,
            r62=r62,
        ),
        tensor_name="V_plus",
        physical_meaning="FOPT projected active-pair p-to-f hopping blocks",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="V_plus(f_out,p_out,f_in,p_in), with the f^n leg projected to k",
        logical_shape=[],
        payload_files=[f"V_plus_f{fs}_lig{lig}.npz" for fs in (1, 2) for lig in (1, 2)],
        extra={
            "hopping_name": cfg.get("inputs", {}).get("hopping_name"),
            "kramer_name": kramer_name,
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        cfg["paths"]["output_root"],
        {
            "key": meta["key"],
            "module": "fopt.contraction",
            "level": "L2",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "hopping_name": cfg.get("inputs", {}).get("hopping_name"),
            "kramer_name": kramer_name,
            "content_hash": ":".join(hashes),
        },
    )


def persist_l3_fopt(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
) -> None:
    h = np.asarray(result["h_eff_4"], dtype=np.complex128)
    h_processes = np.asarray(result["h_eff_4_processes"], dtype=np.complex128)
    process_labels = np.asarray(result["process_labels"])
    j_mu = np.asarray(result["J_mu"], dtype=float)
    j_mu_processes = np.asarray(result["J_mu_processes"], dtype=float)
    residual = np.asarray(result["mapping_residual"], dtype=float)
    residual_processes = np.asarray(result["mapping_residual_processes"], dtype=float)
    kramer_name = str(cfg.get("runtime", {}).get("kramer_name", ""))
    content_hash = atomic_write_npz(
        stage_dir / "data.npz",
        h_eff_4=h,
        h_eff_4_processes=h_processes,
        process_labels=process_labels,
        J_mu=j_mu,
        mapping_residual=residual,
        J_mu_processes=j_mu_processes,
        mapping_residual_processes=residual_processes,
        n_k=np.asarray(result["n_k"], dtype=np.int64),
    )
    write_exchange_matrix_txt(
        stage_dir / "exchange.txt",
        J_mu=j_mu,
        mapping_residual=float(residual),
        process_labels=process_labels,
        J_mu_processes=j_mu_processes,
        mapping_residual_processes=residual_processes,
    )
    meta = build_meta(
        module="fopt.contraction",
        level="L3",
        key=level_key("L3", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary=build_resolved_inputs_summary(
            cfg,
            n_ele=n_ele,
            r42=r42,
            r62=r62,
        ),
        tensor_name="h_eff_4, h_eff_4_processes, J_mu, J_mu_processes",
        physical_meaning="FOPT fourth-order effective Hamiltonian and spin-1/2 exchange, total and process-resolved",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="(f1_fin,f2_fin,f1_init,f2_init)",
        logical_shape=[*h.shape],
        payload_files=["data.npz"],
        extra={
            "human_readable_files": ["exchange.txt"],
            "process_labels": process_labels.tolist(),
            "mapping_residual": float(residual),
            "mapping_residual_processes": residual_processes.tolist(),
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        cfg["paths"]["output_root"],
        {
            "key": meta["key"],
            "module": "fopt.contraction",
            "level": "L3",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "U": float(cfg["fsite"]["U"]),
            "Jh": float(cfg["fsite"]["Jh"]),
            "hopping_name": cfg.get("inputs", {}).get("hopping_name"),
            "kramer_name": kramer_name,
            "content_hash": content_hash,
        },
    )


def persist_l1(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
    soc0: dict[str, Any],
) -> None:
    output_root = cfg["paths"]["output_root"]
    content_hash = atomic_write_npz(stage_dir / "data.npz", A=result["A"], B=result["B"])
    subspace_id = str(soc0.get("subspace_id", "soc_lowest_hunds_v1"))
    subspace_meta = dict(soc0.get("meta", {}))
    if subspace_id == "soc_lowest_hunds_v1":
        subspace_meta = {
            "alpha0": soc0["alpha0"],
            "L0": soc0["L0"],
            "S0": soc0["S0"],
            "J0": soc0["J0"],
            "n_j": soc0["n_j"],
        }
    meta = build_meta(
        module="sopt.precompute",
        level="L1",
        key=level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={
            "n": n_ele,
            "n_u": result["n_u"],
            "n_v": result["n_v"],
            "n_j": result["n_j"],
        },
        tensor_name="A/B vertices",
        physical_meaning="LSJM-rotated local transition vertices",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="A(kappa,u,j), B(kappa,j,v)",
        logical_shape=[*result["A"].shape],
        payload_files=["data.npz"],
        extra={
            "vertex_axis_order_id": "A(kappa,u,j),B(kappa,j,v)",
            "fn_ground_subspace_id": subspace_id,
            "main_subspace_meta": subspace_meta,
            "soc0_meta": subspace_meta if subspace_id == "soc_lowest_hunds_v1" else {},
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
            "module": "sopt.precompute",
            "level": "L1",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "content_hash": content_hash,
        },
    )


def persist_l2(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
) -> None:
    output_root = cfg["paths"]["output_root"]
    source_names = cfg.get("inputs", {})
    hopping_name = str(source_names.get("hopping_name", ""))
    kramer_name = str(cfg.get("runtime", {}).get("kramer_name", ""))
    content_hash = atomic_write_npz(stage_dir / "data.npz", M_A=result["M_A"], M_B=result["M_B"])
    meta = build_meta(
        module="sopt.contraction",
        level="L2",
        key=level_key("L2", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary=build_resolved_inputs_summary(
            cfg,
            n_ele=n_ele,
            r42=r42,
            r62=r62,
        ),
        tensor_name="M_A/M_B",
        physical_meaning="Projected route factors for denominator contraction",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="M_A(u,v,a,b), M_B(r,s,a,b)",
        logical_shape=[*result["M_A"].shape],
        payload_files=["data.npz"],
        extra={
            "axis_order_id": {"M_A": "uvab_v1", "M_B": "rsab_v1"},
            "hopping_name": hopping_name,
            "kramer_name": kramer_name,
            "n_j": int(result.get("n_j", 0)),
            "n_k": int(result.get("n_k", np.asarray(result["M_A"]).shape[2])),
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": level_key("L2", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
            "module": "sopt.contraction",
            "level": "L2",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "hopping_name": hopping_name,
            "kramer_name": kramer_name,
            "content_hash": content_hash,
        },
    )


def persist_l3_sopt(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
    labels: np.ndarray,
) -> None:
    output_root = cfg["paths"]["output_root"]
    fsite = cfg["fsite"]
    hopping_name = str(cfg.get("inputs", {}).get("hopping_name", ""))
    kramer_name = str(cfg.get("runtime", {}).get("kramer_name", ""))
    payload = {
        "h_mu_abcd": result["h_mu_abcd"],
        "Heff_mu_abcd": result["Heff_mu_abcd"],
        "labels_abcd": labels,
    }
    if "J_mu" in result:
        payload["J_mu"] = result["J_mu"]
        payload["mapping_residual"] = np.asarray(result["mapping_residual"], dtype=float)
    content_hash = atomic_write_npz(stage_dir / "data.npz", **payload)
    human_readable_files: list[str] = []
    if "J_mu" in result and "mapping_residual" in result:
        write_exchange_matrix_txt(
            stage_dir / "exchange.txt",
            J_mu=result["J_mu"],
            mapping_residual=float(result["mapping_residual"]),
        )
        human_readable_files.append("exchange.txt")
    meta = build_meta(
        module="sopt.contraction",
        level="L3",
        key=level_key("L3", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary=build_resolved_inputs_summary(
            cfg,
            n_ele=n_ele,
            r42=r42,
            r62=r62,
        ),
        tensor_name="h_mu_abcd, Heff_mu_abcd",
        physical_meaning="Final projected effective exchange tensor",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="(c,d,a,b) with labels_abcd=(a,b,c,d)",
        logical_shape=[*result["Heff_mu_abcd"].shape],
        payload_files=["data.npz"],
        extra={
            "labels_order_id": "abcd_lex_v1",
            "jmu_available": "J_mu" in result,
            "mapping_residual": result.get("mapping_residual"),
            "human_readable_files": human_readable_files,
            "hopping_name": hopping_name,
            "kramer_name": kramer_name,
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": level_key("L3", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
            "module": "sopt.contraction",
            "level": "L3",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "U": float(fsite["U"]),
            "Jh": float(fsite["Jh"]),
            "zeta": float(fsite["zeta"]),
            "hopping_name": hopping_name,
            "kramer_name": kramer_name,
            "content_hash": content_hash,
        },
    )
