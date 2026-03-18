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
)
from fexchange.pipeline.keys import level_key, three_sectors
from fexchange.utils.numerics import numerics_meta

logger = logging.getLogger("fexchange")

# Artifact file spec: maps level -> list of (filename, minimum_required_npz_keys_or_None_for_json).
# These are the minimum keys needed for loading/validation, not the full set of stored keys.
# Used by both try_load_* and validate_upstream_artifacts to maintain one source of truth.
ARTIFACT_FILE_SPEC: dict[str, list[tuple[str, list[str] | None]]] = {
    "LMSM": [
        ("V.npz", ["V_fock"]),
        ("E_terms.npz", ["coef_F0", "coef_F2", "coef_F4", "coef_F6"]),
        ("meta.json", None),
    ],
    "LSJM": [
        ("V.npz", ["V_fock"]),
        ("E_terms.npz", ["coef_F0", "coef_F2", "coef_F4", "coef_F6", "coef_zeta"]),
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
        ("data.npz", ["h_pre_j_mu"]),
        ("meta.json", None),
    ],
    "L4": [
        ("data.npz", ["h_mu_abcd", "Heff_mu_abcd"]),
        ("meta.json", None),
    ],
}


def try_load_stateset(stage_dir: Path, *, level: str, n_ele: int) -> dict[str, Any] | None:
    if not stage_dir.exists():
        return None
    try:
        V = load_npz_checked(stage_dir / "V.npz", ["V_fock"])["V_fock"]
        E_terms = load_npz_checked(
            stage_dir / "E_terms.npz",
            ["coef_F0", "coef_F2", "coef_F4", "coef_F6"],
        )
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        result = {
            "V_fock": np.asarray(V, dtype=np.complex128),
            "labels": meta.get("labels", []),
            "basis_id": meta.get("basis_id", f"fock14_n{n_ele}_lex_v1"),
            "n_ele": n_ele,
            "coef_F0": np.asarray(E_terms["coef_F0"], dtype=float),
            "coef_F2": np.asarray(E_terms["coef_F2"], dtype=float),
            "coef_F4": np.asarray(E_terms["coef_F4"], dtype=float),
            "coef_F6": np.asarray(E_terms["coef_F6"], dtype=float),
            "state_order_id": meta.get("state_order_id", f"{level.lower()}_canonical_v1"),
            "orbital_order_id": meta.get("orbital_order_id"),
            "j_order_id": meta.get("j_order_id"),
        }
        if "coef_zeta" in E_terms:
            result["coef_zeta"] = np.asarray(E_terms["coef_zeta"], dtype=float)

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


def try_load_l0(stage_dir: Path, *, n_ele: int) -> dict[str, Any] | None:
    try:
        x = load_npz_checked(stage_dir / f"n{n_ele}{n_ele + 1}.npz", ["X"])["X"]
        y = load_npz_checked(stage_dir / f"n{n_ele - 1}{n_ele}.npz", ["Y"])["Y"]
        for sec in three_sectors(n_ele):
            meta = load_json_checked(stage_dir / f"meta_n{sec}.json")
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


def try_load_l1(stage_dir: Path) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["A", "B"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
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


def try_load_l2(stage_dir: Path) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["M_A", "M_B"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        M_A = np.asarray(d["M_A"], dtype=np.complex128)
        M_B = np.asarray(d["M_B"], dtype=np.complex128)
        return {
            "M_A": M_A,
            "M_B": M_B,
            "n_u": int(M_A.shape[0]),
            "n_v": int(M_A.shape[1]),
            "n_j": int(M_A.shape[2]),
        }
    except Exception as exc:
        logger.warning("Invalid cached L2 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l3(stage_dir: Path) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["h_pre_j_mu"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        h = np.asarray(d["h_pre_j_mu"], dtype=np.complex128)
        return {"h_pre_j_mu": h, "n_j": int(h.shape[0])}
    except Exception as exc:
        logger.warning("Invalid cached L3 at %s (%s), recomputing", stage_dir, exc)
        return None


def try_load_l4(stage_dir: Path) -> dict[str, Any] | None:
    try:
        d = load_npz_checked(stage_dir / "data.npz", ["h_mu_abcd", "Heff_mu_abcd"])
        meta = load_json_checked(stage_dir / "meta.json")
        validate_meta(meta)
        h = np.asarray(d["h_mu_abcd"], dtype=np.complex128)
        heff = np.asarray(d["Heff_mu_abcd"], dtype=np.complex128)
        result = {"h_mu_abcd": h, "Heff_mu_abcd": heff, "n_k": int(h.shape[0])}
        if "J_mu" in d:
            result["J_mu"] = np.asarray(d["J_mu"], dtype=float)
        if "mapping_residual" in d:
            result["mapping_residual"] = float(np.asarray(d["mapping_residual"]).item())
        return result
    except Exception as exc:
        logger.warning("Invalid cached L4 at %s (%s), recomputing", stage_dir, exc)
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
    hash_v = atomic_write_npz(stage_dir / "V.npz", V_fock=result["V_fock"])
    e_payload: dict[str, Any] = {
        "coef_F0": result["coef_F0"],
        "coef_F2": result["coef_F2"],
        "coef_F4": result["coef_F4"],
        "coef_F6": result["coef_F6"],
    }
    if "coef_zeta" in result:
        e_payload["coef_zeta"] = result["coef_zeta"]
    hash_e = atomic_write_npz(stage_dir / "E_terms.npz", **e_payload)

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
        payload_files=["V.npz", "E_terms.npz"],
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
            "content_hash": f"{hash_v}:{hash_e}",
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
    hash_x = atomic_write_npz(stage_dir / f"n{n_ele}{n_ele + 1}.npz", X=result["X"])
    hash_y = atomic_write_npz(stage_dir / f"n{n_ele - 1}{n_ele}.npz", Y=result["Y"])
    for sec in three_sectors(n_ele):
        meta = build_meta(
            module="sopt.precompute",
            level="L0",
            key=level_key("L0", n_ele=n_ele, r42=0.0, r62=0.0, cfg=cfg),
            inputs_summary={"n": n_ele},
            tensor_name="X/Y transitions",
            physical_meaning="Fock basis transition tensors",
            basis_id=f"fock14_n{sec}_lex_v1",
            index_definition="X(kappa,alpha,beta), Y(kappa,beta,gamma)",
            logical_shape=[*result["X"].shape],
            payload_files=[f"n{n_ele}{n_ele + 1}.npz", f"n{n_ele - 1}{n_ele}.npz"],
            extra={"numerics_meta": numerics_meta()},
        )
        atomic_write_json(stage_dir / f"meta_n{sec}.json", meta)
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
            "fn_ground_subspace_id": "soc_lowest_hunds_v1",
            "soc0_meta": {
                "alpha0": soc0["alpha0"],
                "L0": soc0["L0"],
                "S0": soc0["S0"],
                "J0": soc0["J0"],
                "n_j": soc0["n_j"],
            },
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
    content_hash = atomic_write_npz(stage_dir / "data.npz", M_A=result["M_A"], M_B=result["M_B"])
    meta = build_meta(
        module="sopt.contraction",
        level="L2",
        key=level_key("L2", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={"n": n_ele},
        tensor_name="M_A/M_B",
        physical_meaning="Route factors for denominator contraction",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="M_A(u,v,j1,j2), M_B(r,s,j1,j2)",
        logical_shape=[*result["M_A"].shape],
        payload_files=["data.npz"],
        extra={
            "axis_order_id": {"M_A": "uvj1j2_v1", "M_B": "rsj1j2_v1"},
            "hopping_name": cfg["sources"]["hopping_name"],
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
            "hopping_name": cfg["sources"]["hopping_name"],
            "content_hash": content_hash,
        },
    )


def persist_l3(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
) -> None:
    output_root = cfg["paths"]["output_root"]
    sopt = cfg["sopt"]
    content_hash = atomic_write_npz(stage_dir / "data.npz", h_pre_j_mu=result["h_pre_j_mu"])
    meta = build_meta(
        module="sopt.contraction",
        level="L3",
        key=level_key("L3", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={
            "n": n_ele,
            "U": float(sopt["U"]),
            "Jh": float(sopt["Jh"]),
            "zeta": float(sopt["zeta"]),
            "hopping_name": cfg["sources"]["hopping_name"],
        },
        tensor_name="h_pre_j_mu",
        physical_meaning="Intermediate projected kernel before W projection",
        basis_id=f"fock14_n{n_ele}_lex_v1",
        index_definition="(j3,j4,j1,j2)",
        logical_shape=[*result["h_pre_j_mu"].shape],
        payload_files=["data.npz"],
        extra={"numerics_meta": numerics_meta()},
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
            "U": float(sopt["U"]),
            "Jh": float(sopt["Jh"]),
            "zeta": float(sopt["zeta"]),
            "hopping_name": cfg["sources"]["hopping_name"],
            "content_hash": content_hash,
        },
    )


def persist_l4(
    stage_dir: Path,
    cfg: dict[str, Any],
    result: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
    labels: np.ndarray,
    W: np.ndarray,
) -> None:
    output_root = cfg["paths"]["output_root"]
    sopt = cfg["sopt"]
    payload = {
        "h_mu_abcd": result["h_mu_abcd"],
        "Heff_mu_abcd": result["Heff_mu_abcd"],
        "labels_abcd": labels,
        "W": W,
    }
    if "J_mu" in result:
        payload["J_mu"] = result["J_mu"]
        payload["mapping_residual"] = np.asarray(result["mapping_residual"], dtype=float)
    content_hash = atomic_write_npz(stage_dir / "data.npz", **payload)
    meta = build_meta(
        module="sopt.contraction",
        level="L4",
        key=level_key("L4", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
        inputs_summary={
            "n": n_ele,
            "U": float(sopt["U"]),
            "Jh": float(sopt["Jh"]),
            "zeta": float(sopt["zeta"]),
            "hopping_name": cfg["sources"]["hopping_name"],
            "kramer_name": cfg["sources"]["kramer_name"],
        },
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
            "hopping_name": cfg["sources"]["hopping_name"],
            "kramer_name": cfg["sources"]["kramer_name"],
            "numerics_meta": numerics_meta(),
        },
    )
    atomic_write_json(stage_dir / "meta.json", meta)
    append_index_record(
        output_root,
        {
            "key": level_key("L4", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg),
            "module": "sopt.contraction",
            "level": "L4",
            "path": str(stage_dir),
            "n": n_ele,
            "r42": r42,
            "r62": r62,
            "U": float(sopt["U"]),
            "Jh": float(sopt["Jh"]),
            "zeta": float(sopt["zeta"]),
            "hopping_name": cfg["sources"]["hopping_name"],
            "kramer_name": cfg["sources"]["kramer_name"],
            "content_hash": content_hash,
        },
    )
