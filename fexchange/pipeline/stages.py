"""
Stage ensure functions for LMSM/LSJM/L0/L1/L2/L3/L4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fexchange.io.disk import build_stage_path
from fexchange.io.matrix_io import load_hopping_matrix
from fexchange.models.energy import compute_intermediate_energies
from fexchange.models.ground_doublets import load_or_build_W
from fexchange.pipeline.artifacts import (
    persist_l0,
    persist_l1,
    persist_l2,
    persist_l3,
    persist_l4,
    persist_stateset,
    try_load_l0,
    try_load_l1,
    try_load_l2,
    try_load_l3,
    try_load_l4,
    try_load_stateset,
)
from fexchange.pipeline.keys import labels_abcd_lex, three_sectors
from fexchange.pipeline.validation import validate_labels_abcd
from fexchange.representations.lsjm import build_lsjm, select_soc_lowest_subspace
from fexchange.representations.lsms import build_lsms
from fexchange.utils.errors import InputError
from fexchange.utils.parallel import ParallelContext


def ensure_lsms_all_three(
    cfg: dict[str, Any],
    ctx: ParallelContext,
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    output_root = cfg["paths"]["output_root"]
    physics = cfg.get("physics", {})
    for sec in three_sectors(n_ele):
        key = f"lsms_{sec}"
        if key in state:
            continue
        stage_dir = build_stage_path(output_root, "LMSM", n=sec, r42=r42, r62=r62, scheme=scheme)
        loaded = try_load_stateset(stage_dir, level="LMSM", n_ele=sec)
        if loaded is not None:
            state[key] = loaded
            continue

        result = build_lsms(sec, n_orb, r42=r42, r62=r62)
        state[key] = result
        persist_stateset(
            stage_dir,
            result,
            level="LMSM",
            n_ele=sec,
            cfg=cfg,
            physics=physics,
            r42=r42,
            r62=r62,
            ctx=ctx,
        )


def ensure_lsjm_all_three(
    cfg: dict[str, Any],
    ctx: ParallelContext,
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    output_root = cfg["paths"]["output_root"]
    physics = cfg.get("physics", {})
    ensure_lsms_all_three(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)

    for sec in three_sectors(n_ele):
        key = f"lsjm_{sec}"
        if key in state:
            continue
        stage_dir = build_stage_path(output_root, "LSJM", n=sec, r42=r42, r62=r62, scheme=scheme)
        loaded = try_load_stateset(stage_dir, level="LSJM", n_ele=sec)
        if loaded is not None:
            state[key] = loaded
            continue

        lsms = state[f"lsms_{sec}"]
        result = build_lsjm(lsms, n_orb)
        state[key] = result
        persist_stateset(
            stage_dir,
            result,
            level="LSJM",
            n_ele=sec,
            cfg=cfg,
            physics=physics,
            r42=r42,
            r62=r62,
            ctx=ctx,
        )


def ensure_l0(
    cfg: dict[str, Any],
    ctx: ParallelContext,
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
) -> None:
    from fexchange.sopt.precompute import build_L0

    if "l0" in state:
        return
    output_root = cfg["paths"]["output_root"]
    stage_dir = build_stage_path(output_root, "fock")
    loaded = try_load_l0(stage_dir, n_ele=n_ele)
    if loaded is not None:
        state["l0"] = loaded
        return

    result = build_L0(n_ele, n_orb)
    state["l0"] = result
    persist_l0(cfg, ctx, result, n_ele=n_ele)


def ensure_l1(
    cfg: dict[str, Any],
    ctx: ParallelContext,
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
    output_root = cfg["paths"]["output_root"]
    stage_dir = build_stage_path(output_root, "L1", n=n_ele, r42=r42, r62=r62, scheme=scheme)
    loaded = try_load_l1(stage_dir)
    if loaded is not None:
        state["l1"] = loaded
        return

    ensure_l0(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb)
    ensure_lsjm_all_three(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)

    lsjm_n = state[f"lsjm_{n_ele}"]
    soc0 = select_soc_lowest_subspace(lsjm_n)
    state["soc0"] = soc0
    result = build_L1(
        state["l0"],
        state[f"lsjm_{n_ele + 1}"]["V_fock"],
        soc0["U_n_soc0"],
        state[f"lsjm_{n_ele - 1}"]["V_fock"],
    )
    state["l1"] = result
    persist_l1(
        cfg,
        ctx,
        result,
        n_ele=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        soc0=soc0,
    )


def ensure_l2(
    cfg: dict[str, Any],
    ctx: ParallelContext,
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
    output_root = cfg["paths"]["output_root"]
    hopping_name = cfg["sources"]["hopping_name"]
    stage_dir = build_stage_path(
        output_root,
        "L2",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        hopping_name=hopping_name,
    )
    loaded = try_load_l2(stage_dir)
    if loaded is not None:
        state["l2"] = loaded
        return

    ensure_l1(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    hop_payload = load_hopping_matrix(cfg, n_orb=n_orb)
    t_mu = hop_payload["t_mu"]
    state["hopping_meta"] = hop_payload.get("meta", {})
    state["t_mu"] = t_mu
    result = build_L2(state["l1"], t_mu)
    state["l2"] = result
    persist_l2(
        cfg,
        ctx,
        result,
        n_ele=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        hopping_name=hopping_name,
        hopping_meta=state.get("hopping_meta", {}),
    )


def ensure_l3(
    cfg: dict[str, Any],
    ctx: ParallelContext,
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
    output_root = cfg["paths"]["output_root"]
    hopping_name = cfg["sources"]["hopping_name"]
    sopt = cfg["sopt"]
    stage_dir = build_stage_path(
        output_root,
        "L3",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        hopping_name=hopping_name,
        U=float(sopt["U"]),
        Jh=float(sopt["Jh"]),
        zeta=float(sopt["zeta"]),
    )
    loaded = try_load_l3(stage_dir)
    if loaded is not None:
        state["l3"] = loaded
        return

    ensure_l2(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    ensure_lsjm_all_three(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    E_u_np1, E_u_nm1 = compute_intermediate_energies(cfg, state, n_ele=n_ele)
    result = build_L3(state["l2"], E_u_np1, E_u_nm1, n_ele=n_ele)
    state["l3"] = result
    persist_l3(
        cfg,
        ctx,
        result,
        n_ele=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        hopping_name=hopping_name,
    )


def ensure_l4(
    cfg: dict[str, Any],
    ctx: ParallelContext,
    state: dict[str, Any],
    *,
    n_ele: int,
    n_orb: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    from fexchange.sopt.contraction import build_L4

    if "l4" in state:
        return
    output_root = cfg["paths"]["output_root"]
    hopping_name = cfg["sources"]["hopping_name"]
    kramer_name = cfg["sources"]["kramer_name"]
    sopt = cfg["sopt"]
    stage_dir = build_stage_path(
        output_root,
        "L4",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        hopping_name=hopping_name,
        U=float(sopt["U"]),
        Jh=float(sopt["Jh"]),
        zeta=float(sopt["zeta"]),
        kramer_name=kramer_name,
    )
    loaded = try_load_l4(stage_dir)
    if loaded is not None:
        state["l4"] = loaded
        return

    ensure_l3(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    ensure_lsjm_all_three(cfg, ctx, state, n_ele=n_ele, n_orb=n_orb, r42=r42, r62=r62, scheme=scheme)
    W_info = load_or_build_W(cfg, state, n_ele=n_ele)
    W = W_info["W"]
    labels = labels_abcd_lex(W.shape[1])
    validate_labels_abcd(labels, n_k=W.shape[1])

    result = build_L4(state["l3"], W)
    state["l4"] = result
    state["labels_abcd"] = labels
    state["W"] = W
    persist_l4(
        cfg,
        ctx,
        result,
        n_ele=n_ele,
        r42=r42,
        r62=r62,
        scheme=scheme,
        hopping_name=hopping_name,
        kramer_name=kramer_name,
        labels=labels,
        W=W,
        W_info=W_info,
    )
