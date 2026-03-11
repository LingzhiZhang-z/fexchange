"""
Validation helpers for pipeline preflight and projection labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fexchange.io.disk import build_stage_path, load_json_checked, load_npz_checked, validate_meta
from fexchange.pipeline.keys import three_sectors
from fexchange.utils.errors import BindError, IOError_, SchemaError


def validate_labels_abcd(labels_abcd: NDArray[np.int64], *, n_k: int) -> None:
    if labels_abcd.ndim != 2 or labels_abcd.shape[1] != 4:
        raise BindError(
            "FXE-BIND-003",
            "labels_abcd must have shape (N,4)",
            actual={"shape": list(labels_abcd.shape)},
        )
    if labels_abcd.size and int(labels_abcd.max()) >= n_k:
        raise BindError(
            "FXE-BIND-003",
            "labels_abcd index exceeds target kramers dimension",
            actual={"max_label": int(labels_abcd.max()), "n_k": n_k},
        )
    tuples = [tuple(int(x) for x in row) for row in labels_abcd]
    if len(set(tuples)) != len(tuples):
        raise BindError("FXE-BIND-003", "labels_abcd contains duplicate rows")
    if tuples != sorted(tuples):
        raise BindError("FXE-BIND-003", "labels_abcd must be lexicographically sorted")


def validate_upstream_artifacts(
    cfg: dict,
    required: tuple[str, ...],
    *,
    n_ele: int,
    r42: float,
    r62: float,
    scheme: str,
) -> None:
    if not required:
        return
    output_root = cfg["paths"]["output_root"]
    missing: list[str] = []
    invalid: list[str] = []

    def _check(path: Path, keys: list[str]) -> None:
        if not path.exists():
            missing.append(str(path))
            return
        try:
            load_npz_checked(path, keys)
        except Exception:
            invalid.append(str(path))
            return

    if "LMSM" in required:
        for sec in three_sectors(n_ele):
            d = build_stage_path(output_root, "LMSM", n=sec, r42=r42, r62=r62, scheme=scheme)
            _check(d / "V.npz", ["V_fock"])
            _check(d / "E_terms.npz", ["coef_F0", "coef_F2", "coef_F4", "coef_F6"])
            try:
                validate_meta(load_json_checked(d / "meta.json"))
            except Exception:
                invalid.append(str(d / "meta.json"))
    if "LSJM" in required:
        for sec in three_sectors(n_ele):
            d = build_stage_path(output_root, "LSJM", n=sec, r42=r42, r62=r62, scheme=scheme)
            _check(d / "V.npz", ["V_fock"])
            _check(d / "E_terms.npz", ["coef_F0", "coef_F2", "coef_F4", "coef_F6", "coef_zeta"])
            try:
                validate_meta(load_json_checked(d / "meta.json"))
            except Exception:
                invalid.append(str(d / "meta.json"))
    if "L0" in required:
        d = build_stage_path(output_root, "fock")
        _check(d / f"n{n_ele}{n_ele + 1}.npz", ["X"])
        _check(d / f"n{n_ele - 1}{n_ele}.npz", ["Y"])
        for sec in three_sectors(n_ele):
            try:
                validate_meta(load_json_checked(d / f"meta_n{sec}.json"))
            except Exception:
                invalid.append(str(d / f"meta_n{sec}.json"))
    if "L1" in required:
        d = build_stage_path(output_root, "L1", n=n_ele, r42=r42, r62=r62, scheme=scheme)
        _check(d / "data.npz", ["A", "B"])
        try:
            validate_meta(load_json_checked(d / "meta.json"))
        except Exception:
            invalid.append(str(d / "meta.json"))
    if "L2" in required:
        d = build_stage_path(
            output_root,
            "L2",
            n=n_ele,
            r42=r42,
            r62=r62,
            scheme=scheme,
            hopping_name=cfg["sources"]["hopping_name"],
        )
        _check(d / "data.npz", ["M_A", "M_B"])
        try:
            validate_meta(load_json_checked(d / "meta.json"))
        except Exception:
            invalid.append(str(d / "meta.json"))
    if "L3" in required:
        s = cfg["sopt"]
        d = build_stage_path(
            output_root,
            "L3",
            n=n_ele,
            r42=r42,
            r62=r62,
            scheme=scheme,
            hopping_name=cfg["sources"]["hopping_name"],
            U=float(s["U"]),
            Jh=float(s["Jh"]),
            zeta=float(s["zeta"]),
        )
        _check(d / "data.npz", ["h_pre_j_mu"])
        try:
            validate_meta(load_json_checked(d / "meta.json"))
        except Exception:
            invalid.append(str(d / "meta.json"))
    if "L4" in required:
        s = cfg["sopt"]
        d = build_stage_path(
            output_root,
            "L4",
            n=n_ele,
            r42=r42,
            r62=r62,
            scheme=scheme,
            hopping_name=cfg["sources"]["hopping_name"],
            U=float(s["U"]),
            Jh=float(s["Jh"]),
            zeta=float(s["zeta"]),
            kramer_name=cfg["sources"]["kramer_name"],
        )
        _check(d / "data.npz", ["h_mu_abcd", "Heff_mu_abcd"])
        try:
            validate_meta(load_json_checked(d / "meta.json"))
        except Exception:
            invalid.append(str(d / "meta.json"))

    if missing:
        raise IOError_(
            "FXE-IO-001",
            "Missing required upstream artifacts",
            expected={"required_upstream": list(required)},
            actual={"missing_paths": missing},
            paths={"output_root": output_root},
        )
    if invalid:
        raise SchemaError(
            "FXE-SCHEMA-002",
            "Invalid required upstream artifacts",
            expected={"required_upstream": list(required)},
            actual={"invalid_paths": invalid},
            paths={"output_root": output_root},
        )

