"""
Validation helpers for pipeline preflight and projection labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fexchange.io.disk import build_stage_path, load_json_checked, load_npz_checked, validate_meta
from fexchange.pipeline.artifacts import ARTIFACT_FILE_SPEC
from fexchange.pipeline.keys import (
    branch_signature,
    denominator_signature,
    extract_source_names,
    sector_ratios,
    three_sectors,
)
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
    sn = extract_source_names(cfg)
    kramer_name = sn.kramer_name
    run_name = str(cfg.get("runtime", {}).get("run_name", ""))
    branch = str(cfg.get("runtime", {}).get("branch", "sopt"))
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
    
    def _check_spec(d: Path, level: str) -> None:
        for filename, keys in ARTIFACT_FILE_SPEC[level]:
            if keys is not None:
                _check(d / filename, keys)
            else:
                try:
                    validate_meta(load_json_checked(d / filename))
                except Exception:
                    invalid.append(str(d / filename))

    if "LMSM" in required:
        for sec in three_sectors(n_ele):
            sec_r42, sec_r62 = sector_ratios(cfg, sec, default_r42=r42, default_r62=r62)
            d = build_stage_path(output_root, "LMSM", n=sec, r42=sec_r42, r62=sec_r62, scheme=scheme)
            _check_spec(d, "LMSM")
    if "LSJM" in required:
        for sec in three_sectors(n_ele):
            sec_r42, sec_r62 = sector_ratios(cfg, sec, default_r42=r42, default_r62=r62)
            d = build_stage_path(output_root, "LSJM", n=sec, r42=sec_r42, r62=sec_r62, scheme=scheme)
            _check_spec(d, "LSJM")
    if "L0" in required:
        d = build_stage_path(output_root, "L0", branch=branch)
        _check(d / f"f_create_{n_ele}_to_{n_ele + 1}.npz", ["data"])
        _check(d / f"f_create_{n_ele - 1}_to_{n_ele}.npz", ["data"])
        try:
            validate_meta(load_json_checked(d / "meta.json"))
        except Exception:
            invalid.append(str(d / "meta.json"))
    if "L1" in required:
        d = build_stage_path(
            output_root,
            "L1",
            n=n_ele,
            r42=r42,
            r62=r62,
            scheme=scheme,
            branch_signature=branch_signature(cfg, n_ele=n_ele),
        )
        _check_spec(d, "L1")
    if "L2" in required:
        d = build_stage_path(
            output_root,
            "L2",
            branch=branch,
            run_name=run_name,
        )
        _check_spec(d, "L2")
    if "L3" in required:
        s = cfg["fsite"]
        ligands = cfg.get("ligand", {})
        d = build_stage_path(
            output_root,
            "L3",
            branch=branch,
            run_name=run_name,
            U=float(s["U"]),
            Jh=float(s["Jh"]),
            lig1_U_p=float(ligands.get("1", {}).get("U_p", 0.0)) if isinstance(ligands, dict) else 0.0,
            lig2_U_p=float(ligands.get("2", {}).get("U_p", 0.0)) if isinstance(ligands, dict) else 0.0,
        )
        _check_spec(d, "L3_FOPT" if branch == "fopt" else "L3")
    if "L4" in required:
        s = cfg["fsite"]
        d = build_stage_path(
            output_root,
            "L4",
            branch=branch,
            run_name=run_name,
            U=float(s["U"]),
            Jh=float(s["Jh"]),
            kramer_name=kramer_name,
        )
        _check_spec(d, "L4")

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
