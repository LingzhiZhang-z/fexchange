#!/usr/bin/env python3
"""Prepare a reusable SOPT core cache from legacy REChX_REX3 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fexchange.io.disk import build_stage_path, fmt12
from fexchange.pipeline.artifacts import persist_l1, persist_stateset, try_load_l1, try_load_stateset
from fexchange.pipeline.keys import branch_signature, level_key, three_sectors
from fexchange.pipeline.source_writer import write_core_source_txt
from fexchange.utils.constants import N_ORB, RE_DEFAULTS_BY_N_ELE, RE_TO_N_ELE, STANDARD_VERSION


DEFAULT_SOURCE_ROOT = Path("/work/k0282/k028230/fexchange/REChX_REX3")
DEFAULT_DEST_ROOT = Path("/work/k0282/k028230/fexchange/SOPT_CORE_REUSE_INTERNAL_F")
DEFAULT_RE = ("Nd", "Sm", "Dy", "Er", "Yb")
SCHEME = "RS"


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    re_list = tuple(args.re_list)

    legacy_index = build_legacy_index(source_root)
    manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "standard_version": STANDARD_VERSION,
        "scheme": SCHEME,
        "branch": "sopt",
        "re_list": list(re_list),
        "note": (
            "This cache migrates reusable LMSM/LSJM/L1 artifacts for SOPT RS runs "
            "using the program's internal F2/F4/F6 ratios. L0 is intentionally not "
            "materialized; when L1 is present, direct L2/L4 runs do not load L0."
        ),
        "systems": {},
    }

    for re_name in re_list:
        manifest["systems"][re_name] = prepare_re(
            re_name,
            source_root=source_root,
            dest_root=dest_root,
            legacy_index=legacy_index,
            force=bool(args.force),
        )

    write_readme(dest_root, manifest)
    (dest_root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    complete = [name for name, data in manifest["systems"].items() if data["complete"]]
    incomplete = [name for name, data in manifest["systems"].items() if not data["complete"]]
    print(f"destination: {dest_root}")
    print(f"complete: {' '.join(complete) if complete else '(none)'}")
    print(f"incomplete: {' '.join(incomplete) if incomplete else '(none)'}")
    if args.strict and incomplete:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--re", dest="re_list", nargs="+", default=list(DEFAULT_RE), choices=sorted(RE_TO_N_ELE))
    parser.add_argument("--force", action="store_true", help="Rewrite target artifacts even when they validate.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any requested RE is incomplete.")
    return parser.parse_args()


def build_legacy_index(source_root: Path) -> dict[tuple[str, str], list[Path]]:
    index: dict[tuple[str, str], list[Path]] = {}
    for level in ("LMSM", "LSJM", "L1"):
        for path in source_root.rglob(level):
            if not path.is_dir():
                continue
            token = path.parent.name
            if not token.startswith("n-") or "_scheme-" not in token:
                continue
            index.setdefault((token, level), []).append(path)
    for key in index:
        index[key].sort(key=lambda p: str(p))
    return index


def prepare_re(
    re_name: str,
    *,
    source_root: Path,
    dest_root: Path,
    legacy_index: dict[tuple[str, str], list[Path]],
    force: bool,
) -> dict[str, Any]:
    n_ele = RE_TO_N_ELE[re_name]
    params = RE_DEFAULTS_BY_N_ELE[n_ele]
    F2 = float(params["F2_per_Jh"])
    F4 = float(params["F4_per_Jh"])
    F6 = float(params["F6_per_Jh"])
    zeta = float(params["zeta"])
    r42 = F4 / F2
    r62 = F6 / F2
    cfg = cfg_for_re(dest_root, re_name, n_ele, F2=F2, F4=F4, F6=F6, zeta=zeta)

    entries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for sec in three_sectors(n_ele):
        for level in ("LMSM", "LSJM"):
            stage_dir = build_stage_path(str(dest_root), level, n=sec, r42=r42, r62=r62, scheme=SCHEME)
            if not force and try_load_stateset(stage_dir, level=level, n_ele=sec) is not None:
                entries.append({"level": level, "n": sec, "status": "present", "path": str(stage_dir)})
                continue

            legacy_dir = pick_legacy_dir(legacy_index, sec, r42, r62, level)
            if legacy_dir is None:
                missing.append({"level": level, "n": sec, "token": legacy_token(sec, r42, r62)})
                continue

            result = load_legacy_stateset(legacy_dir, level=level, n_ele=sec)
            persist_stateset(
                stage_dir,
                result,
                level=level,
                n_ele=sec,
                cfg=cfg,
                physics={"F2": F2, "F4": F4, "F6": F6, "zeta": zeta},
                r42=r42,
                r62=r62,
            )
            write_core_source_txt(
                stage_dir,
                {
                    "level": level,
                    "n_ele": sec,
                    "r42": f"{r42:.12f}",
                    "r62": f"{r62:.12f}",
                    "source": str(legacy_dir.relative_to(source_root)),
                },
            )
            entries.append(
                {"level": level, "n": sec, "status": "migrated", "path": str(stage_dir), "source": str(legacy_dir)}
            )

    l1_dir = build_stage_path(
        str(dest_root),
        "L1",
        branch="sopt",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=SCHEME,
        branch_signature=branch_signature(cfg, n_ele=n_ele),
    )
    expected_l1_key = level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)
    if not force and try_load_l1(l1_dir, expected_key=expected_l1_key) is not None:
        entries.append({"level": "L1", "n": n_ele, "status": "present", "path": str(l1_dir)})
    else:
        legacy_l1_dir = pick_legacy_dir(legacy_index, n_ele, r42, r62, "L1")
        if legacy_l1_dir is None:
            missing.append({"level": "L1", "n": n_ele, "token": legacy_token(n_ele, r42, r62)})
        else:
            result, soc0 = load_legacy_l1(legacy_l1_dir, n_ele=n_ele)
            persist_l1(l1_dir, cfg, result, n_ele=n_ele, r42=r42, r62=r62, soc0=soc0)
            write_core_source_txt(
                l1_dir,
                {
                    "level": "L1",
                    "branch": "sopt",
                    "n_ele": n_ele,
                    "r42": f"{r42:.12f}",
                    "r62": f"{r62:.12f}",
                    "source": str(legacy_l1_dir.relative_to(source_root)),
                },
            )
            entries.append(
                {"level": "L1", "n": n_ele, "status": "migrated", "path": str(l1_dir), "source": str(legacy_l1_dir)}
            )

    validation = validate_re(dest_root, cfg, n_ele=n_ele, r42=r42, r62=r62)
    return {
        "n_ele": n_ele,
        "F2_per_Jh": F2,
        "F4_per_Jh": F4,
        "F6_per_Jh": F6,
        "zeta": zeta,
        "r42": r42,
        "r62": r62,
        "complete": bool(validation["ok"]),
        "entries": entries,
        "missing": missing,
        "validation": validation,
    }


def cfg_for_re(
    dest_root: Path,
    re_name: str,
    n_ele: int,
    *,
    F2: float,
    F4: float,
    F6: float,
    zeta: float,
) -> dict[str, Any]:
    return {
        "standard_version": STANDARD_VERSION,
        "runtime": {"branch": "sopt", "run_name": "", "kramer_name": ""},
        "model": {"scheme": SCHEME},
        "paths": {"output_root": str(dest_root)},
        "fsite": {
            "RE": re_name,
            "n_ele": n_ele,
            "U": 0.0,
            "Jh": 1.0,
            "F2": F2,
            "F4": F4,
            "F6": F6,
            "zeta": zeta,
        },
        "physics": {"F2": F2, "F4": F4, "F6": F6, "zeta": zeta},
        "_derived": {"r42": F4 / F2, "r62": F6 / F2},
    }


def pick_legacy_dir(
    legacy_index: dict[tuple[str, str], list[Path]],
    n_ele: int,
    r42: float,
    r62: float,
    level: str,
) -> Path | None:
    candidates = legacy_index.get((legacy_token(n_ele, r42, r62), level), [])
    for path in candidates:
        if level in ("LMSM", "LSJM"):
            if (path / "V.npz").exists() and (path / "E_terms.npz").exists() and (path / "meta.json").exists():
                return path
        elif level == "L1":
            if (path / "data.npz").exists() and (path / "meta.json").exists():
                return path
    return None


def legacy_token(n_ele: int, r42: float, r62: float) -> str:
    return f"n-{n_ele}_r42-{fmt12(r42)}_r62-{fmt12(r62)}_scheme-{SCHEME}"


def load_legacy_stateset(legacy_dir: Path, *, level: str, n_ele: int) -> dict[str, Any]:
    v_payload = np.load(legacy_dir / "V.npz", allow_pickle=False)
    e_payload = np.load(legacy_dir / "E_terms.npz", allow_pickle=False)
    meta = json.loads((legacy_dir / "meta.json").read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "V_fock": np.asarray(v_payload["V_fock"], dtype=np.complex128),
        "coef_F0": np.asarray(e_payload["coef_F0"], dtype=float),
        "coef_F2": np.asarray(e_payload["coef_F2"], dtype=float),
        "coef_F4": np.asarray(e_payload["coef_F4"], dtype=float),
        "coef_F6": np.asarray(e_payload["coef_F6"], dtype=float),
        "labels": meta.get("labels", []),
        "basis_id": meta.get("basis_id", f"fock{N_ORB}_n{n_ele}_lex_v1"),
        "n_ele": n_ele,
        "n_orb": N_ORB,
        "state_order_id": meta.get("state_order_id", f"{level.lower()}_canonical_v1"),
        "orbital_order_id": meta.get("orbital_order_id"),
        "j_order_id": meta.get("j_order_id"),
    }
    if "coef_zeta" in e_payload:
        result["coef_zeta"] = np.asarray(e_payload["coef_zeta"], dtype=float)
    return result


def load_legacy_l1(legacy_dir: Path, *, n_ele: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = np.load(legacy_dir / "data.npz", allow_pickle=False)
    meta = json.loads((legacy_dir / "meta.json").read_text(encoding="utf-8"))
    A = np.asarray(payload["A"], dtype=np.complex128)
    B = np.asarray(payload["B"], dtype=np.complex128)
    result = {
        "A": A,
        "B": B,
        "n_orb": int(A.shape[0]),
        "n_u": int(A.shape[1]),
        "n_j": int(A.shape[2]),
        "n_v": int(B.shape[2]),
        "n_ele": n_ele,
    }
    soc0 = meta.get("soc0_meta")
    if not isinstance(soc0, dict):
        soc0 = {"alpha0": 0, "L0": 0, "S0": 0.0, "J0": 0.0, "n_j": int(A.shape[2])}
    soc0 = {
        "alpha0": int(soc0.get("alpha0", 0)),
        "L0": float(soc0.get("L0", 0.0)),
        "S0": float(soc0.get("S0", 0.0)),
        "J0": float(soc0.get("J0", 0.0)),
        "n_j": int(soc0.get("n_j", A.shape[2])),
    }
    return result, soc0


def validate_re(dest_root: Path, cfg: dict[str, Any], *, n_ele: int, r42: float, r62: float) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for sec in three_sectors(n_ele):
        for level in ("LMSM", "LSJM"):
            stage_dir = build_stage_path(str(dest_root), level, n=sec, r42=r42, r62=r62, scheme=SCHEME)
            ok = try_load_stateset(stage_dir, level=level, n_ele=sec) is not None
            checks.append({"level": level, "n": sec, "ok": ok, "path": str(stage_dir)})

    l1_dir = build_stage_path(
        str(dest_root),
        "L1",
        branch="sopt",
        n=n_ele,
        r42=r42,
        r62=r62,
        scheme=SCHEME,
        branch_signature=branch_signature(cfg, n_ele=n_ele),
    )
    expected_key = level_key("L1", n_ele=n_ele, r42=r42, r62=r62, cfg=cfg)
    ok = try_load_l1(l1_dir, expected_key=expected_key) is not None
    checks.append({"level": "L1", "n": n_ele, "ok": ok, "path": str(l1_dir)})
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def write_readme(dest_root: Path, manifest: dict[str, Any]) -> None:
    complete = [name for name, data in manifest["systems"].items() if data["complete"]]
    incomplete = [name for name, data in manifest["systems"].items() if not data["complete"]]
    lines = [
        "SOPT reusable core cache",
        "",
        f"source_root: {manifest['source_root']}",
        f"dest_root:   {manifest['dest_root']}",
        "branch:      sopt",
        "scheme:      RS",
        "parameters:  internal F2/F4/F6 ratios from fexchange.utils.constants",
        "",
        "Prepared levels:",
        "- LMSM and LSJM for n-1, n, n+1 sectors",
        "- L1 for the main n sector",
        "",
        "L0 is not materialized here. Direct SOPT runs to L2/L4 load L1 first,",
        "so L0 is only needed if L1 is missing or if end_level=L0 is requested.",
        "",
        f"complete:   {' '.join(complete) if complete else '(none)'}",
        f"incomplete: {' '.join(incomplete) if incomplete else '(none)'}",
        "",
        "Point paths.output_root at this directory to reuse the cache.",
        "",
    ]
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
