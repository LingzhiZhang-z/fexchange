"""Human-readable source.txt writers for pipeline artifacts."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any


def write_run_source_txt(run_dir: Path, cfg: dict[str, Any]) -> None:
    """Write ``source.txt`` under the resolved run output directory."""
    path = run_dir / "source.txt"
    derived = cfg.get("_derived", {})
    fsite = cfg.get("fsite", {})
    n_ele = int(fsite.get("n_ele", 0)) if isinstance(fsite, dict) else 0
    summary = build_resolved_inputs_summary(
        cfg,
        n_ele=n_ele,
        r42=float(derived.get("r42", 0.0)) if isinstance(derived, dict) else 0.0,
        r62=float(derived.get("r62", 0.0)) if isinstance(derived, dict) else 0.0,
    )
    lines = [f"created: {_now()}"]
    for key in sorted(summary):
        lines.append(f"{key}: {summary[key]}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_core_source_txt(stage_dir: Path, inputs_summary: dict[str, Any]) -> None:
    """Write a compact ``source.txt`` beside a core artifact."""
    path = stage_dir / "source.txt"
    if path.exists():
        return
    lines = [f"created: {_now()}"]
    for key in sorted(inputs_summary):
        lines.append(f"{key}: {inputs_summary[key]}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_resolved_inputs_summary(
    cfg: dict[str, Any],
    *,
    n_ele: int,
    r42: float,
    r62: float,
) -> dict[str, Any]:
    """Flatten the current run's resolved scalar inputs for artifact provenance."""
    runtime = _as_dict(cfg.get("runtime"))
    model = _as_dict(cfg.get("model"))
    inputs = _as_dict(cfg.get("inputs"))
    paths = _as_dict(cfg.get("paths"))
    fsite = _as_dict(cfg.get("fsite"))
    summary: dict[str, Any] = {
        "standard_version": cfg.get("standard_version", ""),
        "scheme": model.get("scheme", "RS"),
        "branch": runtime.get("branch", ""),
        "run_name": runtime.get("run_name", ""),
        "output_root": paths.get("output_root", ""),
        "output_run": paths.get("output_run", ""),
        "n": int(n_ele),
        "r42": float(r42),
        "r62": float(r62),
        "energy_reference": fsite.get("energy_reference", "lsjm_ground"),
        "kramer_source": runtime.get("kramer_source", "stevens"),
        "hopping_file": inputs.get("hopping_file", ""),
        "kramer_file": inputs.get("kramer_file", ""),
        "hcef_file": inputs.get("hcef_file", ""),
    }

    branches = _as_dict(cfg.get("_branches"))
    if branches:
        for branch_name, prefix in (("n", "fsite"), ("nm1", "fsite_nm1"), ("np1", "fsite_np1")):
            branch = _as_dict(branches.get(branch_name))
            _add_prefixed_scalars(summary, prefix, _as_dict(branch.get("fsite")))
            derived = _as_dict(branch.get("derived"))
            _add_prefixed_scalars(summary, f"{prefix}.derived", derived)
    else:
        _add_prefixed_scalars(summary, "fsite", fsite)

    ligands = _as_dict(cfg.get("ligand"))
    for idx in sorted(ligands):
        ligand = _as_dict(ligands.get(idx))
        if ligand:
            _add_prefixed_scalars(summary, f"ligand.{idx}", ligand)
    return summary


def _add_prefixed_scalars(dst: dict[str, Any], prefix: str, values: dict[str, Any]) -> None:
    for key in sorted(values):
        value = values[key]
        if isinstance(value, bool):
            dst[f"{prefix}.{key}"] = value
        elif isinstance(value, (int, float, str)) or value is None:
            dst[f"{prefix}.{key}"] = value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
