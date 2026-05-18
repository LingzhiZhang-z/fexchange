"""Human-readable source.txt writers for pipeline artifacts."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from fexchange.io.disk import fmt8


def write_run_source_txt(run_dir: Path, cfg: dict[str, Any]) -> None:
    """Write ``<run_name>/source.txt`` once; do not enforce cache consistency."""
    path = run_dir / "source.txt"
    if path.exists():
        return
    runtime = cfg.get("runtime", {})
    fsite = cfg.get("fsite", {})
    inputs = cfg.get("inputs", {})
    ligands = cfg.get("ligand", {})
    lines = [
        f"run_name: {runtime.get('run_name', '')}",
        f"branch:   {runtime.get('branch', '')}",
        f"created:  {_now()}",
        "",
        f"r42:            {fmt8(float(cfg.get('_derived', {}).get('r42', 0.0)))}",
        f"r62:            {fmt8(float(cfg.get('_derived', {}).get('r62', 0.0)))}",
        f"RE:             {fsite.get('RE', 'auto')}",
        f"scheme:         {cfg.get('model', {}).get('scheme', 'RS')}",
        f"zeta:           {fmt8(float(fsite.get('zeta', 0.0)))}",
    ]
    if isinstance(ligands, dict):
        for idx in ("1", "2"):
            lig = ligands.get(idx, {})
            if isinstance(lig, dict):
                lines.extend([
                    f"lig{idx}.Delta:     {fmt8(float(lig.get('Delta', 0.0)))}",
                    f"lig{idx}.lambda_p:  {fmt8(float(lig.get('lambda_p', 0.0)))}",
                ])
    lines.extend([
        "",
        f"hopping_file:   {inputs.get('hopping_file', '')}",
        f"projector_file: {inputs.get('projector_file', '')}",
        f"kramer_name:    {runtime.get('kramer_name', '')}",
        "",
    ])
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
