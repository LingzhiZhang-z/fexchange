#!/usr/bin/env python3
"""Prepare FOPT run inputs from the DFT SOPT manifest."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOPT_MANIFEST = Path(
    "/work/k0282/k028230/fexchange/SOPT_CORE_REUSE_INTERNAL_F/run_inputs/"
    "data_DFT_downfold_U6_RS/manifest.tsv"
)
DEFAULT_INPUT_ROOT = Path(
    "/work/k0282/k028230/fexchange/SOPT_CORE_REUSE_INTERNAL_F/run_inputs/"
    "data_DFT_fopt_U6_RS"
)
DEFAULT_OUTPUT_ROOT = Path("/work/k0282/k028230/fexchange/SOPT_CORE_REUSE_INTERNAL_F")
SCIPY_PATH = Path(
    "/home2/k0282/k028230/optz/miniforge3/pkgs/"
    "scipy-1.17.1-py312h54fa4ab_0/lib/python3.12/site-packages"
)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    rows = read_manifest(args.sopt_manifest)
    generated_at = datetime.now(UTC).isoformat()

    input_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    run_list: list[Path] = []
    yb_run_list: list[Path] = []
    non_yb_run_list: list[Path] = []

    for row in rows:
        constants = constants_from_row(row)
        delta1 = parse_float(constants, "Delta_lig1_eV")
        delta2 = parse_float(constants, "Delta_lig2_eV")
        hopping_fp = Path(row["hopping_file"]).with_name("hopping_fp.txt")
        if not hopping_fp.exists():
            raise FileNotFoundError(hopping_fp)
        projector = Path(row["projector_file"])
        if not projector.exists():
            raise FileNotFoundError(projector)

        material = row["material"]
        bond = row["bond"]
        ratio_token = ratio_to_token(float(row["Jh_over_U"]))
        out_dir = input_root / row["family"] / material / bond
        out_dir.mkdir(parents=True, exist_ok=True)
        input_file = out_dir / f"run_input_U06_r{ratio_token}.toml"
        run_name = f"{material}_FOPT_{bond}"
        hopping_name = f"{material}_{bond}_fp"
        run_id = f"{material}_{bond}_{row['kramer_variant']}_FOPT_U06_r{ratio_token}"

        write_run_input(
            input_file,
            generated_at=generated_at,
            source_constants=Path(row["hopping_file"]).with_name("constants.txt"),
            hopping_source=hopping_fp,
            projector_source=projector,
            row=row,
            run_id=run_id,
            run_name=run_name,
            hopping_name=hopping_name,
            output_root=output_root,
            delta1=delta1,
            delta2=delta2,
            up_e_v=args.Up_eV,
            lambda_p_e_v=args.lambda_p_eV,
        )
        run_list.append(input_file)
        if row["RE"] == "Yb":
            yb_run_list.append(input_file)
        else:
            non_yb_run_list.append(input_file)

        manifest_rows.append(
            {
                **{key: row[key] for key in MANIFEST_KEEP_COLUMNS},
                "branch": "fopt",
                "end_level": "L3",
                "run_name": run_name,
                "hopping_name": hopping_name,
                "Delta_lig1_eV": f"{delta1:.12f}",
                "Delta_lig2_eV": f"{delta2:.12f}",
                "U_p_eV": f"{args.Up_eV:.12f}",
                "lambda_p_eV": f"{args.lambda_p_eV:.12f}",
                "hopping_file": str(hopping_fp),
                "projector_file": str(projector),
                "input_file": str(input_file),
            }
        )

    write_manifest(input_root / "manifest.tsv", manifest_rows)
    write_list(input_root / "production_run_list.txt", run_list)
    write_list(input_root / "run_list.txt", run_list)
    write_list(input_root / "yb_production_run_list.txt", yb_run_list)
    write_list(input_root / "non_yb_production_run_list.txt", non_yb_run_list)
    write_commands(input_root / "run_commands.txt", run_list)
    write_runner(input_root / "run_all.sh", input_root / "run_list.txt")
    write_yb_runner(input_root / "run_yb_local.sh", input_root / "yb_production_run_list.txt", input_root / "yb_local.log")
    write_slurm_array_runner(
        input_root / "submit_non_yb_array.sh",
        input_root / "non_yb_production_run_list.txt",
        input_root / "slurm_logs",
        len(non_yb_run_list),
    )
    write_readme(
        input_root / "README.txt",
        generated_at=generated_at,
        output_root=output_root,
        total=len(run_list),
        yb_total=len(yb_run_list),
        non_yb_total=len(non_yb_run_list),
        base_total=len(unique_base_entries(rows)),
        up_e_v=args.Up_eV,
        lambda_p_e_v=args.lambda_p_eV,
    )

    print(f"wrote {input_root}")
    print(f"production_inputs={len(run_list)}")
    print(f"yb_inputs={len(yb_run_list)}")
    print(f"non_yb_inputs={len(non_yb_run_list)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sopt-manifest", type=Path, default=DEFAULT_SOPT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--Up-eV", type=float, default=0.0)
    parser.add_argument("--lambda-p-eV", type=float, default=0.0)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def constants_from_row(row: dict[str, str]) -> dict[str, str]:
    path = Path(row["hopping_file"]).with_name("constants.txt")
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            data[parts[0]] = parts[1].strip().strip('"')
    return data


def parse_float(data: dict[str, str], key: str) -> float:
    try:
        return float(data[key])
    except KeyError as exc:
        raise KeyError(f"missing {key} in constants") from exc


def ratio_to_token(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def write_run_input(
    path: Path,
    *,
    generated_at: str,
    source_constants: Path,
    hopping_source: Path,
    projector_source: Path,
    row: dict[str, str],
    run_id: str,
    run_name: str,
    hopping_name: str,
    output_root: Path,
    delta1: float,
    delta2: float,
    up_e_v: float,
    lambda_p_e_v: float,
) -> None:
    title = f"{row['family']} {row['material']} {row['bond']} {row['kramer_variant']} U=6eV Jh/U={row['Jh_over_U']}"
    lines = [
        f"# Generated {generated_at}",
        "# FOPT input: DFT f-p hopping; RS intermediate states.",
        f"# source_constants = {source_constants}",
        f"# hopping_source = {hopping_source}",
        f"# projector_source = {projector_source}",
        f"# projector_variant = {row['kramer_variant']}",
        f"# zeta_source = {row['zeta_source']}",
        'schema_version = "fxe.run_input.v1"',
        'standard_version = "2026-02"',
        f"run_id = {toml_string(run_id)}",
        f"title = {toml_string(title)}",
        "",
        "[units]",
        'energy = "eV"',
        "",
        "[fsite]",
        f"n_ele = {int(row['n_ele'])}",
        f"RE = {toml_string(row['RE'])}",
        f"U = {float(row['U_eV']):.12f}",
        f"Jh = {float(row['Jh_eV']):.12f}",
        f"zeta = {float(row['zeta_eV']):.12f}",
        "offset = 0.000000000000",
        'energy_reference = "lsjm_ground"',
        "",
        "[model]",
        'scheme = "RS"',
        "",
        "[ligand.1]",
        f"Delta = {delta1:.12f}",
        f"U_p = {up_e_v:.12f}",
        f"lambda_p = {lambda_p_e_v:.12f}",
        "",
        "[ligand.2]",
        f"Delta = {delta2:.12f}",
        f"U_p = {up_e_v:.12f}",
        f"lambda_p = {lambda_p_e_v:.12f}",
        "",
        "[inputs]",
        f"hopping_file = {toml_string(str(hopping_source))}",
        f"kramer_file = {toml_string(str(projector_source))}",
        f"hopping_name = {toml_string(hopping_name)}",
        "",
        "[paths]",
        f"output_root = {toml_string(str(output_root))}",
        "",
        "[runtime]",
        'branch = "fopt"',
        'end_level = "L3"',
        f"run_name = {toml_string(run_name)}",
        "",
        "[checks]",
        "strict_mode = true",
        'eps_profile = "default"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


MANIFEST_KEEP_COLUMNS = (
    "family",
    "category",
    "material",
    "bond",
    "kramer_variant",
    "RE",
    "n_ele",
    "U_eV",
    "Jh_over_U",
    "Jh_eV",
    "kramer_name",
    "zeta_source",
    "zeta_eV",
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        *MANIFEST_KEEP_COLUMNS,
        "branch",
        "end_level",
        "run_name",
        "hopping_name",
        "Delta_lig1_eV",
        "Delta_lig2_eV",
        "U_p_eV",
        "lambda_p_eV",
        "hopping_file",
        "projector_file",
        "input_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_list(path: Path, items: list[Path]) -> None:
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def write_commands(path: Path, items: list[Path]) -> None:
    lines = [
        "cd "
        + shlex.quote(str(REPO_ROOT))
        + " && PYTHONPATH="
        + shlex.quote(f"{REPO_ROOT}:{SCIPY_PATH}:$PYTHONPATH")
        + " python -m fexchange.cli run "
        + shlex.quote(str(item))
        + " --log-level INFO"
        for item in items
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runner(path: Path, default_list: Path) -> None:
    text = f"""#!/bin/bash
set -euo pipefail
REPO={REPO_ROOT}
SCIPY={SCIPY_PATH}
INPUT_LIST="${{1:-{default_list}}}"
cd "${{REPO}}"
export PYTHONPATH="${{REPO}}:${{SCIPY}}:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
while IFS= read -r input; do
    [ -n "${{input}}" ] || continue
    python -m fexchange.cli run "${{input}}" --log-level INFO
done < "${{INPUT_LIST}}"
"""
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o755)


def write_yb_runner(path: Path, yb_list: Path, log_path: Path) -> None:
    text = f"""#!/bin/bash
set -euo pipefail
REPO={REPO_ROOT}
SCIPY={SCIPY_PATH}
RUN_LIST={yb_list}
LOG={log_path}
cd "${{REPO}}"
export PYTHONPATH="${{REPO}}:${{SCIPY}}:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
printf 'START %s total=%s run_list=%s\\n' "$(date --iso-8601=seconds)" "$(wc -l < "${{RUN_LIST}}")" "${{RUN_LIST}}" | tee "${{LOG}}"
i=0
while IFS= read -r input; do
    [ -n "${{input}}" ] || continue
    i=$((i+1))
    printf '=== BEGIN %04d %s %s ===\\n' "${{i}}" "$(date --iso-8601=seconds)" "${{input}}" | tee -a "${{LOG}}"
    python -m fexchange.cli run "${{input}}" --log-level INFO 2>&1 | tee -a "${{LOG}}"
    printf '=== END %04d %s %s ===\\n' "${{i}}" "$(date --iso-8601=seconds)" "${{input}}" | tee -a "${{LOG}}"
done < "${{RUN_LIST}}"
printf 'COMPLETE %s total=%s\\n' "$(date --iso-8601=seconds)" "${{i}}" | tee -a "${{LOG}}"
"""
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o755)


def write_slurm_array_runner(path: Path, run_list: Path, log_dir: Path, total: int) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    text = f"""#!/bin/bash
#SBATCH --job-name=fopt_dft
#SBATCH --partition=F1cpu
#SBATCH --array=1-{total}%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err

set -euo pipefail
REPO={REPO_ROOT}
SCIPY={SCIPY_PATH}
RUN_LIST={run_list}
INPUT="$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "${{RUN_LIST}}")"
if [ -z "${{INPUT}}" ]; then
    echo "empty input for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
    exit 2
fi
cd "${{REPO}}"
export PYTHONPATH="${{REPO}}:${{SCIPY}}:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
python -m fexchange.cli run "${{INPUT}}" --log-level INFO
"""
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o755)


def write_readme(
    path: Path,
    *,
    generated_at: str,
    output_root: Path,
    total: int,
    yb_total: int,
    non_yb_total: int,
    base_total: int,
    up_e_v: float,
    lambda_p_e_v: float,
) -> None:
    lines = [
        "# data_DFT_fopt_U6_RS inputs",
        "",
        f"generated_utc {generated_at}",
        f"repo {REPO_ROOT}",
        f"output_root {output_root}",
        "",
        "Physics:",
        "- model.scheme = RS",
        "- branch = fopt, end_level = L3",
        "- U = 6 eV",
        "- Jh/U = 0.02, 0.04, ..., 0.40 (20 points)",
        f"- U_p = {up_e_v:g} eV for both ligands",
        f"- lambda_p = {lambda_p_e_v:g} eV for both ligands",
        "- ligand Delta values are read from each DFT constants.txt",
        "- hopping_file is always DFT f-p: hopping_fp.txt",
        "- Projector choices are copied from the SOPT DFT manifest",
        "- REX3-C2m RECl3 materials remain excluded because the SOPT manifest excludes them",
        "",
        "Files:",
        "- production_run_list.txt: production FOPT L3 inputs only",
        "- run_list.txt: same as production_run_list.txt",
        "- yb_production_run_list.txt: Yb-only subset",
        "- non_yb_production_run_list.txt: non-Yb subset for Slurm array",
        "- run_commands.txt: expanded shell commands",
        "- manifest.tsv: one row per production input",
        "- run_all.sh: sequential runner; accepts an optional input-list path",
        "- run_yb_local.sh: sequential Yb runner with tee logging",
        "- submit_non_yb_array.sh: Slurm array runner for non-Yb inputs",
        "",
        "Counts:",
        f"- base material/bond/projector entries: {base_total}",
        f"- production inputs: {total}",
        f"- Yb inputs: {yb_total}",
        f"- non-Yb inputs: {non_yb_total}",
        "",
        "Notes:",
        "- run_name format is material_FOPT_bond, e.g. YbOCl_FOPT_J1 or DyBr3_FOPT_z.",
        "- f9 Oh has both Gamma6 and Gamma7 inputs through distinct kramer_name values.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def unique_base_entries(rows: list[dict[str, str]]) -> set[tuple[str, str, str, str]]:
    return {
        (row["family"], row["material"], row["bond"], row["kramer_name"])
        for row in rows
    }


if __name__ == "__main__":
    raise SystemExit(main())
