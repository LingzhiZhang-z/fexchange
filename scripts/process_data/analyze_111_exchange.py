#!/usr/bin/env python3
"""Analyze exchange tensors in a fixed local frame with z || cubic [111].

The input is the organized exchange tree produced by extract_exchange.sh and,
for REChX, already rotated into the cubic frame under REChX_cubic:

  <root>/<family>/<material>/<bond>/<branch>/<run>/U_<NN>.txt

Each data row stores ratio, label, residual, and the 3x3 exchange tensor in eV.
This script rotates that tensor to the local basis

  a = ( 1,-1, 0)/sqrt(2)
  b = ( 1, 1,-2)/sqrt(6)
  c = ( 1, 1, 1)/sqrt(3)

and reports the corresponding J, K, Gamma, Gamma', D, D' parameters. It also
computes simple-model residuals:

  ising111_resid : symmetric tensor vs only S_cc kept
  xxz111_resid   : symmetric tensor vs diag(J_ab, J_ab, S_cc)
  rank1_resid    : symmetric tensor vs its best one-axis eigenmode

Small residuals and a large |S_cc| / max(non-Ising symmetric terms) indicate
that the exchange itself is Ising-like along [111].
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized/REChX_cubic")
DEFAULT_TARGETS = (
    "REOF/ErOF/ErOF_baseline_Er_J15_2_projector",
    "REOF/DyOF/DyOF_baseline_Dy_J15_2_projector",
)
DEFAULT_BRANCHES = ("fopt_plus_sopt_direct",)
ALL_BRANCHES = ("sopt", "sopt_direct", "fopt", "fopt_plus_sopt_direct")
EPS = 1.0e-14

E_A = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
E_B = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
E_C = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
BASIS_111 = np.column_stack([E_A, E_B, E_C])

DETAIL_FIELDS = [
    "family",
    "material",
    "bond",
    "branch",
    "run",
    "U",
    "ratio",
    "input_residual",
    "J111_mev",
    "K111_mev",
    "Gamma111_mev",
    "Gamma_prime111_mev",
    "Gamma_ac111_mev",
    "Gamma_bc111_mev",
    "D111_mev",
    "D_prime111_mev",
    "D_a111_mev",
    "D_b111_mev",
    "D_c111_mev",
    "Scc111_mev",
    "max_nonising_sym111_mev",
    "ising_ratio_abs",
    "ising111_resid",
    "ising111_full_resid",
    "xxz111_resid",
    "xxz111_full_resid",
    "c3_axial_resid",
    "c3_nonaxial_norm_mev",
    "rank1_resid",
    "axis_alignment111",
    "lambda_max_mev",
    "eig_anisotropy_abs",
    "dm_norm_mev",
    "dm_over_sym",
]

SUMMARY_FIELDS = [
    "family",
    "material",
    "bond",
    "branch",
    "run",
    "U",
    "n",
    "ratio_min",
    "ratio_max",
    "max_abs_Scc111_mev",
    "max_abs_nonising_sym111_mev",
    "min_ising_ratio_abs",
    "median_ising_ratio_abs",
    "median_ising111_resid",
    "max_ising111_resid",
    "median_ising111_full_resid",
    "median_xxz111_resid",
    "max_xxz111_resid",
    "median_xxz111_full_resid",
    "median_c3_axial_resid",
    "max_c3_axial_resid",
    "max_c3_nonaxial_norm_mev",
    "median_rank1_resid",
    "max_rank1_resid",
    "min_axis_alignment111",
    "median_axis_alignment111",
    "max_dm_norm_mev",
    "median_dm_over_sym",
]


def parse_target(text: str) -> tuple[str, str, str]:
    parts = [part for part in text.strip("/").split("/") if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("target must be FAMILY/MATERIAL/RUN")
    return parts[0], parts[1], parts[2]


def discover_targets(root: Path, branches: set[str], bond_filter: set[str] | None) -> list[tuple[str, str, str]]:
    targets: set[tuple[str, str, str]] = set()
    for family_root in sorted(path for path in root.iterdir() if path.is_dir()):
        for material_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            for bond_root in sorted(path for path in material_root.iterdir() if path.is_dir()):
                if bond_filter is not None and bond_root.name not in bond_filter:
                    continue
                for branch in branches:
                    branch_root = bond_root / branch
                    if not branch_root.exists():
                        continue
                    for run_root in sorted(path for path in branch_root.iterdir() if path.is_dir()):
                        if any(run_root.glob("U_*.txt")):
                            targets.add((family_root.name, material_root.name, run_root.name))
    return sorted(targets)


def iter_u_files(root: Path, targets: list[tuple[str, str, str]], branches: set[str], bond_filter: set[str] | None):
    for family, material, run_filter in targets:
        material_root = root / family / material
        if not material_root.exists():
            print(f"warning: missing {material_root}", file=sys.stderr)
            continue
        for bond_root in sorted(path for path in material_root.iterdir() if path.is_dir()):
            bond = bond_root.name
            if bond_filter is not None and bond not in bond_filter:
                continue
            for branch in sorted(branches):
                run_root = bond_root / branch / run_filter
                if not run_root.exists():
                    continue
                for path in sorted(run_root.glob("U_*.txt")):
                    try:
                        u_value = int(path.stem.split("_", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    yield family, material, bond, branch, run_filter, u_value, path


def read_exchange_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        tensor = np.array([float(x) for x in cols[3:12]], dtype=float).reshape(3, 3)
        yield float(cols[0]), float(cols[2]), tensor


def finite_ratio(num: float, den: float) -> float:
    if abs(den) < EPS:
        return float("inf") if abs(num) >= EPS else 0.0
    return num / den


def decompose_111(tensor_ev: np.ndarray) -> dict[str, float]:
    m = BASIS_111.T @ tensor_ev @ BASIS_111
    s = 0.5 * (m + m.T)
    dm = 0.5 * (m - m.T)

    j_ab = 0.5 * (m[0, 0] + m[1, 1])
    gamma = 0.5 * (m[0, 1] + m[1, 0])
    gamma_ac = 0.5 * (m[0, 2] + m[2, 0])
    gamma_bc = 0.5 * (m[1, 2] + m[2, 1])
    d_a = 0.5 * (m[1, 2] - m[2, 1])
    d_b = 0.5 * (m[2, 0] - m[0, 2])
    d_c = 0.5 * (m[0, 1] - m[1, 0])

    sym_norm = float(np.linalg.norm(s))
    full_norm = float(np.linalg.norm(m))
    dm_norm = float(np.sqrt(d_a * d_a + d_b * d_b + d_c * d_c))
    scc_only = np.zeros((3, 3), dtype=float)
    scc_only[2, 2] = s[2, 2]
    xxz = np.diag([j_ab, j_ab, s[2, 2]])
    ising_full = np.zeros((3, 3), dtype=float)
    ising_full[2, 2] = m[2, 2]
    xxz_full = np.diag([j_ab, j_ab, m[2, 2]])
    c3_axial = np.array([
        [j_ab, d_c, 0.0],
        [-d_c, j_ab, 0.0],
        [0.0, 0.0, m[2, 2]],
    ], dtype=float)
    c3_nonaxial_norm = float(np.linalg.norm(m - c3_axial))

    evals, evecs = np.linalg.eigh(s)
    imax = int(np.argmax(np.abs(evals)))
    lam = float(evals[imax])
    vec = evecs[:, imax]
    rank1 = lam * np.outer(vec, vec)
    other = [abs(float(evals[i])) for i in range(3) if i != imax]
    other_max = max(other) if other else 0.0

    nonising_sym_terms = [
        s[0, 0],
        s[1, 1],
        gamma,
        gamma_ac,
        gamma_bc,
    ]
    max_nonising = float(max(abs(x) for x in nonising_sym_terms))
    ising_ratio = finite_ratio(abs(float(s[2, 2])), max_nonising)

    return {
        "J111_mev": float(j_ab * 1000.0),
        "K111_mev": float((m[2, 2] - j_ab) * 1000.0),
        "Gamma111_mev": float(gamma * 1000.0),
        "Gamma_prime111_mev": float(0.5 * (gamma_ac + gamma_bc) * 1000.0),
        "Gamma_ac111_mev": float(gamma_ac * 1000.0),
        "Gamma_bc111_mev": float(gamma_bc * 1000.0),
        "D111_mev": float(d_c * 1000.0),
        "D_prime111_mev": float(0.5 * (d_a + d_b) * 1000.0),
        "D_a111_mev": float(d_a * 1000.0),
        "D_b111_mev": float(d_b * 1000.0),
        "D_c111_mev": float(d_c * 1000.0),
        "Scc111_mev": float(s[2, 2] * 1000.0),
        "max_nonising_sym111_mev": float(max_nonising * 1000.0),
        "ising_ratio_abs": float(ising_ratio),
        "ising111_resid": finite_ratio(float(np.linalg.norm(s - scc_only)), sym_norm),
        "ising111_full_resid": finite_ratio(float(np.linalg.norm(m - ising_full)), full_norm),
        "xxz111_resid": finite_ratio(float(np.linalg.norm(s - xxz)), sym_norm),
        "xxz111_full_resid": finite_ratio(float(np.linalg.norm(m - xxz_full)), full_norm),
        "c3_axial_resid": finite_ratio(c3_nonaxial_norm, full_norm),
        "c3_nonaxial_norm_mev": float(c3_nonaxial_norm * 1000.0),
        "rank1_resid": finite_ratio(float(np.linalg.norm(s - rank1)), sym_norm),
        "axis_alignment111": float(abs(vec[2])),
        "lambda_max_mev": float(lam * 1000.0),
        "eig_anisotropy_abs": finite_ratio(abs(lam), other_max),
        "dm_norm_mev": float(dm_norm * 1000.0),
        "dm_over_sym": finite_ratio(dm_norm, sym_norm),
    }


def fmt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isinf(value):
            return "inf"
        if np.isnan(value):
            return "nan"
        return f"{value:.10g}"
    return str(value)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in ("family", "material", "bond", "branch", "run", "U"))
        grouped[key].append(row)

    out = []
    for key, items in sorted(grouped.items()):
        ratio = np.array([float(row["ratio"]) for row in items], dtype=float)
        abs_scc = np.array([abs(float(row["Scc111_mev"])) for row in items], dtype=float)
        abs_non = np.array([abs(float(row["max_nonising_sym111_mev"])) for row in items], dtype=float)
        ising_ratio = np.array([float(row["ising_ratio_abs"]) for row in items], dtype=float)
        ising_resid = np.array([float(row["ising111_resid"]) for row in items], dtype=float)
        ising_full_resid = np.array([float(row["ising111_full_resid"]) for row in items], dtype=float)
        xxz_resid = np.array([float(row["xxz111_resid"]) for row in items], dtype=float)
        xxz_full_resid = np.array([float(row["xxz111_full_resid"]) for row in items], dtype=float)
        c3_resid = np.array([float(row["c3_axial_resid"]) for row in items], dtype=float)
        c3_norm = np.array([float(row["c3_nonaxial_norm_mev"]) for row in items], dtype=float)
        rank1_resid = np.array([float(row["rank1_resid"]) for row in items], dtype=float)
        axis_align = np.array([float(row["axis_alignment111"]) for row in items], dtype=float)
        dm_norm = np.array([float(row["dm_norm_mev"]) for row in items], dtype=float)
        dm_over_sym = np.array([float(row["dm_over_sym"]) for row in items], dtype=float)
        finite_ising = ising_ratio[np.isfinite(ising_ratio)]
        if finite_ising.size == 0:
            min_ising = med_ising = float("inf")
        else:
            min_ising = float(np.min(finite_ising))
            med_ising = float(np.median(finite_ising))
        out.append({
            "family": key[0],
            "material": key[1],
            "bond": key[2],
            "branch": key[3],
            "run": key[4],
            "U": key[5],
            "n": len(items),
            "ratio_min": float(np.min(ratio)),
            "ratio_max": float(np.max(ratio)),
            "max_abs_Scc111_mev": float(np.max(abs_scc)),
            "max_abs_nonising_sym111_mev": float(np.max(abs_non)),
            "min_ising_ratio_abs": min_ising,
            "median_ising_ratio_abs": med_ising,
            "median_ising111_resid": float(np.median(ising_resid)),
            "max_ising111_resid": float(np.max(ising_resid)),
            "median_ising111_full_resid": float(np.median(ising_full_resid)),
            "median_xxz111_resid": float(np.median(xxz_resid)),
            "max_xxz111_resid": float(np.max(xxz_resid)),
            "median_xxz111_full_resid": float(np.median(xxz_full_resid)),
            "median_c3_axial_resid": float(np.median(c3_resid)),
            "max_c3_axial_resid": float(np.max(c3_resid)),
            "max_c3_nonaxial_norm_mev": float(np.max(c3_norm)),
            "median_rank1_resid": float(np.median(rank1_resid)),
            "max_rank1_resid": float(np.max(rank1_resid)),
            "min_axis_alignment111": float(np.min(axis_align)),
            "median_axis_alignment111": float(np.median(axis_align)),
            "max_dm_norm_mev": float(np.max(dm_norm)),
            "median_dm_over_sym": float(np.median(dm_over_sym)),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target", action="append", type=parse_target,
                        help="FAMILY/MATERIAL/RUN; repeatable")
    parser.add_argument("--all-targets", action="store_true",
                        help="discover every FAMILY/MATERIAL/RUN present under --root")
    parser.add_argument("--branch", action="append", choices=ALL_BRANCHES,
                        help="branch to include; repeatable; default: fopt_plus_sopt_direct")
    parser.add_argument("--bond", action="append", help="bond to include; repeatable")
    parser.add_argument("--detail-output", type=Path,
                        default=Path("outputs/analysis/rechx_111_exchange_detail.tsv"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("outputs/analysis/rechx_111_exchange_summary.tsv"))
    args = parser.parse_args()

    branches = set(args.branch or DEFAULT_BRANCHES)
    bond_filter = set(args.bond) if args.bond else None
    if args.all_targets and args.target:
        print("error: --all-targets cannot be combined with --target", file=sys.stderr)
        return 2
    targets = discover_targets(args.root, branches, bond_filter) if args.all_targets else (
        args.target or [parse_target(text) for text in DEFAULT_TARGETS]
    )

    rows: list[dict[str, object]] = []
    files_seen = 0
    for family, material, bond, branch, run, u_value, path in iter_u_files(args.root, targets, branches, bond_filter):
        files_seen += 1
        for ratio, residual, tensor in read_exchange_rows(path):
            row: dict[str, object] = {
                "family": family,
                "material": material,
                "bond": bond,
                "branch": branch,
                "run": run,
                "U": u_value,
                "ratio": ratio,
                "input_residual": residual,
            }
            row.update(decompose_111(tensor))
            rows.append(row)

    if files_seen == 0:
        print("error: no matching U_*.txt files found", file=sys.stderr)
        return 1

    summary = summarize(rows)
    write_tsv(args.detail_output, DETAIL_FIELDS, rows)
    write_tsv(args.summary_output, SUMMARY_FIELDS, summary)
    print(f"wrote {len(rows)} detail rows from {files_seen} files: {args.detail_output}")
    print(f"wrote {len(summary)} summary rows: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
