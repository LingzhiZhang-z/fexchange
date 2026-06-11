#!/usr/bin/env python3
"""Correlate REChX NNN DM exchange with DFT hopping diagnostics.

Exchange inputs come from the organized REChX_cubic tree. Hopping inputs come
from data/data-DFT-input/<family>/<material>/<bond>/w90.

For every NNN J2 bond, branch/run/U, this reports bond-local DM values from the
current J/K/Gamma/Gamma'/D decomposition and attaches hopping measures:

  direct_imag_frac    ||Im T_direct|| / ||T_direct||
  downfold_imag_frac  ||Im T_downfold|| / ||T_downfold||
  correction_over_direct ||T_downfold - T_direct|| / ||T_direct||
  fp_cancel_frac      ||T_lig1/Delta1 + T_lig2/Delta2|| /
                      (||T_lig1/Delta1|| + ||T_lig2/Delta2||)

The fp cancellation measure uses the same block convention as the fixed-ligand
downfold code, but with scalar 1/Delta propagators; it is intended as a path
cancellation diagnostic, not as a replacement for the actual downfold matrix.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_bond_exchange import PERM, decompose  # noqa: E402

DEFAULT_EXCHANGE_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized/REChX_cubic")
DEFAULT_DFT_INPUT_ROOT = Path("data/data-DFT-input")
DEFAULT_BRANCHES = ("fopt_plus_sopt_direct",)
ALL_BRANCHES = ("sopt", "sopt_direct", "fopt", "fopt_plus_sopt_direct")
J2_AXIS = {
    "J2-000": "z",
    "J2-180": "z",
    "J2-060": "y",
    "J2-240": "y",
    "J2-120": "x",
    "J2-300": "x",
}
EPS = 1.0e-14

FIELDS = [
    "family",
    "material",
    "bond",
    "axis",
    "branch",
    "run",
    "U",
    "n",
    "ratio_min",
    "ratio_max",
    "max_dm_norm_mev",
    "median_dm_norm_mev",
    "median_dm_over_sym",
    "max_abs_D_mev",
    "max_abs_Dprime_mev",
    "median_abs_D_mev",
    "median_abs_Dprime_mev",
    "max_sym_norm_mev",
    "direct_norm_eV",
    "direct_imag_frac",
    "direct_transpose_antisym_frac",
    "downfold_norm_eV",
    "downfold_imag_frac",
    "downfold_transpose_antisym_frac",
    "correction_norm_eV",
    "correction_over_direct",
    "fp_lig1_norm_eV",
    "fp_lig2_norm_eV",
    "fp_sum_norm_eV",
    "fp_cancel_frac",
    "delta_lig1_eV",
    "delta_lig2_eV",
    "lambda_lig1_eV",
    "lambda_lig2_eV",
]


def parse_target(text: str) -> tuple[str, str]:
    parts = [part for part in text.strip("/").split("/") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target must be FAMILY/MATERIAL")
    return parts[0], parts[1]


def parse_tokens(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("["):
            continue
        if line.startswith("#"):
            line = line[1:].strip()
        if "=" in line:
            key, val = (x.strip() for x in line.split("=", 1))
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts
        out[key] = val.strip().strip('"')
    return out


def read_blocks(path: Path) -> dict[str, np.ndarray]:
    blocks: dict[str, list[complex]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            blocks[current] = []
            continue
        if current is None:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        blocks[current].append(float(parts[0]) + 1j * float(parts[1]))

    out: dict[str, np.ndarray] = {}
    for name, values in blocks.items():
        arr = np.array(values, dtype=complex)
        if arr.size == 196:
            out[name] = arr.reshape(14, 14)
        elif arr.size == 84:
            out[name] = arr.reshape(14, 6)
        else:
            out[name] = arr
    return out


def norm(mat: np.ndarray | None) -> float:
    if mat is None:
        return float("nan")
    return float(np.linalg.norm(mat))


def ratio(num: float, den: float) -> float:
    if np.isnan(num) or np.isnan(den):
        return float("nan")
    if abs(den) < EPS:
        return float("inf") if abs(num) >= EPS else 0.0
    return num / den


def matrix_metrics(mat: np.ndarray | None, prefix: str) -> dict[str, float]:
    if mat is None:
        return {
            f"{prefix}_norm_eV": float("nan"),
            f"{prefix}_imag_frac": float("nan"),
            f"{prefix}_transpose_antisym_frac": float("nan"),
        }
    nrm = norm(mat)
    return {
        f"{prefix}_norm_eV": nrm,
        f"{prefix}_imag_frac": ratio(float(np.linalg.norm(mat.imag)), nrm),
        f"{prefix}_transpose_antisym_frac": ratio(float(np.linalg.norm(mat - mat.T)), nrm),
    }


def hopping_metrics(dft_root: Path, family: str, material: str, bond: str) -> dict[str, float]:
    w90 = dft_root / family / material / bond / "w90"
    direct = downfold = None
    fp_blocks: dict[str, np.ndarray] = {}

    direct_path = w90 / "hopping_ff_direct.txt"
    if direct_path.exists():
        direct = read_blocks(direct_path).get("t_mu")
    downfold_path = w90 / "hopping_ff_downfold.txt"
    if downfold_path.exists():
        downfold = read_blocks(downfold_path).get("t_mu")
    fp_path = w90 / "hopping_fp.txt"
    if fp_path.exists():
        fp_blocks = read_blocks(fp_path)

    info = parse_tokens(w90 / "downfold.toml")
    delta1 = float(info.get("delta_lig1", "nan"))
    delta2 = float(info.get("delta_lig2", "nan"))
    lambda1 = float(info.get("lambda_lig1", "nan"))
    lambda2 = float(info.get("lambda_lig2", "nan"))

    out: dict[str, float] = {}
    out.update(matrix_metrics(direct, "direct"))
    out.update(matrix_metrics(downfold, "downfold"))
    if direct is not None and downfold is not None:
        corr = downfold - direct
        corr_norm = norm(corr)
    else:
        corr_norm = float("nan")
    out["correction_norm_eV"] = corr_norm
    out["correction_over_direct"] = ratio(corr_norm, out["direct_norm_eV"])

    t11 = fp_blocks.get("t_f1_lig1")
    t21 = fp_blocks.get("t_f2_lig1")
    t12 = fp_blocks.get("t_f1_lig2")
    t22 = fp_blocks.get("t_f2_lig2")
    path1 = path2 = None
    if t11 is not None and t21 is not None and np.isfinite(delta1) and abs(delta1) >= EPS:
        path1 = t11 @ t21.conj().T / delta1
    if t12 is not None and t22 is not None and np.isfinite(delta2) and abs(delta2) >= EPS:
        path2 = t12 @ t22.conj().T / delta2
    n1 = norm(path1)
    n2 = norm(path2)
    if path1 is not None and path2 is not None:
        nsum = norm(path1 + path2)
        cancel = ratio(nsum, n1 + n2)
    else:
        nsum = cancel = float("nan")
    out.update({
        "fp_lig1_norm_eV": n1,
        "fp_lig2_norm_eV": n2,
        "fp_sum_norm_eV": nsum,
        "fp_cancel_frac": cancel,
        "delta_lig1_eV": delta1,
        "delta_lig2_eV": delta2,
        "lambda_lig1_eV": lambda1,
        "lambda_lig2_eV": lambda2,
    })
    return out


def read_exchange_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        tensor = np.array([float(x) for x in cols[3:12]], dtype=float).reshape(3, 3)
        yield float(cols[0]), tensor


def sym_norm_in_bond_frame(tensor: np.ndarray, axis: str) -> float:
    p = list(PERM[axis])
    m = tensor[np.ix_(p, p)]
    return float(np.linalg.norm(0.5 * (m + m.T)))


def iter_u_files(exchange_root: Path, targets: set[tuple[str, str]] | None, branches: set[str]):
    for family_root in sorted(path for path in exchange_root.iterdir() if path.is_dir()):
        family = family_root.name
        for material_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            material = material_root.name
            if targets is not None and (family, material) not in targets:
                continue
            for bond, axis in J2_AXIS.items():
                bond_root = material_root / bond
                if not bond_root.exists():
                    continue
                for branch in sorted(branches):
                    branch_root = bond_root / branch
                    if not branch_root.exists():
                        continue
                    for path in sorted(branch_root.glob("*/U_*.txt")):
                        try:
                            u_value = int(path.stem.split("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        run = path.parts[-2]
                        yield family, material, bond, axis, branch, run, u_value, path


def collect(exchange_root: Path, dft_root: Path, targets: set[tuple[str, str]] | None, branches: set[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, float]]] = defaultdict(list)
    for family, material, bond, axis, branch, run, u_value, path in iter_u_files(exchange_root, targets, branches):
        for ratio_value, tensor in read_exchange_rows(path):
            vals = decompose(tensor, axis)
            d_norm = float(np.sqrt(vals["D_a"] ** 2 + vals["D_b"] ** 2 + vals["D_c"] ** 2))
            s_norm = sym_norm_in_bond_frame(tensor, axis)
            grouped[(family, material, bond, axis, branch, run, u_value)].append({
                "ratio": ratio_value,
                "dm_norm": d_norm,
                "dm_over_sym": ratio(d_norm, s_norm),
                "D": vals["D_c"],
                "Dprime": 0.5 * (vals["D_a"] + vals["D_b"]),
                "sym_norm": s_norm,
            })

    hop_cache: dict[tuple[str, str, str], dict[str, float]] = {}
    out: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        family, material, bond, axis, branch, run, u_value = key
        hop_key = (str(family), str(material), str(bond))
        if hop_key not in hop_cache:
            hop_cache[hop_key] = hopping_metrics(dft_root, str(family), str(material), str(bond))

        ratio_values = np.array([row["ratio"] for row in rows], dtype=float)
        dm_norms = np.array([row["dm_norm"] for row in rows], dtype=float)
        dm_over_sym = np.array([row["dm_over_sym"] for row in rows], dtype=float)
        d_values = np.array([row["D"] for row in rows], dtype=float)
        dp_values = np.array([row["Dprime"] for row in rows], dtype=float)
        sym_norms = np.array([row["sym_norm"] for row in rows], dtype=float)

        item: dict[str, object] = {
            "family": family,
            "material": material,
            "bond": bond,
            "axis": axis,
            "branch": branch,
            "run": run,
            "U": u_value,
            "n": len(rows),
            "ratio_min": float(np.min(ratio_values)),
            "ratio_max": float(np.max(ratio_values)),
            "max_dm_norm_mev": float(np.max(dm_norms) * 1000.0),
            "median_dm_norm_mev": float(np.median(dm_norms) * 1000.0),
            "median_dm_over_sym": float(np.median(dm_over_sym)),
            "max_abs_D_mev": float(np.max(np.abs(d_values)) * 1000.0),
            "max_abs_Dprime_mev": float(np.max(np.abs(dp_values)) * 1000.0),
            "median_abs_D_mev": float(np.median(np.abs(d_values)) * 1000.0),
            "median_abs_Dprime_mev": float(np.median(np.abs(dp_values)) * 1000.0),
            "max_sym_norm_mev": float(np.max(sym_norms) * 1000.0),
        }
        item.update(hop_cache[hop_key])
        out.append(item)
    return out


def fmt(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf"
        return f"{value:.10g}"
    return str(value)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-root", type=Path, default=DEFAULT_EXCHANGE_ROOT)
    parser.add_argument("--dft-input-root", type=Path, default=DEFAULT_DFT_INPUT_ROOT)
    parser.add_argument("--target", action="append", type=parse_target,
                        help="FAMILY/MATERIAL; repeatable")
    parser.add_argument("--branch", action="append", choices=ALL_BRANCHES,
                        help="branch to include; repeatable; default: fopt_plus_sopt_direct")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/analysis/rechx_nnn_dm_hopping_summary.tsv"))
    args = parser.parse_args()

    targets = set(args.target) if args.target else None
    branches = set(args.branch or DEFAULT_BRANCHES)
    rows = collect(args.exchange_root, args.dft_input_root, targets, branches)
    if not rows:
        print("error: no matching NNN exchange rows found", file=sys.stderr)
        return 1
    write_tsv(args.output, rows)
    print(f"wrote {len(rows)} rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
