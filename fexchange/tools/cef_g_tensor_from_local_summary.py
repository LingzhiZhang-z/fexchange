"""Build CEF Kramers states and g-tensors from local-summary Stevens fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fexchange.hamiltonian.hcef import build_hcef_matrix_J
from fexchange.spectrum.ground import select_kramers_doublet


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _complex_scalar(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def _complex_matrix_payload(value: np.ndarray) -> list[list[dict[str, float]]]:
    return [[_complex_scalar(x) for x in row] for row in np.asarray(value)]


def _real_matrix_payload(value: np.ndarray) -> list[list[float]]:
    arr = np.asarray(value)
    return [[float(x.real) for x in row] for row in arr]


def _lande_g(L: float, S: float, J: float) -> float:
    if J == 0.0:
        raise ValueError("Landé g factor is undefined for J=0")
    return float(1.0 + (J * (J + 1.0) + S * (S + 1.0) - L * (L + 1.0)) / (2.0 * J * (J + 1.0)))


def _decompose_summary_path(row: dict[str, Any]) -> Path:
    return Path(str(row["output_prefix"]) + "_decompose_summary.json")


def _load_lsj_quantum_numbers(row: dict[str, Any]) -> tuple[float, float, float]:
    path = _decompose_summary_path(row)
    decomp = _load_json(path)
    scheme_b = decomp["scheme_b"]
    return float(scheme_b["L0"]), float(scheme_b["S0"]), float(scheme_b["J0"])


def _npz_scalar(value: Any) -> np.ndarray:
    return np.asarray(value)


def compute_row(
    row: dict[str, Any],
    *,
    c_g_mode: str,
    write_per_material: bool,
) -> dict[str, Any]:
    material = str(row["material"])
    n_ele = int(row["n_ele"])
    if n_ele % 2 != 1:
        raise ValueError(f"{material}: Kramers g-tensor requires odd n_ele, got {n_ele}")

    symmetry = str(row["symmetry"])
    mode_q3 = str(row.get("mode_q3", "cos"))
    J = float(row["J0"])
    B_params = {str(k): float(v) for k, v in row["B_params_meV"].items()}
    L, S, J_from_decomp = _load_lsj_quantum_numbers(row)
    if abs(J_from_decomp - J) > 1e-10:
        raise ValueError(f"{material}: J0 mismatch local_summary={J} decompose_summary={J_from_decomp}")

    if c_g_mode == "lande":
        c_g = _lande_g(L, S, J)
    elif c_g_mode == "unit":
        c_g = 1.0
    else:
        raise ValueError(f"Unknown c_g mode: {c_g_mode}")

    H = build_hcef_matrix_J(J, B_params, symmetry=symmetry, mode_q3=mode_q3)
    evals, evecs = np.linalg.eigh(H)
    k_out = select_kramers_doublet(J, evals, evecs, c_g=c_g, point_group=symmetry)

    output_prefix = Path(row["output_prefix"])
    npz_path = output_prefix.with_name(output_prefix.name + "_kramers.npz")
    json_path = output_prefix.with_name(output_prefix.name + "_kramers_summary.json")

    g_tensor = np.asarray(k_out["g_tensor"])
    Lambda = np.asarray(k_out["Lambda"])
    g_principal = np.linalg.svd(g_tensor, compute_uv=False)
    g_real = np.real(g_tensor)
    g_offdiag = g_real.copy()
    np.fill_diagonal(g_offdiag, 0.0)
    splitting = evals - evals[0]

    summary = {
        "schema_version": "fxe.cef_g_tensor_from_local_summary.v1",
        "material": material,
        "family": row.get("family"),
        "site": row.get("site"),
        "source_config": row.get("config"),
        "source_output_prefix": row.get("output_prefix"),
        "kramer_npz": str(npz_path),
        "kramer_summary_json": str(json_path),
        "n_ele": n_ele,
        "L": L,
        "S": S,
        "J": J,
        "point_group": symmetry,
        "mode_q3": mode_q3,
        "c_g_mode": c_g_mode,
        "c_g": c_g,
        "B_params_meV": B_params,
        "cef_eigenvalues_meV": [float(x) for x in evals],
        "cef_splitting_meV": [float(x) for x in splitting],
        "ground_irrep": k_out.get("irrep"),
        "allowed_multipoles": k_out.get("allowed_multipoles", []),
        "g_tensor": _complex_matrix_payload(g_tensor),
        "g_tensor_real": _real_matrix_payload(g_tensor),
        "g_tensor_imag_fro": float(np.linalg.norm(g_tensor.imag, ord="fro")),
        "g_diagonal_real": [float(g_real[i, i]) for i in range(3)],
        "g_offdiag_real_fro": float(np.linalg.norm(g_offdiag, ord="fro")),
        "g_principal_values": [float(x) for x in g_principal],
        "Lambda": _complex_matrix_payload(Lambda),
        "time_reversal_residual": float(k_out["gauge_meta"]["tr_residual"]),
        "j_basis_order": "M=-J..J ascending",
        "kramer_columns": ["k1", "k2"],
    }

    if write_per_material:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            W=np.asarray(k_out["kramer_vectors"]),
            kramer_vectors=np.asarray(k_out["kramer_vectors"]),
            kramer_labels=np.asarray([0, 1]),
            eigenvalues_meV=np.asarray(evals),
            cef_splitting_meV=np.asarray(splitting),
            Lambda=Lambda,
            g_tensor=g_tensor,
            M_Jx=np.asarray(k_out["M_Jx"]),
            M_Jy=np.asarray(k_out["M_Jy"]),
            M_Jz=np.asarray(k_out["M_Jz"]),
            c_g=_npz_scalar(c_g),
            c_g_mode=_npz_scalar(c_g_mode),
            L=_npz_scalar(L),
            S=_npz_scalar(S),
            J=_npz_scalar(J),
            n_ele=_npz_scalar(n_ele),
            material=_npz_scalar(material),
            point_group=_npz_scalar(symmetry),
            mode_q3=_npz_scalar(mode_q3),
            ground_irrep=_npz_scalar(str(k_out.get("irrep", "unknown"))),
            doublet_type=_npz_scalar("kramers"),
            schema_version=_npz_scalar("fxe.cef_g_tensor_from_local_summary.v1"),
            unit=_npz_scalar("meV"),
            j_basis_order=_npz_scalar("M=-J..J ascending"),
        )
        _dump_json(json_path, summary)

    return summary


def run(
    *,
    summary_path: Path,
    output_path: Path,
    family: str,
    materials: set[str] | None,
    c_g_mode: str,
    write_per_material: bool,
) -> list[dict[str, Any]]:
    rows = _load_json(summary_path)
    selected = []
    for row in rows:
        if family and row.get("family") != family:
            continue
        if materials is not None and row.get("material") not in materials:
            continue
        selected.append(row)

    out = [
        compute_row(row, c_g_mode=c_g_mode, write_per_material=write_per_material)
        for row in selected
    ]
    _dump_json(output_path, out)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CEF Kramers states and g-tensors from local_summary.json."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/REChX_REX3/local_summary.json"),
        help="Input local summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/REChX_REX3/rechx_g_tensor_summary.json"),
        help="Output aggregate g-tensor summary JSON.",
    )
    parser.add_argument(
        "--family",
        default="REChX",
        help="Only process rows with this family value. Use an empty string for all.",
    )
    parser.add_argument(
        "--material",
        action="append",
        help="Restrict to one material. May be passed multiple times.",
    )
    parser.add_argument(
        "--c-g",
        choices=["lande", "unit"],
        default="lande",
        help="Magnetic prefactor for g_tensor = c_g * Lambda.",
    )
    parser.add_argument(
        "--no-per-material",
        action="store_true",
        help="Only write the aggregate JSON; skip per-material NPZ/JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(
        summary_path=args.summary,
        output_path=args.output,
        family=args.family,
        materials=set(args.material) if args.material else None,
        c_g_mode=args.c_g,
        write_per_material=not args.no_per_material,
    )
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
