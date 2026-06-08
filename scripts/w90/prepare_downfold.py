#!/usr/bin/env python3
"""Prepare w90_downfold TOML and diagnostics from one bond w90 directory."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

N_F = 14
N_P = 6
SOC_LIGANDS = {"S", "Br", "I"}
ZERO_SOC_LIGANDS = {"O", "F", "Cl"}


def toml_string(path: str | Path) -> str:
    text = str(path)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_txt_blocks(path: Path) -> dict[str, NDArray[np.complexfloating]]:
    blocks: dict[str, list[complex]] = {}
    key: str | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            key = text[1:-1]
            blocks[key] = []
            continue
        if key is None:
            raise ValueError(f"{path}:{line_no}: data before block header")
        parts = text.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: expected 'real imag'")
        blocks[key].append(float(parts[0]) + 1j * float(parts[1]))
    return {name: np.asarray(values, dtype=np.complex128) for name, values in blocks.items()}


def load_onsite(path: Path) -> dict[str, NDArray[np.complexfloating]]:
    raw = load_txt_blocks(path)
    expected = {"h_f1": N_F * N_F, "h_f2": N_F * N_F, "h_lig1": N_P * N_P, "h_lig2": N_P * N_P}
    missing = sorted(set(expected) - set(raw))
    if missing:
        raise ValueError(f"{path}: missing blocks {missing}")
    return {
        key: raw[key].reshape((N_F, N_F) if key.startswith("h_f") else (N_P, N_P))
        for key in expected
    }


def relative_norm(delta: NDArray[np.complexfloating], ref: NDArray[np.complexfloating]) -> float:
    denom = float(np.linalg.norm(ref))
    numer = float(np.linalg.norm(delta))
    return numer / denom if denom > 0.0 else numer


def ls_operator(ell: int) -> NDArray[np.complexfloating]:
    dim_orb = 2 * int(ell) + 1
    m_vals = np.arange(-ell, ell + 1, dtype=float)
    lz = np.diag(m_vals).astype(complex)
    lp = np.zeros((dim_orb, dim_orb), dtype=complex)
    for col, m in enumerate(m_vals[:-1]):
        lp[col + 1, col] = np.sqrt(ell * (ell + 1) - m * (m + 1))
    lm = lp.conj().T
    lx = 0.5 * (lp + lm)
    ly = -0.5j * (lp - lm)
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = -0.5j * np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    sz = 0.5 * np.diag([-1.0, 1.0]).astype(complex)
    return np.kron(lx, sx) + np.kron(ly, sy) + np.kron(lz, sz)


def soc_diagnostics(
    h_site: NDArray[np.complexfloating],
    *,
    ell: int,
    hermitian_atol: float,
) -> dict[str, Any]:
    h = np.asarray(h_site, dtype=np.complex128)
    dim_orb = 2 * int(ell) + 1
    dim = 2 * dim_orb
    if h.shape != (dim, dim):
        raise ValueError(f"expected {dim}x{dim} matrix for ell={ell}, got {h.shape}")

    hermitian_residual = relative_norm(h - h.conj().T, h)
    if not np.allclose(h, h.conj().T, atol=hermitian_atol):
        raise ValueError(f"onsite block is not Hermitian: residual={hermitian_residual:.6e}")

    trace_avg = np.trace(h) / dim
    h0 = h - trace_avg * np.eye(dim, dtype=complex)

    dn = np.arange(0, dim, 2)
    up = np.arange(1, dim, 2)
    h_uu = h0[np.ix_(up, up)]
    h_dd = h0[np.ix_(dn, dn)]
    h_du = h0[np.ix_(dn, up)]
    h_ud = h0[np.ix_(up, dn)]

    m_vals = np.arange(-ell, ell + 1, dtype=float)
    mask = m_vals != 0
    diag_channels = np.diag(h_uu - h_dd)[mask].real / m_vals[mask]

    coeffs = np.sqrt([ell * (ell + 1) - m * (m + 1) for m in range(-ell, ell)])
    du_channels = 2.0 * np.array([h_du[col + 1, col] for col in range(2 * ell)]).real / coeffs
    ud_channels = 2.0 * np.array([h_ud[col, col + 1] for col in range(2 * ell)]).real / coeffs

    zeta = float(np.mean(diag_channels))
    h_orb = 0.5 * (h_uu + h_dd)
    h_orb_trace = np.trace(h_orb) / dim_orb
    h_orb_traceless = h_orb - h_orb_trace * np.eye(dim_orb, dtype=complex)
    h_model = np.kron(h_orb, np.eye(2, dtype=complex)) + zeta * ls_operator(int(ell))

    all_channels = np.concatenate([diag_channels, du_channels, ud_channels])
    channel_mean = float(np.mean(all_channels))
    channel_range = float(np.max(all_channels) - np.min(all_channels))
    spread_rel = channel_range / abs(channel_mean) if abs(channel_mean) > 0.0 else float("inf")

    return {
        "lambda_eV": zeta,
        "channel_spread_rel": spread_rel,
        "soc_decomposition_residual": relative_norm(h0 - h_model, h0),
        "hermitian_residual": hermitian_residual,
        "orbital_bandwidth_eV": float(np.ptp(np.linalg.eigvalsh(h_orb_traceless)).real),
    }


def trace_center(h: NDArray[np.complexfloating]) -> float:
    return float(np.trace(h).real / h.shape[0])


def p5_denominator(delta: float, lambda_p: float) -> dict[str, Any]:
    coeffs = np.array([-0.5, -0.5, -0.5, -0.5, 1.0, 1.0], dtype=float)
    values = float(delta) + float(lambda_p) * coeffs
    return {
        "values_eV": [float(x) for x in values],
        "min_eV": float(np.min(values)),
        "max_eV": float(np.max(values)),
        "min_abs_eV": float(np.min(np.abs(values))),
        "has_negative": bool(np.any(values < 0.0)),
    }


def read_ligand_elements(w90_dir: Path) -> dict[str, str]:
    bond_info = w90_dir.parent / "bond_info.txt"
    if not bond_info.exists():
        raise FileNotFoundError(bond_info)

    elements: list[str] = []
    for line in bond_info.read_text(encoding="utf-8").splitlines():
        match = re.search(r"\bbridge\s+([A-Z][a-z]?)#\d+", line)
        if match:
            elements.append(match.group(1))
    if len(elements) != 2:
        raise ValueError(f"{bond_info}: expected 2 bridge ligands, got {len(elements)}")
    return {"lig1": elements[0], "lig2": elements[1]}


def lambda_for_downfold(element: str, fitted_lambda: float) -> tuple[float, str]:
    if element in SOC_LIGANDS:
        return float(fitted_lambda), "fit"
    if element in ZERO_SOC_LIGANDS:
        return 0.0, "set_zero"
    raise ValueError(f"unknown ligand element for lambda_p rule: {element}")


def write_downfold_toml(
    path: Path,
    *,
    w90_dir: Path,
    params: dict[str, float],
    report: dict[str, Any],
    degenerate_tol: float,
) -> None:
    lines = [
        "# Generated by scripts/w90/prepare_downfold.py",
        "# Required fields below are consumed by fexchange.tools.w90_downfold.",
        "",
        f"hopping_fp_in = {toml_string((w90_dir / 'hopping_fp.txt').resolve())}",
        f"direct_t_mu_in = {toml_string((w90_dir / 'hopping_ff_direct.txt').resolve())}",
        f"output = {toml_string((w90_dir / 'hopping_ff_downfold.txt').resolve())}",
        "",
        f"delta_lig1 = {params['delta_lig1']:.16e}",
        f"delta_lig2 = {params['delta_lig2']:.16e}",
        f"lambda_lig1 = {params['lambda_lig1']:.16e}",
        f"lambda_lig2 = {params['lambda_lig2']:.16e}",
        f"degenerate_tol = {float(degenerate_tol):.16e}",
        "",
        "# Checks, not read by w90_downfold:",
    ]
    for lig in ("lig1", "lig2"):
        lig_report = report["ligands"][lig]
        lines.append(f"# {lig}_element = {lig_report['element']}")
        lines.append(f"# {lig}_lambda_source = {lig_report['lambda_source']}")
        lines.append(f"# {lig}_lambda_fit_eV = {lig_report['lambda_fit_eV']:.12e}")
        lines.append(f"# {lig}_lambda_fit_residual = {lig_report['lambda_fit_residual']:.12e}")
    for key, value in report["delta_consistency"].items():
        lines.append(f"# {key} = {value:.12e}")
    for lig, stats in report["p5_denominator"].items():
        lines.append(f"# p5_{lig}_min_abs_eV = {stats['min_abs_eV']:.12e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    params = report["params"]
    print("downfold.toml")
    for key in ("delta_lig1", "delta_lig2", "lambda_lig1", "lambda_lig2"):
        print(f"  {key}_eV = {params[key]:.12e}")
    for lig in ("lig1", "lig2"):
        lig_report = report["ligands"][lig]
        print(
            f"  {lig} {lig_report['element']}: "
            f"lambda_fit={lig_report['lambda_fit_eV']:.12e} eV, "
            f"residual={lig_report['lambda_fit_residual']:.12e}, "
            f"source={lig_report['lambda_source']}"
        )
    for key, value in report["delta_consistency"].items():
        print(f"  {key} = {value:.12e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("w90_dir", type=Path, help="Bond w90 directory containing onsite/hopping outputs")
    parser.add_argument("--degenerate-tol", type=float, default=1.0e-6)
    parser.add_argument("--hermitian-atol", type=float, default=1.0e-8)
    args = parser.parse_args()

    w90_dir = args.w90_dir.resolve(strict=True)
    ligand_elements = read_ligand_elements(w90_dir)
    onsite = w90_dir / "onsite.txt"
    hopping_fp = w90_dir / "hopping_fp.txt"
    hopping_ff_direct = w90_dir / "hopping_ff_direct.txt"
    for path in (onsite, hopping_fp, hopping_ff_direct):
        if not path.exists():
            raise FileNotFoundError(path)

    blocks = load_onsite(onsite)
    centers = {site: trace_center(block) for site, block in blocks.items()}
    raw_delta = {
        "f1_lig1": centers["h_f1"] - centers["h_lig1"],
        "f2_lig1": centers["h_f2"] - centers["h_lig1"],
        "f1_lig2": centers["h_f1"] - centers["h_lig2"],
        "f2_lig2": centers["h_f2"] - centers["h_lig2"],
    }
    delta_lig1 = 0.5 * (raw_delta["f1_lig1"] + raw_delta["f2_lig1"])
    delta_lig2 = 0.5 * (raw_delta["f1_lig2"] + raw_delta["f2_lig2"])

    soc = {
        "h_lig1": soc_diagnostics(blocks["h_lig1"], ell=1, hermitian_atol=float(args.hermitian_atol)),
        "h_lig2": soc_diagnostics(blocks["h_lig2"], ell=1, hermitian_atol=float(args.hermitian_atol)),
    }
    lambda_lig1, lambda_lig1_source = lambda_for_downfold(ligand_elements["lig1"], soc["h_lig1"]["lambda_eV"])
    lambda_lig2, lambda_lig2_source = lambda_for_downfold(ligand_elements["lig2"], soc["h_lig2"]["lambda_eV"])

    params = {
        "delta_lig1": float(delta_lig1),
        "delta_lig2": float(delta_lig2),
        "lambda_lig1": float(lambda_lig1),
        "lambda_lig2": float(lambda_lig2),
    }
    report = {
        "schema_version": "fxe.w90.prepare_downfold.v1",
        "w90_dir": str(w90_dir),
        "onsite": str(onsite),
        "params": params,
        "ligands": {
            "lig1": {
                "element": ligand_elements["lig1"],
                "lambda_source": lambda_lig1_source,
                "lambda_fit_eV": float(soc["h_lig1"]["lambda_eV"]),
                "lambda_fit_residual": float(soc["h_lig1"]["soc_decomposition_residual"]),
            },
            "lig2": {
                "element": ligand_elements["lig2"],
                "lambda_source": lambda_lig2_source,
                "lambda_fit_eV": float(soc["h_lig2"]["lambda_eV"]),
                "lambda_fit_residual": float(soc["h_lig2"]["soc_decomposition_residual"]),
            },
        },
        "delta_consistency": {
            "lig1_f1_minus_f2_abs_eV": abs(raw_delta["f1_lig1"] - raw_delta["f2_lig1"]),
            "lig2_f1_minus_f2_abs_eV": abs(raw_delta["f1_lig2"] - raw_delta["f2_lig2"]),
        },
        "p5_denominator": {
            "lig1": p5_denominator(params["delta_lig1"], params["lambda_lig1"]),
            "lig2": p5_denominator(params["delta_lig2"], params["lambda_lig2"]),
        },
    }

    write_downfold_toml(
        w90_dir / "downfold.toml",
        w90_dir=w90_dir,
        params=params,
        report=report,
        degenerate_tol=args.degenerate_tol,
    )
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
