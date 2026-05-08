"""Local p-site downfolding from Wannier90 blocks to an effective f-f hopping.

This tool implements the minimal local ligand folding formula

    T_eff = T_ff + sum_p T_f1p (Ef I - H_pp)^(-1) T_pf2

where the output direction is the same as ``T_ff = block(f_site_i, f_site_j)``.
With the fexchange convention this is the hopping from site-j to site-i.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fexchange.tools.w90_extract import (
    _ENERGY_SCALE,
    _build_U_r2c,
    _expand_to_spinor,
    _extract_block,
    _ud_to_du_perm,
    read_hr,
)


def _load_toml(path: str | Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve_counts(
    n_orb_per_atom: list[int] | tuple[int, ...],
    *,
    num_wann: int,
    spinor: bool,
) -> list[int]:
    counts = [int(x) for x in n_orb_per_atom]
    total = sum(counts)
    if total == num_wann:
        return counts
    if spinor and 2 * total == num_wann:
        return [2 * x for x in counts]
    raise ValueError(
        "Could not reconcile n_orb_per_atom with num_wann: "
        f"sum={total}, spinor={spinor}, num_wann={num_wann}"
    )


def _offsets(counts: list[int]) -> list[int]:
    out = [0]
    for count in counts:
        out.append(out[-1] + count)
    return out


def _site_indices(counts: list[int], atom: int) -> list[int]:
    if atom < 0 or atom >= len(counts):
        raise ValueError(f"Atom index out of range: {atom}")
    offs = _offsets(counts)
    return list(range(offs[atom], offs[atom + 1]))


def _site(atom: int, cell: list[int] | tuple[int, int, int]) -> tuple[int, tuple[int, int, int]]:
    cell_tuple = tuple(int(x) for x in cell)
    if len(cell_tuple) != 3:
        raise ValueError(f"site cell must contain exactly 3 integers, got {cell!r}")
    return int(atom), cell_tuple


def _block(
    H_R: dict[tuple[int, int, int], np.ndarray],
    counts: list[int],
    lhs: tuple[int, tuple[int, int, int]],
    rhs: tuple[int, tuple[int, int, int]],
) -> np.ndarray:
    rows = _site_indices(counts, lhs[0])
    cols = _site_indices(counts, rhs[0])
    R = tuple(int(rhs[1][k]) - int(lhs[1][k]) for k in range(3))
    return _extract_block(H_R, rows, cols, R)


def _resolve_ef(
    ef: str | float | int | None,
    *,
    h_ii: np.ndarray,
    h_jj: np.ndarray,
) -> float:
    if ef is None or ef == "auto":
        mean_i = float(np.trace(h_ii).real / h_ii.shape[0])
        mean_j = float(np.trace(h_jj).real / h_jj.shape[0])
        return 0.5 * (mean_i + mean_j)
    return float(ef)


def _finalize_t_mu(t_raw: np.ndarray, *, spinor: bool) -> np.ndarray:
    if spinor:
        perm = _ud_to_du_perm(t_raw.shape[0])
        t_du = t_raw[np.ix_(perm, perm)].copy()
    else:
        t_du = _expand_to_spinor(t_raw)
    U = _build_U_r2c(spinor=True)
    return U @ t_du @ U.conj().T


def _fold_one_p_site(
    t_f1p: np.ndarray,
    t_pf2: np.ndarray,
    h_pp: np.ndarray,
    *,
    ef: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    evals = np.linalg.eigvalsh(0.5 * (h_pp + h_pp.conj().T))
    denom = ef - evals
    if np.any(np.isclose(denom, 0.0, atol=1e-12, rtol=0.0)):
        raise ValueError(f"Local p-site denominator is near zero at Ef={ef}")

    resolvent = ef * np.eye(h_pp.shape[0], dtype=complex) - h_pp
    try:
        response_times_t = np.linalg.solve(resolvent, t_pf2)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Singular local p-site resolvent at Ef={ef}") from exc
    delta_t = t_f1p @ response_times_t

    return delta_t, {
        "p_dim": int(h_pp.shape[0]),
        "eigenvalues_eV": [float(x) for x in evals],
        "denominators_eV": [float(x) for x in denom],
        "min_abs_denominator_eV": float(np.min(np.abs(denom))) if denom.size else 0.0,
        "correction_frobenius_eV": float(np.linalg.norm(delta_t)),
    }


def w90_ff_local_downfold(
    hr_path: str | Path,
    n_orb_per_atom: list[int] | tuple[int, ...],
    f_sites: list[tuple[int, list[int] | tuple[int, int, int]]] | tuple[
        tuple[int, list[int] | tuple[int, int, int]],
        tuple[int, list[int] | tuple[int, int, int]],
    ],
    p_sites: list[tuple[int, list[int] | tuple[int, int, int]]] | None = None,
    *,
    spinor: bool = False,
    energy_unit: str = "eV",
    ef: str | float | int | None = "auto",
    include_direct_ff: bool = True,
) -> dict[str, Any]:
    if energy_unit not in _ENERGY_SCALE:
        raise ValueError(f"Unknown energy_unit: {energy_unit!r}. Choose from {list(_ENERGY_SCALE)}")
    if len(f_sites) != 2:
        raise ValueError("Exactly two f sites are required")

    H_R, num_wann = read_hr(hr_path)
    scale = _ENERGY_SCALE[energy_unit]
    if scale != 1.0:
        H_R = {R: scale * mat for R, mat in H_R.items()}

    counts = _resolve_counts(n_orb_per_atom, num_wann=num_wann, spinor=spinor)
    f_i = _site(int(f_sites[0][0]), f_sites[0][1])
    f_j = _site(int(f_sites[1][0]), f_sites[1][1])
    p_list = [_site(int(site[0]), site[1]) for site in (p_sites or [])]

    h_ii = _block(H_R, counts, f_i, f_i)
    h_jj = _block(H_R, counts, f_j, f_j)
    if h_ii.shape != h_jj.shape:
        raise ValueError(f"f-site onsite block shape mismatch: {h_ii.shape} != {h_jj.shape}")
    if spinor and h_ii.shape[0] != 14:
        raise ValueError(f"spinor=True requires 14 f spin-orbitals per f site, got {h_ii.shape[0]}")
    if not spinor and h_ii.shape[0] != 7:
        raise ValueError(f"spinor=False requires 7 f orbitals per f site, got {h_ii.shape[0]}")

    ef_eV = _resolve_ef(ef, h_ii=h_ii, h_jj=h_jj)
    t_direct = _block(H_R, counts, f_i, f_j)
    t_indirect = np.zeros_like(t_direct, dtype=complex)
    diagnostics: list[dict[str, Any]] = []

    for idx, p_site in enumerate(p_list):
        t_f1p = _block(H_R, counts, f_i, p_site)
        t_pf2 = _block(H_R, counts, p_site, f_j)
        h_pp = _block(H_R, counts, p_site, p_site)
        delta_t, diag = _fold_one_p_site(t_f1p, t_pf2, h_pp, ef=ef_eV)
        t_indirect += delta_t
        diag.update(
            {
                "index": idx,
                "atom": int(p_site[0]),
                "cell": [int(x) for x in p_site[1]],
            }
        )
        diagnostics.append(diag)

    t_eff_raw = (t_direct if include_direct_ff else np.zeros_like(t_direct)) + t_indirect
    t_mu = _finalize_t_mu(t_eff_raw, spinor=spinor)

    return {
        "t_mu": t_mu,
        "t_eff_raw": t_eff_raw,
        "t_direct_raw": t_direct,
        "t_indirect_raw": t_indirect,
        "h_onsite_i_raw": h_ii,
        "h_onsite_j_raw": h_jj,
        "ef": ef_eV,
        "include_direct_ff": bool(include_direct_ff),
        "p_site_spectrum": diagnostics,
    }


def _save_matrix(path: Path, matrix: np.ndarray, *, scale: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, scale * matrix, fmt="%.12f")


def save_outputs(prefix: str | Path, result: dict[str, Any]) -> dict[str, Path]:
    prefix = Path(prefix)
    outputs = {
        "t_mu": prefix.with_name(prefix.name + "_t_mu.dat"),
        "t_eff_raw": prefix.with_name(prefix.name + "_t_eff_raw.dat"),
        "t_direct_raw": prefix.with_name(prefix.name + "_t_direct_raw.dat"),
        "t_indirect_raw": prefix.with_name(prefix.name + "_t_indirect_raw.dat"),
        "p_site_spectrum": prefix.with_name(prefix.name + "_p_site_spectrum.json"),
    }
    _save_matrix(outputs["t_mu"], result["t_mu"], scale=1000.0)
    _save_matrix(outputs["t_eff_raw"], result["t_eff_raw"])
    _save_matrix(outputs["t_direct_raw"], result["t_direct_raw"])
    _save_matrix(outputs["t_indirect_raw"], result["t_indirect_raw"])
    payload = {
        "ef_eV": float(result["ef"]),
        "include_direct_ff": bool(result["include_direct_ff"]),
        "units": {
            "t_mu": "meV",
            "raw_matrices": "eV",
            "p_site_spectrum": "eV",
        },
        "p_sites": result["p_site_spectrum"],
    }
    outputs["p_site_spectrum"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return outputs


def w90_ff_local_downfold_from_toml(toml_path: str | Path) -> dict[str, Any]:
    cfg = _load_toml(toml_path)
    fi = cfg["f_site_i"]
    fj = cfg["f_site_j"]
    p_cfg = cfg.get("p_sites", cfg.get("bridge_sites", cfg.get("ligands", [])))
    result = w90_ff_local_downfold(
        hr_path=cfg["hr_path"],
        n_orb_per_atom=cfg["n_orb_per_atom"],
        f_sites=[
            (int(fi["atom"]), [int(x) for x in fi["cell"]]),
            (int(fj["atom"]), [int(x) for x in fj["cell"]]),
        ],
        p_sites=[(int(site["atom"]), [int(x) for x in site["cell"]]) for site in p_cfg],
        spinor=bool(cfg.get("spinor", False)),
        energy_unit=str(cfg.get("energy_unit", "eV")),
        ef=cfg.get("ef", "auto"),
        include_direct_ff=bool(cfg.get("include_direct_ff", True)),
    )
    save_outputs(cfg.get("output", "ff_downfold"), result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python w90_ff_local_downfold.py config.toml")
        sys.exit(1)
    w90_ff_local_downfold_from_toml(sys.argv[1])
