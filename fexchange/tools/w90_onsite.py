"""Analyze ``w90_extract`` onsite output for RE local SOC and CEF parameters.

The input is the multi-block ``onsite.txt`` written by
``fexchange.tools.w90_extract.extract_w90_two_bond``.  The two f-site onsite
blocks are averaged into

    H_f_local = 0.5 * (h_f1 + h_f2)

and then decomposed in the fexchange canonical f basis
``m=-3..3`` with interleaved ``(down, up)`` spin order.

For REChX use, the CEF fit defaults to C3v with the q=3 ``sin`` convention.
The Stevens fit follows the direct-local route: project the full 14x14
``H_f_local`` into the LSJM SOC-lowest J manifold, remove the scalar trace, and
fit real Stevens coefficients there.  The public output is intentionally small:
``zeta`` plus the CEF Stevens parameters needed by ``cef_states.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fexchange.core.space_ls import build_space_ls_operator
from fexchange.core.stevens import build_cef_stevens_operators
from fexchange.io.matrix import load_txt_blocks
from fexchange.utils.constants import N_ORB, RE_DEFAULTS_BY_N_ELE, RE_TO_N_ELE
from fexchange.utils.errors import FexchangeError, InputError, NumError

Symmetry = Literal["Oh", "C3v"]
ModeQ3 = Literal["cos", "sin"]


def load_hlocal_from_onsite(path: str | Path) -> dict[str, NDArray[np.complexfloating] | complex]:
    """Load ``onsite.txt`` and return averaged f-site local Hamiltonians."""
    onsite = Path(path)
    blocks = load_txt_blocks(onsite, {"h_f1": 196, "h_f2": 196, "h_lig1": 36, "h_lig2": 36})
    h_f1 = blocks["h_f1"].reshape(14, 14)
    h_f2 = blocks["h_f2"].reshape(14, 14)
    h_local = 0.5 * (h_f1 + h_f2)
    trace_avg = np.trace(h_local) / 14.0
    h_local_traceless = h_local - trace_avg * np.eye(14, dtype=complex)
    return {
        "h_f1": h_f1,
        "h_f2": h_f2,
        "h_local": h_local,
        "h_local_traceless": h_local_traceless,
        "trace_avg": trace_avg,
    }


def decompose_hlocal_soc_cef(
    h_local: NDArray[np.complexfloating],
    *,
    hermitian_atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Extract zeta/lambda_RE and an orbital CEF matrix from a 14x14 Hlocal."""
    h = np.asarray(h_local, dtype=np.complex128)
    if h.shape != (14, 14):
        raise InputError("FXE-INPUT-003", "h_local must be 14x14", actual={"shape": list(h.shape)})
    hermitian_residual = _relative_norm(h - h.conj().T, h)
    if not np.allclose(h, h.conj().T, atol=hermitian_atol):
        raise NumError(
            "FXE-NUM-001",
            "h_local is not Hermitian within tolerance",
            module="w90_onsite",
            actual={"hermitian_residual": hermitian_residual, "atol": hermitian_atol},
        )

    trace_avg = np.trace(h) / 14.0
    h0 = h - trace_avg * np.eye(14, dtype=complex)

    dn = np.arange(0, 14, 2)
    up = np.arange(1, 14, 2)
    h_uu = h0[np.ix_(up, up)]
    h_dd = h0[np.ix_(dn, dn)]
    h_du = h0[np.ix_(dn, up)]
    h_ud = h0[np.ix_(up, dn)]

    ell = 3
    m_vals = np.arange(-ell, ell + 1, dtype=float)
    mask = m_vals != 0
    zeta_diag_channels = np.diag(h_uu - h_dd)[mask].real / m_vals[mask]

    coeffs = np.sqrt([ell * (ell + 1) - m * (m + 1) for m in range(-ell, ell)])
    zeta_du_channels = 2.0 * np.array([h_du[c + 1, c] for c in range(2 * ell)]).real / coeffs
    zeta_ud_channels = 2.0 * np.array([h_ud[c, c + 1] for c in range(2 * ell)]).real / coeffs

    zeta_diag = float(np.mean(zeta_diag_channels))
    zeta_offdiag = float(0.5 * (np.mean(zeta_du_channels) + np.mean(zeta_ud_channels)))

    # The project convention takes the spin-diagonal channel as the primary
    # zeta estimate, matching the main-branch diagnostic tool.
    zeta = zeta_diag
    h_cef = 0.5 * (h_uu + h_dd)
    h_cef_trace = np.trace(h_cef) / 7.0
    h_cef_traceless = h_cef - h_cef_trace * np.eye(7, dtype=complex)

    h_model = np.kron(h_cef, np.eye(2, dtype=complex)) + zeta * _build_hsoc_unit_operator_1b()
    residual = _relative_norm(h0 - h_model, h0)
    all_channels = np.concatenate([zeta_diag_channels, zeta_du_channels, zeta_ud_channels])
    mean_channels = np.mean(all_channels)
    spread = (
        float((np.max(all_channels) - np.min(all_channels)) / abs(mean_channels))
        if abs(mean_channels) > 0.0
        else float("inf")
    )
    bandwidth = float(np.ptp(np.linalg.eigvalsh(h_cef_traceless)).real)

    return {
        "trace_avg": trace_avg,
        "h_local_traceless": h0,
        "zeta": zeta,
        "lambda_RE": zeta,
        "zeta_diag": zeta_diag,
        "zeta_offdiag": zeta_offdiag,
        "zeta_diag_channels": zeta_diag_channels,
        "zeta_du_channels": zeta_du_channels,
        "zeta_ud_channels": zeta_ud_channels,
        "zeta_channel_spread_rel": spread,
        "soc_decomposition_residual": residual,
        "hermitian_residual": hermitian_residual,
        "h_cef_orbital": h_cef,
        "h_cef_orbital_traceless": h_cef_traceless,
        "h_cef_orbital_bandwidth": bandwidth,
    }


def fit_stevens_direct_local(
    h_local: NDArray[np.complexfloating],
    *,
    n_ele: int,
    zeta: float,
    symmetry: Symmetry = "C3v",
    mode_q3: ModeQ3 = "sin",
) -> dict[str, Any]:
    """Project full Hlocal to the LSJM ground J manifold and fit Stevens B's."""
    h = np.asarray(h_local, dtype=np.complex128)
    if h.shape != (14, 14):
        raise InputError("FXE-INPUT-003", "h_local must be 14x14", actual={"shape": list(h.shape)})
    if not 1 <= int(n_ele) <= 13:
        raise InputError("FXE-INPUT-003", "n_ele must be in [1, 13]", actual={"n_ele": int(n_ele)})

    try:
        from fexchange.core.fermion import one_body_operator_matrix
        from fexchange.spectrum.lsjm import select_soc_lowest_subspace
    except ModuleNotFoundError as exc:
        raise InputError(
            "FXE-INPUT-003",
            "Stevens LSJM projection requires the project runtime dependency scipy; install project dependencies",
            actual={"missing_module": exc.name},
        ) from exc

    trace_avg = np.trace(h) / 14.0
    h0 = h - trace_avg * np.eye(14, dtype=complex)
    h_fock = one_body_operator_matrix(h0, int(n_ele), N_ORB)

    lsjm = _lsjm_for_n(int(n_ele))
    ratios = _slater_ratio_defaults(int(n_ele))
    soc0 = select_soc_lowest_subspace(
        lsjm,
        F2=1.0,
        F4=ratios["r42"],
        F6=ratios["r62"],
        zeta=float(zeta),
    )
    U_j0 = soc0["U_n_soc0"]
    h_j0 = U_j0.conj().T @ h_fock @ U_j0

    trace_avg_j0 = np.trace(h_j0) / h_j0.shape[0]
    h_j0_0 = h_j0 - trace_avg_j0 * np.eye(h_j0.shape[0], dtype=complex)

    templates = _build_stevens_templates(float(soc0["J0"]), symmetry=symmetry, mode_q3=mode_q3)
    names, coeffs, fit_0 = _real_least_squares_coeffs(templates, h_j0_0)

    return {
        "symmetry": symmetry,
        "mode_q3": mode_q3,
        "alpha0": soc0["alpha0"],
        "L0": soc0["L0"],
        "S0": soc0["S0"],
        "J0": soc0["J0"],
        "n_j": soc0["n_j"],
        "stevens_fit_residual": _relative_norm(h_j0_0 - fit_0, h_j0_0),
        "B_params": {name: float(coeffs[idx]) for idx, name in enumerate(names)},
    }


def analyze_onsite(
    onsite: str | Path,
    *,
    n_ele: int,
    symmetry: Symmetry = "C3v",
    mode_q3: ModeQ3 = "sin",
) -> dict[str, Any]:
    """Full onsite analysis returning only zeta and CEF parameters."""
    loaded = load_hlocal_from_onsite(onsite)
    h_local = np.asarray(loaded["h_local"], dtype=np.complex128)
    decomp = decompose_hlocal_soc_cef(h_local)
    stevens = fit_stevens_direct_local(
        h_local,
        n_ele=int(n_ele),
        zeta=float(decomp["zeta"]),
        symmetry=symmetry,
        mode_q3=mode_q3,
    )
    return {
        "schema_version": "fxe.w90_onsite.v1",
        "onsite": str(onsite),
        "n_ele": int(n_ele),
        "zeta_eV": float(decomp["zeta"]),
        "zeta_meV": 1000.0 * float(decomp["zeta"]),
        "cef": {
            "point_group": stevens["symmetry"],
            "mode_q3": stevens["mode_q3"],
            "J": float(stevens["J0"]),
            "B_params_eV": {name: float(value) for name, value in stevens["B_params"].items()},
            "B_params_meV": {name: 1000.0 * float(value) for name, value in stevens["B_params"].items()},
        },
    }


def write_analysis_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    """Write compact zeta/CEF parameter files."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    lines = _summary_lines(result)
    (target / "onsite_params.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (target / "onsite_params.json").write_text(
        json.dumps(_json_summary(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "cef_REChX_C3v_sin.toml").write_text(
        "\n".join(_cef_states_toml_lines(result)) + "\n",
        encoding="utf-8",
    )


def infer_re_from_path(path: str | Path) -> str | None:
    """Infer the RE symbol from a data/data-DFT style onsite path."""
    parts = Path(path).parts
    for part in reversed(parts):
        match = re.match(r"^([A-Z][a-z]?)[A-Z]", part)
        if match and match.group(1) in RE_TO_N_ELE:
            return match.group(1)
    return None


def resolve_n_ele(*, n_ele: int | None, re_name: str | None, onsite: str | Path) -> tuple[int, str | None]:
    """Resolve f electron count from explicit n_ele, RE, or path inference."""
    if n_ele is not None:
        return int(n_ele), re_name
    resolved_re = None if re_name in {None, "auto"} else str(re_name)
    if resolved_re is None:
        resolved_re = infer_re_from_path(onsite)
    if resolved_re is None or resolved_re not in RE_TO_N_ELE:
        raise InputError(
            "FXE-INPUT-003",
            "Cannot infer RE/n_ele; pass --RE or --n-ele",
            actual={"RE": re_name, "onsite": str(onsite)},
        )
    return RE_TO_N_ELE[resolved_re], resolved_re


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onsite", type=Path, help="w90_extract onsite.txt")
    parser.add_argument("--RE", default="auto", help="Rare-earth symbol; default: infer from path")
    parser.add_argument("--n-ele", type=int, help="Override f electron count")
    parser.add_argument("--symmetry", choices=("C3v", "Oh"), default="C3v")
    parser.add_argument("--mode-q3", choices=("sin", "cos"), default="sin")
    parser.add_argument("--output-dir", type=Path, help="Write zeta/CEF params and a cef_states TOML file")
    parser.add_argument("--json", action="store_true", help="Print JSON summary instead of text")
    args = parser.parse_args(argv)

    try:
        n_ele, re_name = resolve_n_ele(n_ele=args.n_ele, re_name=args.RE, onsite=args.onsite)
        result = analyze_onsite(args.onsite, n_ele=n_ele, symmetry=args.symmetry, mode_q3=args.mode_q3)
    except FexchangeError as exc:
        print(exc.payload_json(), file=sys.stderr)
        return 1
    result["RE"] = re_name

    if args.output_dir:
        write_analysis_outputs(result, args.output_dir)

    if args.json:
        print(json.dumps(_json_summary(result), indent=2, sort_keys=True))
    else:
        print("\n".join(_summary_lines(result)))
    return 0


@lru_cache(maxsize=None)
def _lsjm_for_n(n_ele: int) -> dict[str, Any]:
    try:
        from fexchange.spectrum.lsjm import build_lsjm
        from fexchange.spectrum.lsms import build_lsms
    except ModuleNotFoundError as exc:
        raise InputError(
            "FXE-INPUT-003",
            "LSJM construction requires the project runtime dependency scipy; install project dependencies",
            actual={"missing_module": exc.name},
        ) from exc

    ratios = _slater_ratio_defaults(n_ele)
    lsms = build_lsms(n_ele, N_ORB, r42=ratios["r42"], r62=ratios["r62"])
    return build_lsjm(lsms, N_ORB)


def _slater_ratio_defaults(n_ele: int) -> dict[str, float]:
    defaults = RE_DEFAULTS_BY_N_ELE[int(n_ele)]
    f2 = float(defaults["F2_per_Jh"])
    return {
        "r42": float(defaults["F4_per_Jh"]) / f2,
        "r62": float(defaults["F6_per_Jh"]) / f2,
    }


@lru_cache(maxsize=1)
def _build_hsoc_unit_operator_1b() -> NDArray[np.complexfloating]:
    one = build_space_ls_operator()
    return one["Lx"] @ one["Sx"] + one["Ly"] @ one["Sy"] + one["Lz"] @ one["Sz"]


def _build_stevens_templates(J0: float, *, symmetry: Symmetry, mode_q3: ModeQ3) -> dict[str, NDArray[np.complexfloating]]:
    ops = build_cef_stevens_operators(J0, symmetry=symmetry, mode_q3=mode_q3)
    if symmetry == "Oh":
        return {
            "B4": ops["O40"] + 5.0 * ops["O44c"],
            "B6": ops["O60"] - 21.0 * ops["O64c"],
        }
    return {
        "B20": ops["O20"],
        "B40": ops["O40"],
        "B60": ops["O60"],
        "B66": ops["O66"],
        "B43": ops["O43_eta"],
        "B63": ops["O63_eta"],
    }


def _real_least_squares_coeffs(
    template_map: dict[str, NDArray[np.complexfloating]],
    target: NDArray[np.complexfloating],
) -> tuple[list[str], NDArray[np.floating], NDArray[np.complexfloating]]:
    names = list(template_map)
    design = np.column_stack([template_map[name].reshape(-1) for name in names])
    design_r = np.vstack([design.real, design.imag])
    target_r = np.concatenate([target.reshape(-1).real, target.reshape(-1).imag])
    coeffs, _, _, _ = np.linalg.lstsq(design_r, target_r, rcond=None)
    fit = sum(coeffs[idx] * template_map[name] for idx, name in enumerate(names))
    return names, coeffs, fit


def _relative_norm(delta: NDArray[np.complexfloating], ref: NDArray[np.complexfloating]) -> float:
    denom = float(np.linalg.norm(ref))
    numer = float(np.linalg.norm(delta))
    if denom < 1.0e-14:
        return 0.0 if numer < 1.0e-14 else float("inf")
    return numer / denom


def _summary_lines(result: dict[str, Any]) -> list[str]:
    cef = result["cef"]
    lines = [
        "# w90_onsite zeta and REChX CEF parameters",
        f"RE {result.get('RE') or 'unknown'}",
        f"n_ele {int(result['n_ele'])}",
        f"zeta_eV {float(result['zeta_eV']):.12e}",
        f"zeta_meV {float(result['zeta_meV']):.12e}",
        f"point_group {cef['point_group']}",
        f"mode_q3 {cef['mode_q3']}",
        f"J {float(cef['J']):.12e}",
    ]
    for name, value in cef["B_params_eV"].items():
        lines.append(f"{name}_eV {float(value):.12e}")
        lines.append(f"{name}_meV {float(cef['B_params_meV'][name]):.12e}")
    return lines


def _json_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result["schema_version"],
        "RE": result.get("RE"),
        "n_ele": int(result["n_ele"]),
        "zeta_eV": float(result["zeta_eV"]),
        "zeta_meV": float(result["zeta_meV"]),
        "cef": result["cef"],
    }


def _cef_states_toml_lines(result: dict[str, Any]) -> list[str]:
    cef = result["cef"]
    lines = [
        f'point_group = "{cef["point_group"]}"',
        f"J = {float(cef['J']):.12e}",
        f'mode_q3 = "{cef["mode_q3"]}"',
        f'kramer_name = "{_default_kramer_name(result)}"',
        "",
    ]
    for name, value in cef["B_params_meV"].items():
        lines.append(f"{name} = {float(value):.12e}")
    return lines


def _default_kramer_name(result: dict[str, Any]) -> str:
    re_name = result.get("RE") or "RE"
    return f"{re_name}_REChX_C3v_sin"


if __name__ == "__main__":
    raise SystemExit(main())
