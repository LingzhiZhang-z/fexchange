"""Analyze ``w90_extract`` onsite output for per-site SOC, ligand Delta, and f1 CEF.

The input is the multi-block ``onsite.txt`` written by
``fexchange.tools.w90_extract.extract_w90_two_bond``.  This module performs
exactly three analyses, per the project spec:

1. **Per-site SOC strength.** For each f site (14x14, l=3) and each ligand
   p site (6x6, l=1), decompose the onsite block into ``E*I + zeta*L.S +
   h_orbital`` via the spin-channel L.S projection.  Output: ``zeta_f1,
   zeta_f2, lambda_p_lig1, lambda_p_lig2``.
2. **Per-(f, ligand) charge-transfer Delta.**  For every pair
   ``(fi, lig_k)``, ``Delta = trace(h_fi)/14 - trace(h_lig_k)/6``.  Four
   values total.
3. **CEF Stevens fit on f1 only.**  Project ``h_f1`` into the LSJM SOC-lowest
   ``J`` manifold (using ``zeta_f1`` and the project's default Slater ratios),
   remove the scalar trace, and least-squares fit real Stevens templates.
   Default symmetry is ``C3v`` with the ``q=3`` ``sin`` convention.

The public output (``analyze_onsite``) is structured per-site/per-pair; see
``schema_version = "fxe.w90_onsite.v2"``.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fexchange.core.stevens import build_cef_stevens_operators
from fexchange.io.matrix import load_txt_blocks
from fexchange.utils.constants import N_ORB, RE_DEFAULTS_BY_N_ELE, RE_TO_N_ELE
from fexchange.utils.errors import FexchangeError, InputError, NumError

Symmetry = Literal["Oh", "C3v"]
ModeQ3 = Literal["cos", "sin"]

_ELL_BY_SHELL: dict[str, int] = {"f": 3, "p": 1}
_F_SITES: tuple[str, ...] = ("f1", "f2")
_LIG_SITES: tuple[str, ...] = ("lig1", "lig2")


# ---------------------------------------------------------------------------
# Block I/O
# ---------------------------------------------------------------------------

def load_onsite_blocks(path: str | Path) -> dict[str, NDArray[np.complexfloating]]:
    """Load ``onsite.txt`` and return the four raw site blocks.

    Returned keys: ``h_f1``, ``h_f2`` (14x14 each), ``h_lig1``, ``h_lig2``
    (6x6 each).  No aggregation or trace removal is performed here.
    """
    blocks = load_txt_blocks(Path(path), {"h_f1": 196, "h_f2": 196, "h_lig1": 36, "h_lig2": 36})
    return {
        "h_f1": blocks["h_f1"].reshape(14, 14),
        "h_f2": blocks["h_f2"].reshape(14, 14),
        "h_lig1": blocks["h_lig1"].reshape(6, 6),
        "h_lig2": blocks["h_lig2"].reshape(6, 6),
    }


# ---------------------------------------------------------------------------
# Single-site L.S decomposition (works for any l)
# ---------------------------------------------------------------------------

def decompose_site_soc(
    h_site: NDArray[np.complexfloating],
    *,
    ell: int,
    hermitian_atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Decompose a single ``(2l+1)*2``-dim onsite block into ``zeta + orbital``.

    Convention: spin index is fast (interleaved ``(down, up)`` per ``m``),
    matching ``fexchange.core.space_ls.orbital_map``.  The decomposition reads
    ``zeta`` from three independent spin-channel projections; the
    spin-diagonal channel is the primary estimate.  The orbital-only residual
    ``h_orbital_traceless`` and its bandwidth are returned as diagnostics
    (expected to be ~0 for a clean ligand p shell).
    """
    h = np.asarray(h_site, dtype=np.complex128)
    dim_orb = 2 * int(ell) + 1
    dim = dim_orb * 2
    if h.shape != (dim, dim):
        raise InputError(
            "FXE-INPUT-003",
            f"h_site must be {dim}x{dim} for ell={ell}",
            actual={"shape": list(h.shape), "ell": int(ell)},
        )
    hermitian_residual = _relative_norm(h - h.conj().T, h)
    if not np.allclose(h, h.conj().T, atol=hermitian_atol):
        raise NumError(
            "FXE-NUM-001",
            "h_site is not Hermitian within tolerance",
            module="w90_onsite",
            actual={
                "hermitian_residual": hermitian_residual,
                "atol": hermitian_atol,
                "ell": int(ell),
            },
        )

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
    zeta_diag_channels = np.diag(h_uu - h_dd)[mask].real / m_vals[mask]

    coeffs = np.sqrt([ell * (ell + 1) - m * (m + 1) for m in range(-ell, ell)])
    zeta_du_channels = 2.0 * np.array([h_du[c + 1, c] for c in range(2 * ell)]).real / coeffs
    zeta_ud_channels = 2.0 * np.array([h_ud[c, c + 1] for c in range(2 * ell)]).real / coeffs

    zeta = float(np.mean(zeta_diag_channels))
    h_orb = 0.5 * (h_uu + h_dd)
    h_orb_trace = np.trace(h_orb) / dim_orb
    h_orb_traceless = h_orb - h_orb_trace * np.eye(dim_orb, dtype=complex)

    h_model = np.kron(h_orb, np.eye(2, dtype=complex)) + zeta * _build_ls_operator(int(ell))
    residual = _relative_norm(h0 - h_model, h0)

    all_channels = np.concatenate([zeta_diag_channels, zeta_du_channels, zeta_ud_channels])
    mean_channels = float(np.mean(all_channels))
    spread = (
        float((np.max(all_channels) - np.min(all_channels)) / abs(mean_channels))
        if abs(mean_channels) > 0.0
        else float("inf")
    )
    bandwidth = float(np.ptp(np.linalg.eigvalsh(h_orb_traceless)).real)

    return {
        "ell": int(ell),
        "trace_avg": complex(trace_avg),
        "zeta": zeta,
        "zeta_channel_spread_rel": spread,
        "soc_decomposition_residual": residual,
        "hermitian_residual": hermitian_residual,
        "h_orbital_traceless": h_orb_traceless,
        "h_orbital_bandwidth": bandwidth,
    }


# ---------------------------------------------------------------------------
# Per-(f, ligand) Delta
# ---------------------------------------------------------------------------

def compute_delta_pairs(blocks: dict[str, NDArray[np.complexfloating]]) -> dict[str, float]:
    """Return ``Delta[fi_lig_k] = trace(h_fi)/14 - trace(h_lig_k)/6`` for the 4 pairs."""

    def center(h: NDArray[np.complexfloating]) -> float:
        return float(np.trace(h).real / h.shape[0])

    return {
        f"{f}_{lig}": center(blocks[f"h_{f}"]) - center(blocks[f"h_{lig}"])
        for f in _F_SITES
        for lig in _LIG_SITES
    }


# ---------------------------------------------------------------------------
# f1 Stevens CEF fit
# ---------------------------------------------------------------------------

def fit_stevens_direct_local(
    h_f1: NDArray[np.complexfloating],
    *,
    n_ele: int,
    zeta: float,
    symmetry: Symmetry = "C3v",
    mode_q3: ModeQ3 = "sin",
) -> dict[str, Any]:
    """Project ``h_f1`` to the LSJM lowest-J manifold and fit Stevens B's."""
    h = np.asarray(h_f1, dtype=np.complex128)
    if h.shape != (14, 14):
        raise InputError("FXE-INPUT-003", "h_f1 must be 14x14", actual={"shape": list(h.shape)})
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
        r42=ratios["r42"],
        r62=ratios["r62"],
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


# ---------------------------------------------------------------------------
# Top-level analysis (schema v2)
# ---------------------------------------------------------------------------

def analyze_onsite(
    onsite: str | Path,
    *,
    n_ele: int,
    symmetry: Symmetry = "C3v",
    mode_q3: ModeQ3 = "sin",
    fit_cef: bool = True,
) -> dict[str, Any]:
    """Run the three onsite analyses and return a structured per-site result."""
    blocks = load_onsite_blocks(onsite)

    soc_f = {
        label: decompose_site_soc(blocks[f"h_{label}"], ell=_ELL_BY_SHELL["f"])
        for label in _F_SITES
    }
    soc_p = {
        label: decompose_site_soc(blocks[f"h_{label}"], ell=_ELL_BY_SHELL["p"])
        for label in _LIG_SITES
    }
    delta = compute_delta_pairs(blocks)

    stevens = None
    if fit_cef:
        stevens = fit_stevens_direct_local(
            blocks["h_f1"],
            n_ele=int(n_ele),
            zeta=float(soc_f["f1"]["zeta"]),
            symmetry=symmetry,
            mode_q3=mode_q3,
        )

    return {
        "schema_version": "fxe.w90_onsite.v2",
        "onsite": str(onsite),
        "n_ele": int(n_ele),
        "zeta": {
            label: {
                "eV": float(soc_f[label]["zeta"]),
                "meV": 1000.0 * float(soc_f[label]["zeta"]),
                "channel_spread_rel": float(soc_f[label]["zeta_channel_spread_rel"]),
                "residual": float(soc_f[label]["soc_decomposition_residual"]),
            }
            for label in _F_SITES
        },
        "lambda_p": {
            label: {
                "eV": float(soc_p[label]["zeta"]),
                "meV": 1000.0 * float(soc_p[label]["zeta"]),
                "channel_spread_rel": float(soc_p[label]["zeta_channel_spread_rel"]),
                "residual": float(soc_p[label]["soc_decomposition_residual"]),
                "orbital_bandwidth": float(soc_p[label]["h_orbital_bandwidth"]),
            }
            for label in _LIG_SITES
        },
        "delta": {
            key: {"eV": float(value), "meV": 1000.0 * float(value)}
            for key, value in delta.items()
        },
        "cef": None if stevens is None else {
            "point_group": stevens["symmetry"],
            "mode_q3": stevens["mode_q3"],
            "J": float(stevens["J0"]),
            "B_params_eV": {name: float(value) for name, value in stevens["B_params"].items()},
            "B_params_meV": {name: 1000.0 * float(value) for name, value in stevens["B_params"].items()},
            "stevens_fit_residual": float(stevens["stevens_fit_residual"]),
        },
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_analysis_outputs(result: dict[str, Any], out_dir: str | Path) -> None:
    """Write concise text summary and CEF-state TOML."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    (target / "onsite_params.txt").write_text("\n".join(_summary_lines(result)) + "\n", encoding="utf-8")
    cef_toml = target / "cef_REChX_C3v_sin.toml"
    if result.get("cef") is None:
        cef_toml.unlink(missing_ok=True)
    else:
        cef_toml.write_text(
            "\n".join(_cef_states_toml_lines(result)) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# RE / n_ele resolution
# ---------------------------------------------------------------------------

def infer_re_from_path(path: str | Path) -> str | None:
    """Infer the RE symbol from a ``data/`` or ``data-DFT/`` style onsite path."""
    parts = Path(path).parts
    for part in reversed(parts):
        match = re.match(r"^([A-Z][a-z]?)[A-Z]", part)
        if match and match.group(1) in RE_TO_N_ELE:
            return match.group(1)
    return None


def resolve_n_ele(*, n_ele: int | None, re_name: str | None, onsite: str | Path) -> tuple[int, str | None]:
    """Resolve f electron count from explicit ``n_ele``, ``--RE``, or path inference."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onsite", type=Path, help="w90_extract onsite.txt")
    parser.add_argument("--RE", default="auto", help="Rare-earth symbol; default: infer from path")
    parser.add_argument("--n-ele", type=int, help="Override f electron count")
    parser.add_argument("--symmetry", choices=("C3v", "Oh"), default="C3v")
    parser.add_argument("--mode-q3", choices=("sin", "cos"), default="sin")
    parser.add_argument("--no-cef", action="store_true", help="Only extract onsite SOC/Delta; skip LSJM Stevens fit")
    parser.add_argument("--output-dir", type=Path, help="Write parameter / CEF TOML files")
    args = parser.parse_args(argv)

    try:
        n_ele, re_name = resolve_n_ele(n_ele=args.n_ele, re_name=args.RE, onsite=args.onsite)
        result = analyze_onsite(
            args.onsite,
            n_ele=n_ele,
            symmetry=args.symmetry,
            mode_q3=args.mode_q3,
            fit_cef=not args.no_cef,
        )
    except FexchangeError as exc:
        print(exc.payload_json(), file=sys.stderr)
        return 1
    result["RE"] = re_name

    if args.output_dir:
        write_analysis_outputs(result, args.output_dir)

    print("\n".join(_summary_lines(result)))
    return 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

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


@lru_cache(maxsize=4)
def _build_ls_operator(ell: int) -> NDArray[np.complexfloating]:
    """Return the ``(2l+1)*2``-dim ``L.S`` matrix in the ``(m, sigma=down/up)`` interleaved basis."""
    dim_orb = 2 * int(ell) + 1
    m_vals = np.arange(-ell, ell + 1, dtype=float)
    Lz = np.diag(m_vals).astype(complex)
    Lp = np.zeros((dim_orb, dim_orb), dtype=complex)
    for c, m in enumerate(m_vals[:-1]):
        Lp[c + 1, c] = np.sqrt(ell * (ell + 1) - m * (m + 1))
    Lm = Lp.conj().T
    Lx = 0.5 * (Lp + Lm)
    Ly = -0.5j * (Lp - Lm)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = -0.5j * np.array([[0, -1], [1, 0]], dtype=complex)
    Sz = 0.5 * np.diag([-1.0, 1.0]).astype(complex)
    return np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)


def _build_stevens_templates(
    J0: float, *, symmetry: Symmetry, mode_q3: ModeQ3
) -> dict[str, NDArray[np.complexfloating]]:
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
        "# w90_onsite summary",
        f"RE {result.get('RE') or 'unknown'}",
        f"n_ele {int(result['n_ele'])}",
        "",
        "[re_soc]",
    ]
    for label in _F_SITES:
        entry = result["zeta"][label]
        lines.append(f"zeta_{label}_eV {float(entry['eV']):.12e}")
        lines.append(f"zeta_{label}_fit_residual {float(entry['residual']):.12e}")
        lines.append(f"zeta_{label}_channel_spread_rel {float(entry['channel_spread_rel']):.12e}")
    zeta_diff = abs(float(result["zeta"]["f1"]["eV"]) - float(result["zeta"]["f2"]["eV"]))
    lines.append(f"zeta_f1_f2_diff_eV {zeta_diff:.12e}")
    lines.append("")
    lines.append("[ligand]")
    for label in _LIG_SITES:
        entry = result["lambda_p"][label]
        lines.append(f"lambda_p_{label}_eV {float(entry['eV']):.12e}")
        lines.append(f"lambda_p_{label}_fit_residual {float(entry['residual']):.12e}")
        lines.append(f"lambda_p_{label}_channel_spread_rel {float(entry['channel_spread_rel']):.12e}")
    lines.append("")
    lines.append("[delta]")
    for key, entry in result["delta"].items():
        lines.append(f"delta_{key}_eV {float(entry['eV']):.12e}")
    delta_lig1_diff = abs(float(result["delta"]["f1_lig1"]["eV"]) - float(result["delta"]["f2_lig1"]["eV"]))
    delta_lig2_diff = abs(float(result["delta"]["f1_lig2"]["eV"]) - float(result["delta"]["f2_lig2"]["eV"]))
    lines.append(f"delta_lig1_f1_f2_diff_eV {delta_lig1_diff:.12e}")
    lines.append(f"delta_lig2_f1_f2_diff_eV {delta_lig2_diff:.12e}")
    if cef is None:
        lines.append("")
        lines.append("[cef]")
        lines.append("skipped true")
    else:
        lines.append("")
        lines.append("[cef]")
        lines.append(f"point_group {cef['point_group']}")
        lines.append(f"mode_q3 {cef['mode_q3']}")
        lines.append(f"J {float(cef['J']):.12e}")
        lines.append(f"stevens_fit_residual {float(cef['stevens_fit_residual']):.12e}")
        for name, value in cef["B_params_meV"].items():
            lines.append(f"{name}_meV {float(value):.12e}")
    return lines


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
