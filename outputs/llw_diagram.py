"""
Lea-Leask-Wolf energy level diagrams for Oh symmetry.

Generates E/W vs x plots for J=4, 6, 8 with irrep coloring.
Reference: Lea, Leask, Wolf, J. Phys. Chem. Solids 23, 1381 (1962).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fexchange.core.stevens import build_cef_stevens_operators
from fexchange.spectrum.classify import classify_irreps
from fexchange.hamiltonian.hcef import build_hcef_matrix_J


J_CONFIG: dict[int, dict[str, list[str]]] = {
    4: {
        "ions_3p": ["Pr3+ (f2)", "Pm3+ (f4)"],
        "ions_4p": ["Nd4+ (f2)", "Sm4+ (f4)"],
    },
    6: {
        "ions_3p": ["Tb3+ (f8)", "Tm3+ (f12)"],
        "ions_4p": ["Dy4+ (f8)", "Yb4+ (f12)"],
    },
    8: {
        "ions_3p": ["Ho3+ (f10)"],
        "ions_4p": ["Er4+ (f10)"],
    },
}

IRREP_STYLE: dict[str, dict[str, str]] = {
    "Gamma1": {"color": "#C1121F", "label": "Gamma1 (A1)"},
    "Gamma2": {"color": "#1D3557", "label": "Gamma2 (A2)"},
    "Gamma3": {"color": "#2A9D8F", "label": "Gamma3 (E)"},
    "Gamma4": {"color": "#BC6C25", "label": "Gamma4 (T1)"},
    "Gamma5": {"color": "#7B2CBF", "label": "Gamma5 (T2)"},
    "unknown": {"color": "#8D99AE", "label": "unknown"},
}


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "fexchange-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))


def compute_F_factors(J: float) -> tuple[float, float]:
    """Compute LLW normalization factors F(4) and F(6) for a given J."""
    ops = build_cef_stevens_operators(J, symmetry="Oh")
    O4 = ops["O40"] + 5.0 * ops["O44c"]
    O6 = ops["O60"] - 21.0 * ops["O64c"]
    F4 = float(np.max(np.abs(np.linalg.eigvalsh(O4))))
    F6 = float(np.max(np.abs(np.linalg.eigvalsh(O6))))
    return F4, F6


def llw_scan(J: float, n_points: int = 801) -> dict[str, Any]:
    """Scan x in [-1, 1] and compute sorted energies and irrep labels."""
    F4, F6 = compute_F_factors(J)
    x_grid = np.linspace(-1.0, 1.0, n_points)
    dim = int(round(2.0 * J + 1.0))
    energies = np.zeros((n_points, dim), dtype=float)
    irreps: list[list[str]] = []

    for idx, x in enumerate(x_grid):
        H = build_hcef_matrix_J(
            J,
            {
                "B4": float(x / F4),
                "B6": float((1.0 - abs(x)) / F6),
            },
            symmetry="Oh",
        )
        evals, evecs = np.linalg.eigh(H)
        energies[idx, :] = np.real_if_close(evals)
        irreps.append(classify_irreps(J, evecs, point_group="Oh"))

    return {
        "J": float(J),
        "F4": F4,
        "F6": F6,
        "x_grid": x_grid,
        "energies": energies,
        "irreps": irreps,
    }


def plot_llw_diagram(J: int, scan: dict[str, Any], outpath: str | Path) -> None:
    """Plot E/W vs x for a single J manifold."""
    _configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    x_grid = np.asarray(scan["x_grid"], dtype=float)
    energies = np.asarray(scan["energies"], dtype=float)
    irreps = np.asarray(scan["irreps"], dtype=object)
    x_mesh = np.repeat(x_grid[:, None], energies.shape[1], axis=1)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    for column in range(energies.shape[1]):
        ax.plot(
            x_grid,
            energies[:, column],
            color="#D0D4DA",
            linewidth=0.8,
            alpha=0.7,
            zorder=1,
        )

    plotted_irreps: set[str] = set()
    for irrep_name, style in IRREP_STYLE.items():
        mask = irreps == irrep_name
        if not np.any(mask):
            continue
        ax.scatter(
            x_mesh[mask],
            energies[mask],
            s=8.0,
            color=style["color"],
            linewidths=0.0,
            label=style["label"] if irrep_name not in plotted_irreps else None,
            zorder=2,
        )
        plotted_irreps.add(irrep_name)

    y_min = float(np.min(energies))
    y_max = float(np.max(energies))
    pad = 0.06 * max(y_max - y_min, 1.0)
    ions = " / ".join(J_CONFIG[J]["ions_3p"] + J_CONFIG[J]["ions_4p"])

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("x (LLW parameter)")
    ax.set_ylabel("E / W")
    ax.set_title(f"LLW diagram for Oh symmetry, J = {J}\n{ions}")
    ax.grid(True, color="#E5E7EB", linewidth=0.8, alpha=0.9)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240)
    plt.close(fig)
    print(f"Saved: {outpath}")


def main() -> None:
    outdir = Path(__file__).resolve().parent
    for J in (4, 6, 8):
        print(f"\n=== J = {J} (2J+1 = {2 * J + 1} states) ===")
        F4, F6 = compute_F_factors(float(J))
        print(f"  F(4) = {F4:.6f}")
        print(f"  F(6) = {F6:.6f}")
        scan = llw_scan(float(J), n_points=801)
        plot_llw_diagram(J, scan, outdir / f"llw_J{J}.png")


if __name__ == "__main__":
    main()
