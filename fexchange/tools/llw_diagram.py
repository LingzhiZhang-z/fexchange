"""
Lea-Leask-Wolf energy level diagrams for Oh symmetry, all f^n ground-state J.

Covers f1-f13 (skipping f6 J=0). For each unique J, sweeps the LLW
parameter x in [-1, 1] with B4 = x/F(4), B6 = (1-|x|)/F(6), and plots
E/W vs x with irrep coloring.

Usage:
    # All unique J values
    python tools/llw_diagram.py

    # Single J
    python tools/llw_diagram.py --J 4

    # Specific f^n
    python tools/llw_diagram.py --fn 3

    # Custom resolution and output directory
    python tools/llw_diagram.py --npoints 401 --outdir plots/
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fexchange.core.stevens import build_cef_stevens_operators
from fexchange.hamiltonian.hcef import build_hcef_matrix_J
from fexchange.spectrum.classify import build_projectors

# ---------------------------------------------------------------------------
# f^n ground-state table (Hund's rules)
# ---------------------------------------------------------------------------

FN_TABLE: list[dict[str, Any]] = [
    {"n": 1,  "J": 2.5,  "term": "²F₅/₂",  "ions": "Ce³⁺"},
    {"n": 2,  "J": 4.0,  "term": "³H₄",    "ions": "Pr³⁺"},
    {"n": 3,  "J": 4.5,  "term": "⁴I₉/₂",  "ions": "Nd³⁺"},
    {"n": 4,  "J": 4.0,  "term": "⁵I₄",    "ions": "Pm³⁺"},
    {"n": 5,  "J": 2.5,  "term": "⁶H₅/₂",  "ions": "Sm³⁺"},
    # f6: J=0, skip
    {"n": 7,  "J": 3.5,  "term": "⁸S₇/₂",  "ions": "Gd³⁺"},
    {"n": 8,  "J": 6.0,  "term": "⁷F₆",    "ions": "Tb³⁺"},
    {"n": 9,  "J": 7.5,  "term": "⁶H₁₅/₂", "ions": "Dy³⁺"},
    {"n": 10, "J": 8.0,  "term": "⁵I₈",    "ions": "Ho³⁺"},
    {"n": 11, "J": 7.5,  "term": "⁴I₁₅/₂", "ions": "Er³⁺"},
    {"n": 12, "J": 6.0,  "term": "³H₆",    "ions": "Tm³⁺"},
    {"n": 13, "J": 3.5,  "term": "²F₇/₂",  "ions": "Yb³⁺"},
]

# Group by unique J for plotting
def _entries_for_J(J: float) -> list[dict[str, Any]]:
    return [e for e in FN_TABLE if e["J"] == J]

UNIQUE_J: list[float] = sorted(set(e["J"] for e in FN_TABLE))

# ---------------------------------------------------------------------------
# Irrep styles: single-valued (integer J) + double-valued (half-integer J)
# ---------------------------------------------------------------------------

IRREP_STYLE: dict[str, dict[str, str]] = {
    # Single-valued (integer J)
    "Gamma1": {"color": "#C1121F", "label": "Γ₁ (A₁)"},
    "Gamma2": {"color": "#1D3557", "label": "Γ₂ (A₂)"},
    "Gamma3": {"color": "#2A9D8F", "label": "Γ₃ (E)"},
    "Gamma4": {"color": "#BC6C25", "label": "Γ₄ (T₁)"},
    "Gamma5": {"color": "#7B2CBF", "label": "Γ₅ (T₂)"},
    # Double-valued (half-integer J)
    "Gamma6": {"color": "#E76F51", "label": "Γ₆"},
    "Gamma7": {"color": "#264653", "label": "Γ₇"},
    "Gamma8": {"color": "#E9C46A", "label": "Γ₈"},
    "unknown": {"color": "#8D99AE", "label": "unknown"},
}


# ---------------------------------------------------------------------------
# LLW computation
# ---------------------------------------------------------------------------

def compute_F_factors(J: float) -> tuple[float, float]:
    """Compute LLW normalization factors F(4) and F(6) for a given J."""
    ops = build_cef_stevens_operators(J, symmetry="Oh")
    O4 = ops["O40"] + 5.0 * ops["O44c"]
    O6 = ops["O60"] - 21.0 * ops["O64c"]
    F4 = float(np.max(np.abs(np.linalg.eigvalsh(O4))))
    F6 = float(np.max(np.abs(np.linalg.eigvalsh(O6))))
    return F4, F6


def _classify_subspace_irreps(
    projectors: dict[str, Any],
    evecs: Any,
    energy_groups: list[list[int]],
) -> list[list[str]]:
    """Classify irreps at the subspace level for each energy group."""
    level_irreps: list[list[str]] = []
    for grp in energy_groups:
        deg = len(grp)
        Psi = evecs[:, grp]
        weights: dict[str, float] = {}
        for name, P in projectors.items():
            w = sum(float(np.linalg.norm(P @ Psi[:, j])) ** 2 for j in range(deg))
            if w > 1e-6:
                weights[name] = w
        level_irreps.append(sorted(weights, key=lambda k: -weights[k]))
    return level_irreps


def _group_by_energy(eigenvalues: Any, tol: float = 1e-6) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i - 1]) < tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def llw_scan(J: float, n_points: int = 801) -> dict[str, Any]:
    """Scan x in [-1, 1] and compute energies with subspace-level irrep labels."""
    F4, F6 = compute_F_factors(J)
    x_grid = np.linspace(-1.0, 1.0, n_points)
    dim = int(round(2.0 * J + 1.0))
    projectors = build_projectors(J, point_group="Oh")

    energies = np.zeros((n_points, dim), dtype=float)
    # Per-state irrep label for scatter coloring
    irreps = np.empty((n_points, dim), dtype=object)

    for idx, x in enumerate(x_grid):
        H = build_hcef_matrix_J(
            J,
            {"B4": float(x / F4), "B6": float((1.0 - abs(x)) / F6)},
            symmetry="Oh",
        )
        evals, evecs = np.linalg.eigh(H)
        energies[idx, :] = evals

        groups = _group_by_energy(evals)
        level_irreps = _classify_subspace_irreps(projectors, evecs, groups)
        for grp, irr_list in zip(groups, level_irreps):
            label = irr_list[0] if irr_list else "unknown"
            for j in grp:
                irreps[idx, j] = label

    return {
        "J": float(J),
        "F4": F4,
        "F6": F6,
        "x_grid": x_grid,
        "energies": energies,
        "irreps": irreps,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "fexchange-matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))


def plot_llw_diagram(J: float, scan: dict[str, Any], outpath: str | Path) -> None:
    """Plot E/W vs x for a single J manifold."""
    _configure_matplotlib_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_grid = scan["x_grid"]
    energies = scan["energies"]
    irreps = scan["irreps"]
    x_mesh = np.repeat(x_grid[:, None], energies.shape[1], axis=1)

    fig, ax = plt.subplots(figsize=(10.5, 7.0))

    # Background lines
    for col in range(energies.shape[1]):
        ax.plot(x_grid, energies[:, col], color="#D0D4DA", linewidth=0.8, alpha=0.7, zorder=1)

    # Colored scatter by irrep
    plotted: set[str] = set()
    for irrep_name, style in IRREP_STYLE.items():
        mask = irreps == irrep_name
        if not np.any(mask):
            continue
        ax.scatter(
            x_mesh[mask], energies[mask],
            s=8.0, color=style["color"], linewidths=0.0,
            label=style["label"] if irrep_name not in plotted else None,
            zorder=2,
        )
        plotted.add(irrep_name)

    # Labels
    entries = _entries_for_J(J)
    ions_str = ", ".join(f"f{e['n']} {e['ions']} ({e['term']})" for e in entries)
    twoJ = int(round(2 * J))
    J_str = f"{twoJ}/2" if twoJ % 2 == 1 else str(int(J))

    y_min, y_max = float(np.min(energies)), float(np.max(energies))
    pad = 0.06 * max(y_max - y_min, 1.0)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("x (LLW parameter)")
    ax.set_ylabel("E / W")
    ax.set_title(f"LLW diagram — Oh, J = {J_str}\n{ions_str}")
    ax.grid(True, color="#E5E7EB", linewidth=0.8, alpha=0.9)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=240)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLW diagrams for Oh, all f^n ground-state J")
    parser.add_argument("--J", type=float, help="Only plot this J value")
    parser.add_argument("--fn", type=int, help="Only plot J for this f^n")
    parser.add_argument("--npoints", type=int, default=801, help="Number of x points (default: 801)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: tools/)")
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    if args.fn is not None:
        entries = [e for e in FN_TABLE if e["n"] == args.fn]
        if not entries:
            print(f"No entry for f{args.fn} (f6 J=0 is excluded)")
            return
        j_values = [entries[0]["J"]]
    elif args.J is not None:
        j_values = [args.J]
    else:
        j_values = UNIQUE_J

    for J in j_values:
        entries = _entries_for_J(J)
        ions_str = ", ".join(f"f{e['n']} {e['ions']}" for e in entries)
        twoJ = int(round(2 * J))
        J_str = f"{twoJ}half" if twoJ % 2 == 1 else str(int(J))
        print(f"\n=== J = {J} ({ions_str}) ===")

        F4, F6 = compute_F_factors(J)
        print(f"  F(4) = {F4:.6f},  F(6) = {F6:.6f}")

        scan = llw_scan(J, n_points=args.npoints)
        plot_llw_diagram(J, scan, outdir / f"llw_J{J_str}.png")


if __name__ == "__main__":
    main()
