#!/usr/bin/env python3
"""Plot unrotated REChX z-axis XXZ parameters versus Jh/U.

This uses the original organized REChX trees, not REChX_cubic.  In these files
crystallographic z is the C3 axis.  Each figure shows the z-axis XXZ parameters

  J = (Jxx + Jyy) / 2
  K = Jzz - J

and the remaining full-tensor components:

  Gxy, Gxz, Gyz, Dx, Dy, Dz

For NN bonds the D components are omitted from the plot.

The U curves are scaled by U/8, matching the current jkgamma plot convention.

With --combine-bonds, one figure is written per shell instead of per bond: J and
K are drawn from one representative bond, while all non-XXZ gray components from
every bond in the shell are overlaid.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_ROOT = Path("/work/k0282/k028230/code/fexchange/outputs_organized")
DEFAULT_OUT = Path("c3z_xxz_figures")
FAMILIES = ("REOCl", "REOF", "RESI")
BRANCHES = ("sopt", "fopt_plus_sopt_direct")
REFERENCE_U = 8

COMPONENT_PARAMS = [
    ("J", "#2e8b3d", 2.8),
    ("K", "#e8413a", 2.8),
    ("Gxy", "#f5821f", 1.8),
    ("Gxz", "#1c3f94", 1.8),
    ("Gyz", "#6fa8dc", 1.8),
    ("Dx", "#8c564b", 1.65),
    ("Dy", "#e7298a", 1.65),
    ("Dz", "#111111", 1.65),
]
XXZ_GRAY_PARAMS = [
    ("J", "#2e8b3d", 2.8),
    ("K", "#e8413a", 2.8),
    ("Gxy", "#707070", 1.45),
    ("Gxz", "#707070", 1.45),
    ("Gyz", "#707070", 1.45),
    ("Dx", "#707070", 1.25),
    ("Dy", "#707070", 1.25),
    ("Dz", "#707070", 1.25),
]
INVARIANT_PARAMS = [
    ("J", "#2e8b3d", 2.8),
    ("K", "#e8413a", 2.8),
    ("Axy", "#6a3d9a", 1.8),
    ("Gz", "#1c3f94", 1.8),
    ("Dperp", "#e7298a", 1.65),
    ("Dz", "#111111", 1.65),
]

U_STYLES = {
    8: ("-", 3.0, 1.0),
    6: ("--", 1.8, 0.92),
    4: (":", 1.8, 0.92),
}

REPRESENTATIVE_BONDS = {
    "J1": ("J1-x", "J1-y", "J1-z"),
    "J2": ("J2-000", "J2-060", "J2-120", "J2-180", "J2-240", "J2-300"),
}


def scale_for_u(u_value: int) -> float:
    return u_value / REFERENCE_U


def shell_for_bond(bond: str) -> str:
    if bond.startswith("J1-"):
        return "J1"
    if bond.startswith("J2-"):
        return "J2"
    return "other"


def read_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 12 or cols[1] != "total":
            continue
        tensor = np.array([float(x) for x in cols[3:12]], dtype=float).reshape(3, 3)
        yield float(cols[0]), decompose(tensor)


def decompose(m: np.ndarray) -> dict[str, float]:
    j = 0.5 * (m[0, 0] + m[1, 1])
    e = 0.5 * (m[0, 0] - m[1, 1])
    gxy = 0.5 * (m[0, 1] + m[1, 0])
    gxz = 0.5 * (m[0, 2] + m[2, 0])
    gyz = 0.5 * (m[1, 2] + m[2, 1])
    dx = 0.5 * (m[1, 2] - m[2, 1])
    dy = 0.5 * (m[2, 0] - m[0, 2])
    dz = 0.5 * (m[0, 1] - m[1, 0])
    return {
        "J": j,
        "K": m[2, 2] - j,
        "Gxy": gxy,
        "Gxz": gxz,
        "Gyz": gyz,
        "Dx": dx,
        "Dy": dy,
        "Dz": dz,
        "Axy": float(np.sqrt(e * e + gxy * gxy)),
        "Gz": float(np.sqrt(gxz * gxz + gyz * gyz)),
        "Dperp": float(np.sqrt(dx * dx + dy * dy)),
    }


def params_for_bond(bond: str, mode: str, keep_nn_dm: bool):
    if mode == "invariants":
        params = INVARIANT_PARAMS
    elif mode == "xxz-gray":
        params = XXZ_GRAY_PARAMS
    else:
        params = COMPONENT_PARAMS
    if bond.startswith("J1-") and not keep_nn_dm:
        return [item for item in params if item[0] not in {"Dx", "Dy", "Dz", "Dperp"}]
    return params


def representative_bond(shell: str, bonds: set[str]) -> str:
    for bond in REPRESENTATIVE_BONDS.get(shell, ()):
        if bond in bonds:
            return bond
    return sorted(bonds)[0]


def discover_cases(
    root: Path,
    branches: set[str],
    families: set[str],
    bond_filter: set[str] | None,
    material_filter: set[str] | None,
    run_filter: set[str] | None,
):
    cases: dict[tuple[str, str, str, str, str], dict[int, Path]] = defaultdict(dict)
    for family in sorted(families):
        family_root = root / family
        if not family_root.exists():
            continue
        for material_root in sorted(path for path in family_root.iterdir() if path.is_dir()):
            if material_filter is not None and material_root.name not in material_filter:
                continue
            for bond_root in sorted(path for path in material_root.iterdir() if path.is_dir()):
                if bond_filter is not None and bond_root.name not in bond_filter:
                    continue
                for branch in sorted(branches):
                    branch_root = bond_root / branch
                    if not branch_root.exists():
                        continue
                    for path in sorted(branch_root.glob("*/U_*.txt")):
                        run = path.parts[-2]
                        if run_filter is not None and run not in run_filter:
                            continue
                        try:
                            u_value = int(path.stem.split("_", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        cases[(family, material_root.name, bond_root.name, branch, run)][u_value] = path
    return cases


def yspan(files: dict[int, Path], params) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for u_value, path in files.items():
        scale = scale_for_u(u_value)
        for _ratio, values in read_rows(path):
            for key, _colour, _lw in params:
                y = values[key] * 1000.0 * scale
                lo, hi = min(lo, y), max(hi, y)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if abs(hi - lo) < 1.0e-12:
        return lo - 1.0, hi + 1.0
    pad_low = 0.07 * (hi - lo)
    pad_high = 0.23 * (hi - lo)
    return lo - pad_low, hi + pad_high


def yspan_shell(bond_files: dict[str, dict[int, Path]], mode: str, keep_nn_dm: bool) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    rep = representative_bond(shell_for_bond(next(iter(bond_files))), set(bond_files))
    for bond, files in bond_files.items():
        for key, _colour, _lw in params_for_bond(bond, mode, keep_nn_dm):
            if key in {"J", "K"} and bond != rep:
                continue
            for u_value, path in files.items():
                scale = scale_for_u(u_value)
                for _ratio, values in read_rows(path):
                    y = values[key] * 1000.0 * scale
                    lo, hi = min(lo, y), max(hi, y)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if abs(hi - lo) < 1.0e-12:
        return lo - 1.0, hi + 1.0
    pad_low = 0.07 * (hi - lo)
    pad_high = 0.23 * (hi - lo)
    return lo - pad_low, hi + pad_high


def plot_case(files: dict[int, Path], params, out_png: Path, yrange: tuple[float, float], show_u_legend: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for key, colour, base_lw in params:
        for u_value in sorted(files, reverse=True):
            linestyle, u_lw, alpha = U_STYLES.get(u_value, ("-", 1.6, 0.9))
            scale = scale_for_u(u_value)
            ratios: list[float] = []
            values: list[float] = []
            for ratio_value, row in read_rows(files[u_value]):
                ratios.append(ratio_value)
                values.append(row[key] * 1000.0 * scale)
            ax.plot(
                ratios,
                values,
                color=colour,
                linestyle=linestyle,
                linewidth=base_lw * u_lw / 2.4,
                alpha=alpha,
            )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)
    ax.set_xlim(0.0, 0.4)
    ax.set_ylim(*yrange)
    ax.set_xlabel(r"$J_H/U$")
    ax.set_ylabel("COUPLING CONST. (meV)")
    ax.tick_params(direction="out", top=True, right=True)
    ax.grid(False)

    if show_u_legend:
        u_handles = []
        for u_value in sorted(files, reverse=True):
            linestyle, u_lw, _alpha = U_STYLES.get(u_value, ("-", 1.6, 0.9))
            scale = scale_for_u(u_value)
            label = f"U={u_value} eV" if u_value == REFERENCE_U else f"U={u_value} eV (x{scale:g})"
            u_handles.append(Line2D([0], [0], color="black", linestyle=linestyle, lw=u_lw, label=label))
        ax.legend(handles=u_handles, loc="upper left", fontsize=10, frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_shell_case(
    bond_files: dict[str, dict[int, Path]],
    mode: str,
    keep_nn_dm: bool,
    out_png: Path,
    yrange: tuple[float, float],
    show_u_legend: bool,
) -> None:
    rep = representative_bond(shell_for_bond(next(iter(bond_files))), set(bond_files))
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for bond in sorted(bond_files):
        files = bond_files[bond]
        for key, colour, base_lw in params_for_bond(bond, mode, keep_nn_dm):
            if key in {"J", "K"} and bond != rep:
                continue
            for u_value in sorted(files, reverse=True):
                linestyle, u_lw, alpha = U_STYLES.get(u_value, ("-", 1.6, 0.9))
                scale = scale_for_u(u_value)
                ratios: list[float] = []
                values: list[float] = []
                for ratio_value, row in read_rows(files[u_value]):
                    ratios.append(ratio_value)
                    values.append(row[key] * 1000.0 * scale)
                ax.plot(
                    ratios,
                    values,
                    color=colour,
                    linestyle=linestyle,
                    linewidth=base_lw * u_lw / 2.8,
                    alpha=alpha if key in {"J", "K"} else min(alpha, 0.66),
                )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)
    ax.set_xlim(0.0, 0.4)
    ax.set_ylim(*yrange)
    ax.set_xlabel(r"$J_H/U$")
    ax.set_ylabel("COUPLING CONST. (meV)")
    ax.tick_params(direction="out", top=True, right=True)
    ax.grid(False)

    if show_u_legend:
        u_values = sorted({u_value for files in bond_files.values() for u_value in files}, reverse=True)
        u_handles = []
        for u_value in u_values:
            linestyle, u_lw, _alpha = U_STYLES.get(u_value, ("-", 1.6, 0.9))
            scale = scale_for_u(u_value)
            label = f"U={u_value} eV" if u_value == REFERENCE_U else f"U={u_value} eV (x{scale:g})"
            u_handles.append(Line2D([0], [0], color="black", linestyle=linestyle, lw=u_lw, label=label))
        ax.legend(handles=u_handles, loc="upper left", fontsize=10, frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def group_cases_by_shell(cases: dict[tuple[str, str, str, str, str], dict[int, Path]]):
    shell_cases: dict[tuple[str, str, str, str, str], dict[str, dict[int, Path]]] = defaultdict(dict)
    for (family, material, bond, branch, run), files in cases.items():
        shell = shell_for_bond(bond)
        if shell not in {"J1", "J2"}:
            continue
        shell_cases[(family, material, shell, branch, run)][bond] = files
    return shell_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--branch", action="append", choices=BRANCHES)
    parser.add_argument("--family", action="append", choices=FAMILIES)
    parser.add_argument("--material", action="append")
    parser.add_argument("--bond", action="append")
    parser.add_argument("--run", action="append")
    parser.add_argument("--single-output", type=Path)
    parser.add_argument("--no-u-legend", action="store_true")
    parser.add_argument("--misc-mode", choices=("components", "invariants", "xxz-gray"), default="components")
    parser.add_argument("--keep-nn-dm", action="store_true")
    parser.add_argument("--combine-bonds", action="store_true", help="overlay all non-XXZ components by shell")
    args = parser.parse_args()

    branches = set(args.branch or BRANCHES)
    families = set(args.family or FAMILIES)
    bond_filter = set(args.bond) if args.bond else None
    material_filter = set(args.material) if args.material else None
    run_filter = set(args.run) if args.run else None
    cases = discover_cases(args.root, branches, families, bond_filter, material_filter, run_filter)
    if not cases:
        print(f"no unrotated REChX cases found under {args.root}")
        return 1

    fig_dir = args.out / "REChX" / "figures"
    if args.combine_bonds:
        shell_cases = group_cases_by_shell(cases)
        ranges: dict[tuple[str, str, str], tuple[float, float]] = {}
        lohi: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [np.inf, -np.inf])
        for (family, material, shell, _branch, _run), bond_files in shell_cases.items():
            lo, hi = yspan_shell(bond_files, args.misc_mode, args.keep_nn_dm)
            key = (material[:2], family, shell)
            lohi[key][0] = min(lohi[key][0], lo)
            lohi[key][1] = max(lohi[key][1], hi)
        for key, (lo, hi) in lohi.items():
            ranges[key] = (lo, hi)
            print(f"# {key[0]} {key[1]} {key[2]} y-range: [{lo:.4f}, {hi:.4f}] meV")

        count = 0
        for (family, material, shell, branch, run), bond_files in sorted(shell_cases.items()):
            stem = f"{family}_{material}_{shell}_{branch}_{run}"
            out_png = args.single_output if args.single_output and len(shell_cases) == 1 else fig_dir / f"{stem}.png"
            plot_shell_case(
                bond_files,
                args.misc_mode,
                args.keep_nn_dm,
                out_png,
                ranges[(material[:2], family, shell)],
                not args.no_u_legend,
            )
            count += 1
        print(f"# wrote {count} shell-combined figures under {fig_dir}")
    else:
        ranges: dict[tuple[str, str, str], tuple[float, float]] = {}
        lohi: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [np.inf, -np.inf])
        for (family, material, bond, _branch, _run), files in cases.items():
            lo, hi = yspan(files, params_for_bond(bond, args.misc_mode, args.keep_nn_dm))
            key = (material[:2], family, shell_for_bond(bond))
            lohi[key][0] = min(lohi[key][0], lo)
            lohi[key][1] = max(lohi[key][1], hi)
        for key, (lo, hi) in lohi.items():
            ranges[key] = (lo, hi)
            print(f"# {key[0]} {key[1]} {key[2]} y-range: [{lo:.4f}, {hi:.4f}] meV")

        count = 0
        for (family, material, bond, branch, run), files in sorted(cases.items()):
            stem = f"{family}_{material}_{bond}_{branch}_{run}"
            out_png = args.single_output if args.single_output and len(cases) == 1 else fig_dir / f"{stem}.png"
            plot_case(
                files,
                params_for_bond(bond, args.misc_mode, args.keep_nn_dm),
                out_png,
                ranges[(material[:2], family, shell_for_bond(bond))],
                not args.no_u_legend,
            )
            count += 1
        print(f"# wrote {count} figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
