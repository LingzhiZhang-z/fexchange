#!/usr/bin/env python3
"""Plot REX3 / REChX J/K/Gamma vs Jh/U from the organized exchange tree (K2PrO3 style).

Each case gets a gnuplot script that plots DIRECTLY from the organized U_<NN>.txt
files. Every coupling is a fixed linear combination of the raw Jxx..Jzz columns
(the decomposition -- and, for REChX, the crystal->cubic rotation -- are folded
into the coefficients), so no intermediate data file is written.

    <organized>/<family>/<material>/<bond>/<branch>/<run_id>/U_<NN>.txt
        cols:  1 ratio | 2 'total' | 3 residual | 4..12 Jxx..Jzz   (eV)

Branches: ``sopt`` and ``fopt_plus_sopt_direct`` (built by combine_branches.sh).
Per family:
  * REX3 (R3, C2m): raw is already cubic -> identity rotation; bonds x/y/z;
    plot J, K, Gamma, Gamma'  (the C2m Gamma_ac/Gamma_bc split is negligible).
  * REChX (REOCl/REOF/RESI): buckled honeycomb, raw z = crystal c = [111]; rotate
    each NN bond to the cubic frame with the fixed geometric U (the three NN bonds
    then give C3-consistent couplings), bonds J1-x/y/z; plot J, K, Gamma,
    Gamma_ac, Gamma_bc  (REChX does NOT force Gamma_ac == Gamma_bc; two blues).
    NNN (J2-*) bonds are not handled yet.

Style: no grid, saturated colours, U=8 is the reference thick curve, other U
curves are rescaled by U/8 (1/U collapse), y=0 dashed line, U/rescale
line-style key, no title/labels. y-range is shared per (RE element, family).

Usage:  python scripts/process_data/plot_jkgamma.py <organized_root> [out_dir]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # decompose_bond_exchange.py is alongside
from decompose_bond_exchange import decompose  # noqa: E402

BRANCHES = ("sopt", "fopt_plus_sopt_direct")
U_DT = ["lw 3.2 dt 1", "lw 1.6 dt 1", "lw 1.6 dt 2", "lw 1.4 dt 3"]   # thick/thin/dashed/dotted
REFERENCE_U = 8
RECHX_FAMS = ("REOCl", "REOF", "RESI")

GREEN, RED, ORANGE, BLUE, DBLUE, LBLUE = "#2e8b3d", "#e8413a", "#f5821f", "#3d4eb8", "#1c3f94", "#6fa8dc"
PARAMS_REX3 = [("J", GREEN), ("K", RED), ("Gamma", ORANGE), ("Gamma_prime", BLUE)]
PARAMS_RECHX = [("J", GREEN), ("K", RED), ("Gamma", ORANGE), ("Gamma_ac", DBLUE), ("Gamma_bc", LBLUE)]
# NNN: clean symmetric 4-param J2/K2/Gamma2/Gamma2' + DM. The DM has only two
# independent components (Da=Db by symmetry): D = Dc (axial), D' = Da=Db (in-plane).
PARAMS_J2 = [("J", GREEN), ("K", RED), ("Gamma", ORANGE), ("Gamma_prime", BLUE),
             ("D_c", "#6a3d9a"), ("D_a", "#e7298a")]   # D = Dc (axial), D' = Da=Db (in-plane)
# all 6 NNN, each its own figure; axis = note label (0/180->z, 60/240->y, 120/300->x).
# symmetric part is the same for all 6; the DM sign flips between reverse pairs (0<->180 etc.).
J2_REP = {"J2-000": ("J2-000", "z"), "J2-180": ("J2-180", "z"),
          "J2-060": ("J2-060", "y"), "J2-240": ("J2-240", "y"),
          "J2-120": ("J2-120", "x"), "J2-300": ("J2-300", "x")}

IDENT = np.eye(3)   # all inputs are already cubic-frame (REX3 natively; REChX via REChX_cubic)


def col(i: int, j: int) -> int:
    """1-based gnuplot column of J[i,j] in a U_<NN>.txt row (4..12 = Jxx..Jzz)."""
    return 4 + 3 * i + j


def case_cfg(fam: str, bond: str):
    """(rotation, params, axis, bond_tag, range_group) for a case, or None to skip."""
    if fam.startswith("REX3"):
        if bond in ("x", "y", "z"):
            return IDENT, PARAMS_REX3, bond, bond, "NN"
    elif fam in RECHX_FAMS:                                  # files pre-rotated (REChX_cubic) -> identity
        if bond.startswith("J1-"):                           # NN is clean 4-param after R(111,theta).U
            return IDENT, PARAMS_REX3, bond.split("-")[1], bond, "NN"
        if bond in J2_REP:                                  # NNN: two inequivalent C3 triads
            tag, axis = J2_REP[bond]
            return IDENT, PARAMS_J2, axis, tag, "J2"
    return None


def coeff_expr(R: np.ndarray, axis: str, key: str) -> str:
    """A coupling as a gnuplot column expression: linear in raw Jxx..Jzz (in eV)."""
    terms = []
    for k in range(3):
        for m in range(3):
            E = np.zeros((3, 3))
            E[k, m] = 1.0
            c = decompose(R @ E @ R.T, axis)[key]            # coupling is linear -> coeff of J[k,m]
            if abs(c) > 1e-10:
                terms.append(f"({c:.7g})*${col(k, m)}")
    return "+".join(terms) or "0"


def yspan(files: dict[int, Path], params, R: np.ndarray, axis: str, scale: dict[int, float]):
    """Min/max of every plotted (rescaled) coupling across a case's U files."""
    ymin, ymax = np.inf, -np.inf
    for u, f in files.items():
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or ln.startswith("#"):
                continue
            d = decompose(R @ np.array([float(x) for x in ln.split()[3:12]]).reshape(3, 3) @ R.T, axis)
            for key, _ in params:
                y = d[key] * 1000.0 * scale[u]
                ymin, ymax = min(ymin, y), max(ymax, y)
    return float(ymin), float(ymax)


def scale_for_u(u: int) -> float:
    """Plot scaling for the 1/U collapse, normalised to U=8."""
    return u / REFERENCE_U


def style_for_us(us: list[int]) -> dict[int, str]:
    """Reverse the old U-style order so the largest/reference U is visually dominant."""
    styles = list(reversed(U_DT[: len(us)]))
    return {u: styles[i] for i, u in enumerate(us)}


def write_gp(path: Path, params, files: dict[int, Path], R: np.ndarray, axis: str,
             png: Path, lo: float, hi: float) -> None:
    us = sorted(files)
    scale = {u: scale_for_u(u) for u in us}
    u_styles = style_for_us(us)
    styles, ulegend, plots = [], [], []
    for ui, u in enumerate(sorted(us, reverse=True)):            # neutral line-style legend = reference U first
        styles.append(f"set style line {201 + ui} lc rgb 'black' {u_styles[u]}")   # 201+: clear of quantity ls
        tag = f"{{/Helvetica-Italic U}}={u} eV" if u == REFERENCE_U else f"{{/Helvetica-Italic U}}={u} eV (x{scale[u]:g})"
        ulegend.append(f"  NaN with lines ls {201 + ui} title '{tag}'")
    zero = "  0 with lines lc rgb 'black' dt 2 lw 1.2 notitle"     # y=0 reference line, behind curves
    for pi, (key, colour) in enumerate(params):
        expr = coeff_expr(R, axis, key)
        for ui, u in enumerate(us):
            ls = (pi + 1) * 10 + ui + 1
            styles.append(f"set style line {ls} lc rgb '{colour}' {u_styles[u]}")
            plots.append(f"  '{files[u]}' using 1:(({expr})*1000*{scale[u]:g}) with lines ls {ls} notitle")

    path.write_text(
        "set terminal pngcairo size 820,640 enhanced font 'Helvetica,18'\n"
        f"set output '{png}'\n"
        "set datafile commentschars '#'\n"
        "unset grid\nset border 31 lw 1.4\nset tics nomirror out scale 1.3,0.7\n"
        "set xrange [0:0.40]\nset xtics 0,0.10,0.40\nset mxtics 2\nset mytics 2\n"
        f"set yrange [{lo:.5f}:{hi:.5f}]\n"
        "set key at graph 0.03,0.97 left top reverse Left samplen 2.4 spacing 1.3 font ',13'\n"
        "set ylabel 'COUPLING CONST. (meV)'\n"
        "set xlabel '{/Helvetica-Italic J}_H/{/Helvetica-Italic U}'\n"
        "unset title\n"
        + "\n".join(styles) + "\nplot \\\n"
        + ", \\\n".join(ulegend + [zero] + plots) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if not 2 <= len(sys.argv) <= 3:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("jkgamma_figures")
    if not root.exists():
        print(f"no organized root at {root}")
        return 1

    cases: dict = defaultdict(dict)
    for f in root.rglob("U_*.txt"):
        fam, mat, bond, branch, run = f.parts[-6:-1]
        if branch not in BRANCHES or not case_cfg(fam, bond):
            continue
        if fam in RECHX_FAMS and "REChX_cubic" not in f.parts:
            continue                                        # raw REChX -> use the rotated REChX_cubic copy
        cases[(fam, mat, bond, branch, run)][int(f.stem.split("_")[1])] = f
    if not cases:
        print(f"no plottable U_*.txt under {root}")
        return 1

    # shared y-range per (RE element, family, group): NN and J2 get separate axes
    grp_lo: dict = defaultdict(lambda: np.inf)
    grp_hi: dict = defaultdict(lambda: -np.inf)
    for (fam, mat, bond, branch, run), files in cases.items():
        R, params, axis, tag, group = case_cfg(fam, bond)
        lo, hi = yspan(files, params, R, axis, {u: scale_for_u(u) for u in files})
        g = (mat[:2], fam, group)
        grp_lo[g], grp_hi[g] = min(grp_lo[g], lo), max(grp_hi[g], hi)
    yr = {}
    for g in grp_lo:
        rng = (grp_hi[g] - grp_lo[g]) or 1.0
        yr[g] = (grp_lo[g] - 0.06 * rng, grp_hi[g] + 0.22 * rng)
        print(f"# {g[0]} {g[1]} {g[2]} y-range: [{yr[g][0]:.4f}, {yr[g][1]:.4f}] meV")

    for (fam, mat, bond, branch, run), files in sorted(cases.items()):
        R, params, axis, tag, group = case_cfg(fam, bond)
        gdir = out / ("REX3" if fam.startswith("REX3") else "REChX")   # separate REX3 / REChX trees
        (gdir / "figures").mkdir(parents=True, exist_ok=True)
        stem = f"{fam}_{mat}_{tag}_{branch}_{run}"
        write_gp(gdir / f"{stem}.gp", params, files, R, axis,
                 (gdir / "figures" / f"{stem}.png").resolve(), *yr[(mat[:2], fam, group)])

    print(f"# wrote {len(cases)} gnuplot scripts under {out}/REX3 and {out}/REChX (branches: {', '.join(BRANCHES)})")
    print(f"# render: find {out}/REX3 {out}/REChX -name '*.gp' -print0 | xargs -0 -n1 gnuplot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
