"""Plot fexchange sweep results against the literature, in the J–K–Γ–Γ′ decomposition.

Datasets / references reproduced:
* gau    -> J, K vs the analytical formulas of arXiv:1802.03024 (per Γ6, Γ7).
* nature -> J, K vs hopping ratio, per sector (Fig 3, doi:10.1038/s43246-024-00634-w).
* prb    -> full J, K, Γ, Γ′ vs J_H/U with the paper's U/2 scaling
            (Fig 5, K2/Rb2/Cs2PrO3; arXiv:1912.03422).

Exchange decomposition from the 3x3 tensor in L3/exchange.txt (Z-bond J-K-Γ-Γ′):
    J  = Jyy                            # Heisenberg  (Jxx=Jyy for nature/prb; matches the gau paper J)
    K  = Jxx + Jzz - 2*Jyy             # Kitaev       (= Jzz - Jxx when Jxx=Jyy)
    Γ  = (Jxy + Jyx) / 2               # symmetric off-diagonal
    Γ′ = (Jxz + Jyz + Jzx + Jzy) / 4   # symmetric off-diagonal (z)

CAVEAT: for gau, K << J, and K = Jxx+Jzz-2Jyy is a small difference of large numbers, so
its numerical precision is poor -- J reproduces the formula to ~5 sig figs but K does not.
gau's J and K are therefore drawn on SEPARATE axes so the (real) K scatter is not hidden by J.

Usage:  python scripts/plot_sweep_results.py [gau|nature|prb|all]
Reads outputs/<set>/.../L3/exchange.txt ; writes outputs/plots/*.png .
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PLOTS = Path("outputs/plots")
_KEYS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]


def _set_cjk_font() -> None:
    """Use a macOS CJK font for the Chinese labels; fall back silently if absent."""
    for path in (
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ):
        if Path(path).exists():
            fm.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=path).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _tensor(path: str) -> dict[str, float] | None:
    for line in Path(path).read_text().splitlines():
        if line.startswith("total"):
            return dict(zip(_KEYS, (float(x) for x in line.split()[2:11])))
    return None


def jkgg(path: str) -> dict[str, float] | None:
    """Decompose the exchange tensor into J, K, Γ, Γ′ (see module docstring)."""
    t = _tensor(path)
    if t is None:
        return None
    J = t["yy"]
    return {
        "J": J,
        "K": t["xx"] + t["zz"] - 2 * J,
        "Γ": (t["xy"] + t["yx"]) / 2,
        "Γ′": (t["xz"] + t["yz"] + t["zx"] + t["zy"]) / 4,
    }


# --------------------------------------------------------------------------- gau
_GAU_POLY = {
    ("g6", "J"): (5.955, -4.010, 0.554),
    ("g6", "K"): (0.071, 0.099, 0.027),
    ("g7", "J"): (-0.012, 0.008, 0.315),
    ("g7", "K"): (0.036, 0.038, -0.016),
}
_GAU_T4 = 1e-3  # t_pfσ^4 of the checked-in gau hopping (matches J(Γ6) to ~5 sig figs)


def _gau_formula(g: str, kind: str, rho):
    a2, a3, a4 = _GAU_POLY[(g, kind)]
    return _GAU_T4 * (a2 * rho**2 + a3 * rho**3 + a4 * rho**4)


def plot_gau() -> None:
    rr = np.linspace(-1.05, 0.0, 200)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, (g, gl) in enumerate([("g6", r"$\Gamma_6$"), ("g7", r"$\Gamma_7$")]):
        o = {}
        for ex in glob.glob(f"outputs/gau/{g}/rho_*/L3/exchange.txt"):
            rho = -float(re.search(r"rho_m([0-9p]+)/", ex).group(1).replace("p", "."))
            jk = jkgg(ex)
            if jk:
                o[round(rho, 3)] = jk
        xs = sorted(o)
        for col, kind in enumerate(["J", "K"]):
            ax = axes[row, col]
            ax.plot(rr, _gau_formula(g, kind, rr), "-", color="C0", lw=2, label="论文公式")
            ax.plot(xs, [o[r][kind] for r in xs], "o", color="C3", ms=6, label="我们")
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(f"{kind} (eV)")
            ax.set_title(f"gau {gl}: {kind}" + ("   (K<<J，单独轴)" if kind == "K" else ""))
            ax.legend()
            ax.grid(alpha=0.3)
    fig.suptitle("gau: J、K vs 论文公式（J、K 各自一个 y 轴）", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "gau_compare.png", dpi=120)
    plt.close(fig)
    print("gau_compare.png")


# ------------------------------------------------------------------------ nature
_NATURE_SECTORS = [
    ("f01", "g7"), ("f03", "g6"), ("f05", "g7"), ("f09", "g6"),
    ("f09", "g7"), ("f11", "g7"), ("f13", "g6"),
]


def plot_nature() -> None:
    found = [(f, g) for f, g in _NATURE_SECTORS
             if glob.glob(f"outputs/nature/{f}/{g}/ratio*/L3/exchange.txt")]
    if not found:
        return
    nrow = (len(found) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 4.2 * nrow), squeeze=False)
    for ax, (f, g) in zip(axes.flat, found):
        pts = []
        for ex in glob.glob(f"outputs/nature/{f}/{g}/ratio*/L3/exchange.txt"):
            r = float(re.search(r"ratio([0-9.]+)/", ex).group(1))
            jk = jkgg(ex)
            if jk:
                pts.append((r, jk["J"], jk["K"]))
        pts.sort()
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], "-o", color="C2", ms=4, label="J")
        ax.plot(xs, [p[2] for p in pts], "-s", color="C3", ms=4, label="K")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("hopping ratio")
        ax.set_ylabel("J, K (eV)")
        ax.set_title(f"nature f{f[1:].lstrip('0')} $\\Gamma_{g[1]}$")
        ax.legend()
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(found):]:
        ax.axis("off")
    fig.suptitle("nature: J、K vs ratio（对应 Fig 3）", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "nature_JK.png", dpi=120)
    plt.close(fig)
    print("nature_JK.png")


# --------------------------------------------------------------------------- prb
_PRB_U_STYLE = {2.0: ("-", 2.6), 3.0: ("-", 1.1), 4.0: ("--", 1.3), 6.0: (":", 1.6)}
_PRB_COLOR = {"J": "green", "K": "red", "Γ′": "blue", "Γ": "darkorange"}
_PRB_PANELS = [("K", "K$_2$PrO$_3$ (a)"), ("Rb", "Rb$_2$PrO$_3$ (b)"), ("Cs", "Cs$_2$PrO$_3$ (c)")]


def plot_prb() -> None:
    """Fig 5: J, K, Γ, Γ′ vs J_H/U; each U curve scaled by U/2 (the paper's normalisation)."""
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 13))
    for ax, (mat, title) in zip(axes, _PRB_PANELS):
        by_u: dict[float, list] = {}
        for ex in glob.glob(f"outputs/prb/RS/{mat}/U*/jh*/L3/exchange.txt"):
            m = re.search(r"/U([0-9.]+)/jh([0-9.]+)/", ex)
            jk = jkgg(ex)
            if jk:
                by_u.setdefault(float(m.group(1)), []).append((float(m.group(2)), jk))
        for u in sorted(by_u):
            ls, lw = _PRB_U_STYLE.get(u, ("-", 1.0))
            scale = u / 2.0
            pts = sorted(by_u[u], key=lambda t: t[0])
            xs = [p[0] for p in pts]
            for par in ("J", "K", "Γ′", "Γ"):
                ax.plot(xs, [p[1][par] * scale for p in pts], ls, color=_PRB_COLOR[par], lw=lw)
        ax.set_xlim(0, 0.2)
        ax.set_ylim(-0.5, 3.0)
        ax.set_xlabel("$J_H/U$")
        ax.set_ylabel("COUPLING CONST. (meV)")
        ax.set_title(title)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.grid(alpha=0.25)
        ax.text(0.17, 2.3, "K", color="red", fontweight="bold")
        ax.text(0.02, 1.25, "J", color="green", fontweight="bold")
        ax.text(0.17, 0.7, "Γ′", color="blue")
        ax.text(0.17, 0.05, "Γ", color="darkorange")
    legend = [
        Line2D([0], [0], color="k", ls="-", lw=2.6, label="U=2 eV"),
        Line2D([0], [0], color="k", ls="-", lw=1.1, label="(×1.5) U=3 eV"),
        Line2D([0], [0], color="k", ls="--", lw=1.3, label="(×2) U=4 eV"),
        Line2D([0], [0], color="k", ls=":", lw=1.6, label="(×3) U=6 eV"),
    ]
    axes[0].legend(handles=legend, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "prb_fig5.png", dpi=130)
    plt.close(fig)
    print("prb_fig5.png")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    _set_cjk_font()
    PLOTS.mkdir(parents=True, exist_ok=True)
    if which in ("gau", "all"):
        plot_gau()
    if which in ("nature", "all"):
        plot_nature()
    if which in ("prb", "all"):
        plot_prb()


if __name__ == "__main__":
    main()
