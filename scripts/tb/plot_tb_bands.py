#!/usr/bin/env python3
"""Plot DFT-vs-TB band comparisons for tb_from_enumerate outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import subprocess
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
CALC_RE = Path("/home/k0282/k028230/calculation/RE")
DEFAULT_WINDOWS = [(-0.25, 0.25), (-1.5, 0.8)]
DEFAULT_FIT_WINDOW = (-0.25, 0.25)
FIT_SHIFT_RANGE = (-20.0, 20.0)


@dataclass(frozen=True)
class PdosSpec:
    path: Path
    f72_col: int
    f52_col: int
    p_cols: tuple[int, ...]
    re_label: str
    ligand_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="tb directories, enumerate_nn_bonds.txt files, or band text files. Defaults to all data/data-DFT-input/*/*/tb.",
    )
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        action="append",
        metavar=("EMIN", "EMAX"),
        default=None,
        help=(
            "Energy window to draw after subtracting E_F. May be repeated. "
            "Defaults to near-Fermi and the reference -1.5..0.8 eV window."
        ),
    )
    parser.add_argument(
        "--fit-window",
        nargs=2,
        type=float,
        metavar=("EMIN", "EMAX"),
        default=DEFAULT_FIT_WINDOW,
        help="Near-Fermi window whose lower envelope is used to fit the rigid TB energy shift.",
    )
    parser.add_argument(
        "--tb-shift-mode",
        choices=("fit", "header", "efermi", "zero"),
        default="fit",
        help="How to align TB energies in plots. Default fits one rigid shift per TB curve.",
    )
    parser.add_argument("--no-run", action="store_true", help="Only write gnuplot scripts; do not run gnuplot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    windows = [tuple(window) for window in args.window] if args.window else list(DEFAULT_WINDOWS)
    fit_window = tuple(args.fit_window)
    for tb_dir in resolve_tb_dirs(args.targets):
        plot_tb_dir(
            tb_dir,
            windows,
            fit_window=fit_window,
            run=not args.no_run,
            tb_shift_mode=args.tb_shift_mode,
        )
    return 0


def resolve_tb_dirs(targets: list[Path]) -> list[Path]:
    if not targets:
        return sorted((REPO / "data" / "data-DFT-input").glob("*/*/tb"))

    out: list[Path] = []
    for target in targets:
        path = target if target.is_absolute() else REPO / target
        if path.is_dir():
            out.append(path if path.name == "tb" else path / "tb")
        elif path.name == "enumerate_nn_bonds.txt":
            out.append(path.parent / "tb")
        elif re.search(r"_bands(?:_downfold|_pruned_bondpp)?(?:_[A-Za-z0-9]+)?\.txt$", path.name):
            out.append(path.parent)
        else:
            raise SystemExit(f"Unsupported target: {target}")
    return sorted(set(out))


def plot_tb_dir(
    tb_dir: Path,
    windows: list[tuple[float, float]],
    *,
    fit_window: tuple[float, float],
    run: bool,
    tb_shift_mode: str,
) -> None:
    pairs = find_band_pairs(tb_dir)
    if not pairs:
        raise SystemExit(f"No <name>_bands.txt / <name>_bands_downfold.txt pair under {tb_dir}")

    for stem, variant, pruned_band, downfold_band in pairs:
        dft_band = resolve_dft_band(tb_dir, stem)
        efermi = resolve_efermi(tb_dir, dft_band)
        dft_nodes = read_dft_nodes(dft_band)
        tb_ticks = read_tb_ticks(pruned_band)
        pdos_spec = resolve_pdos(dft_band, stem)
        if not dft_nodes:
            raise SystemExit(f"Cannot read DFT k labels from {dft_band}")
        if not tb_ticks:
            raise SystemExit(f"Cannot read TB ticks from {pruned_band}")

        model_specs = [
            ("pruned", pruned_band, "pruned TB"),
        ]
        pruned_bondpp_band = tb_dir / f"{stem}_bands_pruned_bondpp{variant}.txt"
        if pruned_bondpp_band.is_file():
            model_specs.append(("pruned_bondpp", pruned_bondpp_band, "pruned TB (+bond p-p)"))
        model_specs.append(("downfold", downfold_band, "downfold TB"))

        for model, tb_band, title in model_specs:
            variant_title = f"{title} {variant_label(variant)}".strip()
            shift_mode = read_tb_shift_mode(tb_band) if tb_shift_mode == "header" else tb_shift_mode
            tb_shift = resolve_tb_shift(
                mode=shift_mode,
                dft_band=dft_band,
                tb_band=tb_band,
                efermi=efermi,
                window=fit_window,
            )
            for index, window in enumerate(windows):
                tag = range_tag(window)
                outputs = [(tb_dir / f"{stem}_dft_vs_{model}{variant}.near.{tag}.gnuplot",
                            tb_dir / f"{stem}_dft_vs_{model}{variant}.near.{tag}.png")]
                if index == 0:
                    outputs.append((tb_dir / f"{stem}_dft_vs_{model}{variant}.gnuplot",
                                    tb_dir / f"{stem}_dft_vs_{model}{variant}.png"))

                for script, png in outputs:
                    script.write_text(
                        build_gnuplot(
                            dft_band=dft_band,
                            tb_band=tb_band,
                            output=png,
                            stem=stem,
                            tb_title=variant_title,
                            dft_nodes=dft_nodes,
                            tb_ticks=tb_ticks,
                            efermi=efermi,
                            tb_shift=tb_shift,
                            window=window,
                            pdos_spec=pdos_spec,
                        ),
                        encoding="utf-8",
                    )
                    if run:
                        subprocess.run(["gnuplot", str(script)], check=True)
                    print(
                        f"wrote {png} "
                        f"(DFT={dft_band}, E_F={efermi:.4f} eV, TB shift={tb_shift:.4f} eV)"
                    )


def find_band_pairs(tb_dir: Path) -> list[tuple[str, str, Path, Path]]:
    pairs = []
    pattern = re.compile(r"^(?P<stem>.+)_bands(?P<variant>(?:_[A-Za-z0-9]+)?)\.txt$")
    for pruned_band in sorted(tb_dir.glob("*_bands*.txt")):
        match = pattern.match(pruned_band.name)
        if match is None or pruned_band.name.startswith("."):
            continue
        stem = match.group("stem")
        variant = match.group("variant")
        if stem.endswith("_bands_downfold"):
            continue
        downfold_band = tb_dir / f"{stem}_bands_downfold{variant}.txt"
        if downfold_band.is_file():
            pairs.append((stem, variant, pruned_band, downfold_band))
    return pairs


def variant_label(variant: str) -> str:
    if not variant:
        return ""
    if variant == "_2d":
        return "(Rz=0)"
    if variant == "_socdelta":
        return "(SOC+Delta onsite)"
    return f"({variant.removeprefix('_')})"


def resolve_dft_band(tb_dir: Path, stem: str) -> Path:
    group, material_dir = material_context(tb_dir)
    candidates = sorted(calc_band_root(group).glob(f"**/figures/{stem}.bands.dat"))
    candidates = [path for path in candidates if band_candidate_ok(path, group, material_dir)]
    if not candidates:
        raise SystemExit(f"No DFT bands.dat found for {group}/{material_dir} ({stem})")
    return sorted(candidates, key=lambda path: band_score(path, group, material_dir))[0]


def material_context(tb_dir: Path) -> tuple[str, str]:
    try:
        rel = tb_dir.resolve().relative_to((REPO / "data" / "data-DFT-input").resolve())
    except ValueError as exc:
        raise SystemExit(f"tb directory is not under data/data-DFT-input: {tb_dir}") from exc
    parts = rel.parts
    if len(parts) < 3:
        raise SystemExit(f"Cannot infer material context from {tb_dir}")
    return parts[0], parts[1]


def calc_band_root(group: str) -> Path:
    if group == "REOF":
        return CALC_RE / "REChX" / "REChX_SCF_SOC_ACB"
    if group in {"REOCl", "RESI"}:
        return CALC_RE / "REChX" / "REChX_SCF_SOC_ABC"
    if group == "REX3-C2m":
        return CALC_RE / "REX3" / "REX3_SOC" / "REX3_C2m"
    if group == "REX3-R3":
        return CALC_RE / "REX3" / "REX3_SOC" / "REX3_R3"
    raise SystemExit(f"Unknown material group: {group}")


def band_candidate_ok(path: Path, group: str, material_dir: str) -> bool:
    parts = set(path.parts)
    wants_nc = material_dir.endswith("-NC")
    has_nc = any(part.endswith("-NC") for part in path.parts)
    if wants_nc != has_nc and group == "REX3-C2m":
        return False
    if group == "RESI" and "I-nore" in parts:
        return False
    return True


def band_score(path: Path, group: str, material_dir: str) -> tuple[int, str]:
    score = 0
    parts = set(path.parts)
    if "data" in parts:
        score += 10
    if group == "RESI" and "I" in parts:
        score -= 5
    if material_dir.endswith("-NC") and any(part.endswith("-NC") for part in path.parts):
        score -= 5
    return score, str(path)


def resolve_efermi(tb_dir: Path, dft_band: Path) -> float:
    group, material_dir = material_context(tb_dir)
    fermi_material = material_dir.removesuffix("-re")
    search_dirs = [
        CALC_RE / "data-DFT" / group / fermi_material / "nscf",
        CALC_RE / "data-DFT-position" / group / fermi_material / "nscf",
        dft_band.parent.parent / "output",
        dft_band.parent.parent / "nscf",
    ]
    for directory in search_dirs:
        found = read_last_fermi(directory)
        if found is not None:
            return found
    raise SystemExit(f"No Fermi energy found for {group}/{material_dir}")


def read_last_fermi(directory: Path) -> float | None:
    if not directory.is_dir():
        return None
    pattern = re.compile(r"the Fermi energy is\s+([-+0-9.]+)\s+ev", re.IGNORECASE)
    found = None
    for path in sorted(directory.glob("*.out")):
        for match in pattern.finditer(path.read_text(errors="ignore")):
            found = float(match.group(1))
    return found


def read_dft_nodes(path: Path) -> list[tuple[str, float]]:
    nodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# klabel"):
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 4:
            continue
        nodes.append((parts[3], float(parts[2])))
    return nodes


def read_tb_ticks(path: Path) -> list[tuple[str, float]]:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    if not first.startswith("# ticks:"):
        return []
    ticks = []
    for item in first.removeprefix("# ticks:").split():
        label, value = item.rsplit("=", 1)
        ticks.append((label, float(value)))
    return ticks


def read_tb_shift_mode(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines()[:8]:
        if line.startswith("# tb_shift_mode:"):
            value = line.split(":", 1)[1].strip().lower()
            if value == "fermi":
                return "efermi"
            if value in {"efermi", "zero", "fit"}:
                return value
        if not line.startswith("# subtract_efermi_in_plot:"):
            continue
        value = line.split(":", 1)[1].strip().lower()
        return "zero" if value in {"0", "false", "no", "off"} else "efermi"
    return "efermi"


def resolve_pdos(dft_band: Path, stem: str) -> PdosSpec | None:
    roots = [
        dft_band.parent.parent / "wannier90" / "out_tb_full" / "pdos.dat",
        dft_band.parent.parent / "data" / "wannier90" / "out_tb_full" / "pdos.dat",
    ]
    for path in roots:
        if not path.is_file():
            continue
        spec = read_pdos_spec(path, stem)
        if spec is not None:
            return spec
    return None


def read_pdos_spec(path: Path, stem: str) -> PdosSpec | None:
    try:
        header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except IndexError:
        return None
    if not header.startswith("#"):
        return None
    names = header.removeprefix("#").split()
    index = {name: col for col, name in enumerate(names, start=1)}
    f72_col = index.get("f1_72_hole", index.get("f1_72"))
    f52_col = index.get("f1_52_hole", index.get("f1_52"))
    p_cols = tuple(col for name, col in index.items() if re.fullmatch(r"p\d+", name))
    if f72_col is None or f52_col is None or not p_cols:
        return None
    re_label, ligand_label = material_labels(stem)
    return PdosSpec(
        path=path,
        f72_col=f72_col,
        f52_col=f52_col,
        p_cols=p_cols,
        re_label=re_label,
        ligand_label=ligand_label,
    )


def material_labels(stem: str) -> tuple[str, str]:
    match = re.match(r"^(Ce|Nd|Sm|Dy|Er|Yb)(.*)$", stem)
    if match is None:
        return "RE", "ligand"
    re_label = match.group(1)
    ligand_part = match.group(2)
    ligands = re.findall(r"[A-Z][a-z]?", ligand_part)
    if not ligands:
        return re_label, "ligand"
    return re_label, "/".join(ligands)


def pdos_xmax(spec: PdosSpec, efermi: float, window: tuple[float, float]) -> float:
    try:
        data = np.loadtxt(spec.path, comments="#")
    except OSError:
        return 1.0
    if data.ndim == 1:
        data = data[None, :]
    energies = data[:, 0] - float(efermi)
    lo, hi = window
    mask = (energies >= lo) & (energies <= hi)
    if not np.any(mask):
        mask = np.ones_like(energies, dtype=bool)
    curves = [data[mask, spec.f72_col - 1], data[mask, spec.f52_col - 1]]
    curves.append(np.sum([data[mask, col - 1] for col in spec.p_cols], axis=0) * 5.0)
    max_value = max(float(np.nanmax(curve)) for curve in curves if curve.size)
    if not np.isfinite(max_value) or max_value <= 0.0:
        return 1.0
    return max_value * 1.08


def resolve_tb_shift(*, mode: str, dft_band: Path, tb_band: Path, efermi: float, window: tuple[float, float]) -> float:
    if mode == "efermi":
        return float(efermi)
    if mode == "zero":
        return 0.0
    if mode == "fit":
        return fit_tb_shift(dft_band, tb_band, efermi=efermi, window=window)
    raise SystemExit(f"Unknown tb_shift_mode {mode!r} in {tb_band}")


def fit_tb_shift(dft_band: Path, tb_band: Path, *, efermi: float, window: tuple[float, float]) -> float:
    dft_xs, dft_es = read_band_blocks(dft_band)
    tb_xs, tb_es = read_band_blocks(tb_band)
    tb_x_full = tb_xs[0]
    step = max(1, len(tb_x_full) // 140)
    tb_x = tb_x_full[::step]
    tb_mat = np.vstack([e[::step] for e in tb_es]).T
    dft_x = tb_x * (dft_xs[0][-1] / tb_x_full[-1])
    dft_mat = eval_band_blocks(dft_xs, dft_es, dft_x) - float(efermi)

    lo, hi = window
    target_blocks = []
    target_count = 0
    for dft_levels, tb_levels in zip(dft_mat, tb_mat):
        near = dft_levels[(dft_levels >= lo) & (dft_levels <= hi)]
        if near.size == 0:
            continue
        dft_lower_edge = float(np.min(near))
        target_blocks.append(tb_levels - dft_lower_edge)
        target_count += 1
    if target_count == 0:
        return float(efermi)

    def rmse_grid(grid: np.ndarray) -> np.ndarray:
        sumsq = np.zeros(grid.shape, dtype=float)
        for targets in target_blocks:
            errs = np.min(np.abs(grid[:, None] - targets[None, :]), axis=1)
            sumsq += errs * errs
        return np.sqrt(sumsq / target_count)

    coarse = np.linspace(FIT_SHIFT_RANGE[0], FIT_SHIFT_RANGE[1], 801)
    coarse_vals = rmse_grid(coarse)
    center = float(coarse[int(np.argmin(coarse_vals))])
    fine = np.linspace(center - 0.04, center + 0.04, 161)
    fine_vals = rmse_grid(fine)
    return float(fine[int(np.argmin(fine_vals))])


def read_band_blocks(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs, es = [], []
    cur_x, cur_e = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            if cur_x:
                xs.append(np.asarray(cur_x, dtype=float))
                es.append(np.asarray(cur_e, dtype=float))
                cur_x, cur_e = [], []
            continue
        if text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        cur_x.append(float(parts[0]))
        cur_e.append(float(parts[1]))
    if cur_x:
        xs.append(np.asarray(cur_x, dtype=float))
        es.append(np.asarray(cur_e, dtype=float))
    return xs, es


def eval_band_blocks(xs: list[np.ndarray], es: list[np.ndarray], xnew: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(xnew, x, e) for x, e in zip(xs, es)]).T


def build_gnuplot(
    *,
    dft_band: Path,
    tb_band: Path,
    output: Path,
    stem: str,
    tb_title: str,
    dft_nodes: list[tuple[str, float]],
    tb_ticks: list[tuple[str, float]],
    efermi: float,
    tb_shift: float,
    window: tuple[float, float],
    pdos_spec: PdosSpec | None,
) -> str:
    emin, emax = window
    labels = labels_for_plot(dft_nodes, tb_ticks)
    x0, x1 = dft_nodes[0][1], dft_nodes[-1][1]
    label_cmds = label_commands(labels, x1 - x0)
    arrows = [
        f"set arrow {i} from {x:.12f}, {emin:g} to {x:.12f}, {emax:g} nohead lw 1.2 lc rgb 'black'"
        for i, (_, x) in enumerate(unique_inner_positions(labels), start=1)
    ]
    unset_annotations = unset_annotation_commands(label_count=len(labels), arrow_count=len(arrows))
    if pdos_spec is None:
        return "\n".join(
            [
                'set terminal pngcairo size 1560,820 enhanced font ",16"',
                f'dft_file = "{dft_band}"',
                f'tb_file = "{tb_band}"',
                f"efermi = {efermi:.12f}",
                f"tb_shift = {tb_shift:.12f}",
                "",
                "stats tb_file u 1 nooutput",
                f"tb_xscale = {x1:.12f} / STATS_max",
                "",
                f'set output "{output}"',
                "set tmargin at screen 0.91",
                "set bmargin at screen 0.19",
                "set lmargin at screen 0.12",
                "set rmargin at screen 0.91",
                f"set xrange [{x0:.12f}:{x1:.12f}]",
                f"set yrange [{emin:g}:{emax:g}]",
                'set ylabel "Energy - E_F (eV)"',
                "set ytics nomirror",
                'set format y "%g"',
                "unset xtics",
                "set zeroaxis lt -1 dt 2 lw 1.6",
                'set key top right noopaque font ",11"',
                *label_cmds,
                *arrows,
                "plot \\",
                "    dft_file u 1:($2-efermi) w l lw 1.3 lc rgb 'black' title 'DFT', \\",
                f"    tb_file u ($1*tb_xscale):($2-tb_shift) w l lw 1.6 dt 2 lc rgb 'red' title '{escape_title(tb_title)}'",
                "",
            ]
        )

    p_expr = "+".join(f"column({col})" for col in pdos_spec.p_cols)
    pdos_max = pdos_xmax(pdos_spec, efermi, window)
    return "\n".join(
        [
            'set terminal pngcairo size 1560,820 enhanced font ",16"',
            f'dft_file = "{dft_band}"',
            f'tb_file = "{tb_band}"',
            f'pdos_file = "{pdos_spec.path}"',
            f"efermi = {efermi:.12f}",
            f"tb_shift = {tb_shift:.12f}",
            "",
            "stats tb_file u 1 nooutput",
            f"tb_xscale = {x1:.12f} / STATS_max",
            "",
            f'set output "{output}"',
            "set multiplot",
            "set tmargin at screen 0.91",
            "set bmargin at screen 0.19",
            f"set yrange [{emin:g}:{emax:g}]",
            "set zeroaxis lt -1 dt 2 lw 1.6",
            "",
            "set lmargin at screen 0.1231",
            "set rmargin at screen 0.6646",
            f"set xrange [{x0:.12f}:{x1:.12f}]",
            'set ylabel "Energy - E_F (eV)"',
            "set ytics nomirror",
            'set format y "%g"',
            "unset xtics",
            'set key top right noopaque font ",11"',
            *label_cmds,
            *arrows,
            "plot \\",
            "    dft_file u 1:($2-efermi) w l lw 1.3 lc rgb 'black' title 'DFT', \\",
            f"    tb_file u ($1*tb_xscale):($2-tb_shift) w l lw 1.6 dt 2 lc rgb 'red' title '{escape_title(tb_title)}'",
            "",
            *unset_annotations,
            "unset ylabel",
            'set format y ""',
            "unset xtics",
            "",
            "set lmargin at screen 0.6646",
            "set rmargin at screen 0.8985",
            f"set xrange [0:{pdos_max:.6f}]",
            "set ytics nomirror",
            'set key top right noopaque samplen 1.2 spacing 0.95 font ",11"',
            'set label 2001 "PDOS" at screen 0.7815, screen 0.968 center font ",20" tc rgb "black"',
            "plot \\",
            (
                f"    pdos_file u (column({pdos_spec.f72_col})):($1-efermi) "
                f"w l lw 1.8 lc rgb 'red' title '{pdos_spec.re_label} F_{{7/2}}', \\"
            ),
            (
                f"    pdos_file u (column({pdos_spec.f52_col})):($1-efermi) "
                f"w l lw 1.8 lc rgb 'blue' title '{pdos_spec.re_label} F_{{5/2}}', \\"
            ),
            (
                f"    pdos_file u ((({p_expr}))*5):($1-efermi) "
                f"w l lw 1.8 lc rgb 'green' title '{pdos_spec.ligand_label} p x5'"
            ),
            "unset label 2001",
            "unset multiplot",
            "",
        ]
    )


def range_tag(window: tuple[float, float]) -> str:
    return f"{format_bound(window[0])}_{format_bound(window[1])}"


def format_bound(value: float) -> str:
    return f"{value:g}"


def label_commands(labels: list[tuple[str, float]], span: float) -> list[str]:
    commands = []
    previous_x = None
    previous_y = -0.055
    for idx, (label, x) in enumerate(labels, start=1001):
        y = -0.055
        is_close = previous_x is not None and abs(x - previous_x) < 0.07 * span
        if label == "M/K":
            y = -0.095
        elif is_close and previous_y < -0.07:
            y = -0.055
        elif is_close and "/" in label:
            y = -0.095
        commands.append(
            f"set label {idx} '{format_label(label)}' at {x:.12f}, graph {y:g} "
            "center front font ',12'"
        )
        previous_x = x
        previous_y = y
    return commands


def unset_annotation_commands(*, label_count: int, arrow_count: int) -> list[str]:
    commands = [f"unset arrow {idx}" for idx in range(1, arrow_count + 1)]
    commands.extend(f"unset label {idx}" for idx in range(1001, 1001 + label_count))
    return commands


def escape_title(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def labels_for_plot(dft_nodes: list[tuple[str, float]], tb_ticks: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if dft_nodes:
        named_nodes = [(kpoint_name(label), x) for label, x in dft_nodes]
        if all(label is not None for label, _ in named_nodes):
            return merge_same_x_labels([(label, x) for label, x in named_nodes if label is not None])
    if dft_nodes and all(not label.startswith("(") for label, _ in dft_nodes):
        return merge_same_x_labels(dft_nodes)
    scale = dft_nodes[-1][1] / tb_ticks[-1][1]
    return [(label, x * scale) for label, x in tb_ticks]


def kpoint_name(label: str) -> str | None:
    values = re.findall(r"[-+0-9.]+", label)
    if len(values) != 3:
        return None
    coord = tuple(float(value) for value in values)
    known = {
        (0.0, 0.0, 0.0): "G",
        (0.5, 0.0, 0.0): "M",
        (1.0 / 3.0, 1.0 / 3.0, 0.0): "K",
        (0.0, 0.0, 0.5): "A",
        (0.5, 0.0, 0.5): "L",
        (1.0 / 3.0, 1.0 / 3.0, 0.5): "H",
    }
    for ref, name in known.items():
        if all(abs(x - y) < 1e-5 for x, y in zip(coord, ref)):
            return name
    return None


def merge_same_x_labels(nodes: list[tuple[str, float]]) -> list[tuple[str, float]]:
    merged: list[tuple[list[str], float]] = []
    for label, x in nodes:
        if merged and abs(merged[-1][1] - x) < 1e-8:
            if label not in merged[-1][0]:
                merged[-1][0].append(label)
        else:
            merged.append(([label], x))
    return [("/".join(labels), x) for labels, x in merged]


def unique_inner_positions(labels: list[tuple[str, float]]) -> list[tuple[str, float]]:
    out = []
    seen = set()
    for label, x in labels[1:-1]:
        key = round(x, 10)
        if key in seen:
            continue
        seen.add(key)
        out.append((label, x))
    return out


def format_label(label: str) -> str:
    if label in {"G", "Gamma", "GAMMA"}:
        return "{/Symbol G}"
    return label.replace("_", "\\_").replace('"', '\\"')


if __name__ == "__main__":
    raise SystemExit(main())
