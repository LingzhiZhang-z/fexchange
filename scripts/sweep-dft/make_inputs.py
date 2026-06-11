#!/usr/bin/env python3
"""Build DFT-based sweep TOMLs for sopt, fopt, and sopt_direct."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

U_VALUES = (4.0, 6.0, 8.0)
JH_MAX = 0.40
JH_POINTS = 41
U_P = 3.0
OUTPUT_ROOT = "/work/k0282/k028230/code/fexchange/outputs"

# DFT onsite SOC means summarized from the completed w90 onsite extraction.
ZETA_BY_RE = {
    "Ce": 0.089441917,
    "Nd": 0.124237963,
    "Sm": 0.163421194,
    "Dy": 0.261607194,
    "Er": 0.319688417,
    "Yb": 0.390238722,
}

# Only Br/I ligand SOC is enabled in the exchange inputs. Everything else,
# including S, is written as zero.
LAMBDA_P_BY_ELEMENT = {
    "Br": 0.270838611,
    "I": 0.570592185,
}

RE_ORDER = ("Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb")
RE_TO_N = {re: i + 1 for i, re in enumerate(RE_ORDER)}
N_TO_RE = {i + 1: re for i, re in enumerate(RE_ORDER)}
RE_RE = re.compile(r"^(Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb)")

REX3_KRAMERS = {
    "Ce": (("ohg7", "f1_ce_g7.txt"),),
    "Nd": (("ohg6", "f3_nd_g6.txt"),),
    "Sm": (("ohg7", "f5_sm_g7.txt"),),
    "Dy": (("ohg6", "f9_dy_g6.txt"), ("ohg7", "f9_dy_g7.txt"), ("c2", "YbCl3_exp_baseline_Dy_J15_2_projector.txt")),
    "Er": (("ohg7", "f11_er_g7.txt"), ("c2", "YbCl3_exp_baseline_Er_J15_2_projector.txt")),
    "Yb": (("ohg6", "f13_yb_g6.txt"), ("c2", "YbCl3_exp_baseline_Yb_J7_2_projector.txt")),
}

RECHX_KRAMERS = {
    ("REOCl", "ErOCl"): (
        "ErOCl_exp_baseline_Er_J15_2_projector.txt",
    ),
    ("REOCl", "YbOCl"): (
        "YbOCl_baseline_Yb_J7_2_projector.txt",
        "YbOCl_exp_baseline_Yb_J7_2_projector.txt",
    ),
    ("REOF", "DyOF"): (
        "YbOCl_exp_baseline_Dy_J15_2_projector.txt",
        "ErOCl_exp_baseline_Dy_J15_2_projector.txt",
        "NdOF_exp_baseline_Dy_J15_2_projector.txt",
        "DyOF_baseline_Dy_J15_2_projector.txt",     # DyOF's own DFT-extracted (Ising-like) doublet
    ),
    ("REOF", "ErOF"): (
        "YbOCl_baseline_Er_J15_2_projector.txt",
        "ErOCl_exp_baseline_Er_J15_2_projector.txt",
        "ErOF_baseline_Er_J15_2_projector.txt",     # ErOF's own DFT-extracted (Ising-like) doublet
    ),
    ("REOF", "NdOF"): (
        "NdOF_exp_baseline_Nd_J9_2_projector.txt",
    ),
    ("REOF", "SmOF"): (
        "CeSm_C3v_Gamma4_J5_2_theta000_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta015_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta030_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta045_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta060_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta075_projector.txt",
    ),
    ("RESI", "NdSI-re"): (
        "YbOCl_baseline_Nd_J9_2_projector.txt",
        "NdOF_exp_baseline_Nd_J9_2_projector.txt",
    ),
    ("RESI", "SmSI-re"): (
        "CeSm_C3v_Gamma4_J5_2_theta000_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta015_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta030_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta045_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta060_projector.txt",
        "CeSm_C3v_Gamma4_J5_2_theta075_projector.txt",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("data/data-DFT-input"))
    ap.add_argument("--output-root", default=OUTPUT_ROOT)
    ap.add_argument("--u-values", default=",".join(f"{u:g}" for u in U_VALUES))
    ap.add_argument("--jh-max", type=float, default=JH_MAX)
    ap.add_argument("--jh-points", type=int, default=JH_POINTS)
    ap.add_argument("--u-p", type=float, default=U_P)
    ap.add_argument("--samples-only", action="store_true")
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--family")
    ap.add_argument("--material")
    ap.add_argument("--projector-file")
    args = ap.parse_args()

    root = args.input_root.resolve()
    u_values = tuple(float(x) for x in args.u_values.split(",") if x.strip())
    ratios = tuple(i * args.jh_max / (args.jh_points - 1) for i in range(args.jh_points))

    materials = material_dirs(root)

    if args.samples_only:
        sample_material = root / "RESI" / "NdSI-re"
        write_samples(sample_material, args.output_root, u_values, ratios, args.u_p)
        print("wrote 3 samples under scripts/sweep-dft/samples")
        return 0
    wrote_samples = False
    if not args.no_samples:
        sample_material = root / "RESI" / "NdSI-re"
        write_samples(sample_material, args.output_root, u_values, ratios, args.u_p)
        wrote_samples = True

    if args.family:
        materials = [mat for mat in materials if mat.parent.name == args.family]
    if args.material:
        materials = [mat for mat in materials if mat.name == args.material]

    n_written = 0
    for mat in materials:
        re = infer_re(mat.name)
        projectors = kramer_choices(mat, re)
        if args.projector_file:
            projectors = tuple(
                item for item in projectors if Path(item[1]).name == args.projector_file or item[0] == args.projector_file
            )
            if not projectors:
                raise ValueError(f"no projector matched --projector-file={args.projector_file!r} for {mat.parent.name}/{mat.name}")
        for bond in bond_dirs(mat):
            for label, kfile in projectors:
                out_dir = bond / "sweep-dft" / path_token(label)
                out_dir.mkdir(parents=True, exist_ok=True)
                for mode in ("sopt", "fopt", "sopt_direct"):
                    text = build_bond_toml(
                        mat,
                        bond,
                        mode,
                        (label, kfile),
                        args.output_root,
                        u_values,
                        ratios,
                        args.u_p,
                    )
                    (out_dir / f"{mode}.toml").write_text(text, encoding="utf-8")
                    n_written += 1
    print(f"wrote {n_written} sweep TOMLs under {root}")
    if wrote_samples:
        print("wrote 3 samples under scripts/sweep-dft/samples")
    else:
        print("skipped samples")
    return 0


def material_dirs(root: Path) -> list[Path]:
    mats = []
    for family in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "kramer"):
        for mat in sorted(p for p in family.iterdir() if p.is_dir()):
            is_rex3 = family.name.startswith("REX3")
            is_selected_rechx = (family.name, mat.name) in RECHX_KRAMERS
            if bond_dirs(mat) and (is_rex3 or is_selected_rechx):
                mats.append(mat)
    return mats


def bond_dirs(mat: Path) -> list[Path]:
    return sorted(p for p in mat.iterdir() if p.is_dir() and (p / "w90" / "downfold.toml").exists())


def read_tokens(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("["):
            continue
        if line.startswith("#"):
            line = line[1:].strip()
        if "=" in line:
            key, val = (x.strip() for x in line.split("=", 1))
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts
        out[key] = val.strip().strip('"')
    return out


def fmt(x: float) -> str:
    x = float(x)
    if abs(x) < 5.0e-13:
        x = 0.0
    s = f"{x:.12f}".rstrip("0").rstrip(".")
    return s if "." in s else f"{s}.0"


def write_samples(
    mat: Path,
    output_root: str,
    u_values: tuple[float, ...],
    ratios: tuple[float, ...],
    u_p: float,
) -> None:
    sample_dir = Path("scripts/sweep-dft/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    bond = [b for b in bond_dirs(mat) if b.name == "J1-y"][0]
    projector = kramer_choices(mat, infer_re(mat.name))[0]
    (sample_dir / "sopt_sample.toml").write_text(
        build_bond_toml(mat, bond, "sopt", projector, output_root, u_values, ratios, u_p),
        encoding="utf-8",
    )
    (sample_dir / "fopt_sample.toml").write_text(
        build_bond_toml(mat, bond, "fopt", projector, output_root, u_values, ratios, u_p),
        encoding="utf-8",
    )
    (sample_dir / "sopt_direct_sample.toml").write_text(
        build_bond_toml(mat, bond, "sopt_direct", projector, output_root, u_values, ratios, u_p),
        encoding="utf-8",
    )


def build_bond_toml(
    mat: Path,
    bond: Path,
    mode: str,
    projector: tuple[str, str],
    output_root: str,
    u_values: tuple[float, ...],
    ratios: tuple[float, ...],
    u_p: float,
) -> str:
    re = infer_re(mat.name)
    n_ele = RE_TO_N[re]
    zeta = ZETA_BY_RE[re]
    first = first_row_values(mat, bond, projector, mode, u_values[0], 0, ratios[0])
    label, kfile = projector

    lines = [
        "# Generated by scripts/sweep-dft/make_inputs.py",
        f"# source_material = {mat.resolve()}",
        f"# source_bond = {bond.resolve()}",
        f"# mode = {mode}",
        f"# kramer_label = {label}",
        f"# Uplus + Uminus values = {', '.join(f'{u:g}' for u in u_values)} eV",
        f"# Jh/U values = 0..{ratios[-1]:.2f}, {len(ratios)} points",
        'schema_version = "fxe.run_input.v1"',
        'standard_version = "2026-02"',
        f'run_id = "{mat.name}_{bond.name}_{mode}_{label}_dft_sweep"',
        f'title = "{mat.parent.name} {mat.name}/{bond.name} {mode} {label} DFT sweep"',
        "",
        "[units]",
        'energy = "eV"',
        "",
    ]
    lines.extend(fsite_lines(re, n_ele, zeta, first["Jh"], first["Uminus"], first["Uplus"]))
    lines.extend([
        "",
        "[model]",
        'scheme = "ED"',
        "",
    ])
    if mode == "fopt":
        lines.extend([
            "[ligand.1]",
            f'Delta = {fmt(first["Delta1"])}',
            f"U_p = {fmt(u_p)}",
            f'lambda_p = {fmt(first["lambda1"])}',
            "",
            "[ligand.2]",
            f'Delta = {fmt(first["Delta2"])}',
            f"U_p = {fmt(u_p)}",
            f'lambda_p = {fmt(first["lambda2"])}',
            "",
        ])
    lines.extend([
        "[inputs]",
        f'hopping_file = "{first["hopping"]}"',
        f'kramer_file = "{kfile}"',
        "",
        "[paths]",
        f'output_root = "{output_root}"',
        f'output_run = "{output_run_base(output_root, mat, bond, mode, label)}"',
        "",
        "[runtime]",
        f'branch = "{branch_for_mode(mode)}"',
        'end_level = "L3"',
        f'run_name = "{first["run_name"]}"',
        'kramer_source = "stevens"',
        "",
        "[checks]",
        "strict_mode = true",
        'eps_profile = "default"',
        "",
    ])
    lines.extend(sweep_table(mat, bond, mode, label, u_values, ratios))
    return "\n".join(lines) + "\n"


def first_row_values(
    mat: Path,
    bond: Path,
    projector: tuple[str, str],
    mode: str,
    u: float,
    jh_index: int,
    ratio: float,
) -> dict[str, float | str]:
    uminus, uplus = side_gaps(infer_re(mat.name), u)
    info = read_tokens(bond / "w90" / "downfold.toml")
    lam1 = ligand_lambda(info.get("lig1_element", ""))
    lam2 = ligand_lambda(info.get("lig2_element", ""))
    return {
        "U": u,
        "Jh": u * ratio,
        "Uminus": uminus,
        "Uplus": uplus,
        "Delta1": float(info["delta_lig1"]),
        "Delta2": float(info["delta_lig2"]),
        "lambda1": lam1,
        "lambda2": lam2,
        "hopping": str(hopping_file(bond, mode)),
        "run_name": run_name(mat, bond, mode, u, jh_index, projector[0], info),
    }


def fsite_lines(re: str, n_ele: int, zeta: float, jh: float, uminus: float, uplus: float) -> list[str]:
    lines = [
        "[fsite]",
        f"n_ele = {n_ele}",
        f'RE = "{re}"',
        "U = 0.0",
        f"Jh = {fmt(jh)}",
        f"zeta = {fmt(zeta)}",
        "offset = 0.0",
        'energy_reference = "zero"',
        "",
        "[fsite_nm1]",
    ]
    if n_ele == 1:
        lines.extend([
            'RE = "Ce"',
            "U = 0.0",
            "Jh = 0.0",
            "zeta = 0.0",
        ])
    else:
        lines.extend([
            f'RE = "{N_TO_RE[n_ele - 1]}"',
            "U = 0.0",
            f"Jh = {fmt(jh)}",
            f"zeta = {fmt(zeta)}",
        ])
    lines.extend([
        f"Uminus = {fmt(uminus)}",
        "",
        "[fsite_np1]",
    ])
    if n_ele == 13:
        lines.extend([
            'RE = "Yb"',
            "U = 0.0",
            "Jh = 0.0",
            "zeta = 0.0",
        ])
    else:
        lines.extend([
            f'RE = "{N_TO_RE[n_ele + 1]}"',
            "U = 0.0",
            f"Jh = {fmt(jh)}",
            f"zeta = {fmt(zeta)}",
        ])
    lines.extend([
        f"Uplus = {fmt(uplus)}",
    ])
    return lines


def sweep_table(
    mat: Path,
    bond: Path,
    mode: str,
    klabel: str,
    u_values: tuple[float, ...],
    ratios: tuple[float, ...],
) -> list[str]:
    cols = "runtime.run_name Jh Uminus Uplus"
    rows = []
    re = infer_re(mat.name)
    info = read_tokens(bond / "w90" / "downfold.toml")
    for u in u_values:
        uminus, uplus = side_gaps(re, u)
        for jh_idx, ratio in enumerate(ratios):
            jh = u * ratio
            row = [
                run_name(mat, bond, mode, u, jh_idx, klabel, info),
                fmt(jh),
                fmt(uminus),
                fmt(uplus),
            ]
            rows.append(" ".join(row))
    return ["[sweep]", 'table = """', str(len(rows)), cols, *rows, '"""']


def infer_re(material: str) -> str:
    m = RE_RE.match(material)
    if not m:
        raise ValueError(f"cannot infer RE from material name: {material}")
    return m.group(1)


def side_gaps(re: str, u: float) -> tuple[float, float]:
    if re == "Ce":
        return (u - 1.0) / 2.0, (u + 1.0) / 2.0
    if re == "Yb":
        return (u + 1.0) / 2.0, (u - 1.0) / 2.0
    return (u + 3.0) / 2.0, (u - 3.0) / 2.0


def ligand_lambda(element: str) -> float:
    return LAMBDA_P_BY_ELEMENT.get(element, 0.0)


def hopping_file(bond: Path, mode: str) -> Path:
    w90 = bond / "w90"
    if mode == "sopt":
        return w90 / "hopping_ff_downfold.txt"
    if mode == "sopt_direct":
        return w90 / "hopping_ff_direct.txt"
    if mode == "fopt":
        return w90 / "hopping_fp.txt"
    raise ValueError(mode)


def branch_for_mode(mode: str) -> str:
    return "fopt" if mode == "fopt" else "sopt"


def kramer_choices(mat: Path, re: str) -> tuple[tuple[str, str], ...]:
    family = mat.parent.name
    root = mat.parents[1]
    if family.startswith("REX3"):
        choices = REX3_KRAMERS.get(re)
        if not choices:
            raise ValueError(f"no REX3 kramer mapping for {re}")
        return tuple(
            (kramer_label(root / "kramer" / "REX3" / name), str(root / "kramer" / "REX3" / name))
            for _, name in choices
        )
    choices = RECHX_KRAMERS.get((family, mat.name))
    if not choices:
        raise ValueError(f"no REChX kramer mapping for {family}/{mat.name}")
    return tuple(
        (kramer_label(root / "kramer" / "REChX" / name), str(root / "kramer" / "REChX" / name))
        for name in choices
    )


def kramer_label(path: Path) -> str:
    if path.stem != "kramer_projector":
        return path.stem
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("# kramer_name "):
            return line.split(None, 2)[2]
    return path.stem


def path_token(s: str) -> str:
    return s.strip().replace("/", "_")


def run_name(mat: Path, bond: Path, mode: str, u: float, jh_index: int, klabel: str, info: dict[str, str]) -> str:
    if abs(u - round(u)) < 1.0e-12:
        u_token = f"U{int(round(u)):02d}"
    else:
        u_token = f"U{u:g}".replace(".", "p")
    return f"{path_token(klabel)}/{u_token}/jh{jh_index:02d}"


def output_run_base(output_root: str, mat: Path, bond: Path, mode: str, klabel: str) -> str:
    return str(Path(output_root) / mat.parent.name / mat.name / bond.name / mode)


if __name__ == "__main__":
    raise SystemExit(main())
