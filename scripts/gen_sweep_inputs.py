"""Generate clean sweep bases, replacing the hand-written per-case TOMLs.

* PRB (mpirun): one base per (material, scheme), [sweep] over U x (JH/U); Jh = ratio*U.
* nature / gau (serial): one base per sector / gamma, [sweep] over the hopping file.

Bases live under data/<set>/sweep/ (inputs). ALL outputs go to ./outputs (data/
holds inputs only). Inert fields dropped; doublet input unified on `kramer_file`.
Physics config for nature/gau is read verbatim from an existing per-case template
so the per-sector fsite/sides/ligand/RE are exact.
"""
from __future__ import annotations

import glob
import re
import tomllib
from pathlib import Path

OUT = "./outputs"


def _emit_val(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return f'"{v}"'


def _emit_section(name: str, d: dict) -> str:
    lines = [f"[{name}]"]
    lines += [f"{k} = {_emit_val(v)}" for k, v in d.items() if not isinstance(v, dict)]
    return "\n".join(lines)


def _sweep_hopping(rows: list[tuple[str, str]]) -> str:
    body = "\n".join(f"{h}   {n}" for h, n in rows)
    return f'[sweep]\ntable = """\n{len(rows)}\ninputs.hopping_file   runtime.run_name\n{body}\n"""'


def _hopping_base(template: str, hoppings: list[str], rows: list[tuple[str, str]], kramer: str) -> str:
    cfg = tomllib.load(open(template, "rb"))
    parts = ['schema_version = "fxe.run_input.v1"', 'standard_version = "2026-02"', ""]
    for sec in ("fsite", "fsite_nm1", "fsite_np1"):
        if sec in cfg:
            parts += [_emit_section(sec, cfg[sec]), ""]
    if "model" in cfg:
        parts += [_emit_section("model", cfg["model"]), ""]
    for lk, lv in cfg.get("ligand", {}).items():
        parts += [_emit_section(f"ligand.{lk}", lv), ""]
    parts += [f'[inputs]\nhopping_file = "{hoppings[0]}"\nkramer_file = "{kramer}"', ""]
    parts += [f'[paths]\noutput_root = "{OUT}"', ""]
    rt = cfg["runtime"]
    parts += [f'[runtime]\nbranch = "{rt["branch"]}"\nend_level = "{rt.get("end_level", "L3")}"\nrun_name = "base"', ""]
    parts += [_sweep_hopping(rows)]
    return "\n".join(parts)


def gen_prb() -> None:
    materials, schemes = ["Cs", "K", "Rb"], ["RS"]
    zeta_by_mat = {"Cs": 110.0, "K": 120.0, "Rb": 110.0}
    u_mev = [2000.0, 3000.0, 4000.0, 6000.0]
    ratios = [i / 100 for i in range(21)]
    f2, f4, f6 = 12.98, 8.163, 5.878
    out = Path("data/prb/sweep")
    out.mkdir(parents=True, exist_ok=True)
    for mat in materials:
        for sch in schemes:
            rows = [
                (u, round(r * u, 6), f"prb/{sch}/{mat}/U{u / 1000:.1f}/jh{r:.2f}")
                for u in u_mev
                for r in ratios
            ]
            tbl = "\n".join(f"{u}   {jh}   {n}" for u, jh, n in rows)
            base = f'''schema_version = "fxe.run_input.v1"
standard_version = "2026-02"

[fsite]
n_ele = 1
F2_ratio = {f2}
F4_ratio = {f4}
F6_ratio = {f6}
zeta = {zeta_by_mat[mat]}
offset = 0.0
energy_reference = "zero"
U = 3000.0
Jh = 240.0

[model]
scheme = "{sch}"

[inputs]
hopping_file = "data/prb/hopping/prb_t_mu_{mat}.txt"
kramer_file = "data/prb/kramer/prb_gamma7_literature.txt"

[paths]
output_root = "{OUT}"

[runtime]
branch = "sopt"
end_level = "L3"
run_name = "base"

[sweep]
table = """
{len(rows)}
U   Jh   runtime.run_name
{tbl}
"""
'''
            (out / f"{mat}_{sch}.toml").write_text(base, encoding="utf-8")
    print(f"PRB: {len(materials) * len(schemes)} bases x {len(u_mev) * len(ratios)} cases")


def gen_nature() -> None:
    out = Path("data/nature/sweep")
    out.mkdir(parents=True, exist_ok=True)
    for sec_dir in sorted(Path("data/nature/hopping/fig3").iterdir()):
        sec = sec_dir.name
        hops = sorted(
            glob.glob(f"data/nature/hopping/fig3/{sec}/t_mu_ratio_*.dat"),
            key=lambda p: float(re.search(r"ratio_([0-9.]+)\.dat", p).group(1)),
        )
        tmpls = sorted(glob.glob(f"data/nature/input/fig3/{sec}/RS/*.toml")) or sorted(
            glob.glob(f"data/nature/input/table2/{sec}/RS/*.toml")
        )
        if not hops or not tmpls:
            print(f"nature {sec}: SKIP (hops={len(hops)}, tmpls={len(tmpls)})")
            continue
        m = re.match(r"f(\d+)_\w+_(g\d)", sec)
        flabel, gamma = f"f{int(m.group(1)):02d}", m.group(2)
        rows = [
            (h, f"nature/{flabel}/{gamma}/ratio{re.search(r'ratio_([0-9.]+)[.]dat', h).group(1)}")
            for h in hops
        ]
        base = _hopping_base(tmpls[0], hops, rows, f"data/nature/kramer/{sec}.txt")
        (out / f"{sec}.toml").write_text(base, encoding="utf-8")
        print(f"nature {sec}: {len(rows)} cases")


def gen_gau() -> None:
    out = Path("data/gau/sweep")
    out.mkdir(parents=True, exist_ok=True)
    hops = sorted(
        glob.glob("data/gau/hopping/t_fp_rho_m*.txt"),
        key=lambda p: re.search(r"rho_(m[0-9p]+)\.txt", p).group(1),
    )
    for gamma in ("G6", "G7"):
        tmpls = sorted(glob.glob(f"data/gau/input/{gamma}/ED/*.toml"))
        if not hops or not tmpls:
            print(f"gau {gamma}: SKIP")
            continue
        rows = [(h, f"gau/{gamma.lower()}/rho_{re.search(r'rho_(m[0-9p]+)', h).group(1)}") for h in hops]
        base = _hopping_base(tmpls[0], hops, rows, f"data/gau/kramer/gau_{gamma}.txt")
        (out / f"{gamma}_ED.toml").write_text(base, encoding="utf-8")
        print(f"gau {gamma}: {len(rows)} cases")


if __name__ == "__main__":
    gen_prb()
    gen_nature()
    gen_gau()
