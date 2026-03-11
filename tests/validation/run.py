#!/usr/bin/env python
"""
LSMS/LSJM validation script.

Usage:
    python tests/validation/run.py 2
    python tests/validation/run.py 1 2 13
    python tests/validation/run.py all
    python tests/validation/run.py 2 --modules=structural,lsms_operators,lsjm_operators
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from tests.validation.checks import (
    DEFAULT_MODULES,
    F2_RATIO,
    F4_RATIO,
    F6_RATIO,
    MODULE_REGISTRY,
    NUMERICAL_SECTION,
    R42,
    R62,
    TERM_NAMES,
    ValidationContext,
)

_NUMERICAL_CHECK_ORDER = {
    "LSMS dimension": 0,
    "LSJM dimension": 1,
    "LSMS orthonormality": 2,
    "LSJM orthonormality": 3,
    "Term multiplicities": 4,
}


# ===================================================================
# Output formatting
# ===================================================================

class Reporter:
    """Collects output lines and writes to stdout + file."""

    def __init__(self):
        self.lines: list[str] = []
        self.n_pass = 0
        self.n_fail = 0

    def line(self, text: str = ""):
        self.lines.append(text)
        print(text)

    def header(self, n_ele: int):
        dim = comb(14, n_ele)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.line("=" * 70)
        self.line("  fexchange LSMS/LSJM Validation Report")
        self.line(f"  Configuration: f{n_ele}  (n_ele={n_ele}, dim={dim})")
        self.line(f"  F2 : F4 : F6 = {F2_RATIO} : {F4_RATIO} : {F6_RATIO}")
        self.line(f"  r42 = {R42:.6f},  r62 = {R62:.6f}")
        self.line(f"  Timestamp: {ts}")
        self.line("=" * 70)

    def section(self, title: str):
        self.line()
        self.line(f"--- {title} ---")
        self.line()

    def check_results(self, results: list[dict]):
        for r in results:
            if r["passed"] is None:
                self.line(f"  [INFO] {r['name']}: {r['detail']}")
            elif r["passed"]:
                self.line(f"  [PASS] {r['name']}: {r['detail']}")
                self.n_pass += 1
            else:
                self.line(f"  [FAIL] {r['name']}: {r['detail']}")
                self.n_fail += 1

    def term_table(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        self.line(
            f"  {'Term':<6} {'a':>2} {'L':>2} {'S':>5} {'States':>7} "
            f"{'coef_F2':>11} {'coef_F4':>11} {'coef_F6':>11}"
        )
        self.line(f"  {'-'*6} {'-'*2} {'-'*2} {'-'*5} {'-'*7} {'-'*11} {'-'*11} {'-'*11}")
        for row in rows:
            name = TERM_NAMES.get((row["L"], row["twoS"]), f"L={row['L']}")
            s_str = _fhi(row["twoS"])
            self.line(
                f"  {name:<6} {row['alpha']:>2} {row['L']:>2} "
                f"{s_str:>5} {row['n_states']:>7} "
                f"{row['coef_F2']:>+11.6f} {row['coef_F4']:>+11.6f} "
                f"{row['coef_F6']:>+11.6f}"
            )

    def multiplet_table(self, rows: list[dict]):
        self.line(
            f"  {'Term':<6} {'a':>2} {'J':>5} {'States':>7} "
            f"{'coef_zeta':>11} {'E_coulomb':>13}"
        )
        self.line(f"  {'-'*6} {'-'*2} {'-'*5} {'-'*7} {'-'*11} {'-'*13}")
        for row in rows:
            name = TERM_NAMES.get((row["L"], row["twoS"]), f"L={row['L']}")
            J_str = f"{row['twoJ']}/2" if row["twoJ"] % 2 else f"{row['twoJ'] // 2}"
            self.line(
                f"  {name:<6} {row['alpha']:>2} {J_str:>5} "
                f"{row['n_states']:>7} "
                f"{row['coef_zeta']:>+11.6f} "
                f"{row['E_coulomb']:>+13.6f}"
            )

    def lsms_operator_table(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        def _triple(num: float, ana: float, spr: float) -> str:
            return f"{num:+8.4f}|{ana:+8.4f}|{spr:.1e}"

        l2_label = f"L\N{SUPERSCRIPT TWO}"
        s2_label = f"S\N{SUPERSCRIPT TWO}"
        self.line(
            f"  {'Term':<6} {'a':>2} {'L':>2} {'S':>5} {'n':>3} | "
            f"{'H0(num|ana|spr)':>28} {'H2(num|ana|spr)':>28} {'H4(num|ana|spr)':>28} {'H6(num|ana|spr)':>28} | "
            f"{(l2_label + '(num|ana|spr)'):>28} {(s2_label + '(num|ana|spr)'):>28}"
        )
        for row in rows:
            name = TERM_NAMES.get((row["L"], row["twoS"]), f"L={row['L']}")
            self.line(
                f"  {name:<6} {row['alpha']:>2} {row['L']:>2} {_fhi(row['twoS']):>5} {row['n_states']:>3} | "
                f"{_triple(row['H0_num'], row['H0_ana'], row['H0_spr']):>28} "
                f"{_triple(row['H2_num'], row['H2_ana'], row['H2_spr']):>28} "
                f"{_triple(row['H4_num'], row['H4_ana'], row['H4_spr']):>28} "
                f"{_triple(row['H6_num'], row['H6_ana'], row['H6_spr']):>28} | "
                f"{_triple(row['L2_num'], row['L2_ana'], row['L2_spr']):>28} "
                f"{_triple(row['S2_num'], row['S2_ana'], row['S2_spr']):>28}"
            )

    def lsjm_operator_table(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        def _triple(num: float, ana: float, spr: float) -> str:
            return f"{num:+8.4f}|{ana:+8.4f}|{spr:.1e}"

        l2_label = f"L\N{SUPERSCRIPT TWO}"
        s2_label = f"S\N{SUPERSCRIPT TWO}"
        j2_label = f"J\N{SUPERSCRIPT TWO}"
        zeta_label = "\N{GREEK SMALL LETTER ZETA}"
        self.line(
            f"  {'Term':<6} {'a':>2} {'J':>5} {'n':>3} | "
            f"{(l2_label + '(n|a|s)'):>24} {(s2_label + '(n|a|s)'):>24} {(j2_label + '(n|a|s)'):>24} {(zeta_label + '(n|a|s)'):>24} | "
            f"{'H0(n|a|s)':>24} {'H2(n|a|s)':>24} {'H4(n|a|s)':>24} {'H6(n|a|s)':>24}"
        )
        for row in rows:
            name = TERM_NAMES.get((row["L"], row["twoS"]), f"L={row['L']}")
            self.line(
                f"  {name:<6} {row['alpha']:>2} {_fhi(row['twoJ']):>5} {row['n_states']:>3} | "
                f"{_triple(row['L2_num'], row['L2_ana'], row['L2_spr']):>24} "
                f"{_triple(row['S2_num'], row['S2_ana'], row['S2_spr']):>24} "
                f"{_triple(row['J2_num'], row['J2_ana'], row['J2_spr']):>24} "
                f"{_triple(row['zeta_num'], row['zeta_ana'], row['zeta_spr']):>24} | "
                f"{_triple(row['H0_num'], row['H0_ana'], row['H0_spr']):>24} "
                f"{_triple(row['H2_num'], row['H2_ana'], row['H2_spr']):>24} "
                f"{_triple(row['H4_num'], row['H4_ana'], row['H4_spr']):>24} "
                f"{_triple(row['H6_num'], row['H6_ana'], row['H6_spr']):>24}"
            )

    def lsms_m_table(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        self.line(
            f"  {'alpha':>5} {'L':>2} {'S':>5} {'ML':>4} {'MS':>5} {'Lz_num':>10} {'Sz_num':>10}"
        )
        self.line(f"  {'-'*5} {'-'*2} {'-'*5} {'-'*4} {'-'*5} {'-'*10} {'-'*10}")
        for row in rows:
            self.line(
                f"  {row['alpha']:>5} {row['L']:>2} {_fhi(row['twoS']):>5} "
                f"{row['ML']:>4} {row['MS']:>5g} {row['Lz_num']:>+10.6f} {row['Sz_num']:>+10.6f}"
            )

    def lsjm_m_table(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        self.line(
            f"  {'alpha':>5} {'L':>2} {'S':>5} {'J':>5} {'M':>5} {'Jz_num':>10}"
        )
        self.line(f"  {'-'*5} {'-'*2} {'-'*5} {'-'*5} {'-'*5} {'-'*10}")
        for row in rows:
            self.line(
                f"  {row['alpha']:>5} {row['L']:>2} {_fhi(row['twoS']):>5} "
                f"{_fhi(row['twoJ']):>5} {_fhi(row['twoM']):>5} {row['Jz_num']:>+10.6f}"
            )

    def lsjm_soc_term_table(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        self.line(
            f"  {'Term':<6} {'a':>2} {'L':>2} {'S':>5} {'n':>3} | "
            f"{'O_soc(max_imag|max_offdiag)':>28}"
        )
        for row in rows:
            name = TERM_NAMES.get((row["L"], row["twoS"]), f"L={row['L']}")
            self.line(
                f"  {name:<6} {row['alpha']:>2} {row['L']:>2} {_fhi(row['twoS']):>5} {row['n_states']:>3} | "
                f"{row['soc_max_imag']:.1e}|{row['soc_max_offdiag']:.1e}"
            )
            blocks = "  ".join(
                f"({_fhi(block['twoJ'])},{block['n_states']},{block['zeta_num']:+.4f},{block['zeta_ana']:+.4f},{block['zeta_spr']:.1e})"
                for block in row["zeta_blocks"]
            )
            self.line(f"    J blocks: {blocks}")

    def angular_table_lsms(self, rows: list[dict]):
        def _fhi(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        l2_label = "<L\N{SUPERSCRIPT TWO}>"
        s2_label = "<S\N{SUPERSCRIPT TWO}>"
        self.line(
            f"  {'alpha':>5} {'L':>2} {'S':>5} {'ML':>4} {'MS':>4} "
            f"{l2_label:>10} {s2_label:>10} {'<Lz>':>10} {'<Sz>':>10}"
        )
        self.line(f"  {'-'*5} {'-'*2} {'-'*5} {'-'*4} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for row in rows:
            s_str = _fhi(row["twoS"])
            self.line(
                f"  {row['alpha']:>5} {row['L']:>2} {s_str:>5} "
                f"{row['ML']:>4} {row['MS']:>4} "
                f"{row['exp_L2']:>10.6f} {row['exp_S2']:>10.6f} "
                f"{row['exp_Lz']:>10.6f} {row['exp_Sz']:>10.6f}"
            )

    def angular_table_lsjm(self, rows: list[dict]):
        def _fmt_half_int(two_x: int) -> str:
            return f"{two_x}/2" if two_x % 2 else f"{two_x // 2}"

        l2_label = "<L\N{SUPERSCRIPT TWO}>"
        s2_label = "<S\N{SUPERSCRIPT TWO}>"
        j2_label = "<J\N{SUPERSCRIPT TWO}>"
        self.line(
            f"  {'alpha':>5} {'L':>2} {'S':>5} {'J':>5} {'M':>5} "
            f"{l2_label:>10} {s2_label:>10} {j2_label:>10} {'<Jz>':>10}"
        )
        self.line(f"  {'-'*5} {'-'*2} {'-'*5} {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for row in rows:
            s_str = _fmt_half_int(row["twoS"])
            j_str = _fmt_half_int(row["twoJ"])
            m_str = _fmt_half_int(row["twoM"])
            self.line(
                f"  {row['alpha']:>5} {row['L']:>2} {s_str:>5} "
                f"{j_str:>5} {m_str:>5} "
                f"{row['exp_L2']:>10.6f} {row['exp_S2']:>10.6f} "
                f"{row['exp_J2']:>10.6f} {row['exp_Jz']:>10.6f}"
            )

    def ref0_table(self, results: list[dict]):
        self.line(
            f"  {'L':>2} {'S':>2} {'J':>2} {'M':>3}   "
            f"{'|<ref|fex>|^2':>14} {'E_our':>13} {'E_ref':>13} {'dE':>10}"
        )
        self.line(f"  {'-'*2} {'-'*2} {'-'*2} {'-'*3}   {'-'*14} {'-'*13} {'-'*13} {'-'*10}")
        for r in results:
            if not r["matched"]:
                self.line(f"  {r['L']:>2} {r['S']:>2} {r['J']:>2} {r['M']:>3}     NOT MATCHED")
                continue
            self.line(
                f"  {r['L']:>2} {r['S']:>2} {r['J']:>2} {r['M']:>3}   "
                f"{r['overlap_sq']:>14.10f} "
                f"{r['E_our']:>+13.6f} {r['E_ref']:>+13.6f} "
                f"{r['dE']:>10.2e}"
            )

    def summary(self, elapsed: float, out_path: str):
        self.line()
        self.line("=" * 70)
        total = self.n_pass + self.n_fail
        self.line(f"  Summary: {total} checks, {self.n_pass} PASSED, {self.n_fail} FAILED")
        self.line(f"  Elapsed: {elapsed:.1f}s")
        self.line(f"  Output saved to: {out_path}")
        self.line("=" * 70)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def _sorted_numerical_checks(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda r: _NUMERICAL_CHECK_ORDER.get(r["name"], len(_NUMERICAL_CHECK_ORDER)))


def _render_payload(rpt: Reporter, ctype: str, data: Any):
    if ctype == "checks":
        rpt.check_results(data)
    elif ctype == "term_table":
        rpt.term_table(data)
    elif ctype == "multiplet_table":
        rpt.multiplet_table(data)
    elif ctype == "angular_lsms":
        rpt.angular_table_lsms(data)
    elif ctype == "angular_lsjm":
        rpt.angular_table_lsjm(data)
    elif ctype == "lsms_operator_table":
        rpt.lsms_operator_table(data)
    elif ctype == "lsms_m_table":
        rpt.lsms_m_table(data)
    elif ctype == "lsjm_operator_table":
        rpt.lsjm_operator_table(data)
    elif ctype == "lsjm_m_table":
        rpt.lsjm_m_table(data)
    elif ctype == "lsjm_soc_term_table":
        rpt.lsjm_soc_term_table(data)
    elif ctype == "analytic_fk_list":
        rpt.line(data)
    elif ctype == "analytic_soc_list":
        rpt.line(data)
    elif ctype == "lsjm_soc_global_reference":
        rpt.line(data)
    elif ctype == "ref0_table":
        rpt.ref0_table(data)
    elif ctype == "text":
        rpt.line(data)
    else:
        raise ValueError(f"Unknown content type: {ctype}")


def _emit_section_group(rpt: Reporter, title: str, group: list[tuple[str, str, Any]]):
    rpt.section(title)
    if title == NUMERICAL_SECTION and all(ctype == "checks" for _, ctype, _ in group):
        results: list[dict] = []
        for _, _, data in group:
            results.extend(data)
        rpt.check_results(_sorted_numerical_checks(results))
        return

    for _, ctype, data in group:
        _render_payload(rpt, ctype, data)


# ===================================================================
# Main validation for one n
# ===================================================================

def validate(n_ele: int, modules: list[str] | None = None):
    rpt = Reporter()
    rpt.header(n_ele)

    selected = modules if modules is not None else DEFAULT_MODULES
    unknown = [name for name in selected if name not in MODULE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown module(s): {', '.join(unknown)}")

    rpt.section("Build")
    t0 = time.time()
    ctx = ValidationContext(n_ele)
    t_build = time.time() - t0
    physics = ctx.lsms["physics"]
    rpt.line(f"  Build completed in {t_build:.1f}s")
    rpt.line(
        f"  Internal F2={float(physics['F2']):.6f}, "
        f"F4={float(physics['F4']):.6f}, "
        f"F6={float(physics['F6']):.6f}"
    )

    payloads: list[tuple[str, str, object]] = []
    for mod_name in selected:
        payloads.extend(MODULE_REGISTRY[mod_name](ctx))

    idx = 0
    while idx < len(payloads):
        title = payloads[idx][0]
        group: list[tuple[str, str, object]] = []
        while idx < len(payloads) and payloads[idx][0] == title:
            group.append(payloads[idx])
            idx += 1
        _emit_section_group(rpt, title, group)

    out_path = str(_project_root / "outputs" / "validation" / f"f{n_ele}.txt")
    elapsed = time.time() - t0
    rpt.summary(elapsed, out_path)
    rpt.save(out_path)
    return rpt.n_fail


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LSMS/LSJM validation")
    parser.add_argument("n", nargs="*", help="n_ele values (1-13) or 'all'")
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated module names to run (default: all)",
    )
    args = parser.parse_args()

    if not args.n:
        parser.print_help()
        sys.exit(1)

    if args.n[0] == "all":
        n_list = list(range(1, 14))
    else:
        n_list = [int(x) for x in args.n]
        for n in n_list:
            if not 1 <= n <= 13:
                print(f"Error: n must be 1..13, got {n}")
                sys.exit(1)

    modules = None
    if args.modules:
        modules = [name.strip() for name in args.modules.split(",") if name.strip()]
        unknown = [name for name in modules if name not in MODULE_REGISTRY]
        if unknown:
            print(f"Error: unknown module(s): {', '.join(unknown)}")
            print(f"Available modules: {', '.join(DEFAULT_MODULES)}")
            sys.exit(1)

    total_fail = 0
    for n in n_list:
        total_fail += validate(n, modules=modules)

    if len(n_list) > 1:
        print()
        print(f"=== All done: {len(n_list)} configurations, {total_fail} total failures ===")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
