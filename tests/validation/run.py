from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.validation.checks import (  # noqa: E402
    VALIDATION_F2,
    VALIDATION_F4,
    VALIDATION_F6,
    VALIDATION_R42,
    VALIDATION_R62,
    build_context,
    compute_lsjm_section,
    compute_lsms_section,
    compute_numerical_section,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "validation"


def validate_one(n_ele: int) -> dict[str, Any]:
    ctx = build_context(n_ele)
    numerical = compute_numerical_section(ctx)
    lsms = compute_lsms_section(ctx)
    lsjm = compute_lsjm_section(ctx)

    checks = numerical["checks"] + lsms["checks"] + lsjm["checks"]
    passed = sum(1 for check in checks if (not check["reference_only"]) and check["passed"])
    failed = sum(1 for check in checks if (not check["reference_only"]) and (not check["passed"]))
    info = sum(1 for check in checks if check["reference_only"])

    return {
        "n_ele": n_ele,
        "dim": ctx.expected_dim,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "build_elapsed_s": ctx.build_elapsed_s,
        "internal_F2": float(ctx.lsms["physics"]["F2"]),
        "internal_F4": float(ctx.lsms["physics"]["F4"]),
        "internal_F6": float(ctx.lsms["physics"]["F6"]),
        "numerical": numerical,
        "lsms": lsms,
        "lsjm": lsjm,
        "checks": checks,
        "summary": {"passed": passed, "failed": failed, "info": info},
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "======================================================================",
        "  fexchange LSMS/LSJM Validation Report",
        f"  Configuration: f{report['n_ele']}  (n_ele={report['n_ele']}, dim={report['dim']})",
        f"  F2 : F4 : F6 = {VALIDATION_F2:.2f} : {VALIDATION_F4:.3f} : {VALIDATION_F6:.3f}",
        f"  r42 = {VALIDATION_R42:.6f},  r62 = {VALIDATION_R62:.6f}",
        f"  Timestamp: {report['timestamp']}",
        "======================================================================",
        "",
        "--- Build ---",
        "",
        f"  Build completed in {report['build_elapsed_s']:.1f}s",
        (
            "  Internal "
            f"F2={report['internal_F2']:.6f}, "
            f"F4={report['internal_F4']:.6f}, "
            f"F6={report['internal_F6']:.6f}"
        ),
        "",
        "--- Numerical Self-Consistency ---",
        "",
    ]
    lines.extend(_render_checks(report["numerical"]["checks"]))
    lines.extend(
        [
            "",
            "--- LSMS Checks ---",
            "",
            "  LSMS Term Summary",
            "",
        ]
    )
    lines.extend(
        _render_table(
            headers=["alpha", "L", "S", "n", "<L2>", "<S2>", "<H0>", "<H2>", "<H4>", "<H6>"],
            rows=[
                [
                    str(row["alpha"]),
                    str(row["L"]),
                    _fmt_quantum(row["S"]),
                    str(row["n"]),
                    _fmt_value(row["L2"]),
                    _fmt_value(row["S2"]),
                    _fmt_value(row["H0"]),
                    _fmt_value(row["H2"]),
                    _fmt_value(row["H4"]),
                    _fmt_value(row["H6"]),
                ]
                for row in report["lsms"]["term_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "  LSMS State Lz/Sz",
            "",
        ]
    )
    lines.extend(
        _render_table(
            headers=["alpha", "L", "S", "ML", "MS", "<Lz>", "<Sz>"],
            rows=[
                [
                    str(row["alpha"]),
                    str(row["L"]),
                    _fmt_quantum(row["S"]),
                    _fmt_quantum(row["ML"]),
                    _fmt_quantum(row["MS"]),
                    _fmt_value(row["Lz"]),
                    _fmt_value(row["Sz"]),
                ]
                for row in report["lsms"]["state_rows"]
            ],
        )
    )
    lines.extend([""])
    lines.extend(_render_checks(report["lsms"]["checks"]))
    lines.extend(
        [
            "",
            "--- LSJM Checks ---",
            "",
            "  LSJM Multiplet Summary",
            "",
        ]
    )
    lines.extend(
        _render_table(
            headers=[
                "alpha",
                "L",
                "S",
                "J",
                "n",
                "<L2>",
                "<S2>",
                "<J2>",
                "<O_soc>",
                "term max|Im diag|",
                "term max residual",
            ],
            rows=[
                [
                    str(row["alpha"]),
                    str(row["L"]),
                    _fmt_quantum(row["S"]),
                    _fmt_quantum(row["J"]),
                    str(row["n"]),
                    _fmt_value(row["L2"]),
                    _fmt_value(row["S2"]),
                    _fmt_value(row["J2"]),
                    _fmt_value(row["O_soc"]),
                    _fmt_exp(row["diag_imag_term"]),
                    _fmt_exp(row["residual_term"]),
                ]
                for row in report["lsjm"]["multiplet_rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "  LSJM State Jz",
            "",
        ]
    )
    lines.extend(
        _render_table(
            headers=["alpha", "L", "S", "J", "M", "<Jz>"],
            rows=[
                [
                    str(row["alpha"]),
                    str(row["L"]),
                    _fmt_quantum(row["S"]),
                    _fmt_quantum(row["J"]),
                    _fmt_quantum(row["M"]),
                    _fmt_value(row["Jz"]),
                ]
                for row in report["lsjm"]["state_rows"]
            ],
        )
    )
    lines.extend([""])
    lines.extend(_render_checks(report["lsjm"]["checks"]))
    global_soc = report["lsjm"]["global_soc"]
    lines.extend(
        [
            "",
            "  LSJM O_soc Global Reference",
            "",
            f"  max |Im diag| = {_fmt_exp(global_soc['diag_imag'])}",
            f"  max residual = {_fmt_exp(global_soc['residual'])}",
            "  diag preview = [" + ", ".join(_fmt_value(v) for v in global_soc["diag_preview"]) + "]",
            "",
            "--- Summary ---",
            "",
            (
                f"  {report['summary']['passed']} PASSED, "
                f"{report['summary']['failed']} FAILED, "
                f"{report['summary']['info']} INFO"
            ),
            f"  Elapsed (n_ele={report['n_ele']}): {report.get('elapsed_s', 0.0):.1f}s",
            "",
        ]
    )
    return "\n".join(lines)


def run_suite(n_values: Sequence[int], output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    import time
    output_dir.mkdir(parents=True, exist_ok=True)
    total_failed = 0
    for n_ele in n_values:
        t0 = time.perf_counter()
        report = validate_one(int(n_ele))
        elapsed = time.perf_counter() - t0
        report["elapsed_s"] = elapsed
        text = render_report(report)
        path = output_dir / f"f{n_ele}.txt"
        path.write_text(text)
        total_failed += int(report["summary"]["failed"])
        print(
            f"f{n_ele}: {report['summary']['passed']} PASSED, "
            f"{report['summary']['failed']} FAILED, "
            f"{report['summary']['info']} INFO  "
            f"({report['build_elapsed_s']:.1f}s) -> {path}"
        )
    return total_failed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LSMS/LSJM validation reports.")
    parser.add_argument("n_ele", nargs="+", type=int, help="electron counts to validate")
    args = parser.parse_args(argv)
    failed = run_suite(args.n_ele, DEFAULT_OUTPUT_DIR)
    return 1 if failed else 0


def _render_checks(checks: Sequence[dict[str, Any]]) -> list[str]:
    lines = []
    for check in checks:
        if check["reference_only"]:
            status = "INFO"
        else:
            status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  [{status}] {check['name']}: {check['detail']}")
    return lines


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    lines = [
        "  " + "  ".join(header.rjust(widths[idx]) for idx, header in enumerate(headers)),
        "  " + "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append("  " + "  ".join(cell.rjust(widths[idx]) for idx, cell in enumerate(row)))
    return lines


def _fmt_quantum(value: float) -> str:
    two = int(round(2.0 * float(value)))
    if two % 2 == 0:
        return str(two // 2)
    return f"{two}/2"


def _fmt_value(value: float) -> str:
    return f"{float(value):+.6f}"


def _fmt_exp(value: float) -> str:
    return f"{float(value):.2e}"


if __name__ == "__main__":
    raise SystemExit(main())
