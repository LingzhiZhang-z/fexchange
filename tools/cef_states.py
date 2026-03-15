"""
晶体场本征态计算工具 - 给定晶体场参数和J，输出本征态及不可约表示分类。

Usage:
    # 标准输出
    python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001

    # 简洁模式（只输出汇总文件）
    python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001 \
        --summary-file summary.txt --silent

    # NPZ输出（主程序输入格式）
    python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001 \
        --format npz -o kramer_input.npz --kramer-name my_cef

    # C3v 对称性 (cos模式，默认)
    python cef_states.py --point-group C3v --J 4 --B40 0.01 --B60 0.0001 --B66 0.0005

    # C3v 对称性 (sin模式)
    python cef_states.py --point-group C3v --J 4 --B40 0.01 --B60 0.0001 --B66 0.0005 --mode-q3 sin

    # 从TOML配置
    python cef_states.py config.toml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path to import fexchange
sys.path.insert(0, str(Path(__file__).parent.parent))

from fexchange.hamiltonian.hcef import build_hcef_matrix_J
from fexchange.spectrum.classify import (
    classify_irreps,
    irrep_metadata,
    allowed_multipoles,
    irrep_dimension,
)
from fexchange.core.space_j import _fix_column_phases, build_time_reversal_operator
from fexchange.spectrum.multipole import analyze_multipole_carrying, format_multipole_summary


SUPPORTED_POINT_GROUPS = ["Oh", "C3v"]

# Irrep display names (Bethe -> Mulliken)
IRREP_NAMES: dict[str, str] = {
    "Gamma1": "Γ₁ (A₁)",
    "Gamma2": "Γ₂ (A₂)",
    "Gamma3": "Γ₃ (E)",
    "Gamma4": "Γ₄ (T₁)",
    "Gamma5": "Γ₅ (T₂)",
    "Gamma6": "Γ₆ (E'')",
    "Gamma7": "Γ₇ (E'')",
    "Gamma8": "Γ₈ (U')",
    "Gamma1+": "Γ₁⁺ (A₁g)",
    "Gamma2+": "Γ₂⁺ (A₂g)",
    "Gamma3+": "Γ₃⁺ (Eg)",
}


def _normalize_eigenvectors(evecs: np.ndarray) -> np.ndarray:
    """规范化本征态：L2归一化 + 固定全局相位。"""
    evecs = evecs / np.linalg.norm(evecs, axis=0, keepdims=True)
    return _fix_column_phases(evecs)


def _group_by_energy(eigenvalues: np.ndarray, tol: float = 1e-6) -> list[list[int]]:
    """按能量分组（考虑简并）。"""
    groups = []
    current_group = [0]
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i - 1]) < tol:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)
    return groups


def _format_eigenvector_compact(vec: np.ndarray, J: float, max_terms: int = 4) -> str:
    """紧凑格式化本征态。"""
    dim = int(round(2 * J + 1))
    terms = []
    for idx in range(dim):
        M = -J + idx
        coeff = vec[idx]
        if abs(coeff) < 0.01:
            continue
        if abs(coeff.imag) < 1e-10:
            coeff_str = f"{coeff.real:+.2f}"
        else:
            coeff_str = f"({coeff.real:+.2f}{coeff.imag:+.2f}i)"
        terms.append(f"{coeff_str}|{M:g}>")
        if len(terms) >= max_terms:
            terms.append("...")
            break
    return " ".join(terms) if terms else "0"


def compute_cef_states(
    point_group: str,
    J: float,
    B_params: dict[str, float],
    mode_q3: str = "cos",
    verbose: bool = True,
) -> dict[str, Any]:
    """计算晶体场本征态及不可约表示分类。"""
    point_group = point_group.strip()
    if point_group not in SUPPORTED_POINT_GROUPS:
        raise ValueError(
            f"Unsupported point group: {point_group!r}. "
            f"Supported: {', '.join(SUPPORTED_POINT_GROUPS)}"
        )

    H = build_hcef_matrix_J(J, B_params, symmetry=point_group, mode_q3=mode_q3)
    eigenvalues, evecs = np.linalg.eigh(H)
    evecs = _normalize_eigenvectors(evecs)
    irreps = classify_irreps(J, evecs, point_group=point_group)

    energy_groups = _group_by_energy(eigenvalues)

    level_info = []
    for grp in energy_groups:
        E = eigenvalues[grp[0]]
        deg = len(grp)
        level_irreps = [irreps[i] for i in grp]
        level_info.append({
            "energy": float(E),
            "degeneracy": deg,
            "state_indices": grp,
            "irreps": level_irreps,
        })

    dim = int(round(2 * J + 1))
    metadata = []
    for i in range(dim):
        meta = irrep_metadata(irreps[i], point_group=point_group, J=J)
        multipoles = allowed_multipoles(irreps[i], point_group=point_group)
        metadata.append({
            "index": i,
            "energy_meV": float(eigenvalues[i]),
            "irrep": irreps[i],
            "irrep_display": meta["irrep_display"],
            "irrep_dimension": irrep_dimension(irreps[i], point_group=point_group),
            "allowed_multipoles": multipoles,
            "eigenvector": evecs[:, i].copy(),
        })

    level_multipoles = []
    for grp in energy_groups:
        deg = len(grp)
        if deg not in (1, 2, 3):
            level_multipoles.append(None)
            continue
        Psi = evecs[:, grp]
        analysis = analyze_multipole_carrying(
            Psi,
            J,
            tol=1e-10,
            point_group=point_group,
        )
        level_multipoles.append(analysis)

    return {
        "point_group": point_group,
        "J": float(J),
        "eigenvalues": eigenvalues,
        "eigenvectors": evecs,
        "irreps": irreps,
        "energy_groups": energy_groups,
        "level_info": level_info,
        "level_multipoles": level_multipoles,
        "metadata": metadata,
        "B_params": B_params,
        "mode_q3": mode_q3,
    }


def _extract_ground_doublet(result: dict[str, Any]) -> dict[str, Any]:
    """提取最低双重态。"""
    eigenvalues = result["eigenvalues"]
    evecs = result["eigenvectors"]
    irreps = result["irreps"]
    J = result["J"]
    dim = int(round(2 * J + 1))

    energy_tol = 1e-6
    if abs(eigenvalues[0] - eigenvalues[1]) > energy_tol:
        raise ValueError(
            f"基态不是双重态: E0={eigenvalues[0]:.6f}, E1={eigenvalues[1]:.6f}"
        )

    ground_states = evecs[:, :2].copy()

    twoJ = int(round(2 * J))
    if twoJ % 2 == 1:  # Kramers双重态
        U_T = build_time_reversal_operator(J)
        k1 = ground_states[:, 0].copy()
        k2_tr = U_T @ k1.conj()
        overlap = np.abs(np.vdot(k2_tr, ground_states[:, 1]))
        if overlap < 0.99:
            k1, k2 = ground_states[:, 1].copy(), ground_states[:, 0].copy()
        else:
            k1, k2 = ground_states[:, 0].copy(), ground_states[:, 1].copy()

        pivot = int(np.argmax(np.abs(k1)))
        phase = np.angle(k1[pivot])
        k1 = k1 * np.exp(-1j * phase)
        k2 = U_T @ k1.conj()

        ground_states = np.column_stack([k1, k2])
        doublet_type = "kramers"
    else:
        doublet_type = "non_kramers"

    return {
        "W": ground_states,
        "ground_irrep": irreps[0],
        "ground_energy": float(eigenvalues[0]),
        "doublet_type": doublet_type,
        "n_j": dim,
    }


def export_for_fexchange(
    result: dict[str, Any],
    output_path: str,
    kramer_name: str,
    basis_id: str = "complex_spherical_j_v1",
    orbital_order_id: str = "f7_m-3..3_v1",
) -> None:
    """导出为主程序可直接使用的NPZ格式。"""
    doublet_info = _extract_ground_doublet(result)

    np.savez(
        output_path,
        W=doublet_info["W"],
        kramer_name=kramer_name,
        kramer_labels=np.array([0, 1]),
        ground_irrep=doublet_info["ground_irrep"],
        ground_energy=doublet_info["ground_energy"],
        doublet_type=doublet_info["doublet_type"],
        n_j=doublet_info["n_j"],
        schema_version="fxe.cef_states.v1",
        standard_version="2026-02",
        basis_id=basis_id,
        orbital_order_id=orbital_order_id,
        unit="meV",
        J=result["J"],
        point_group=result["point_group"],
    )
    print(f"Saved: {output_path}")


def print_results_llw_style(result: dict[str, Any]) -> None:
    """以LLW风格打印结果。"""
    pg = result["point_group"]
    J = result["J"]
    B_params = result["B_params"]
    dim = int(round(2 * J + 1))

    # 头部信息（类似 llw_diagram.py 的 === J = X ===）
    print(f"\n{'='*60}")
    print(f"CEF States Calculation")
    print(f"{'='*60}")
    print(f"  Point group: {pg}")
    print(f"  J = {J}  (2J+1 = {dim} states)")
    print(f"  B_params: {B_params}")
    print(f"{'='*60}")

    # 能级表（紧凑格式）
    print(f"\n{'Level':>6} {'Energy(meV)':>14} {'Deg':>5} {'Irrep':>12} {'Wavefunction (compact)'}")
    print("-" * 80)

    for lvl_idx, level in enumerate(result["level_info"]):
        E = level["energy"]
        deg = level["degeneracy"]
        irrep_set = set(level["irreps"])
        irrep_str = "/".join(sorted(irrep_set))[:10]

        # 显示第一个态的波函数
        first_state_idx = level["state_indices"][0]
        vec = result["eigenvectors"][:, first_state_idx]
        wf_str = _format_eigenvector_compact(vec, J)

        print(f"{lvl_idx:>6} {E:>14.6f} {deg:>5} {irrep_str:>12}  {wf_str}")

    # 基态特殊标注
    ground = result["level_info"][0]
    ground_irrep = ground["irreps"][0]
    multipoles = allowed_multipoles(ground_irrep, point_group=pg)

    print(f"\n{'='*60}")
    print(f"Ground State: {ground_irrep}  E = {ground['energy']:.6f} meV")
    print(f"{'='*60}")
    print(f"  Degeneracy: {ground['degeneracy']}-fold")
    print(f"  Allowed multipoles: {', '.join(multipoles)}")
    print(f"{'='*60}\n")


def print_multipole_analysis(result: dict[str, Any]) -> None:
    """Print multipole projection for each singlet, doublet, or triplet level."""
    level_multipoles = result.get("level_multipoles", [])
    if not level_multipoles:
        return

    print(f"\n{'='*60}")
    print("Multipole Projections")
    print(f"{'='*60}")

    for lvl_idx, (level, analysis) in enumerate(zip(result["level_info"], level_multipoles)):
        if analysis is None:
            continue
        deg = level["degeneracy"]
        E = level["energy"]
        irrep_str = "/".join(sorted(set(level["irreps"])))
        dim_label = {1: "singlet", 2: "doublet", 3: "triplet"}.get(deg, f"{deg}D")
        gauge = analysis.get("used_gauge", "none")
        print(f"\n--- Level {lvl_idx}: {irrep_str} ({dim_label})  E = {E:.6f} meV  gauge={gauge} ---")
        print(format_multipole_summary(analysis))

    print(f"\n{'='*60}\n")


def generate_summary_file(result: dict[str, Any], output_path: str) -> None:
    """生成汇总文件（CSV-like格式）。"""
    lines = []
    lines.append(f"# CEF States Summary")
    lines.append(f"# Point group: {result['point_group']}, J = {result['J']}")
    lines.append(f"# B_params: {result['B_params']}")
    lines.append("#")

    # 能级表
    lines.append(f"# {'Level':>6} {'Energy':>12} {'Deg':>4} {'Irrep':>10}")
    for lvl_idx, level in enumerate(result["level_info"]):
        E = level["energy"]
        deg = level["degeneracy"]
        irrep = "/".join(set(level["irreps"]))
        lines.append(f"# {lvl_idx:>6} {E:>12.4f} {deg:>4} {irrep:>10}")

    lines.append("#")
    lines.append("-" * 70)

    # 详细态表
    lines.append(f"{'Index':>6} {'Energy':>14} {'Irrep':>12} {'Multipoles'}")
    lines.append("-" * 70)

    for meta in result["metadata"]:
        idx = meta["index"]
        E = meta["energy_meV"]
        irrep = meta["irrep"]
        multipoles = ", ".join(meta["allowed_multipoles"][:3])
        if len(meta["allowed_multipoles"]) > 3:
            multipoles += ", ..."
        lines.append(f"{idx:>6} {E:>14.6f} {irrep:>12}  {multipoles}")

    lines.append("-" * 70)

    content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Saved: {output_path}")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="晶体场本征态计算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 标准输出
  python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001

  # 静默模式 + 汇总文件
  python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001 \\
      --summary-file summary.txt --silent

  # NPZ格式（主程序输入）
  python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001 \\
      --format npz -o kramer.npz --kramer-name cef_v1

  # C3v 对称性 (cos/sin模式)
  python cef_states.py --point-group C3v --J 4 --B40 0.01 --B60 0.0001 --B66 0.0005
  python cef_states.py --point-group C3v --J 4 --B40 0.01 --B63 0.001 --mode-q3 sin
        """,
    )

    parser.add_argument("config", nargs="?", help="TOML配置文件路径")

    parser.add_argument(
        "--point-group", "-pg", choices=SUPPORTED_POINT_GROUPS, help="晶体场点群"
    )
    parser.add_argument("--J", type=float, help="总角动量量子数（如 4, 2.5, 6）")

    # Oh 参数
    parser.add_argument("--B4", type=float, default=0.0, help="Oh: 4阶晶体场参数 (meV)")
    parser.add_argument("--B6", type=float, default=0.0, help="Oh: 6阶晶体场参数 (meV)")

    # C3v 参数
    parser.add_argument("--B20", type=float, default=0.0, help="C3v: 2阶参数 (meV)")
    parser.add_argument("--B40", type=float, default=0.0, help="C3v: 4阶参数 (meV)")
    parser.add_argument("--B60", type=float, default=0.0, help="C3v: 6阶参数 (meV)")
    parser.add_argument("--B66", type=float, default=0.0, help="C3v: B66参数 (meV)")
    parser.add_argument("--B43", type=float, default=0.0, help="C3v: B43参数 (meV)")
    parser.add_argument("--B63", type=float, default=0.0, help="C3v: B63参数 (meV)")

    parser.add_argument(
        "--mode-q3", choices=["cos", "sin"], default="cos", help="q=3 Stevens算符模式"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "npz"],
        default="text",
        help="输出格式 (默认: text)",
    )

    parser.add_argument("--output", "-o", help="主输出文件路径")
    parser.add_argument(
        "--summary-file", help="汇总文件路径（CSV-like格式）"
    )
    parser.add_argument("--kramer-name", help="Kramer doublet名称（用于NPZ）")
    parser.add_argument(
        "--silent", action="store_true", help="静默模式（不打印到stdout）"
    )

    return parser.parse_args()


def _load_toml_config(path: str) -> dict[str, Any]:
    """从TOML文件加载配置。"""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _collect_b_params(args: argparse.Namespace, point_group: str) -> dict[str, float]:
    """根据点群收集B参数。"""
    if point_group == "Oh":
        params = {"B4": args.B4, "B6": args.B6}
    elif point_group == "C3v":
        params = {
            "B20": args.B20,
            "B40": args.B40,
            "B60": args.B60,
            "B66": args.B66,
            "B43": args.B43,
            "B63": args.B63,
        }
    else:
        raise ValueError(f"Unknown point group: {point_group}")
    return {k: v for k, v in params.items() if v != 0.0}


def _to_json_serializable(obj: Any) -> Any:
    """将numpy数组转换为JSON可序列化的对象。"""
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return [[float(x.real), float(x.imag)] for x in obj.flatten()]
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.complexfloating):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    return obj


def main() -> int:
    """主入口函数。"""
    args = _parse_args()

    try:
        # 解析配置
        if args.config:
            cfg = _load_toml_config(args.config)
            point_group = cfg.get("point_group") or cfg.get("point-group")
            J = float(cfg.get("J"))
            B_params = cfg.get("B_params", {})
            if not B_params:
                keys = ["B4", "B6"] if point_group == "Oh" else ["B20", "B40", "B60", "B66", "B43", "B63"]
                B_params = {k: cfg[k] for k in keys if k in cfg}
            mode_q3 = cfg.get("mode_q3", "cos")
            kramer_name = cfg.get("kramer_name", cfg.get("kramer-name", "cef_custom"))
        else:
            if args.point_group is None or args.J is None:
                raise ValueError("--point-group 和 --J 是必需参数")
            point_group = args.point_group
            J = args.J
            B_params = _collect_b_params(args, point_group)
            mode_q3 = args.mode_q3
            kramer_name = args.kramer_name or "cef_custom"

        # 执行计算
        result = compute_cef_states(
            point_group=point_group,
            J=J,
            B_params=B_params,
            mode_q3=mode_q3,
            verbose=False,
        )

        # 生成汇总文件
        if args.summary_file:
            generate_summary_file(result, args.summary_file)

        # 根据格式输出
        if args.format == "text":
            if not args.silent:
                print_results_llw_style(result)
                print_multipole_analysis(result)
            if args.output:
                # 将输出重定向到文件
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    print_results_llw_style(result)
                    print_multipole_analysis(result)
                with open(args.output, "w") as fout:
                    fout.write(f.getvalue())
                if not args.silent:
                    print(f"Saved: {args.output}")

        elif args.format == "json":
            if not args.output:
                raise ValueError("JSON格式需要指定 --output 输出文件路径")
            json_result = {
                "point_group": result["point_group"],
                "J": result["J"],
                "eigenvalues": _to_json_serializable(result["eigenvalues"]),
                "eigenvectors": _to_json_serializable(result["eigenvectors"]),
                "irreps": result["irreps"],
                "level_info": result["level_info"],
            }
            with open(args.output, "w") as f:
                json.dump(json_result, f, indent=2)
            if not args.silent:
                print(f"Saved: {args.output}")

        elif args.format == "npz":
            if not args.output:
                raise ValueError("NPZ格式需要指定 --output 输出文件路径")
            export_for_fexchange(result, args.output, kramer_name)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
