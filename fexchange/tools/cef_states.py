"""
晶体场本征态计算工具 - 给定晶体场参数和J，输出本征态及不可约表示分类。

Usage:
    # 标准输出
    python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001

    # Projection输出（主程序 kramer_file 输入格式，stevens 模式），并在stdout打印 gx/gy/gz
    python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001 \
        --format projector -o projector.txt --kramer-name my_cef

    # C3v 对称性 (cos模式，默认)
    python cef_states.py --point-group C3v --J 4 --B40 0.01 --B60 0.0001 --B66 0.0005

    # C3v 对称性 (sin模式)
    python cef_states.py --point-group C3v --J 4 --B40 0.01 --B60 0.0001 --B66 0.0005 --mode-q3 sin

    # 从TOML配置
    python cef_states.py config.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fexchange.hamiltonian.hcef import build_hcef_matrix_J
from fexchange.spectrum.classify import build_projectors, allowed_multipoles
from fexchange.core.space_j import _fix_column_phases
from fexchange.spectrum.ground import select_kramers_doublet, select_non_kramers_doublet
from fexchange.spectrum.multipole import analyze_multipole_carrying, gauge_fix_subspace, format_multipole_summary


SUPPORTED_POINT_GROUPS = ["Oh", "C3v"]


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

    energy_groups = _group_by_energy(eigenvalues)

    # Gauge-fix each degenerate subspace in-place (before irrep classification)
    gauge_labels: list[str] = []
    for grp in energy_groups:
        deg = len(grp)
        if deg in (1, 2, 3):
            fixed, label = gauge_fix_subspace(evecs[:, grp], J, point_group=point_group)
            evecs[:, grp] = fixed
            gauge_labels.append(label)
        else:
            gauge_labels.append("none")

    # Classify on gauge-fixed basis (subspace-level, not per-state)
    projectors = build_projectors(J, point_group=point_group)

    level_info = []
    level_multipoles = []
    for i, grp in enumerate(energy_groups):
        deg = len(grp)
        Psi = evecs[:, grp]
        # Sum ||P_Gamma @ psi_j||^2 over all states in the subspace
        irrep_weights: dict[str, float] = {}
        for name, P in projectors.items():
            w = sum(float(np.linalg.norm(P @ Psi[:, j])) ** 2 for j in range(deg))
            if w > 1e-6:
                irrep_weights[name] = w
        level_irreps = sorted(irrep_weights, key=lambda k: -irrep_weights[k])
        level_info.append({
            "energy": float(eigenvalues[grp[0]]),
            "degeneracy": deg,
            "state_indices": grp,
            "irreps": level_irreps,
            "gauge": gauge_labels[i],
        })
        if deg in (1, 2, 3):
            analysis = analyze_multipole_carrying(evecs[:, grp], J)
            level_multipoles.append(analysis)
        else:
            level_multipoles.append(None)

    return {
        "point_group": point_group,
        "J": float(J),
        "eigenvalues": eigenvalues,
        "eigenvectors": evecs,
        "energy_groups": energy_groups,
        "level_info": level_info,
        "level_multipoles": level_multipoles,
        "B_params": B_params,
        "mode_q3": mode_q3,
    }


def _extract_ground_doublet(result: dict[str, Any]) -> dict[str, Any]:
    """提取最低双重态并返回主代码 projector 所需的 W。"""
    eigenvalues = result["eigenvalues"]
    evecs = result["eigenvectors"]
    J = result["J"]
    dim = int(round(2 * J + 1))

    energy_tol = 1e-6
    if abs(eigenvalues[0] - eigenvalues[1]) > energy_tol:
        raise ValueError(
            f"基态不是双重态: E0={eigenvalues[0]:.6f}, E1={eigenvalues[1]:.6f}"
        )

    twoJ = int(round(2 * J))
    doublet_type = "kramers" if twoJ % 2 == 1 else "non_kramers"
    ground_irreps = result["level_info"][0]["irreps"]

    if doublet_type == "kramers":
        ground = select_kramers_doublet(
            J,
            eigenvalues,
            evecs,
            point_group=result["point_group"],
        )
        W = np.asarray(ground["kramer_vectors"], dtype=np.complex128)
        g_tensor = np.asarray(ground["g_tensor"], dtype=np.complex128)
        g_components = _axis_g_components(g_tensor)
        gauge_meta = ground.get("gauge_meta", {})
    else:
        ground = select_non_kramers_doublet(
            J,
            eigenvalues,
            evecs,
            point_group=result["point_group"],
        )
        W = np.asarray(ground["doublet_vectors"], dtype=np.complex128)
        g_tensor = np.zeros((3, 3), dtype=np.complex128)
        g_components = {"gx": 0.0, "gy": 0.0, "gz": 0.0}
        gauge_meta = ground.get("gauge_meta", {})

    return {
        "W": W,
        "ground_irrep": "/".join(ground_irreps),
        "ground_energy": float(eigenvalues[0]),
        "doublet_type": doublet_type,
        "n_j": dim,
        "g_tensor": g_tensor,
        "g_components": g_components,
        "gauge_meta": gauge_meta,
    }


def export_for_fexchange(
    result: dict[str, Any],
    output_path: str,
    kramer_name: str,
    basis_id: str = "complex_spherical_j_v1",
    orbital_order_id: str = "f7_m-3..3_v1",
) -> None:
    """导出为主程序 ``inputs.kramer_file``（stevens 模式）可直接读取的 projector 格式。"""
    doublet_info = _extract_ground_doublet(result)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_projector_txt(
        target,
        doublet_info,
        result=result,
        kramer_name=kramer_name,
        basis_id=basis_id,
        orbital_order_id=orbital_order_id,
    )
    print_g_components(doublet_info["g_components"])


def _axis_g_components(g_tensor: np.ndarray) -> dict[str, float]:
    """Return axis components from a canonical-gauge 3x3 g tensor."""
    g = np.asarray(g_tensor, dtype=np.complex128)
    return {
        "gx": float(g[0, 0].real),
        "gy": float(g[1, 1].real),
        "gz": float(g[2, 2].real),
    }


def print_g_components(g_components: dict[str, float]) -> None:
    """Print only the axis g components needed after projector generation."""
    print(f"gx {g_components['gx']:.12e}")
    print(f"gy {g_components['gy']:.12e}")
    print(f"gz {g_components['gz']:.12e}")


def _write_projector_txt(
    path: Path,
    doublet_info: dict[str, Any],
    *,
    result: dict[str, Any],
    kramer_name: str,
    basis_id: str,
    orbital_order_id: str,
) -> None:
    """Write multi-block projector text accepted by pipeline._load_stevens."""
    lines = [
        "# schema_version fxe.cef_projector.v1",
        "# standard_version 2026-02",
        f"# kramer_name {kramer_name}",
        f"# point_group {result['point_group']}",
        f"# J {float(result['J']):.12e}",
        f"# basis_id {basis_id}",
        f"# orbital_order_id {orbital_order_id}",
        f"# ground_irrep {doublet_info['ground_irrep']}",
        f"# doublet_type {doublet_info['doublet_type']}",
        "",
    ]
    W = np.asarray(doublet_info["W"], dtype=np.complex128)
    for idx in range(W.shape[1]):
        lines.append(f"[W_state_{idx}]")
        lines.extend(f"{val.real:.12e} {val.imag:.12e}" for val in W[:, idx])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
    print(f"\n{'Level':>6} {'Energy(meV)':>14} {'Deg':>5} {'Irrep':>18} {'Wavefunction (compact)'}")
    print("-" * 90)

    for lvl_idx, level in enumerate(result["level_info"]):
        E = level["energy"]
        deg = level["degeneracy"]
        irrep_str = "/".join(level["irreps"])

        # 显示第一个态的波函数
        first_state_idx = level["state_indices"][0]
        vec = result["eigenvectors"][:, first_state_idx]
        wf_str = _format_eigenvector_compact(vec, J)

        print(f"{lvl_idx:>6} {E:>14.6f} {deg:>5} {irrep_str:>18}  {wf_str}")

    # 基态特殊标注
    ground = result["level_info"][0]
    ground_irrep_str = "/".join(ground["irreps"])
    multipole_set: set[str] = set()
    for irr in ground["irreps"]:
        multipole_set.update(allowed_multipoles(irr, point_group=pg))
    multipoles = sorted(multipole_set)

    print(f"\n{'='*60}")
    print(f"Ground State: {ground_irrep_str}  E = {ground['energy']:.6f} meV")
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
        gauge = level.get("gauge", "none")
        print(f"\n--- Level {lvl_idx}: {irrep_str} ({dim_label})  E = {E:.6f} meV  gauge={gauge} ---")
        print(format_multipole_summary(analysis))

    print(f"\n{'='*60}\n")


def print_ground_projector_summary(result: dict[str, Any]) -> None:
    """Print only the ground-doublet axis g components."""
    try:
        doublet_info = _extract_ground_doublet(result)
    except Exception as exc:
        print(f"Ground projector unavailable: {exc}")
        return

    print_g_components(doublet_info["g_components"])



def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="晶体场本征态计算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 标准输出
  python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001

  # Projector文本格式（主程序 kramer_file 输入，stevens 模式），stdout打印 gx/gy/gz
  python cef_states.py --point-group Oh --J 4 --B4 0.01 --B6 0.0001 \\
      --format projector -o projector.txt --kramer-name cef_v1

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
        choices=["text", "projector"],
        default="text",
        help="输出格式: text/projector (默认: text)",
    )

    parser.add_argument("--output", "-o", help="主输出文件路径")
    parser.add_argument("--kramer-name", help="Kramer/projector名称")
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


def _resolve_config(args: argparse.Namespace) -> tuple[str, float, dict[str, float], str, str]:
    """从命令行参数或TOML文件解析配置，返回 (point_group, J, B_params, mode_q3, kramer_name)。"""
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
    return point_group, J, B_params, mode_q3, kramer_name


def _output_text(result: dict[str, Any], args: argparse.Namespace) -> None:
    """文本格式输出。"""
    if not args.silent:
        print_results_llw_style(result)
        print_multipole_analysis(result)
        print_ground_projector_summary(result)
    if args.output:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_results_llw_style(result)
            print_multipole_analysis(result)
            print_ground_projector_summary(result)
        with open(args.output, "w") as fout:
            fout.write(buf.getvalue())
        if not args.silent:
            print(f"Saved: {args.output}")


def main() -> None:
    """主入口函数。"""
    args = _parse_args()
    if args.format == "projector" and not args.output:
        raise SystemExit(f"--output/-o is required when --format {args.format}")

    point_group, J, B_params, mode_q3, kramer_name = _resolve_config(args)

    result = compute_cef_states(
        point_group=point_group, J=J, B_params=B_params,
        mode_q3=mode_q3,
    )

    if args.format == "text":
        _output_text(result, args)
    elif args.format == "projector":
        export_for_fexchange(result, args.output, kramer_name)


if __name__ == "__main__":
    main()
