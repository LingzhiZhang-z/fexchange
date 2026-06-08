#!/usr/bin/env python3
"""Compute equal nm1/np1 offsets for fixed-Ueff zero-reference scans.

This helper assumes the physical 4f LS-coupling regime where Hund ground terms
are fixed by the usual positive Slater ratios and Jh dominates the SOC scale.
It does not rebuild LSJM states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fexchange.utils.constants import RE_DEFAULTS_BY_N_ELE  # noqa: E402


# Coefficients for the Hund ground LSJ state:
# E0 = c2*F2 + c4*F4 + c6*F6 + cz*zeta, with F0/U omitted.
GROUND_COEFFS: dict[int, tuple[str, float, float, float, float]] = {
    0: ("^1S_0", 0.0, 0.0, 0.0, 0.0),
    1: ("^2F_5/2", 0.0, 0.0, 0.0, -2.0),
    2: ("^3H_4", -0.1111111111, -0.04683195592, -0.001765910857, -3.0),
    3: ("^4I_9/2", -0.2888888889, -0.1294765840, -0.03002048457, -3.5),
    4: ("^5I_4", -0.4222222222, -0.2203856749, -0.1465706011, -3.5),
    5: ("^6H_5/2", -0.5111111111, -0.3195592287, -0.3514162605, -3.0),
    6: ("^7F_0", -0.6666666667, -0.4545454545, -0.5827505828, -2.0),
    7: ("^8S_7/2", -0.9333333333, -0.6363636364, -0.8158508159, 0.0),
    8: ("^7F_6", -0.9333333333, -0.6363636364, -0.8158508159, -1.5),
    9: ("^6H_15/2", -1.0444444440, -0.6831955923, -0.8176167267, -2.5),
    10: ("^5I_8", -1.2222222220, -0.7658402204, -0.8458713004, -3.0),
    11: ("^4I_15/2", -1.3555555560, -0.8567493113, -0.9624214170, -3.0),
    12: ("^3H_6", -1.4444444440, -0.9559228650, -1.1672670760, -2.5),
    13: ("^2F_7/2", -1.6000000000, -1.0909090910, -1.3986013990, -1.5),
    14: ("^1S_0", -1.8666666670, -1.2727272730, -1.6317016320, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ele", type=int, required=True, choices=range(1, 14), help="Center f electron count.")
    parser.add_argument("--ueff", type=float, required=True, help="Target pair charge gap after equal offsets.")
    parser.add_argument("--jh", type=float, help="Use this Jh for both nm1 and np1 unless branch overrides are given.")
    parser.add_argument("--jh-nm1", type=float, help="Jh for f^(n-1). Overrides --jh.")
    parser.add_argument("--jh-np1", type=float, help="Jh for f^(n+1). Overrides --jh.")
    parser.add_argument(
        "--ratio-nm1",
        type=float,
        nargs=3,
        metavar=("F2_RATIO", "F4_RATIO", "F6_RATIO"),
        help="Slater ratio triplet for f^(n-1). Defaults to built-in ratios for that sector.",
    )
    parser.add_argument(
        "--ratio-np1",
        type=float,
        nargs=3,
        metavar=("F2_RATIO", "F4_RATIO", "F6_RATIO"),
        help="Slater ratio triplet for f^(n+1). Defaults to built-in ratios for that sector.",
    )
    parser.add_argument("--zeta-nm1", type=float, help="SOC strength for f^(n-1). Defaults to built-in zeta.")
    parser.add_argument("--zeta-np1", type=float, help="SOC strength for f^(n+1). Defaults to built-in zeta.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def default_ratio(n_ele: int) -> tuple[float, float, float]:
    if n_ele == 0:
        return (1.0, 0.0, 0.0)
    key = 13 if n_ele == 14 else n_ele
    row = RE_DEFAULTS_BY_N_ELE[key]
    return (float(row["F2_per_Jh"]), float(row["F4_per_Jh"]), float(row["F6_per_Jh"]))


def default_zeta(n_ele: int) -> float:
    if n_ele in (0, 14):
        return 0.0
    return float(RE_DEFAULTS_BY_N_ELE[n_ele]["zeta"])


def slater_from_ratio_jh(ratio: tuple[float, float, float], jh: float) -> tuple[float, float, float]:
    f2r, f4r, f6r = ratio
    if f2r == 0.0:
        raise ValueError("F2 ratio must not be zero")
    r42 = f4r / f2r
    r62 = f6r / f2r
    denom = 286.0 + 195.0 * r42 + 250.0 * r62
    if denom == 0.0:
        raise ValueError("Jh denominator is zero for the given ratios")
    if jh == 0.0:
        return (0.0, 0.0, 0.0)
    F2 = 6435.0 * jh / denom
    return (F2, r42 * F2, r62 * F2)


def ground_energy(n_ele: int, *, ratio: tuple[float, float, float], jh: float, zeta: float) -> dict[str, Any]:
    term, c2, c4, c6, cz = GROUND_COEFFS[n_ele]
    F2, F4, F6 = slater_from_ratio_jh(ratio, jh)
    energy = c2 * F2 + c4 * F4 + c6 * F6 + cz * zeta
    return {
        "n_ele": n_ele,
        "term": term,
        "ratio": ratio,
        "Jh": jh,
        "zeta": zeta,
        "F2": F2,
        "F4": F4,
        "F6": F6,
        "energy": energy,
    }


def branch_jh(args: argparse.Namespace, side: str) -> float:
    value = getattr(args, f"jh_{side}")
    if value is not None:
        return float(value)
    if args.jh is not None:
        return float(args.jh)
    raise ValueError(f"missing Jh for {side}: provide --jh or --jh-{side}")


def branch_ratio(args: argparse.Namespace, side: str, n_ele: int) -> tuple[float, float, float]:
    value = getattr(args, f"ratio_{side}")
    if value is None:
        return default_ratio(n_ele)
    return (float(value[0]), float(value[1]), float(value[2]))


def branch_zeta(args: argparse.Namespace, side: str, n_ele: int) -> float:
    value = getattr(args, f"zeta_{side}")
    return default_zeta(n_ele) if value is None else float(value)


def main() -> int:
    args = parse_args()
    n_nm1 = int(args.n_ele) - 1
    n_np1 = int(args.n_ele) + 1

    nm1 = ground_energy(
        n_nm1,
        ratio=branch_ratio(args, "nm1", n_nm1),
        jh=branch_jh(args, "nm1"),
        zeta=branch_zeta(args, "nm1", n_nm1),
    )
    np1 = ground_energy(
        n_np1,
        ratio=branch_ratio(args, "np1", n_np1),
        jh=branch_jh(args, "np1"),
        zeta=branch_zeta(args, "np1", n_np1),
    )

    raw_pair = float(nm1["energy"]) + float(np1["energy"])
    offset = 0.5 * (float(args.ueff) - raw_pair)
    check = raw_pair + 2.0 * offset

    result = {
        "center_n_ele": int(args.n_ele),
        "target_ueff": float(args.ueff),
        "energy_reference": "zero",
        "nm1": nm1,
        "np1": np1,
        "raw_pair": raw_pair,
        "offset_nm1": offset,
        "offset_np1": offset,
        "check_ueff": check,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print("energy_reference = zero")
    print(f"center_n_ele = {args.n_ele}")
    print(f"target_ueff = {args.ueff:.12g}")
    for side in ("nm1", "np1"):
        data = result[side]
        ratio = data["ratio"]
        print(
            f"{side}: n={data['n_ele']} term={data['term']} "
            f"Jh={data['Jh']:.12g} zeta={data['zeta']:.12g} "
            f"ratio=({ratio[0]:.12g}, {ratio[1]:.12g}, {ratio[2]:.12g}) "
            f"E_raw={data['energy']:.12g}"
        )
    print(f"raw_pair = {raw_pair:.12g}")
    print(f"offset_nm1 = {offset:.12g}")
    print(f"offset_np1 = {offset:.12g}")
    print(f"check_ueff = {check:.12g}")
    print()
    print("# TOML side-branch snippet")
    print("[fsite]")
    print('energy_reference = "zero"')
    print("U = 0.0")
    print()
    for section, data in (("fsite_nm1", nm1), ("fsite_np1", np1)):
        r2, r4, r6 = data["ratio"]
        print(f"[{section}]")
        print("U = 0.0")
        print(f"Jh = {data['Jh']:.12g}")
        print(f"zeta = {data['zeta']:.12g}")
        print(f"F2_ratio = {r2:.12g}")
        print(f"F4_ratio = {r4:.12g}")
        print(f"F6_ratio = {r6:.12g}")
        print(f"offset = {offset:.12g}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
