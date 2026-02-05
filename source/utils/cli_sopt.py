"""CLI entry point for SOPT contraction from disk inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..api.io import load_energy_set, load_ground_manifold
from ..core.fock_basis import build_fock_basis
from ..core.orbitals import ORBITAL_ORDER_ID
from ..sopt.export import export_heff
from ..sopt.fermion_ops_cache import build_or_load_ops_cache
from ..sopt.heff import compute_heff
from ..sopt.projected_transitions_cache import build_or_load_projected_transitions_cache
from ..sopt.truncation import truncate_by_energy_window, truncate_by_nstates
from .params import PATH_KEYS, load_params_file

_REQUIRED_KEYS = {
    "n",
    "t_npy",
    "ground_npz",
    "lsjm_np1_npz",
    "e_np1_npz",
    "lsjm_nm1_npz",
    "e_nm1_npz",
    "cache_dir",
    "out_dir",
}


def _merge_args_with_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """Fill missing CLI args with values from a config dict."""
    for key, value in config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    for key in PATH_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if not isinstance(value, Path):
            value = Path(value)
        setattr(args, key, value)
    return args


def _validate_required(args: argparse.Namespace) -> None:
    """Ensure all required args are provided."""
    missing = [key for key in sorted(_REQUIRED_KEYS) if getattr(args, key, None) is None]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")


def _apply_truncation(
    energies: np.ndarray, n_keep: Optional[int], emax: Optional[float]
) -> Tuple[np.ndarray, str]:
    """Return kept indices and a truncation id for caching."""
    indices = np.arange(len(energies))
    trunc_id_parts: list[str] = []
    if emax is not None:
        indices, trunc_id = truncate_by_energy_window(energies, emax)
        trunc_id_parts.append(trunc_id)
    if n_keep is not None:
        if len(indices) == 0:
            return indices, "empty"
        # Re-apply n_keep on the filtered energies
        sub_energies = energies[indices]
        sub_indices, trunc_id = truncate_by_nstates(sub_energies, n_keep)
        indices = indices[sub_indices]
        trunc_id_parts.append(trunc_id)
    trunc_id = "all" if not trunc_id_parts else "+".join(trunc_id_parts)
    return indices, trunc_id


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for SOPT contraction."""
    parser = argparse.ArgumentParser(description="Run SOPT contraction from on-disk inputs")
    parser.add_argument("--config", type=Path, default=None, help="JSON parameter file")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--t-npy", type=Path, default=None)
    parser.add_argument("--ground-npz", type=Path, default=None)
    parser.add_argument("--lsjm-np1-npz", type=Path, default=None)
    parser.add_argument("--e-np1-npz", type=Path, default=None)
    parser.add_argument("--lsjm-nm1-npz", type=Path, default=None)
    parser.add_argument("--e-nm1-npz", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--np1-nstates", type=int, default=None)
    parser.add_argument("--np1-emax", type=float, default=None)
    parser.add_argument("--nm1-nstates", type=int, default=None)
    parser.add_argument("--nm1-emax", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the SOPT contraction pipeline from file-based inputs."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config is not None:
        config = load_params_file(args.config)
        unknown = [key for key in config.keys() if key not in vars(args)]
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
        args = _merge_args_with_config(args, config)
    _validate_required(args)

    t = np.load(args.t_npy)
    ground = load_ground_manifold(args.ground_npz)

    if ground.n != -1 and ground.n != args.n:
        raise ValueError(f"Ground manifold n={ground.n} does not match --n={args.n}")

    if args.n < 1:
        raise ValueError("--n must be >= 1 for nm1 sector")

    basis_n = build_fock_basis(args.n)
    basis_np1 = build_fock_basis(args.n + 1)
    basis_nm1 = build_fock_basis(args.n - 1)

    if ground.basis_id and ground.basis_id != basis_n.basis_id:
        raise ValueError("Ground manifold basis_id does not match expected Fock basis")

    ops_n = build_or_load_ops_cache(args.n, args.cache_dir, basis_n, basis_np1, ORBITAL_ORDER_ID)
    ops_nm1 = build_or_load_ops_cache(args.n - 1, args.cache_dir, basis_nm1, basis_n, ORBITAL_ORDER_ID)

    e_np1 = np.asarray(load_energy_set(args.e_np1_npz).E_total).ravel()
    e_nm1 = np.asarray(load_energy_set(args.e_nm1_npz).E_total).ravel()

    idx_np1, trunc_id_np1 = _apply_truncation(e_np1, args.np1_nstates, args.np1_emax)
    idx_nm1, trunc_id_nm1 = _apply_truncation(e_nm1, args.nm1_nstates, args.nm1_emax)

    cache_np1 = build_or_load_projected_transitions_cache(
        n=args.n,
        cache_dir=args.cache_dir,
        basis_id=basis_n.basis_id,
        orbital_order_id=ORBITAL_ORDER_ID,
        which="np1",
        lsjm_path=args.lsjm_np1_npz,
        E_mid=e_np1,
        truncation_indices=idx_np1 if len(idx_np1) != len(e_np1) else None,
        truncation_id=trunc_id_np1,
        D_create=ops_n.D_create,
    )

    cache_nm1 = build_or_load_projected_transitions_cache(
        n=args.n,
        cache_dir=args.cache_dir,
        basis_id=basis_n.basis_id,
        orbital_order_id=ORBITAL_ORDER_ID,
        which="nm1",
        lsjm_path=args.lsjm_nm1_npz,
        E_mid=e_nm1,
        truncation_indices=idx_nm1 if len(idx_nm1) != len(e_nm1) else None,
        truncation_id=trunc_id_nm1,
        D_create_nm1=ops_nm1.D_create,
    )

    Heff = compute_heff(t, ground.G_fock, cache_np1, cache_nm1, ground.E0)
    export_heff(args.out_dir, Heff, meta={"n": args.n})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
