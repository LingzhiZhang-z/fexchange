"""fopt: 4th-order ligand-mediated superexchange (L0..L3 + spin-1/2 mapping)."""

from fexchange.fopt.precompute import (
    N_ORB_F,
    N_ORB_P,
    compute_L0_f,
    compute_L0_p,
    build_L0,
    compute_L1_f,
    compute_L1_p,
    build_L1,
)
from fexchange.fopt.contraction import build_L2, build_L3, project_W
from fexchange.fopt.runner import (
    run_fopt,
    compute_p_energies_nsoc,
    compute_p_energies_soc,
)

__all__ = [
    "N_ORB_F",
    "N_ORB_P",
    "compute_L0_f",
    "compute_L0_p",
    "build_L0",
    "compute_L1_f",
    "compute_L1_p",
    "build_L1",
    "build_L2",
    "build_L3",
    "project_W",
    "run_fopt",
    "compute_p_energies_nsoc",
    "compute_p_energies_soc",
]
