from __future__ import annotations

from typing import Tuple

from ..api.types import EnergySet, StateSet

scheme_id = "RS_no_LS_mixing_v1"


def build_lsms_basis(*args, **kwargs) -> Tuple[StateSet, EnergySet]:
    """Construct the LSMS basis and associated energies.

    This is a placeholder for the single-site solver. It should return:
    - StateSet: eigenvectors in the Fock basis
    - EnergySet: eigenvalues and decomposition
    """
    raise NotImplementedError("LSMS basis construction is not implemented in this package")
