from __future__ import annotations

from typing import Tuple

from ..api.types import EnergySet, SOCParam, StateSet

scheme_id = "RS_no_LS_mixing_v1"


def build_lsjm_basis(*args, **kwargs) -> Tuple[StateSet, EnergySet]:
    """Construct LSJM basis with SOC coupling (placeholder).

    Expected to return the LSJM eigenvectors (StateSet) and energies (EnergySet).
    """
    raise NotImplementedError("LSJM basis construction is not implemented")
