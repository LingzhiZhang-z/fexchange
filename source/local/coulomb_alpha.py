from __future__ import annotations

from typing import Tuple

from ..api.types import CoulombScale, CoulombShape, EnergySet, StateSet

scheme_id = "RS_no_LS_mixing_v1"


def build_coulomb_alpha(
    *args,
    **kwargs,
) -> Tuple[StateSet, EnergySet]:
    """Compute Coulomb interaction in the alpha basis (placeholder).

    Expected to return a StateSet and EnergySet in the alpha basis.
    """
    raise NotImplementedError("Coulomb alpha basis construction is not implemented")
