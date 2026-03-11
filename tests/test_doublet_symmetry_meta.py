import numpy as np

from fexchange.core.doublet_basis import build_oh_eg_doublet
from fexchange.core.symmetry_tables import MULTIPOLE_RULES
from fexchange.models.ground_doublets import (
    select_kramers_doublet,
    select_non_kramers_doublet,
)


def test_kramers_returns_irrep_and_multipole_metadata():
    evals = np.array([0.0, 0.0], dtype=float)
    evecs = np.eye(2, dtype=complex)

    out = select_kramers_doublet(0.5, evals, evecs, point_group="Oh")

    assert "irrep" in out
    assert "allowed_multipoles" in out
    assert "excited_irreps" in out
    assert out["irrep"].startswith("Gamma")
    assert out["allowed_multipoles"] == MULTIPOLE_RULES["Oh"][out["irrep"]]
    assert all(set(item) == {"index", "irrep"} for item in out["excited_irreps"])


def test_nonkramers_returns_irrep_and_multipole_metadata():
    J = 4.0
    evals = np.array([0.0, 0.0], dtype=float)
    evecs = build_oh_eg_doublet(J, [0.0, 1.0], [1.0]).astype(complex)

    out = select_non_kramers_doublet(
        J,
        evals,
        evecs,
        point_group="Oh",
        eps_mag_ab=10.0,
    )

    assert "irrep" in out
    assert "allowed_multipoles" in out
    assert "excited_irreps" in out
    assert out["irrep"].startswith("Gamma")
    assert out["allowed_multipoles"] == MULTIPOLE_RULES["Oh"][out["irrep"]]
    assert all(set(item) == {"index", "irrep"} for item in out["excited_irreps"])
