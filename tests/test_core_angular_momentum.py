import numpy as np
import pytest

from fexchange.core.space_j import build_space_j_operator, normalize_J
from fexchange.utils.errors import PhysError


def test_normalize_j_snaps_near_half_integer():
    assert normalize_J(2.500000000001) == 2.5


def test_normalize_j_rejects_non_quantized_value():
    with pytest.raises(PhysError, match="not close to integer/half-integer"):
        normalize_J(2.3)


def test_build_j_matrices_include_cartesian_components():
    Jz, Jp, Jm, Jx, Jy = build_space_j_operator(0.5)
    assert Jz.shape == (2, 2)
    assert Jp.shape == (2, 2)
    assert Jm.shape == (2, 2)
    np.testing.assert_allclose(Jx, 0.5 * (Jp + Jm))
    np.testing.assert_allclose(Jy, -0.5j * (Jp - Jm))
