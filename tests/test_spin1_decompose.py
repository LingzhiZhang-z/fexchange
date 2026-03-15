"""Tests for spin1_decompose."""

import numpy as np
import pytest

from fexchange.core.space_j import spin1_decompose
from fexchange.utils.constants import (
    DTYPE_COMPLEX,
    Q_3Z2R2,
    Q_X2Y2,
    Q_XY,
    Q_YZ,
    Q_ZX,
    S1_X,
    S1_Y,
    S1_Z,
)

I3 = np.eye(3, dtype=DTYPE_COMPLEX)


def test_identity_decomposition():
    result = spin1_decompose(I3)
    assert abs(result["s0"] - 1.0) < 1e-12
    for key in ("sx", "sy", "sz", "qx2y2", "qxy", "qzx", "qyz", "q3z2r2"):
        assert abs(result[key]) < 1e-12


def test_sx_decomposition():
    result = spin1_decompose(S1_X)
    assert abs(result["sx"] - 1.0) < 1e-12
    for key in ("s0", "sy", "sz", "qx2y2", "qxy", "qzx", "qyz", "q3z2r2"):
        assert abs(result[key]) < 1e-12


def test_sz_decomposition():
    result = spin1_decompose(S1_Z)
    assert abs(result["sz"] - 1.0) < 1e-12
    for key in ("s0", "sx", "sy", "qx2y2", "qxy", "qzx", "qyz", "q3z2r2"):
        assert abs(result[key]) < 1e-12


def test_quadrupole_decomposition():
    result = spin1_decompose(Q_X2Y2)
    assert abs(result["qx2y2"] - 1.0) < 1e-12
    for key in ("s0", "sx", "sy", "sz", "qxy", "qzx", "qyz", "q3z2r2"):
        assert abs(result[key]) < 1e-12


def test_roundtrip():
    """Reconstruct M from decomposition coefficients."""
    M = 0.3 * I3 + 0.5 * S1_X - 0.2 * S1_Z + 0.7 * Q_XY + 0.1 * Q_3Z2R2
    coeffs = spin1_decompose(M)
    bases = [I3, S1_X, S1_Y, S1_Z, Q_X2Y2, Q_XY, Q_ZX, Q_YZ, Q_3Z2R2]
    keys = ["s0", "sx", "sy", "sz", "qx2y2", "qxy", "qzx", "qyz", "q3z2r2"]
    reconstructed = sum(coeffs[key] * basis for key, basis in zip(keys, bases))
    assert np.allclose(M, reconstructed, atol=1e-12)


def test_wrong_shape_raises():
    with pytest.raises(Exception):
        spin1_decompose(np.eye(2, dtype=DTYPE_COMPLEX))
