import numpy as np
import pytest

from fexchange.core.stevens import build_multipole_operators


@pytest.mark.parametrize("J", [4.0, 3.5])
def test_dipole_matrices_hermitian(J):
    ops = build_multipole_operators(J, "magnetic_dipole")
    assert set(ops) == {"Jx", "Jy", "Jz"}
    for mat in ops.values():
        assert np.allclose(mat, mat.conj().T)


@pytest.mark.parametrize("J", [4.0, 3.5])
def test_quadrupole_operators_hermitian(J):
    ops = build_multipole_operators(J, "electric_quadrupole")
    assert set(ops) == {"O20", "O21c", "O21s", "O22c", "O22s"}
    for mat in ops.values():
        assert np.allclose(mat, mat.conj().T)


@pytest.mark.parametrize("J", [4.0, 3.5])
def test_octupole_operators_hermitian(J):
    ops = build_multipole_operators(J, "magnetic_octupole")
    assert set(ops) == {"O30", "O31c", "O31s", "O32c", "O32s", "O33c", "O33s"}
    for mat in ops.values():
        assert np.allclose(mat, mat.conj().T)


def test_dipole_trace_zero():
    ops = build_multipole_operators(4.0, "magnetic_dipole")
    for mat in ops.values():
        assert abs(np.trace(mat)) < 1e-12


@pytest.mark.parametrize("kind", ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"])
def test_operator_shapes(kind):
    J = 3.5
    dim = int(2 * J + 1)
    ops = build_multipole_operators(J, kind)
    for mat in ops.values():
        assert mat.shape == (dim, dim)


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        build_multipole_operators(4.0, "invalid")
