import numpy as np
import pytest

from fexchange.sopt.contraction import build_L3
from fexchange.utils.errors import NumError


def _l2_stub():
    return {
        "M_A": np.ones((1, 1, 1, 1), dtype=np.complex128),
        "M_B": np.ones((1, 1, 1, 1), dtype=np.complex128),
        "n_j": 1,
    }


def test_build_l3_allows_negative_branch_energies_if_denominators_are_nonzero():
    result = build_L3(
        _l2_stub(),
        np.array([2.0], dtype=float),
        np.array([-0.5], dtype=float),
        n_ele=2,
    )

    assert result["h_pre_j_mu"].shape == (1, 1, 1, 1)
    assert np.isfinite(result["h_pre_j_mu"]).all()


def test_build_l3_reports_near_zero_denominator_even_with_negative_branch():
    with pytest.raises(NumError, match="Invalid denominator Delta_uv"):
        build_L3(
            _l2_stub(),
            np.array([1.0], dtype=float),
            np.array([-1.0], dtype=float),
            n_ele=2,
        )


def test_build_l3_rejects_non_finite_denominator_sum():
    with pytest.raises(NumError, match="Invalid denominator Delta_uv"):
        build_L3(
            _l2_stub(),
            np.array([np.nan], dtype=float),
            np.array([1.0], dtype=float),
            n_ele=2,
        )
