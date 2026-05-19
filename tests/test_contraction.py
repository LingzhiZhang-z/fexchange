import numpy as np
import pytest

from fexchange.sopt.contraction import build_L3, build_L4, build_L4_fast
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


def _random_l2(seed: int = 1234):
    rng = np.random.default_rng(seed)
    n_u, n_v, n_j = 2, 3, 4
    M_A = rng.normal(size=(n_u, n_v, n_j, n_j)) + 1j * rng.normal(size=(n_u, n_v, n_j, n_j))
    M_B = rng.normal(size=(n_v, n_u, n_j, n_j)) + 1j * rng.normal(size=(n_v, n_u, n_j, n_j))
    return {
        "M_A": M_A.astype(np.complex128),
        "M_B": M_B.astype(np.complex128),
        "n_u": n_u,
        "n_v": n_v,
        "n_j": n_j,
    }


def _orthonormal_projector(n_j: int, n_k: int, seed: int = 5678):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_j, n_k)) + 1j * rng.normal(size=(n_j, n_k))
    q, _ = np.linalg.qr(z)
    return q[:, :n_k].astype(np.complex128)


def test_build_l4_fast_matches_l3_for_identity_projector():
    l2 = _random_l2()
    E_np1 = np.array([2.0, 3.5], dtype=float)
    E_nm1 = np.array([1.25, 2.25, 4.0], dtype=float)
    W = np.eye(l2["n_j"], dtype=np.complex128)

    l3 = build_L3(l2, E_np1, E_nm1, n_ele=3)
    fast = build_L4_fast(l2, E_np1, E_nm1, W, n_ele=3)

    np.testing.assert_allclose(fast["Heff_mu_abcd"], l3["h_pre_j_mu"], rtol=1e-12, atol=1e-12)


def test_build_l4_fast_matches_materialized_l4_for_projector():
    l2 = _random_l2()
    E_np1 = np.array([2.0, 3.5], dtype=float)
    E_nm1 = np.array([1.25, 2.25, 4.0], dtype=float)
    W = _orthonormal_projector(l2["n_j"], 2)

    materialized = build_L4(build_L3(l2, E_np1, E_nm1, n_ele=3), W)
    fast = build_L4_fast(l2, E_np1, E_nm1, W, n_ele=3)

    np.testing.assert_allclose(
        fast["Heff_mu_abcd"],
        materialized["Heff_mu_abcd"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_build_l4_fast_reports_near_zero_denominator_like_l3():
    with pytest.raises(NumError, match="Invalid denominator Delta_uv"):
        build_L4_fast(
            _l2_stub(),
            np.array([1.0], dtype=float),
            np.array([-1.0], dtype=float),
            np.ones((1, 1), dtype=np.complex128),
            n_ele=2,
        )
