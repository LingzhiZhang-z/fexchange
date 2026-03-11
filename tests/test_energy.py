import numpy as np
from fexchange.models.energy import compute_intermediate_energies


def _lsjm_payload(coef_f0):
    arr = np.asarray(coef_f0, dtype=float)
    zeros = np.zeros_like(arr)
    return {
        "coef_F0": arr,
        "coef_F2": zeros,
        "coef_F4": zeros,
        "coef_F6": zeros,
        "coef_zeta": zeros,
    }


def test_compute_intermediate_energies_uses_fixed_e0_reference_without_extra_shift():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.0},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_2": _lsjm_payload([5.0, 7.0]),
        "lsjm_3": _lsjm_payload([11.0, 13.0]),
        "lsjm_1": _lsjm_payload([9.0, 10.0]),
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=2)

    np.testing.assert_allclose(e_np1, np.array([11.0, 13.0]))
    np.testing.assert_allclose(e_nm1, np.array([9.0, 10.0]))


def test_compute_intermediate_energies_allows_negative_branch_energies():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.0},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_2": _lsjm_payload([0.0]),
        "lsjm_3": _lsjm_payload([2.0]),
        "lsjm_1": _lsjm_payload([-3.5]),
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=2)

    np.testing.assert_allclose(e_np1, np.array([2.0]))
    np.testing.assert_allclose(e_nm1, np.array([-3.5]))


def test_compute_intermediate_energies_does_not_gate_non_finite_values():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.0},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_2": _lsjm_payload([0.0]),
        "lsjm_3": _lsjm_payload([np.nan]),
        "lsjm_1": _lsjm_payload([1.0]),
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=2)
    assert np.isnan(e_np1[0])
    np.testing.assert_allclose(e_nm1, np.array([1.0]))
