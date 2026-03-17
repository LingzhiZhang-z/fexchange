import numpy as np
from fexchange.spectrum.energy import compute_intermediate_energies


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


def _f13_ground_j0_labels():
    labels = []
    for two_m in range(-7, 8, 2):
        labels.append(
            {
                "alpha": 0,
                "L": 3,
                "S": 0.5,
                "twoS": 1,
                "J": 3.5,
                "twoJ": 7,
                "M": two_m / 2.0,
                "twoM": two_m,
            }
        )
    return labels


def _f13_ground_payload():
    return {
        **_lsjm_payload([78.0] * 8),
        "labels": _f13_ground_j0_labels(),
        "n_ele": 13,
        "V_fock": np.eye(8, dtype=np.complex128),
    }


def test_compute_intermediate_energies_references_branch_energies_to_ground_soc0_energy():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.0},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_14": _lsjm_payload([91.0]),
        "lsjm_13": _f13_ground_payload(),
        "lsjm_12": _lsjm_payload([66.0]),
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=13)

    np.testing.assert_allclose(e_np1, np.array([13.0]))
    np.testing.assert_allclose(e_nm1, np.array([-12.0]))


def test_compute_intermediate_energies_allows_negative_branch_energies():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.0},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_14": _lsjm_payload([80.0]),
        "lsjm_13": _f13_ground_payload(),
        "lsjm_12": _lsjm_payload([-3.5]),
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=13)

    np.testing.assert_allclose(e_np1, np.array([2.0]))
    np.testing.assert_allclose(e_nm1, np.array([-81.5]))


def test_compute_intermediate_energies_does_not_gate_non_finite_values():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.0},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_14": _lsjm_payload([np.nan]),
        "lsjm_13": _f13_ground_payload(),
        "lsjm_12": _lsjm_payload([79.0]),
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=13)
    assert np.isnan(e_np1[0])
    np.testing.assert_allclose(e_nm1, np.array([1.0]))


def test_compute_intermediate_energies_uses_coulomb_only_fn_ground_reference():
    cfg = {
        "sopt": {"U": 1.0, "zeta": 0.5},
        "physics": {"F2": 0.0, "F4": 0.0, "F6": 0.0},
    }
    state = {
        "lsjm_14": {
            **_lsjm_payload([91.0]),
            "coef_zeta": np.array([0.25], dtype=float),
        },
        "lsjm_13": {
            **_f13_ground_payload(),
            "coef_zeta": np.array([-1.5] * 8, dtype=float),
        },
        "lsjm_12": {
            **_lsjm_payload([66.0]),
            "coef_zeta": np.array([0.75], dtype=float),
        },
    }

    e_np1, e_nm1 = compute_intermediate_energies(cfg, state, n_ele=13)

    np.testing.assert_allclose(e_np1, np.array([13.125]))
    np.testing.assert_allclose(e_nm1, np.array([-11.625]))
