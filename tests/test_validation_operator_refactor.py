from __future__ import annotations

import numpy as np

from tests.validation.checks import (
    compute_lsjm_checks_and_table,
    compute_lsms_checks_and_table,
)
from tests.validation.run import Reporter, _render_payload


def _toy_lsms_inputs():
    labels = [
        {"alpha": 0, "L": 1, "twoS": 1, "ML": -1, "MS": -0.5},
        {"alpha": 0, "L": 1, "twoS": 1, "ML": 1, "MS": 0.5},
    ]
    lsms = {
        "n_ele": 3,
        "V_fock": np.eye(2, dtype=complex),
        "labels": labels,
        "coef_F0": np.array([10.0, 10.0]),
        "coef_F2": np.array([1.0, 1.0]),
        "coef_F4": np.array([0.5, 0.5]),
        "coef_F6": np.array([0.25, 0.25]),
        "hint_rank_matrix_map": {
            0: np.diag([10.0, 10.0]),
            2: np.diag([1.0, 1.0]),
            4: np.diag([0.5, 0.5]),
            6: np.diag([0.25, 0.25]),
        },
    }
    angular_ops = {
        "L2": np.diag([2.0, 2.0]),
        "S2": np.diag([0.75, 0.75]),
        "Lz": np.diag([-1.0, 1.0]),
        "Sz": np.diag([-0.5, 0.5]),
    }
    return lsms, angular_ops


def _toy_lsjm_inputs():
    labels = [
        {"alpha": 0, "L": 1, "twoS": 1, "twoJ": 1, "twoM": -1},
        {"alpha": 0, "L": 1, "twoS": 1, "twoJ": 3, "twoM": 1},
        {"alpha": 1, "L": 2, "twoS": 1, "twoJ": 3, "twoM": 1},
    ]

    lsjm = {
        "n_ele": 3,
        "V_fock": np.eye(3, dtype=complex),
        "labels": labels,
        "coef_F0": np.array([10.0, 10.0, 20.0]),
        "coef_F2": np.array([1.0, 1.0, 2.0]),
        "coef_F4": np.array([0.5, 0.5, 1.5]),
        "coef_F6": np.array([0.25, 0.25, 1.25]),
        "coef_zeta": np.array([-1.0, 0.5, 1.5]),
        "hint_rank_matrix_map": {
            0: np.diag([10.0, 10.0, 20.0]),
            2: np.diag([1.0, 1.0, 2.0]),
            4: np.diag([0.5, 0.5, 1.5]),
            6: np.diag([0.25, 0.25, 1.25]),
        },
    }

    angular_ops = {
        "L2": np.diag([2.0, 2.0, 6.0]),
        "S2": np.diag([0.75, 0.75, 0.75]),
        "J2": np.diag([0.75, 3.75, 3.75]),
        "Jz": np.diag([-0.5, 0.5, 0.5]),
    }

    hsoc_unit = np.array(
        [
            [-1.0, 0.0, 0.2],
            [0.0, 0.5, 0.0],
            [0.2, 0.0, 1.5],
        ],
        dtype=complex,
    )

    return lsjm, angular_ops, hsoc_unit


def test_compute_lsms_checks_moves_m_output_to_separate_rows():
    lsms, angular_ops = _toy_lsms_inputs()

    checks, table_rows, m_rows, analytic_text = compute_lsms_checks_and_table(
        lsms,
        lsms["n_ele"],
        angular_ops=angular_ops,
    )

    assert checks
    assert analytic_text is None
    assert len(table_rows) == 1
    assert "lzsz_tuples" not in table_rows[0]
    assert m_rows == [
        {"alpha": 0, "L": 1, "twoS": 1, "ML": -1, "MS": -0.5, "Lz_num": -1.0, "Sz_num": -0.5},
        {"alpha": 0, "L": 1, "twoS": 1, "ML": 1, "MS": 0.5, "Lz_num": 1.0, "Sz_num": 0.5},
    ]


def test_compute_lsjm_checks_moves_m_output_to_separate_rows():
    lsjm, angular_ops, hsoc_unit = _toy_lsjm_inputs()

    checks, multiplet_rows, m_rows, soc_term_rows, analytic_text, soc_reference_text = compute_lsjm_checks_and_table(
        lsjm,
        lsjm["n_ele"],
        angular_ops=angular_ops,
        hsoc_unit=hsoc_unit,
    )

    assert checks
    assert analytic_text is None
    assert soc_reference_text is not None
    assert "max_offdiag" in soc_reference_text
    assert "2.0e-01" in soc_reference_text
    assert "jz_tuples" not in multiplet_rows[0]
    assert m_rows == [
        {"alpha": 0, "L": 1, "twoS": 1, "twoJ": 1, "twoM": -1, "Jz_num": -0.5},
        {"alpha": 0, "L": 1, "twoS": 1, "twoJ": 3, "twoM": 1, "Jz_num": 0.5},
        {"alpha": 1, "L": 2, "twoS": 1, "twoJ": 3, "twoM": 1, "Jz_num": 0.5},
    ]

    offdiag_checks = {check["name"]: check for check in checks if "O_soc" in check["name"] and "off-diagonal residual" in check["name"]}
    assert "LSJM O_soc off-diagonal residual" not in offdiag_checks
    assert offdiag_checks["LSJM O_soc term alpha=0 L=1 twoS=1 off-diagonal residual"]["passed"] is True
    assert offdiag_checks["LSJM O_soc term alpha=1 L=2 twoS=1 off-diagonal residual"]["passed"] is True
    assert len(soc_term_rows) == 2


def test_render_payload_supports_standalone_m_tables_and_reference_blocks():
    reporter = Reporter()

    _render_payload(
        reporter,
        "lsms_operator_table",
        [
            {
                "alpha": 0,
                "L": 1,
                "twoS": 1,
                "n_states": 2,
                "H0_num": 1.0,
                "H0_ana": 1.0,
                "H0_spr": 0.0,
                "H2_num": 0.5,
                "H2_ana": 0.5,
                "H2_spr": 0.0,
                "H4_num": 0.25,
                "H4_ana": 0.25,
                "H4_spr": 0.0,
                "H6_num": 0.125,
                "H6_ana": 0.125,
                "H6_spr": 0.0,
                "L2_num": 2.0,
                "L2_ana": 2.0,
                "L2_spr": 0.0,
                "S2_num": 0.75,
                "S2_ana": 0.75,
                "S2_spr": 0.0,
            }
        ],
    )
    _render_payload(
        reporter,
        "lsms_m_table",
        [
            {"alpha": 0, "L": 1, "twoS": 1, "ML": -1, "MS": -0.5, "Lz_num": -1.0, "Sz_num": -0.5},
            {"alpha": 0, "L": 1, "twoS": 1, "ML": 1, "MS": 0.5, "Lz_num": 1.0, "Sz_num": 0.5},
        ],
    )
    _render_payload(
        reporter,
        "lsjm_operator_table",
        [
            {
                "alpha": 0,
                "L": 1,
                "twoS": 1,
                "twoJ": 1,
                "n_states": 1,
                "L2_num": 2.0,
                "L2_ana": 2.0,
                "L2_spr": 0.0,
                "S2_num": 0.75,
                "S2_ana": 0.75,
                "S2_spr": 0.0,
                "J2_num": 0.75,
                "J2_ana": 0.75,
                "J2_spr": 0.0,
                "zeta_num": -1.0,
                "zeta_ana": -1.0,
                "zeta_spr": 0.0,
                "H0_num": 10.0,
                "H0_ana": 10.0,
                "H0_spr": 0.0,
                "H2_num": 1.0,
                "H2_ana": 1.0,
                "H2_spr": 0.0,
                "H4_num": 0.5,
                "H4_ana": 0.5,
                "H4_spr": 0.0,
                "H6_num": 0.25,
                "H6_ana": 0.25,
                "H6_spr": 0.0,
            }
        ],
    )
    _render_payload(
        reporter,
        "lsjm_m_table",
        [
            {"alpha": 0, "L": 1, "twoS": 1, "twoJ": 1, "twoM": -1, "Jz_num": -0.5},
        ],
    )
    _render_payload(
        reporter,
        "lsjm_soc_term_table",
        [
            {
                "alpha": 0,
                "L": 1,
                "twoS": 1,
                "n_states": 2,
                "soc_max_imag": 0.0,
                "soc_max_offdiag": 0.0,
                "zeta_blocks": [
                    {"twoJ": 1, "n_states": 1, "zeta_num": -1.0, "zeta_ana": -1.0, "zeta_spr": 0.0},
                    {"twoJ": 3, "n_states": 1, "zeta_num": 0.5, "zeta_ana": 0.5, "zeta_spr": 0.0},
                ],
            }
        ],
    )
    _render_payload(
        reporter,
        "lsjm_soc_global_reference",
        "Reference O_soc (global LSJM basis):\n  max_imag     0.0\n  max_offdiag  2.0e-01",
    )

    rendered = "\n".join(reporter.lines)
    assert "Lz/Sz:" not in rendered
    assert "Jz:" not in rendered
    assert "ML" in rendered
    assert "MS" in rendered
    assert "Jz_num" in rendered
    assert "Reference O_soc (global LSJM basis):" in rendered
