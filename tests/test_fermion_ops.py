"""Tests for core/fermion_ops.py (01-02)."""

import pytest
import numpy as np

from fexchange.core.fock import N_ORB, enumerate_dets, dim_sector
from fexchange.core.fermion import (
    apply_cdag, apply_c,
    cdag_matrix, c_matrix,
    one_body_operator_matrix,
)


class TestPrimitiveActions:
    def test_cdag_on_empty(self):
        """c^dag_0 |0> = |1>"""
        result = apply_cdag(0, 0)
        assert result is not None
        sign, det = result
        assert sign == 1
        assert det == 1

    def test_cdag_on_occupied(self):
        """c^dag_0 on orbital 0 occupied -> None"""
        assert apply_cdag(1, 0) is None

    def test_c_on_occupied(self):
        """c_0 |...0 occupied...> removes orbital 0"""
        result = apply_c(1, 0)
        assert result is not None
        sign, det = result
        assert sign == 1
        assert det == 0

    def test_c_on_empty(self):
        assert apply_c(0, 0) is None

    def test_sign_convention(self):
        """Verify fermionic sign: c^dag_2 |0,1> should pick up (-1)^1"""
        det = 0b11  # orbitals 0 and 1 occupied
        result = apply_cdag(det, 2)
        assert result is not None
        sign, _ = result
        assert sign == 1  # 2 bits below bit-2 are set


class TestAnticommutation:
    """Anti-commutation {c_p, c^dag_q} = delta_{pq} (01-02 §3)."""

    @pytest.mark.parametrize("n_ele", [1, 2])
    def test_anticommutation_identity(self, n_ele):
        dim = dim_sector(n_ele)
        for p in range(min(4, N_ORB)):
            for q in range(min(4, N_ORB)):
                # {c_p, c^dag_q} = c_p c^dag_q + c^dag_q c_p
                # In matrix form across sectors:
                # c_p: n -> n-1,  c^dag_q: n-1 -> n
                # So c_p @ c^dag_q: n-1 -> n-1
                # and c^dag_q @ c_p: n -> n (via n-1)
                if n_ele == 0:
                    continue

                cp = c_matrix(p, n_ele)      # (dim_{n-1}, dim_n)
                cdq = cdag_matrix(q, n_ele - 1)  # (dim_n, dim_{n-1})

                # c_p @ c^dag_q acts on n-1 sector: c_p(n->n-1) @ c^dag_q(n-1->n)
                # Wait, dimensions: cp is (dim(n-1), dim(n)), cdq is (dim(n), dim(n-1))
                # So cp @ cdq: (dim(n-1), dim(n-1))
                # And cdq @ cp: (dim(n), dim(n))

                anticomm_small = cp @ cdq  # on n-1 sector
                cdq_n = cdag_matrix(q, n_ele)  # (dim(n+1), dim(n))
                cp_np1 = c_matrix(p, n_ele + 1)  # (dim(n), dim(n+1))
                anticomm_large = cp_np1 @ cdq_n  # on n sector

                # For the n-sector: {c_p, c^dag_q} on sector n
                # = c_p(n+1->n) c^dag_q(n->n+1) + c^dag_q(n-1->n) c_p(n->n-1)
                # Both pieces act as (dim_n, dim_n)
                term1 = cp_np1 @ cdq_n  # (dim_n, dim_n)
                cp_n = c_matrix(p, n_ele)  # (dim_{n-1}, dim_n)
                cdq_nm1 = cdag_matrix(q, n_ele - 1)  # (dim_n, dim_{n-1})
                term2 = cdq_nm1 @ cp_n  # (dim_n, dim_n)

                result = term1 + term2
                expected = np.eye(dim) * (1.0 if p == q else 0.0)
                np.testing.assert_allclose(
                    result, expected, atol=1e-12,
                    err_msg=f"Anti-comm failed for p={p}, q={q}, n={n_ele}"
                )


class TestMatrixHermitianConsistency:
    """c_p matrix should be the conjugate transpose of c^dag_p (01-00 §7)."""

    @pytest.mark.parametrize("n_ele", [1, 2, 3])
    def test_cdag_c_transpose(self, n_ele):
        for p in range(min(4, N_ORB)):
            cd = cdag_matrix(p, n_ele)  # (dim(n+1), dim(n))
            cm = c_matrix(p, n_ele + 1)  # (dim(n), dim(n+1))
            np.testing.assert_allclose(
                cm, cd.conj().T, atol=1e-14,
                err_msg=f"c_p != (c^dag_p)^dag for p={p}, n={n_ele}"
            )
