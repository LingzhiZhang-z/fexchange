"""Tests for models/hint.py (02-01)."""

import pytest
import numpy as np

from fexchange.hamiltonian.hint import build_hint_matrix, build_hint_rank_matrix
from fexchange.utils.checks import check_hermitian


class TestHintHermiticity:
    """H_int must be Hermitian (02-01 §5)."""

    @pytest.mark.parametrize("n_ele", [1, 2])
    def test_hermitian(self, n_ele):
        H = build_hint_matrix(n_ele)
        check_hermitian(H, label="H_int_test")

    @pytest.mark.parametrize("n_ele", [1, 2])
    def test_rank_hermitian(self, n_ele):
        for k in (0, 2, 4, 6):
            H_k = build_hint_rank_matrix(k, n_ele)
            check_hermitian(H_k, label=f"O_{k}")


class TestF2TermStructure:
    """f^2: expect 7 LS terms with total 91 states (prompt verification checklist)."""

    def test_f2_dimension(self):
        from fexchange.core.fock import dim_sector
        assert dim_sector(2) == 91

    def test_f2_hint_shape(self):
        H = build_hint_matrix(2)
        assert H.shape == (91, 91)
