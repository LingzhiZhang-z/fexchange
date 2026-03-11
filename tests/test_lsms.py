"""Tests for representations/lsms.py (03-00)."""

import pytest
import numpy as np

from fexchange.representations.lsms import build_lsms
from fexchange.utils.checks import check_orthonormal


class TestLsmsF2:
    """f^2 must produce exactly 91 LSMS states across known LS terms."""

    @pytest.fixture(autouse=True)
    def build(self):
        self.result = build_lsms(2)

    def test_total_states(self):
        assert self.result["V_fock"].shape[1] == 91

    def test_orthonormality(self):
        check_orthonormal(self.result["V_fock"], label="V_lsms_f2")

    def test_known_terms(self):
        """f^2 terms: 1S, 1D, 1G, 1I, 3P, 3F, 3H."""
        labels = self.result["labels"]
        terms = set()
        for lab in labels:
            terms.add((lab["L"], lab["twoS"]))
        expected_terms = {
            (0, 0),   # 1S
            (2, 0),   # 1D
            (4, 0),   # 1G
            (6, 0),   # 1I
            (1, 2),   # 3P
            (3, 2),   # 3F
            (5, 2),   # 3H
        }
        assert terms == expected_terms

    def test_term_multiplicities(self):
        """Each term should have (2L+1)*(2S+1) states."""
        labels = self.result["labels"]
        term_counts: dict[tuple, int] = {}
        for lab in labels:
            key = (lab["alpha"], lab["L"], lab["twoS"])
            term_counts[key] = term_counts.get(key, 0) + 1

        for (alpha, L, twoS), count in term_counts.items():
            expected = (2 * L + 1) * (twoS + 1)
            assert count == expected, f"Term alpha={alpha},L={L},2S={twoS}: {count} != {expected}"


class TestLsmsF1:
    """f^1: single term 2F with 14 states."""

    def test_f1(self):
        result = build_lsms(1)
        assert result["V_fock"].shape[1] == 14
        check_orthonormal(result["V_fock"], label="V_lsms_f1")
