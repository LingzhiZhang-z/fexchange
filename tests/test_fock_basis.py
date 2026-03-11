"""Tests for core/fock_basis.py (01-00)."""

import pytest
import numpy as np
from math import comb

from fexchange.core.fock_basis import (
    N_ORB, occ, popcount, count_below,
    set_bit, clear_bit,
    dim_sector, enumerate_dets, det_index,
    make_basis_id, check_basis_id,
)


def test_n_orb():
    assert N_ORB == 14


@pytest.mark.parametrize("n", range(0, 15))
def test_dim_sector(n):
    assert dim_sector(n) == comb(14, n)


def test_enumerate_dets_f1():
    dets = enumerate_dets(1)
    assert len(dets) == 14
    assert all(popcount(int(d)) == 1 for d in dets)
    assert list(dets) == sorted(dets)


def test_enumerate_dets_f2():
    dets = enumerate_dets(2)
    assert len(dets) == 91


def test_occ_and_popcount():
    det = 0b1010  # orbitals 1 and 3 occupied
    assert occ(det, 0) == 0
    assert occ(det, 1) == 1
    assert occ(det, 3) == 1
    assert popcount(det) == 2


def test_count_below():
    det = 0b1010
    assert count_below(det, 0) == 0
    assert count_below(det, 1) == 0
    assert count_below(det, 2) == 1
    assert count_below(det, 3) == 1


def test_set_clear_bit():
    det = 0b0000
    det = set_bit(det, 2)
    assert occ(det, 2) == 1
    det = clear_bit(det, 2)
    assert occ(det, 2) == 0


def test_det_index():
    dets = enumerate_dets(2)
    for i, d in enumerate(dets):
        assert det_index(int(d), dets) == i


def test_basis_id():
    bid = make_basis_id(3)
    assert bid == "fock14_n3_lex_v1"
    check_basis_id(bid, bid)  # should not raise


def test_basis_id_mismatch():
    with pytest.raises(Exception):
        check_basis_id("fock14_n3_lex_v1", "fock14_n4_lex_v1")
