"""Shared fixtures for fexchange tests (00-01 §5)."""

from __future__ import annotations

import pytest
import numpy as np

from fexchange.core.fock import N_ORB, enumerate_dets, dim_sector


def random_u2(seed: int) -> np.ndarray:
    """Deterministic random 2x2 unitary for gauge-invariance tests."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, _ = np.linalg.qr(raw)
    return q


@pytest.fixture(params=[1, 2, 3])
def small_n_ele(request):
    """Small electron counts for fast tests."""
    return request.param


@pytest.fixture
def f1_dets():
    """Determinants for f^1 sector."""
    return enumerate_dets(1, N_ORB)


@pytest.fixture
def f2_dets():
    """Determinants for f^2 sector."""
    return enumerate_dets(2, N_ORB)
