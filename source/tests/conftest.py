"""Pytest configuration and fixtures for fexchange tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_fock_basis():
    """Small Fock basis for quick tests."""
    from fexchange.core.fock_basis import build_fock_basis

    return build_fock_basis(2)
