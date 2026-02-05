"""Tests for api/keys.py - Cache key generation.

Expected behavior:
- quantize_float: converts float to stable string representation
- normalize_shape: normalizes (F2, F4, F6) to (r4, r6, F2_abs)
- shape_key: generates deterministic cache key
- energy_key: generates deterministic cache key

Expected outputs:
- quantize_float(1.0) -> "1.00000000000000e+00" (14 digits)
- quantize_float(0.0) -> "0.00000000000000e+00"
- quantize_float(1e-15) with precision=1e-12 -> "0.00..." (zeroed)
- normalize_shape(100, 50, 25) -> (0.5, 0.25, 100.0)
"""
from __future__ import annotations

import math

import pytest

from fexchange.api.keys import energy_key, normalize_shape, quantize_float, shape_key


class TestQuantizeFloat:
    """Test float quantization for cache keys."""

    def test_basic_values(self) -> None:
        """Basic float values should be quantized."""
        result = quantize_float(1.0)
        assert "e" in result or "E" in result  # scientific notation
        assert float(result) == pytest.approx(1.0)

    def test_zero(self) -> None:
        """Zero should quantize to zero."""
        result = quantize_float(0.0)
        assert float(result) == 0.0

    def test_negative(self) -> None:
        """Negative values should preserve sign."""
        result = quantize_float(-1.5)
        assert float(result) == pytest.approx(-1.5)

    def test_tiny_value_zeroed(self) -> None:
        """Values smaller than precision should be zeroed.

        With precision=1e-12, value 1e-15 should become 0.
        """
        result = quantize_float(1e-15, precision=1e-12)
        assert float(result) == 0.0

    def test_large_value(self) -> None:
        """Large values should be handled."""
        result = quantize_float(1e10)
        assert float(result) == pytest.approx(1e10)

    def test_nan(self) -> None:
        """NaN should return 'nan'."""
        result = quantize_float(float("nan"))
        assert "nan" in result.lower()

    def test_inf(self) -> None:
        """Infinity should return 'inf'."""
        result = quantize_float(float("inf"))
        assert "inf" in result.lower()

    def test_deterministic(self) -> None:
        """Same input should produce same output."""
        val = 1.234567890123
        r1 = quantize_float(val)
        r2 = quantize_float(val)
        assert r1 == r2

    def test_precision_effect(self) -> None:
        """Different precisions should affect output."""
        val = 1.23456789
        r1 = quantize_float(val, precision=1e-6)
        r2 = quantize_float(val, precision=1e-12)
        # Both should round-trip to approximately the same value
        assert float(r1) == pytest.approx(float(r2), rel=1e-6)


class TestNormalizeShape:
    """Test Coulomb shape normalization."""

    def test_basic_normalization(self) -> None:
        """(F2, F4, F6) should normalize to (F4/F2, F6/F2, |F2|).

        Example: (100, 50, 25) -> (0.5, 0.25, 100.0)
        """
        r4, r6, F2_abs = normalize_shape(100.0, 50.0, 25.0)
        assert r4 == pytest.approx(0.5)
        assert r6 == pytest.approx(0.25)
        assert F2_abs == pytest.approx(100.0)

    def test_negative_F2(self) -> None:
        """Negative F2 should give positive F2_abs."""
        r4, r6, F2_abs = normalize_shape(-100.0, -50.0, -25.0)
        assert r4 == pytest.approx(0.5)
        assert r6 == pytest.approx(0.25)
        assert F2_abs == pytest.approx(100.0)

    def test_F2_zero_raises(self) -> None:
        """F2=0 should raise ValueError."""
        with pytest.raises(ValueError, match="F2"):
            normalize_shape(0.0, 50.0, 25.0)

    def test_physical_ratios(self) -> None:
        """Test with typical f-electron Slater ratios.

        For lanthanides, typical ratios are r4 ~ 0.6-0.7, r6 ~ 0.4-0.5
        """
        F2, F4, F6 = 400.0, 260.0, 180.0  # Typical Ce values (meV)
        r4, r6, F2_abs = normalize_shape(F2, F4, F6)
        assert 0.5 < r4 < 0.8
        assert 0.3 < r6 < 0.6
        assert F2_abs == F2


class TestShapeKey:
    """Test shape key generation."""

    def test_format(self) -> None:
        """Shape key should contain expected components."""
        key = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS_v1", basis_id="fock14_n2")
        assert "shape|" in key
        assert "n=2" in key
        assert "r4=" in key
        assert "r6=" in key
        assert "scheme=RS_v1" in key
        assert "basis=fock14_n2" in key

    def test_deterministic(self) -> None:
        """Same inputs should produce same key."""
        k1 = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
        k2 = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
        assert k1 == k2

    def test_different_n(self) -> None:
        """Different n should produce different keys."""
        k1 = shape_key(n=2, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
        k2 = shape_key(n=3, r4=0.5, r6=0.25, scheme_id="RS", basis_id="b1")
        assert k1 != k2


class TestEnergyKey:
    """Test energy key generation."""

    def test_format(self) -> None:
        """Energy key should contain shape_key and energy parameters."""
        sk = "shape|n=2|r4=0.5|r6=0.25"
        key = energy_key(sk, F2_abs=100.0, F0_abs=10.0, lambda_abs=50.0)
        assert "energy|" in key
        assert sk in key
        assert "F2=" in key
        assert "F0=" in key
        assert "lambda=" in key

    def test_deterministic(self) -> None:
        """Same inputs should produce same key."""
        sk = "shape|test"
        k1 = energy_key(sk, F2_abs=100.0, F0_abs=10.0, lambda_abs=50.0)
        k2 = energy_key(sk, F2_abs=100.0, F0_abs=10.0, lambda_abs=50.0)
        assert k1 == k2
