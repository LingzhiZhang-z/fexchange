"""
Tests against Reference-0 analytic results.

Reference source: docs/reference0/space_lz.py
  - RS_f2_En: analytic Coulomb+SOC energy for f^2 in Russell-Saunders coupling
  - Analytic SOC energies for f^1 (pure one-electron L·S)

Coulomb convention (from RS_f2_En):
    F2_int = F^2 / 225
    F4_int = F^4 / 1089
    F6_int = F^6 * 25 / 184041
    E_term  = c2*F2_int + c4*F4_int + c6*F6_int
  -> coef_Fk (per unit F^k_big) = c_k / denom_k

SOC convention:
  f^1: coef_zeta = <L·S> = (J(J+1) - L(L+1) - S(S+1)) / 2
  f^2: coef_zeta = <sum_i l_i·s_i> = (J(J+1) - L(L+1) - S(S+1)) / 4
       (factor 1/2 for two equivalent electrons in RS_f2_En: 0.25*(...)*Lambda)
"""
from __future__ import annotations

import numpy as np
import pytest

from fexchange.representations.lsms import build_lsms, build_lsms_spectrum
from fexchange.representations.lsjm import build_lsjm, build_lsjm_spectrum

# ---------------------------------------------------------------------------
# Analytic coefficients from Reference-0 RS_f2_En
# (c2, c4, c6) are the integer coefficients in the un-normalised expansion;
# divide by the respective denominators to get coef_Fk per unit F^k.
# ---------------------------------------------------------------------------
_DENOM2 = 225.0
_DENOM4 = 1089.0
_DENOM6 = 184041.0 / 25.0   # = 7361.64; gives c6 * 25 / 184041

# Keyed by (L, twoS): (c2, c4, c6)
_F2_TERM_COEFFS: dict[tuple[int, int], tuple[int, int, int]] = {
    (5, 2): (-25,  -51,   -13),  # ^3H
    (3, 2): (-10,  -33,  -286),  # ^3F
    (1, 2): ( 45,   33, -1287),  # ^3P
    (6, 0): ( 25,    9,     1),  # ^1I
    (4, 0): (-30,   97,    78),  # ^1G
    (2, 0): ( 19,  -99,   715),  # ^1D
    (0, 0): ( 60,  198,  1716),  # ^1S
}


def _ref0_coef_Fk(L: int, twoS: int) -> tuple[float, float, float]:
    """Analytic (coef_F2, coef_F4, coef_F6) per unit F^k for an f^2 LS term."""
    c2, c4, c6 = _F2_TERM_COEFFS[(L, twoS)]
    return c2 / _DENOM2, c4 / _DENOM4, c6 / _DENOM6


def _ref0_f1_coef_zeta(twoJ: int) -> float:
    """
    Analytic coef_zeta for f^1 (L=3, S=1/2).

    coef_zeta = <L·S> = (J(J+1) - L(L+1) - S(S+1)) / 2
    """
    L, S = 3, 0.5
    J = twoJ / 2.0
    return 0.5 * (J * (J + 1) - L * (L + 1) - S * (S + 1))


def _ref0_f2_coef_zeta(L: int, twoS: int, twoJ: int) -> float:
    """
    Analytic coef_zeta for f^2.

    For two equivalent f electrons, <sum_i l_i·s_i> = <L·S>/2,
    consistent with RS_f2_En using 0.25*(J(J+1)-L(L+1)-S(S+1))*Lambda.

    coef_zeta = (J(J+1) - L(L+1) - S(S+1)) / 4
    """
    S = twoS / 2.0
    J = twoJ / 2.0
    return 0.25 * (J * (J + 1) - L * (L + 1) - S * (S + 1))


# ---------------------------------------------------------------------------
# Shared module-scoped fixtures (avoid repeating expensive builds)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lsms_f1():
    return build_lsms(1)


@pytest.fixture(scope="module")
def lsms_f2():
    return build_lsms(2)


@pytest.fixture(scope="module")
def lsjm_f1(lsms_f1):
    return build_lsjm(lsms_f1)


@pytest.fixture(scope="module")
def lsjm_f2(lsms_f2):
    return build_lsjm(lsms_f2)


# ---------------------------------------------------------------------------
# f^1 – SOC analytic test
# ---------------------------------------------------------------------------
class TestF1SOCAnalytic:
    """f^1 coef_zeta must match <L·S> = (J(J+1)-L(L+1)-S(S+1))/2 exactly."""

    def test_coef_zeta_j52_all_M(self, lsjm_f1):
        """J=5/2: coef_zeta = -2.0 for every M state."""
        expected = _ref0_f1_coef_zeta(twoJ=5)  # -2.0
        labels = lsjm_f1["labels"]
        coef_zeta = lsjm_f1["coef_zeta"]
        found = False
        for idx, lab in enumerate(labels):
            if lab["twoJ"] == 5:
                found = True
                assert abs(coef_zeta[idx] - expected) < 1e-8, (
                    f"J=5/2, M={lab['M']}: coef_zeta={coef_zeta[idx]:.10f}, "
                    f"expected={expected:.10f}"
                )
        assert found, "No J=5/2 states found in lsjm_f1"

    def test_coef_zeta_j72_all_M(self, lsjm_f1):
        """J=7/2: coef_zeta = 1.5 for every M state."""
        expected = _ref0_f1_coef_zeta(twoJ=7)  # 1.5
        labels = lsjm_f1["labels"]
        coef_zeta = lsjm_f1["coef_zeta"]
        found = False
        for idx, lab in enumerate(labels):
            if lab["twoJ"] == 7:
                found = True
                assert abs(coef_zeta[idx] - expected) < 1e-8, (
                    f"J=7/2, M={lab['M']}: coef_zeta={coef_zeta[idx]:.10f}, "
                    f"expected={expected:.10f}"
                )
        assert found, "No J=7/2 states found in lsjm_f1"

    def test_spectrum_soc_energies(self, lsjm_f1):
        """build_lsjm_spectrum E_soc must match analytic SOC for each J multiplet."""
        zeta = 1.0
        spectrum = build_lsjm_spectrum(lsjm_f1, zeta=zeta)
        twoJ_seen: set[int] = set()
        for entry in spectrum:
            twoJ = int(entry["twoJ"])
            E_soc_ref = zeta * _ref0_f1_coef_zeta(twoJ)
            assert abs(entry["E_soc"] - E_soc_ref) < 1e-8, (
                f"J={twoJ/2}: E_soc={entry['E_soc']:.10f}, expected={E_soc_ref:.10f}"
            )
            assert abs(entry["E_coulomb"]) < 1e-8, (
                f"f^1 should have zero Coulomb energy, got {entry['E_coulomb']:.3e}"
            )
            twoJ_seen.add(twoJ)
        assert {5, 7} <= twoJ_seen, f"Expected twoJ {{5,7}} in spectrum, got {twoJ_seen}"


# ---------------------------------------------------------------------------
# f^2 – Coulomb energy coefficient tests
# ---------------------------------------------------------------------------
class TestF2CoulombAnalytic:
    """
    f^2 coef_Fk must match Reference-0 analytic coefficients for all 7 terms.
    All (ML, MS) states in the same term must share the same coefficient
    (Wigner-Eckart: Coulomb is scalar in LS space).
    """

    @pytest.mark.parametrize("L,twoS", sorted(_F2_TERM_COEFFS))
    def test_coef_F2(self, lsms_f2, L, twoS):
        expected, _, _ = _ref0_coef_Fk(L, twoS)
        labels = lsms_f2["labels"]
        coef_F2 = lsms_f2["coef_F2"]
        for idx, lab in enumerate(labels):
            if lab["L"] == L and lab["twoS"] == twoS:
                assert abs(coef_F2[idx] - expected) < 1e-8, (
                    f"L={L},twoS={twoS} ML={lab['ML']} MS={lab['MS']}: "
                    f"coef_F2={coef_F2[idx]:.10f}, expected={expected:.10f}"
                )

    @pytest.mark.parametrize("L,twoS", sorted(_F2_TERM_COEFFS))
    def test_coef_F4(self, lsms_f2, L, twoS):
        _, expected, _ = _ref0_coef_Fk(L, twoS)
        labels = lsms_f2["labels"]
        coef_F4 = lsms_f2["coef_F4"]
        for idx, lab in enumerate(labels):
            if lab["L"] == L and lab["twoS"] == twoS:
                assert abs(coef_F4[idx] - expected) < 1e-8, (
                    f"L={L},twoS={twoS}: coef_F4={coef_F4[idx]:.10f}, expected={expected:.10f}"
                )

    @pytest.mark.parametrize("L,twoS", sorted(_F2_TERM_COEFFS))
    def test_coef_F6(self, lsms_f2, L, twoS):
        _, _, expected = _ref0_coef_Fk(L, twoS)
        labels = lsms_f2["labels"]
        coef_F6 = lsms_f2["coef_F6"]
        for idx, lab in enumerate(labels):
            if lab["L"] == L and lab["twoS"] == twoS:
                assert abs(coef_F6[idx] - expected) < 1e-8, (
                    f"L={L},twoS={twoS}: coef_F6={coef_F6[idx]:.10f}, expected={expected:.10f}"
                )

    def test_uniform_coef_within_term(self, lsms_f2):
        """Within each (alpha,L,twoS) term all states share the same coef_Fk."""
        labels = lsms_f2["labels"]
        coef_F2 = lsms_f2["coef_F2"]
        coef_F4 = lsms_f2["coef_F4"]
        coef_F6 = lsms_f2["coef_F6"]

        first: dict[tuple, tuple] = {}
        for idx, lab in enumerate(labels):
            key = (lab["alpha"], lab["L"], lab["twoS"])
            vals = (float(coef_F2[idx]), float(coef_F4[idx]), float(coef_F6[idx]))
            if key not in first:
                first[key] = vals
            else:
                ref = first[key]
                assert abs(vals[0] - ref[0]) < 1e-8, f"coef_F2 not uniform in term {key}"
                assert abs(vals[1] - ref[1]) < 1e-8, f"coef_F4 not uniform in term {key}"
                assert abs(vals[2] - ref[2]) < 1e-8, f"coef_F6 not uniform in term {key}"


# ---------------------------------------------------------------------------
# f^2 – SOC energy coefficient tests
# ---------------------------------------------------------------------------
class TestF2SOCAnalytic:
    """
    f^2 coef_zeta must match (J(J+1)-L(L+1)-S(S+1))/4 for each LSJM multiplet.

    This uses the analytic result that for two equivalent f electrons,
    <sum_i l_i·s_i> = <L·S>/2, consistent with Reference-0 RS_f2_En
    which uses factor 0.25*(J(J+1)-...).
    """

    @pytest.mark.parametrize("L,twoS", sorted(_F2_TERM_COEFFS))
    def test_coef_zeta_each_J(self, lsjm_f2, L, twoS):
        S = twoS / 2.0
        twoJ_values = range(abs(2 * L - twoS), 2 * L + twoS + 2, 2)
        labels = lsjm_f2["labels"]
        coef_zeta = lsjm_f2["coef_zeta"]

        for twoJ in twoJ_values:
            expected = _ref0_f2_coef_zeta(L, twoS, twoJ)
            found = False
            for idx, lab in enumerate(labels):
                if lab["L"] == L and lab["twoS"] == twoS and lab["twoJ"] == twoJ:
                    found = True
                    assert abs(coef_zeta[idx] - expected) < 1e-8, (
                        f"L={L},twoS={twoS},J={twoJ/2}: "
                        f"coef_zeta={coef_zeta[idx]:.10f}, expected={expected:.10f}"
                    )
            assert found, f"No state found for L={L},twoS={twoS},twoJ={twoJ}"

    def test_uniform_coef_zeta_within_J_multiplet(self, lsjm_f2):
        """coef_zeta is M-independent within a (alpha,L,twoS,twoJ) multiplet."""
        labels = lsjm_f2["labels"]
        coef_zeta = lsjm_f2["coef_zeta"]

        first: dict[tuple, float] = {}
        for idx, lab in enumerate(labels):
            key = (lab["alpha"], lab["L"], lab["twoS"], lab["twoJ"])
            val = float(coef_zeta[idx])
            if key not in first:
                first[key] = val
            else:
                assert abs(val - first[key]) < 1e-8, (
                    f"coef_zeta not uniform in multiplet {key}: {val:.10f} vs {first[key]:.10f}"
                )


# ---------------------------------------------------------------------------
# f^2 – full Coulomb spectrum ordering
# ---------------------------------------------------------------------------
class TestF2SpectrumOrdering:
    """
    Coulomb spectrum ordering from build_lsms_spectrum must match
    Reference-0 analytic ordering for physical ratios r42=0.5, r62=0.25.
    """

    @pytest.fixture(scope="class")
    def lsms_f2_physical(self):
        """Build with r42=0.5, r62=0.25 (representative physical ratio for f ions)."""
        return build_lsms(2, r42=0.5, r62=0.25)

    def test_ground_term_is_3H(self, lsms_f2_physical):
        """Hund's 1st & 2nd rules: f^2 ground term is ^3H (L=5, twoS=2)."""
        spectrum = build_lsms_spectrum(lsms_f2_physical)
        ground = spectrum[0]
        assert ground["L"] == 5 and ground["twoS"] == 2, (
            f"Ground term should be ^3H (L=5,twoS=2), got L={ground['L']},twoS={ground['twoS']}"
        )

    def test_highest_term_is_1S(self, lsms_f2_physical):
        """^1S (L=0, twoS=0) is the highest Coulomb term."""
        spectrum = build_lsms_spectrum(lsms_f2_physical)
        top = spectrum[-1]
        assert top["L"] == 0 and top["twoS"] == 0, (
            f"Highest term should be ^1S (L=0,twoS=0), got L={top['L']},twoS={top['twoS']}"
        )

    def test_analytic_coulomb_energies(self, lsms_f2_physical):
        """
        Verify build_lsms_spectrum E_coulomb against RS_f2_En analytic formula
        for every term, using the F2/F4/F6 stored in the physics dict.
        """
        spectrum = build_lsms_spectrum(lsms_f2_physical)
        physics = lsms_f2_physical["physics"]
        F2 = float(physics["F2"])
        F4 = float(physics["F4"])
        F6 = float(physics["F6"])

        for entry in spectrum:
            L, twoS = entry["L"], entry["twoS"]
            c2, c4, c6 = _ref0_coef_Fk(L, twoS)
            E_ref = F2 * c2 + F4 * c4 + F6 * c6
            assert abs(entry["E_coulomb"] - E_ref) < 1e-7, (
                f"Term L={L},twoS={twoS}: E_coulomb={entry['E_coulomb']:.8f}, "
                f"analytic={E_ref:.8f}"
            )
