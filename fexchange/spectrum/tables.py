"""
Static group-theory data for point-group irrep classification.

This module contains only data dicts. No functions.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Rotational-core character tables (normative, 02-07 Appendix A/B)
# ---------------------------------------------------------------------------

# O* (order 48): 10 sub-classes
O_STAR_CLASSES: tuple[str, ...] = ("E", "R", "3C2", "3RC2", "C4", "RC4", "6C2p", "6RC2p", "C3", "RC3")
O_STAR_SIZES: tuple[int, ...] = (1, 1, 3, 3, 6, 6, 6, 6, 8, 8)
O_STAR_IRREPS: tuple[str, ...] = (
    "Gamma1",
    "Gamma2",
    "Gamma3",
    "Gamma4",
    "Gamma5",
    "Gamma6",
    "Gamma7",
    "Gamma8",
)

_s2 = np.sqrt(2)
O_STAR_CHI = np.array(
    [
        # E, R, 3C2, 3RC2, C4, RC4, 6C2p, 6RC2p, C3, RC3
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Gamma1
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1],  # Gamma2
        [2, 2, 2, 2, 0, 0, 0, 0, -1, -1],  # Gamma3
        [3, 3, -1, -1, 1, 1, -1, -1, 0, 0],  # Gamma4
        [3, 3, -1, -1, -1, -1, 1, 1, 0, 0],  # Gamma5
        [2, -2, 0, 0, _s2, -_s2, 0, 0, 1, -1],  # Gamma6
        [2, -2, 0, 0, -_s2, _s2, 0, 0, 1, -1],  # Gamma7
        [4, -4, 0, 0, 0, 0, 0, 0, -1, 1],  # Gamma8
    ],
    dtype=complex,
)

# C3v* (order 12): 6 classes
C3V_STAR_CLASSES: tuple[str, ...] = ("E", "R", "2C3", "2RC3", "3sigma_v", "3Rsigma_v")
C3V_STAR_SIZES: tuple[int, ...] = (1, 1, 2, 2, 3, 3)
C3V_STAR_IRREPS: tuple[str, ...] = ("Gamma1", "Gamma2", "Gamma3", "Gamma4", "Gamma5", "Gamma6")
C3V_STAR_CHI = np.array(
    [
        # E, R, 2C3, 2RC3, 3sigma_v, 3Rsigma_v
        [1, 1, 1, 1, 1, 1],  # Gamma1
        [1, 1, 1, 1, -1, -1],  # Gamma2
        [2, 2, -1, -1, 0, 0],  # Gamma3
        [2, -2, 1, -1, 0, 0],  # Gamma4
        [1, -1, -1, 1, 1j, -1j],  # Gamma5
        [1, -1, -1, 1, -1j, 1j],  # Gamma6
    ],
    dtype=complex,
)

# ---------------------------------------------------------------------------
# Multipole compatibility rules (02-07 §6)
# ---------------------------------------------------------------------------

MULTIPOLE_RULES: dict[str, dict[str, list[str]]] = {
    "Oh": {
        "Gamma1": ["magnetic_octupole"],
        "Gamma2": ["magnetic_octupole"],
        "Gamma3": ["electric_quadrupole", "magnetic_octupole"],
        "Gamma4": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma5": ["electric_quadrupole", "magnetic_octupole"],
        "Gamma6": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma7": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma8": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
    },
    "D3d": {
        # Parity-tagged (02-07 contract)
        "Gamma1+": [],
        "Gamma2+": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma3+": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
        "Gamma1-": [],
        "Gamma2-": ["magnetic_octupole"],
        "Gamma3-": ["electric_quadrupole", "magnetic_octupole"],
        # Spinor parity-tagged
        "Gamma4+": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma5+": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
        "Gamma6+": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
        "Gamma4-": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma5-": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
        "Gamma6-": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
    },
    "C3v": {
        "Gamma1": [],
        "Gamma2": [],
        "Gamma3": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
        "Gamma4": ["magnetic_dipole", "magnetic_octupole"],
        "Gamma5": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
        "Gamma6": ["magnetic_dipole", "electric_quadrupole", "magnetic_octupole"],
    },
}

# ---------------------------------------------------------------------------
# Display and alias metadata
# ---------------------------------------------------------------------------

IRREP_DISPLAY: dict[str, str] = {
    "Gamma1": "Γ1",
    "Gamma2": "Γ2",
    "Gamma3": "Γ3",
    "Gamma4": "Γ4",
    "Gamma5": "Γ5",
    "Gamma6": "Γ6",
    "Gamma7": "Γ7",
    "Gamma8": "Γ8",
    "Gamma1+": "Γ1+",
    "Gamma2+": "Γ2+",
    "Gamma3+": "Γ3+",
    "Gamma4+": "Γ4+",
    "Gamma5+": "Γ5+",
    "Gamma6+": "Γ6+",
    "Gamma1-": "Γ1-",
    "Gamma2-": "Γ2-",
    "Gamma3-": "Γ3-",
    "Gamma4-": "Γ4-",
    "Gamma5-": "Γ5-",
    "Gamma6-": "Γ6-",
}

ALIAS_MAP: dict[tuple[str, int], dict[str, list[str]]] = {
    ("Oh", +1): {
        "Gamma1": ["A1g"],
        "Gamma2": ["A2g"],
        "Gamma3": ["Eg"],
        "Gamma4": ["T1g"],
        "Gamma5": ["T2g"],
    },
    ("Oh", -1): {
        "Gamma1": ["A1u"],
        "Gamma2": ["A2u"],
        "Gamma3": ["Eu"],
        "Gamma4": ["T1u"],
        "Gamma5": ["T2u"],
    },
    ("D3d", +1): {
        "Gamma1+": ["A1g"],
        "Gamma2+": ["A2g"],
        "Gamma3+": ["Eg"],
    },
    ("D3d", -1): {
        "Gamma1-": ["A1u"],
        "Gamma2-": ["A2u"],
        "Gamma3-": ["Eu"],
    },
}
