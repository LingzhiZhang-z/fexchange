"""Validate fixed-ligand SOC downfold against the literature Eq. 12.

The comparison has two independent sides:

1. Actual side:
   - build f-p hopping blocks with ``slater_koster_hopping_fp_blocks``;
   - fold p holes using ``_p5_hole_propagator``;
   - rotate the resulting 14x14 matrix back to cubic f basis.

2. Reference side:
   - transcribe Eq. 12 directly as a 7x7 matrix of 2x2 Pauli blocks;
   - use the paper definitions
       t0 = 1/2 * sqrt(5/2) * t_pf_pi,
       t1 = 1/2 * sqrt(3/2) * t_pf_pi,
       t2 = t_pf_sigma,
       t_eff = t0^2 * (Delta + lambda/2)
               / ((Delta + lambda) * (Delta - lambda/2)),
       r = lambda / (Delta + lambda/2).

This script intentionally does not import any expected matrix from the tool.
"""

from __future__ import annotations

import numpy as np

from fexchange.tools import slater_koster_pf as sk
from fexchange.tools.w90_downfold import _p5_hole_propagator


def _pauli_down_up() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigma0 = np.eye(2, dtype=complex)
    sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
    sigmay = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    sigmaz = np.array([[-1, 0], [0, 1]], dtype=complex)
    return sigma0, sigmax, sigmay, sigmaz


def _actual_cubic(
    pf_sigma: float,
    pf_pi: float,
    delta: float,
    lambda_p: float,
) -> np.ndarray:
    blocks = dict(sk.slater_koster_hopping_fp_blocks(pf_sigma, pf_pi))
    g_p, _, _ = _p5_hole_propagator(delta, lambda_p, degenerate_tol=1e-12)
    t_complex = (
        blocks["t_f1_lig1"] @ g_p @ blocks["t_f2_lig1"].conj().T
        + blocks["t_f1_lig2"] @ g_p @ blocks["t_f2_lig2"].conj().T
    )
    u_cubic_to_complex = sk._build_U_c2y(spinor=True)
    return u_cubic_to_complex.conj().T @ t_complex @ u_cubic_to_complex


def _reference_eq12(
    pf_sigma: float,
    pf_pi: float,
    delta: float,
    lambda_p: float,
) -> np.ndarray:
    sigma0, sigmax, sigmay, sigmaz = _pauli_down_up()
    zero = np.zeros((2, 2), dtype=complex)

    t0 = 0.5 * np.sqrt(5.0 / 2.0) * pf_pi
    t1 = 0.5 * np.sqrt(3.0 / 2.0) * pf_pi
    t2 = pf_sigma
    t_eff = (
        t0 * t0 * (delta + 0.5 * lambda_p)
        / ((delta + lambda_p) * (delta - 0.5 * lambda_p))
    )
    r = lambda_p / (delta + 0.5 * lambda_p)

    m = [[zero.copy() for _ in range(7)] for __ in range(7)]
    m[0][1] = 1j * r / 2 * sigmaz
    m[0][2] = 1j * r / 2 * sigmay
    m[0][4] = t2 / t0 * sigma0
    m[0][5] = 1j * r * t1 / (2 * t0) * sigmaz
    m[0][6] = -1j * r * t1 / (2 * t0) * sigmay

    m[1][0] = -1j * r / 2 * sigmaz
    m[1][2] = -1j * r / 2 * sigmax
    m[1][4] = 1j * r * t1 / (2 * t0) * sigmaz
    m[1][5] = -t2 / t0 * sigma0
    m[1][6] = -1j * r * t1 / (2 * t0) * sigmax

    m[2][0] = -1j * r / 2 * sigmay
    m[2][1] = 1j * r / 2 * sigmax
    m[2][2] = -2 * sigma0
    m[2][4] = 1j * r * (t1 + t2) / (2 * t0) * sigmay
    m[2][5] = 1j * r * (t1 + t2) / (2 * t0) * sigmax

    m[4][0] = t2 / t0 * sigma0
    m[4][1] = -1j * r * t1 / (2 * t0) * sigmaz
    m[4][2] = -1j * r * (t1 + t2) / (2 * t0) * sigmay
    m[4][4] = -2 * t1 * t2 / (t0 * t0) * sigma0
    m[4][5] = -1j * r * (t1 * t1 + t2 * t2) / (2 * t0 * t0) * sigmaz
    m[4][6] = 1j * r * t1 * (t1 - t2) / (2 * t0 * t0) * sigmay

    m[5][0] = -1j * r * t1 / (2 * t0) * sigmaz
    m[5][1] = -t2 / t0 * sigma0
    m[5][2] = -1j * r * (t1 + t2) / (2 * t0) * sigmax
    m[5][4] = 1j * r * (t1 * t1 + t2 * t2) / (2 * t0 * t0) * sigmaz
    m[5][5] = -2 * t1 * t2 / (t0 * t0) * sigma0
    m[5][6] = -1j * r * t1 * (t1 - t2) / (2 * t0 * t0) * sigmax

    m[6][0] = 1j * r * t1 / (2 * t0) * sigmay
    m[6][1] = 1j * r * t1 / (2 * t0) * sigmax
    m[6][4] = -1j * r * t1 * (t1 - t2) / (2 * t0 * t0) * sigmay
    m[6][5] = 1j * r * t1 * (t1 - t2) / (2 * t0 * t0) * sigmax
    m[6][6] = 2 * t1 * t1 / (t0 * t0) * sigma0

    out = np.zeros((14, 14), dtype=complex)
    for i in range(7):
        for j in range(7):
            out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = t_eff * m[i][j]
    return out


def main() -> int:
    cases = [
        (1.7, -0.4, 2.3, 0.5),
        (1.0, -0.25, 4.0, 0.2),
        (2.1, -0.9, 3.7, 0.8),
    ]
    tol = 1e-12
    for pf_sigma, pf_pi, delta, lambda_p in cases:
        actual = _actual_cubic(pf_sigma, pf_pi, delta, lambda_p)
        expected = _reference_eq12(pf_sigma, pf_pi, delta, lambda_p)
        diff = actual - expected
        max_abs = float(np.max(np.abs(diff)))
        rel_fro = float(np.linalg.norm(diff) / np.linalg.norm(expected))
        print(
            "pf_sigma={:.6g} pf_pi={:.6g} Delta={:.6g} lambda={:.6g} "
            "max_abs={:.16e} rel_fro={:.16e}".format(
                pf_sigma, pf_pi, delta, lambda_p, max_abs, rel_fro
            )
        )
        if max_abs > tol:
            raise SystemExit(1)
    print("PASS Eq12 validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
