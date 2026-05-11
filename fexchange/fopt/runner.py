"""High-level fopt orchestrator: L0 -> L1 -> L2 -> L3 + spin-1/2 mapping.

Entry point for end-to-end ligand-mediated 4th-order superexchange:
  - assembles f-shell L0/L1 (uses sopt primitives via fopt.precompute)
  - assembles ligand p-shell L0/L1 (creation tensors at sectors {4,5,6})
  - builds 12 cluster V_+ blocks (L2)
  - sums 24 path amplitudes with W projection to doublet (L3)
  - hermitizes and feeds spin12_map for Kramers J-tensor extraction

Ligand-SOC support
------------------
`compute_p_energies_soc(Delta, U_p, lambda_p)` builds the single-particle
p-shell SOC Hamiltonian H_SOC^p = lambda_p * L*S (l=1, s=1/2), lifts it to the
many-body p^5 and p^4 sectors, and diagonalizes. The eigenvalues become the
intermediate-state energies; the eigenvector matrices become the U_p_5 /
U_p_4 ligand rotations consumed by `build_L1`. `build_L3` itself stays
byte-compatible with both no-SOC and SOC-ligand modes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from fexchange.fopt.precompute import build_L0, build_L1
from fexchange.fopt.contraction import build_L2, build_L3
from fexchange.spectrum.ligand import build_ligand_spectrum
from fexchange.spectrum.energy import compute_ligand_energies
from fexchange.sopt.spin12 import spin12_map
from fexchange.utils.numerics import DTYPE_COMPLEX

logger = logging.getLogger("fexchange")


# ---------------------------------------------------------------------------
# Ligand p-shell intermediate-state energies.
# ---------------------------------------------------------------------------

def _ligand_payload(
    Delta: float, U_p: float, lambda_p: float,
) -> dict[str, Any]:
    """Assemble per-sector p-shell energies + rotations via spectrum.ligand.

    Dispatch on `lambda_p`:
      lambda_p == 0  -> `build_ligand_spectrum(N, soc=False)`: V_fock = identity,
                        per-state energies degenerate (E_p^5 = Delta,
                        E_p^4 = 2*Delta + U_p).
      lambda_p != 0  -> `build_ligand_spectrum(N, soc=True)`:  SOC eigenbasis +
                        atomic j=3/2 / j=1/2 split.
    """
    use_soc = bool(lambda_p != 0.0)
    spec4 = build_ligand_spectrum(4, soc=use_soc)
    spec5 = build_ligand_spectrum(5, soc=use_soc)
    spec6 = build_ligand_spectrum(6)   # always trivial; soc flag irrelevant
    return {
        "p_4":   compute_ligand_energies(spec4, Delta=Delta, U_p=U_p, lambda_p=lambda_p),
        "p_5":   compute_ligand_energies(spec5, Delta=Delta, U_p=U_p, lambda_p=lambda_p),
        "p_6":   compute_ligand_energies(spec6, Delta=Delta, U_p=U_p, lambda_p=lambda_p),
        "U_p_4": spec4["V_fock"],
        "U_p_5": spec5["V_fock"],
        "U_p_6": spec6["V_fock"],
        "spectrum_4": spec4,
        "spectrum_5": spec5,
        "spectrum_6": spec6,
    }


def compute_p_energies_nsoc(
    Delta: float,
    U_p: float,
) -> dict[str, Any]:
    """Ligand p-shell intermediate energies in the no-SOC limit (lambda_p = 0).

    Thin wrapper over `compute_p_energies_soc(..., lambda_p=0)`. The SOC
    eigenbasis is well-defined even at zero coupling (eigh of a non-zero
    operator), and the rotation drops out of `build_L3` observables because
    the per-state energies are degenerate at lambda_p=0.
    """
    return compute_p_energies_soc(Delta, U_p, lambda_p=0.0)


def compute_p_energies_soc(
    Delta: float,
    U_p: float,
    lambda_p: float,
) -> dict[str, Any]:
    """SOC-ligand p-shell intermediate energies + Fock->eigenbasis rotations.

    Delegates to `spectrum.ligand.build_ligand_spectrum` for the structural
    payload (per-state coefficients + eigenbasis rotation) and
    `spectrum.energy.compute_ligand_energies` for the physical assembly.

    Returned keys
    -------------
    "p_4", "p_5", "p_6" : 1D ndarrays of E_p^N values, length C(6, N).
    "U_p_4", "U_p_5", "U_p_6" : (N x N) Fock->SOC-eigenbasis rotations
        for the corresponding p^N sectors. These feed `build_L1` via the
        `U_p_per_ligand` parameter so the raw p-shell creation tensors are
        delivered in the eigenbasis that diagonalises the per-state E array.
    "spectrum_4", "spectrum_5", "spectrum_6" : the underlying spectrum
        payloads (Fock-basis V_fock + the three coef arrays). Convenient for
        downstream consumers that want to re-compose energies at other
        Delta / U_p / lambda_p values without re-diagonalising.
    """
    return _ligand_payload(Delta, U_p, lambda_p)


# ---------------------------------------------------------------------------
# End-to-end orchestrator.
# ---------------------------------------------------------------------------

def run_fopt(
    *,
    n_ele: int,
    U_f_per_site:   dict[int, tuple],
    t_per_pair:     dict[tuple[int, int], NDArray[np.complexfloating]],
    W:              NDArray[np.complexfloating],
    E_np1:          NDArray[np.floating],
    E_nm1:          NDArray[np.floating],
    Delta:          float,
    U_p:            float,
    lambda_p:       float = 0.0,
    U_p_per_ligand: Optional[dict[int, tuple]] = None,
    hermitize:      bool = True,
) -> dict[str, Any]:
    """Run the full fopt pipeline and (when n_k == 2) the spin-1/2 mapping.

    Parameters
    ----------
    n_ele
        f-shell electron count (1..13).
    U_f_per_site
        {f_site: (U_np1, U_n_soc0, U_nm1)} — Fock->LSJM rotations per f-site.
    t_per_pair
        {(f_site, ligand): t_pf hopping matrix} for all 4 (f_site, ligand) pairs.
        t_pf shape must be (14, 6) (f-orbitals x p-orbitals).
    W
        (2J+1, n_k) projector from f^n LSJM ground multiplet onto the n_k-state
        target subspace (e.g. n_k = 2 for the Kramers doublet).
    E_np1, E_nm1
        1-D arrays of intermediate-state energies for f^{n+1}/f^{n-1}, in the
        same unit as Delta/U_p/lambda_p, referenced to E_GS = 0.
    Delta, U_p
        Ligand-to-f charge-transfer cost and on-ligand Hubbard U.
    lambda_p
        Ligand p-shell SOC strength. ``0`` (default) selects the no-SOC limit
        (`compute_p_energies_nsoc`). Non-zero invokes
        `compute_p_energies_soc` and overrides any ``U_p_per_ligand``
        identity defaults with the diagonalised Fock->eigenbasis rotations.
    U_p_per_ligand
        Explicit override `{lig: (U_p_6, U_p_5, U_p_4)}`. Use this only when
        you want to override the SOC eigenbasis (or the nsoc identity) with
        a custom rotation; pass ``None`` for the auto path.
    hermitize
        If True (default), enforce Heff = (Heff + Heff^dag)/2 before the
        spin-1/2 mapping (`spin12_map` enforces strict Hermiticity).

    Returns
    -------
    dict with keys
        "Heff"             : (n_k,)*4 complex tensor in [c, d, a, b] convention.
        "Heff_raw"         : pre-hermitization tensor (for diagnostics).
        "energies"         : the energies dict that was fed to build_L3.
        "lambda_p"         : the SOC strength actually used.
        "J_mu"             : 3x3 Kramers exchange (only when n_k == 2).
        "mapping_residual" : spin12 reconstruction residual (only when n_k == 2).
    """
    logger.info(
        "fopt.run_fopt: n_ele=%d, n_pairs=%d, lambda_p=%g",
        n_ele, len(t_per_pair), lambda_p,
    )

    if lambda_p == 0.0:
        p_data = compute_p_energies_nsoc(Delta, U_p)
    else:
        p_data = compute_p_energies_soc(Delta, U_p, lambda_p)
        # Override identity defaults with the SOC eigenbasis. Caller can still
        # supply U_p_per_ligand explicitly (e.g. lig-specific spectra).
        if U_p_per_ligand is None:
            soc_us = (p_data["U_p_6"], p_data["U_p_5"], p_data["U_p_4"])
            U_p_per_ligand = {lig: soc_us for lig in (1, 2)}

    l0 = build_L0(n_ele)
    l1 = build_L1(l0, U_f_per_site, U_p_per_ligand)
    l2 = build_L2(l1, t_per_pair)

    energies = {
        "f_np1": np.asarray(E_np1, dtype=np.float64),
        "f_nm1": np.asarray(E_nm1, dtype=np.float64),
        "p_5":   p_data["p_5"],
        "p_4":   p_data["p_4"],
    }

    Heff_raw = build_L3(l2, energies, W, n_ele=n_ele)
    n_k = Heff_raw.shape[0]

    if hermitize:
        flat = Heff_raw.reshape(n_k * n_k, n_k * n_k)
        flat = 0.5 * (flat + flat.conj().T)
        Heff = flat.reshape((n_k,) * 4).astype(DTYPE_COMPLEX)
    else:
        Heff = Heff_raw

    out: dict[str, Any] = {
        "Heff":     Heff,
        "Heff_raw": Heff_raw,
        "energies": energies,
        "lambda_p": float(lambda_p),
    }

    if n_k == 2:
        spin = spin12_map(Heff)
        out["J_mu"] = spin["J_mu"]
        out["mapping_residual"] = spin["mapping_residual"]
        logger.info(
            "fopt.run_fopt: spin-1/2 mapping residual=%.2e", spin["mapping_residual"],
        )

    return out
