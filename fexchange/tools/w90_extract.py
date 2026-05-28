"""Wannier90 hr.dat -> fexchange basis two-bond extractor.

Reads one Wannier90 ``hr.dat`` for a 4-site cluster (2 f-sites + 2 ligand p-sites,
each with its own R-vector), and emits three multi-block text files compatible
with :func:`fexchange.io.matrix.load_txt_blocks`:

* ``onsite_out``      — blocks ``[h_f1] [h_f2] [h_lig1] [h_lig2]``.
* ``hopping_fp_out``  — blocks ``[t_f1_lig1] [t_f1_lig2] [t_f2_lig1] [t_f2_lig2]``
  (the FOPT contract from ``standards/05-04-RUN_INPUT.md §5``).
* ``hopping_ff_out``  — single block ``[t_mu]`` of the direct f1↔f2 hopping
  (the SOPT contract).

All matrices are in the fexchange complex spherical-harmonic basis with spin
(σ=−1/2, σ=+1/2) interleaved per m, matching
``fexchange/core/space_ls.py::orbital_map``.

Wannier90 default real-harmonic order (User Guide §3.1):

* ``l=1`` (p):  ``pz, px, py``                                ⇒ m_l = ``[0, +1, −1]``
* ``l=3`` (f):  ``fz3, fxz², fyz², fz(x²−y²), fxyz,
                 fx(x²−3y²), fy(3x²−y²)``                     ⇒ m_l = ``[0, +1, −1, +2, −2, +3, −3]``

If the ``.win`` file uses a different projection order, override
``real_order_f`` / ``real_order_p`` accordingly.

This function does **raw bilinear extraction only** — no trace removal, no
Hermitization, no f–p–f downfolding. Those concerns belong to downstream
CEF/FOPT prep tools.
"""

from __future__ import annotations

import argparse
import logging
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fexchange.utils.errors import InputError, IOError_, PhysError, W90Error

logger = logging.getLogger("fexchange")

ELL_F: int = 3
ELL_P: int = 1
DIM_F_REAL: int = 2 * ELL_F + 1
DIM_P_REAL: int = 2 * ELL_P + 1

W90_F_REAL_ORDER: tuple[int, ...] = (0, +1, -1, +2, -2, +3, -3)
W90_P_REAL_ORDER: tuple[int, ...] = (0, +1, -1)

_SPIN_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# hr.dat parser
# ---------------------------------------------------------------------------

def read_w90_hr(
    path: str | Path,
) -> tuple[dict[tuple[int, int, int], NDArray[np.complexfloating]], int]:
    """Parse ``wannier90_hr.dat`` -> ``({R: H(R)}, num_wann)``.

    Each H(R) block is divided by its degeneracy weight so the returned
    matrices are the bare real-space Hamiltonian.
    """
    p = Path(path)
    if not p.exists():
        raise IOError_("FXE-IO-001", f"hr file missing: {p}", paths={"path": str(p)})

    num_wann = int(np.loadtxt(str(p), skiprows=1, max_rows=1))
    nrpts = int(np.loadtxt(str(p), skiprows=2, max_rows=1))

    n_wt_lines, rem = divmod(nrpts, 15)
    weights = (
        np.loadtxt(str(p), skiprows=3, max_rows=n_wt_lines).flatten()
        if n_wt_lines > 0
        else np.array([], dtype=float)
    )
    if rem != 0:
        extra = np.atleast_1d(np.loadtxt(str(p), skiprows=3 + n_wt_lines, max_rows=1))
        weights = np.concatenate([weights, extra])
        n_wt_lines += 1
    if weights.size != nrpts:
        raise W90Error(
            "FXE-W90-001",
            "weight count does not match nrpts",
            expected={"nrpts": nrpts},
            actual={"weights": int(weights.size)},
            paths={"path": str(p)},
        )

    body = np.loadtxt(str(p), skiprows=3 + n_wt_lines)
    block = num_wann * num_wann

    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]] = {}
    for i in range(nrpts):
        rows = body[block * i : block * (i + 1)]
        R = tuple(int(x) for x in rows[0, 0:3])
        m_idx = rows[:, 3].astype(int) - 1
        n_idx = rows[:, 4].astype(int) - 1
        mat = np.zeros((num_wann, num_wann), dtype=complex)
        mat[m_idx, n_idx] = (rows[:, 5] + 1j * rows[:, 6]) / weights[i]
        H_R[R] = mat

    return H_R, num_wann


# ---------------------------------------------------------------------------
# Basis transforms
# ---------------------------------------------------------------------------

def _real_to_complex_unitary(ell: int, real_order: tuple[int, ...]) -> NDArray[np.complexfloating]:
    """Build the (2ℓ+1)×(2ℓ+1) unitary ``U`` for one orbital shell.

    Convention: column ``r`` holds the expansion of the r-th Wannier90 real
    spherical harmonic in the complex spherical-harmonic basis ordered
    ``m = −ℓ..+ℓ``. The W90 real harmonic at position r has m_l = ``real_order[r]``.

    Real-harmonic ↔ complex-harmonic definitions (Condon-Shortley phase):

      m_l = 0    →   Y_l^0
      m_l = +|m| →  (Y_l^{−|m|} + (−1)^m Y_l^{+|m|}) / √2
      m_l = −|m| →  i (Y_l^{−|m|} − (−1)^m Y_l^{+|m|}) / √2

    A Hamiltonian transforms as ``h_complex = U @ h_real @ U.conj().T``.
    """
    dim = 2 * ell + 1
    if sorted(real_order) != list(range(-ell, ell + 1)):
        raise InputError(
            "FXE-INPUT-003",
            f"real_order must be a permutation of [-{ell}..+{ell}]",
            actual={"real_order": list(real_order)},
        )
    sq2 = np.sqrt(2.0)
    U = np.zeros((dim, dim), dtype=complex)
    for r_idx, m_real in enumerate(real_order):
        if m_real == 0:
            U[ell, r_idx] = 1.0
        elif m_real > 0:
            phase = (-1) ** m_real
            U[ell - m_real, r_idx] = 1.0 / sq2
            U[ell + m_real, r_idx] = phase / sq2
        else:
            absm = -m_real
            phase = (-1) ** absm
            U[ell - absm, r_idx] = 1j / sq2
            U[ell + absm, r_idx] = -1j * phase / sq2
    return U


def _shell_transform(shell: str, real_order: tuple[int, ...], *, spinor_input: bool) -> NDArray[np.complexfloating]:
    """Full spin–orbital transform U_full for one site.

    Applied as ``h_fxe = U_full @ h_W90 @ U_full.conj().T`` when the input is
    already a spinor block. For NSOC input the caller first Kronecker-embeds
    the orbital matrix into the spinor layout (see :func:`_to_fxe`).

    Spin convention:
      * W90 (spinor=True, default ``proj=X: l``): per orbital ``(↑, ↓)`` pair.
      * fxe (``space_ls.orbital_map``): per m, ``(σ=−½, σ=+½)`` pair (i.e. ``(↓, ↑)``).
    We compose orbital real→complex with a spin swap ``[[0,1],[1,0]]`` so a
    single ``U_full`` does everything.
    """
    if shell == "f":
        ell = ELL_F
    elif shell == "p":
        ell = ELL_P
    else:
        raise InputError(
            "FXE-INPUT-003",
            f"Unsupported shell: {shell!r} (only 'f' and 'p' implemented)",
            actual={"shell": shell},
        )
    U_orb = _real_to_complex_unitary(ell, real_order)
    spin_block = _SPIN_SWAP if spinor_input else np.eye(2, dtype=complex)
    return np.kron(U_orb, spin_block)


# ---------------------------------------------------------------------------
# Site index resolution
# ---------------------------------------------------------------------------

def _orbital_offsets(n_orb_per_atom: list[int], *, num_wann: int, spinor: bool) -> list[int]:
    """Compute cumulative Wannier-index offsets per atom.

    Accepts narrow counts (``[7, 7, 3, 3]`` for SOC) or already-wide counts
    (``[14, 14, 6, 6]``); both reconcile against ``num_wann``.
    """
    narrow = [int(x) for x in n_orb_per_atom]
    if sum(narrow) == num_wann:
        return [sum(narrow[:i]) for i in range(len(narrow) + 1)]
    if spinor and 2 * sum(narrow) == num_wann:
        wide = [2 * c for c in narrow]
        return [sum(wide[:i]) for i in range(len(wide) + 1)]
    raise InputError(
        "FXE-INPUT-003",
        "Cannot reconcile n_orb_per_atom with num_wann",
        expected={"sum_or_2x_sum": num_wann},
        actual={"sum": sum(narrow), "spinor": spinor, "num_wann": num_wann},
    )


def _site_indices(offsets: list[int], atom: int, *, expected: int) -> list[int]:
    if not 0 <= atom < len(offsets) - 1:
        raise InputError(
            "FXE-INPUT-003",
            f"atom index out of range: {atom} (have {len(offsets) - 1} atoms)",
            actual={"atom": atom},
        )
    got = offsets[atom + 1] - offsets[atom]
    if got != expected:
        raise W90Error(
            "FXE-W90-001",
            f"atom {atom} has {got} W90 orbitals; expected {expected}",
            expected={"orbitals": expected},
            actual={"orbitals": got},
        )
    return list(range(offsets[atom], offsets[atom + 1]))


def _resolve_site(
    site: dict[str, Any],
    *,
    label: str,
    shell: str,
    offsets: list[int],
    spinor: bool,
) -> dict[str, Any]:
    if "atom" not in site or "cell" not in site:
        raise InputError(
            "FXE-INPUT-003",
            f"{label}: missing required field atom/cell",
            actual={"site": site},
        )
    cell = tuple(int(x) for x in site["cell"])
    if len(cell) != 3:
        raise InputError(
            "FXE-INPUT-003",
            f"{label}: cell must contain exactly 3 integers",
            actual={"cell": list(cell)},
        )
    per_orb = DIM_F_REAL if shell == "f" else DIM_P_REAL
    expected_wann = 2 * per_orb if spinor else per_orb
    indices = _site_indices(offsets, int(site["atom"]), expected=expected_wann)
    return {
        "label": label,
        "shell": shell,
        "atom": int(site["atom"]),
        "cell": cell,
        "indices": indices,
    }


# ---------------------------------------------------------------------------
# Block extraction with Hermitian fallback
# ---------------------------------------------------------------------------

def _extract_block(
    H_R: dict[tuple[int, int, int], NDArray[np.complexfloating]],
    rows: list[int],
    cols: list[int],
    R: tuple[int, int, int],
) -> NDArray[np.complexfloating]:
    """Extract ``H_R[R][rows, cols]``; fall back to ``H_R[-R].conj().T`` if R absent."""
    R = tuple(int(x) for x in R)
    if R in H_R:
        return H_R[R][np.ix_(rows, cols)].copy()
    R_neg = tuple(-x for x in R)
    if R_neg in H_R:
        return H_R[R_neg][np.ix_(cols, rows)].conj().T.copy()
    raise W90Error(
        "FXE-W90-001",
        f"R-vector {R} not present in hr.dat (neither R nor -R)",
        actual={"R": list(R)},
    )


# ---------------------------------------------------------------------------
# Multi-block text writer
# ---------------------------------------------------------------------------

def _write_blocks_txt(path: Path, blocks: list[tuple[str, NDArray[np.complexfloating]]]) -> None:
    """Write blocks in the ``[key] / real imag`` format read by ``io.matrix.load_txt_blocks``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for key, mat in blocks:
        flat = np.asarray(mat, dtype=complex).reshape(-1)
        out.append(f"[{key}]")
        out.extend(f"{val.real:.12e} {val.imag:.12e}" for val in flat)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------

def extract_w90_two_bond(
    *,
    hr_path: str | Path,
    n_orb_per_atom: list[int],
    f1_site: dict[str, Any],
    f2_site: dict[str, Any],
    lig1_site: dict[str, Any],
    lig2_site: dict[str, Any],
    spinor: bool,
    onsite_out: str | Path,
    hopping_fp_out: str | Path,
    hopping_ff_out: str | Path,
    energy_unit: str = "eV",
    fermi_level: float = 0.0,
    f_trace_tol: float = 1e-6,
    real_order_f: tuple[int, ...] = W90_F_REAL_ORDER,
    real_order_p: tuple[int, ...] = W90_P_REAL_ORDER,
) -> dict[str, Any]:
    """Extract a 4-site (2 f + 2 p) cluster from Wannier90 ``hr.dat``.

    Parameters
    ----------
    hr_path: path to ``wannier90_hr.dat``.
    n_orb_per_atom: orbital counts per atom in the W90 projection. Either
        narrow (e.g. ``[7, 7, 3, 3]``) or already spin-doubled (``[14, 14, 6, 6]``);
        both reconcile against ``num_wann``.
    f1_site, f2_site, lig1_site, lig2_site: each is ``{"atom": int, "cell": [i,j,k]}``.
    spinor: whether ``hr.dat`` was produced with ``spinors = .true.``.
    onsite_out, hopping_fp_out, hopping_ff_out: paths for the three output text files.
    energy_unit: metadata label for the raw input/output energy unit.
    fermi_level: subtracted from the onsite diagonal in the same raw unit
        (``onsite_out_diag - fermi_level``). Hopping blocks are unaffected.
        Default 0.0 (no shift). Set to the DFT Fermi energy in the same unit as
        ``hr.dat`` when ``hr.dat`` does not pre-align E_F to zero.
        Note: the reported ligand Delta = ⟨h_f⟩ − ⟨h_lig⟩ is invariant under
        this shift (both ends shift by the same constant).
    f_trace_tol: max allowed |⟨h_f1⟩ − ⟨h_f2⟩| (in the raw unit). Two f
        sites should be physically equivalent, hence have equal trace
        averages. Default 1e-6.
    real_order_f, real_order_p: Wannier90 m_l ordering for each shell (default
        is the W90 hybrid order; override when ``.win`` uses non-default
        projections).

    Returns
    -------
    A dict ``{"onsite": ..., "hopping": ..., "meta": ...}`` echoing the
    in-memory matrices and metadata.
    """
    if not isinstance(energy_unit, str) or not energy_unit.strip():
        raise InputError(
            "FXE-INPUT-003",
            "energy_unit must be a non-empty string",
            actual={"energy_unit": energy_unit},
        )
    energy_unit = energy_unit.strip()

    H_R, num_wann = read_w90_hr(hr_path)

    offsets = _orbital_offsets(n_orb_per_atom, num_wann=num_wann, spinor=spinor)

    sites = [
        _resolve_site(f1_site, label="f1", shell="f", offsets=offsets, spinor=spinor),
        _resolve_site(f2_site, label="f2", shell="f", offsets=offsets, spinor=spinor),
        _resolve_site(lig1_site, label="lig1", shell="p", offsets=offsets, spinor=spinor),
        _resolve_site(lig2_site, label="lig2", shell="p", offsets=offsets, spinor=spinor),
    ]
    by_label = {s["label"]: s for s in sites}

    U_for = {
        "f": _shell_transform("f", real_order_f, spinor_input=spinor),
        "p": _shell_transform("p", real_order_p, spinor_input=spinor),
    }

    def _to_fxe(raw: NDArray[np.complexfloating], shell_row: str, shell_col: str) -> NDArray[np.complexfloating]:
        if not spinor:
            # Promote NSOC orbital block to fxe (orbital, spin) interleaved layout.
            raw = np.kron(raw, np.eye(2, dtype=complex))
        return U_for[shell_row] @ raw @ U_for[shell_col].conj().T

    H0 = H_R.get((0, 0, 0))
    if H0 is None:
        raise W90Error(
            "FXE-W90-001",
            "hr.dat is missing the R=(0,0,0) block (required for onsite extraction)",
        )

    onsite: dict[str, NDArray[np.complexfloating]] = {}
    for s in sites:
        raw = H0[np.ix_(s["indices"], s["indices"])].copy()
        h_fxe = _to_fxe(raw, s["shell"], s["shell"])
        if fermi_level != 0.0:
            h_fxe = h_fxe - fermi_level * np.eye(h_fxe.shape[0], dtype=complex)
        onsite[s["label"]] = h_fxe

    hopping_fp: dict[tuple[str, str], NDArray[np.complexfloating]] = {}
    for f_lbl in ("f1", "f2"):
        for lig_lbl in ("lig1", "lig2"):
            f = by_label[f_lbl]
            lig = by_label[lig_lbl]
            R_fl = tuple(int(lig["cell"][k]) - int(f["cell"][k]) for k in range(3))
            raw = _extract_block(H_R, f["indices"], lig["indices"], R_fl)
            hopping_fp[(f_lbl, lig_lbl)] = _to_fxe(raw, "f", "p")

    f1, f2 = by_label["f1"], by_label["f2"]
    R_ff = tuple(int(f2["cell"][k]) - int(f1["cell"][k]) for k in range(3))
    raw_ff = _extract_block(H_R, f1["indices"], f2["indices"], R_ff)
    t_ff = _to_fxe(raw_ff, "f", "f")

    # Trace averages (unitary-invariant; invariant under fermi shift across sites).
    def _trace_avg(h: NDArray[np.complexfloating]) -> float:
        return float(np.trace(h).real / h.shape[0])

    e_avg = {lbl: _trace_avg(onsite[lbl]) for lbl in ("f1", "f2", "lig1", "lig2")}
    f_trace_diff = abs(e_avg["f1"] - e_avg["f2"])
    if f_trace_diff > f_trace_tol:
        raise PhysError(
            "FXE-PHYS-001",
            f"f1 and f2 onsite trace averages differ by {f_trace_diff:.6e} ({energy_unit}) > tol {f_trace_tol:.6e}",
            expected={"f_trace_diff_max": float(f_trace_tol)},
            actual={"f1_avg": e_avg["f1"], "f2_avg": e_avg["f2"], "diff": f_trace_diff},
        )
    e_f = 0.5 * (e_avg["f1"] + e_avg["f2"])
    # Delta = E_f - E_lig (ligand-to-f charge-transfer cost; positive when
    # the ligand p-shell sits below f). Matches the convention used by
    # fexchange/spectrum/ligand.py::build_ligand_spectrum.
    delta = {"lig1": e_f - e_avg["lig1"], "lig2": e_f - e_avg["lig2"]}

    onsite_path = Path(onsite_out)
    hopping_fp_path = Path(hopping_fp_out)
    hopping_ff_path = Path(hopping_ff_out)
    _write_blocks_txt(
        onsite_path,
        [(f"h_{lbl}", onsite[lbl]) for lbl in ("f1", "f2", "lig1", "lig2")],
    )
    _write_blocks_txt(
        hopping_fp_path,
        [
            (f"t_{f_lbl}_{lig_lbl}", hopping_fp[(f_lbl, lig_lbl)])
            for f_lbl in ("f1", "f2")
            for lig_lbl in ("lig1", "lig2")
        ],
    )
    _write_blocks_txt(hopping_ff_path, [("t_mu", t_ff)])
    logger.info(
        "extract_w90_two_bond wrote: onsite=%s hopping_fp=%s hopping_ff=%s (num_wann=%d, spinor=%s)",
        onsite_path,
        hopping_fp_path,
        hopping_ff_path,
        num_wann,
        spinor,
    )

    meta = {
        "hr_path": str(hr_path),
        "num_wann": int(num_wann),
        "spinor": bool(spinor),
        "energy_unit": energy_unit,
        "fermi_level": float(fermi_level),
        "n_orb_per_atom": [int(x) for x in n_orb_per_atom],
        "real_order_f": list(real_order_f),
        "real_order_p": list(real_order_p),
        "sites": {
            s["label"]: {"atom": s["atom"], "cell": list(s["cell"]), "shell": s["shell"]}
            for s in sites
        },
        "onsite_out": str(onsite_path),
        "hopping_fp_out": str(hopping_fp_path),
        "hopping_ff_out": str(hopping_ff_path),
        "trace_avg": e_avg,
        "e_f_ref": float(e_f),
        "delta": {lbl: float(delta[lbl]) for lbl in ("lig1", "lig2")},
        "f_trace_tol": float(f_trace_tol),
        "f_trace_diff": float(f_trace_diff),
    }
    return {
        "onsite": onsite,
        "hopping_fp": hopping_fp,
        "hopping_ff": t_ff,
        "delta": delta,
        "trace_avg": e_avg,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# TOML CLI entry point
# ---------------------------------------------------------------------------

def extract_from_toml(toml_path: str | Path) -> dict[str, Any]:
    """Load a config file and invoke :func:`extract_w90_two_bond`.

    Config schema::

        hr_path = "wannier90_hr.dat"
        spinor = true
        energy_unit = "eV"
        n_orb_per_atom = [7, 7, 3, 3]
        onsite_out = "out/onsite.txt"
        hopping_fp_out = "out/hopping_fp.txt"
        hopping_ff_out = "out/hopping_ff.txt"

        # Optional: subtract DFT Fermi energy from onsite diagonals (raw unit)
        # fermi_level = 5.123

        # Optional W90 real-harmonic order overrides
        # real_order_f = [0, 1, -1, 2, -2, 3, -3]
        # real_order_p = [0, 1, -1]

        [f1_site]
        atom = 0
        cell = [0, 0, 0]

        [f2_site]
        atom = 1
        cell = [1, 0, 0]

        [lig1_site]
        atom = 2
        cell = [0, 0, 0]

        [lig2_site]
        atom = 3
        cell = [0, 0, 0]
    """
    path = Path(toml_path)
    if not path.exists():
        raise IOError_("FXE-IO-001", f"config not found: {path}", paths={"path": str(path)})
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    required = (
        "hr_path",
        "n_orb_per_atom",
        "spinor",
        "f1_site",
        "f2_site",
        "lig1_site",
        "lig2_site",
        "onsite_out",
        "hopping_fp_out",
        "hopping_ff_out",
    )
    missing = [k for k in required if k not in cfg]
    if missing:
        raise InputError(
            "FXE-INPUT-003",
            f"config missing fields: {missing}",
            expected={"required": list(required)},
            actual={"present": sorted(cfg.keys())},
        )
    kwargs: dict[str, Any] = dict(
        hr_path=cfg["hr_path"],
        n_orb_per_atom=cfg["n_orb_per_atom"],
        f1_site=cfg["f1_site"],
        f2_site=cfg["f2_site"],
        lig1_site=cfg["lig1_site"],
        lig2_site=cfg["lig2_site"],
        spinor=bool(cfg["spinor"]),
        onsite_out=cfg["onsite_out"],
        hopping_fp_out=cfg["hopping_fp_out"],
        hopping_ff_out=cfg["hopping_ff_out"],
        energy_unit=cfg.get("energy_unit", "eV"),
        fermi_level=float(cfg.get("fermi_level", 0.0)),
        f_trace_tol=float(cfg.get("f_trace_tol", 1e-6)),
    )
    if "real_order_f" in cfg:
        kwargs["real_order_f"] = tuple(int(x) for x in cfg["real_order_f"])
    if "real_order_p" in cfg:
        kwargs["real_order_p"] = tuple(int(x) for x in cfg["real_order_p"])
    return extract_w90_two_bond(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract fexchange onsite, f-p, and f-f blocks from Wannier90 hr.dat.")
    parser.add_argument("config", help="TOML config path")
    args = parser.parse_args(argv)
    extract_from_toml(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
