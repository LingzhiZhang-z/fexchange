# fexchange

Fock-basis fermionic operators and a second‑order perturbation (SOPT) pipeline driven by files on disk.

## Status

Working modules:
- `core/` (Fock basis, fermion operators)
- `api/` (types, keys, IO)
- `sopt/` (caches, truncation, contraction, export)
- `utils/cli_sopt.py` (end‑to‑end SOPT)

Stubs (placeholders only):
- `local/` (LSMS/alpha/LSJM construction)
- `cef/` (CEF Stevens + ground‑manifold extraction)
- `utils/cli_singlesite.py`

## File Formats

### StateSet (`.npz`)
Keys:
- `V_fock`: `(dim_fock, N)` eigenvectors in the Fock basis
- `labels_json`: JSON string (or empty string)
- `meta_json`: JSON string

### EnergySet (`.npz`)
Keys:
- `E_total`: `(N,)`
- `E_parts_json`: JSON map `{name: key}` for arrays stored separately
- `meta_json`: JSON string
- additional arrays referenced by `E_parts_json` (e.g. `part__Coulomb`)

### GroundManifold (`.npz`)
Keys:
- `G_fock`: `(dim_fock, d)` ground‑manifold vectors in the Fock basis
- `E0`: float
- `degeneracy`: int
- `tol`: float
- `meta_json`: JSON string

## CLI: SOPT

Example (paths are illustrative):

```bash
fexchange-sopt \
  --n 2 \
  --t-npy /path/to/t.npy \
  --ground-npz /path/to/ground.npz \
  --lsjm-np1-npz /path/to/lsjm_np1.npz \
  --e-np1-npz /path/to/e_np1.npz \
  --lsjm-nm1-npz /path/to/lsjm_nm1.npz \
  --e-nm1-npz /path/to/e_nm1.npz \
  --cache-dir /path/to/cache \
  --out-dir /path/to/out
```

### Parameter File

You can pass all parameters via a JSON file using `--config`. Relative paths are resolved relative to the JSON file location.

Example `params.json`:

```json
{
  "n": 2,
  "t_npy": "t.npy",
  "ground_npz": "ground.npz",
  "lsjm_np1_npz": "lsjm_np1.npz",
  "e_np1_npz": "e_np1.npz",
  "lsjm_nm1_npz": "lsjm_nm1.npz",
  "e_nm1_npz": "e_nm1.npz",
  "cache_dir": "cache",
  "out_dir": "out",
  "np1_nstates": null,
  "np1_emax": null,
  "nm1_nstates": null,
  "nm1_emax": null
}
```

Run:

```bash
fexchange-sopt --config /path/to/params.json
```

Optional truncation flags:
- `--np1-nstates`, `--np1-emax`
- `--nm1-nstates`, `--nm1-emax`

Outputs:
- `Heff.npy`
- `meta.json`

## Notes

- `core/orbitals.py` defines a fixed ordering of the 14 spin‑orbitals (m = −3..3, sigma = −1/2 then +1/2).
- Two programs communicate only via files: a single‑site solver writes LSJM/ground files, and the SOPT CLI reads them.
