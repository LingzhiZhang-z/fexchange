# 05-03-WANNIER90_PARSING

This file defines deterministic parsing and mapping rules for Wannier90 inputs.
It is the parsing companion of `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`.

## 1) Scope (MUST)
MUST:
- Apply when `hopping_source = "wannier90"`.
- Cover file parsing, atom/orbital/spin mapping, unit metadata, and order checks.
- Do not define SOPT contraction formulas.

Code form:
```text
if hopping_source != "wannier90": skip this file
```

Validation:
- If this mode is active, all gates in this file are mandatory.

## 2) Required Input Files (MUST)
MUST:
- Required files:
  - `wannier90_hr.dat`
  - `wannier.win`
- Optional helper files may exist, but must not replace required files.

Code form:
```text
required_w90_files = {hr_path, win_path}
```

Validation:
- Missing file is `FXE-W90-001`.

### 2.1) `wannier90_hr.dat` File Format (MUST)
The file layout is:
```text
Line 1:    comment string (free-form, ignored by parser)
Line 2:    num_wann                          (integer)
Line 3:    nrpts                             (integer, number of R-vectors)
Next ceil(nrpts/15) lines:
           degeneracy weights, 15 integers per line
Data block (nrpts * num_wann^2 lines, each):
           R1  R2  R3  i  j  Re(H_ij(R))  Im(H_ij(R))
```

Physical definition of each matrix element:

Math:
$$
H_{ij}(\mathbf R)
=
\langle \mathbf 0,\,i \rvert \hat H \lvert \mathbf R,\,j \rangle,
$$

where $\lvert \mathbf R,\,j\rangle$ is the $j$-th Wannier function centered at
lattice vector $\mathbf R$, and $\lvert \mathbf 0,\,i\rangle$ is the $i$-th
Wannier function at the home cell.
When $\mathbf R=\mathbf 0$ and $i=j$, $H_{ii}(\mathbf 0)$ is the onsite energy
$\epsilon_i=\langle \mathbf 0,\,i\rvert\hat H\lvert\mathbf 0,\,i\rangle$.

Parsing rules:
- `R = (R1, R2, R3)` is the integer lattice-vector index.
- Onsite block corresponds to `R = (0,0,0)`.
- `i, j` are **1-indexed** Wannier function indices.
- Hamiltonian value: `H_ij(R) = Re + i*Im` in the energy unit declared by `energy_unit`.
- Each `H_ij(R)` must be divided by its R-vector degeneracy weight.
- The data block is ordered: outer loop over R-vectors, inner loop over `(i,j)` with `j` fast.

Code form:
```text
for r_idx in range(nrpts):
  for j in range(1, num_wann+1):
    for i in range(1, num_wann+1):
      read R1, R2, R3, i, j, re_h, im_h
      H[R][(i-1, j-1)] = complex(re_h, im_h) / weight[r_idx]
```

### 2.2) `wannier.win` Minimal Required Fields (MUST)
Parser must extract at minimum:
- `num_wann`: number of Wannier functions.
- `begin projections` / `end projections`: atom sites and orbital types.
- `begin atoms_frac` / `end atoms_frac` (or `atoms_cart`): atomic positions.
- `begin unit_cell_cart` / `end unit_cell_cart`: lattice vectors (needed for R-vector interpretation).

The `.win` file uses Fortran-style key-value format (`keyword = value`) with block
sections delimited by `begin`/`end` tags.

Code form:
```text
num_wann     = parse_int(win, "num_wann")
projections  = parse_block(win, "projections")
atoms        = parse_block(win, "atoms_frac") or parse_block(win, "atoms_cart")
unit_cell    = parse_block(win, "unit_cell_cart")
```

Validation:
- Missing `num_wann` or projection block is `FXE-W90-001`.
- `num_wann` in `.win` must match `num_wann` in `_hr.dat`; mismatch is `FXE-W90-003`.

## 3) Atom-Site Selection Rules (MUST)
MUST:
- `f_site_i != f_site_j`.
- `f_site_i_cell` and `f_site_j_cell` must be integer triplets.
- `ligand_indices` may be empty (direct `f-f` only, no ligand-mediated correction).
- if non-empty, `ligand_indices` must exclude `f_site_i/f_site_j`.
- `ligand_cells` must be provided with one integer triplet per `ligand_indices` entry.
- `all_wannier_atom_indices` must include all selected sites.
- Duplicated atom index in any site list is forbidden.

Code form:
```text
require f_site_i != f_site_j
require disjoint({f_site_i,f_site_j}, set(ligand_indices))
require subset({f_site_i,f_site_j} ∪ ligand_indices, all_wannier_atom_indices)
require len(f_site_i_cell) == 3 and len(f_site_j_cell) == 3
require len(ligand_cells) == len(ligand_indices)
for cell in ligand_cells: require len(cell) == 3

R_ij = f_site_j_cell - f_site_i_cell
for each ligand o:
  R_io[o] = ligand_cells[o] - f_site_i_cell
  R_jo[o] = ligand_cells[o] - f_site_j_cell
```

Validation:
- Violations are `FXE-W90-002`.

Code form:
```text
if len(ligand_indices) == 0:
  use_direct_ff_only = true
  ligand_correction = 0
```

Validation:
- Invalid cell-triplet binding or inconsistent ligand-cell length is `FXE-W90-002`.
- Derived relative vectors must be recorded in metadata.

### 3.1) Relative-R Fetch and Hermitian Completion Rule (MUST)
MUST:
- All hopping reads must use relative lattice vectors derived in Section 3
  (`R_ij`, `R_io`, `R_jo`), regardless of whether any selected site lies in `000`.
- Parser/loader must support Hermitian completion for missing direct entries:

Math:
$$
H_{mn}(\mathbf R)=H_{nm}^{\ast}(-\mathbf R).
$$

- If neither direct nor Hermitian-completed entry exists, fail hard.

Code form:
```text
def fetch_H(m, n, R):
  if exists(H[R][m,n]): return H[R][m,n]
  if exists(H[-R][n,m]): return conj(H[-R][n,m])
  fail(FXE-W90-002)
```

Validation:
- Missing required `R` entry without Hermitian completion is `FXE-W90-002`.
- Uniform common-cell shift invariance check is recommended:
  shift all selected cells by same vector, derived tensors remain unchanged.

## 4) Orbital Mapping Rules (MUST)
MUST:
- For `f` channels, orbital order id must be explicit (`orbital_order_id`).
- For default real-harmonic `f` order:
  `m = [0, 1, -1, 2, -2, 3, -3]`.
- Parser must produce explicit index maps:
  - `map_f_i[u] -> (atom, orbital, spin?)`
  - `map_f_j[v] -> (atom, orbital, spin?)`
  - `map_lig[o,p] -> (atom, orbital, spin?)`
- Ambiguous orbital labels are forbidden.

Code form:
```text
build map_f_i, map_f_j, map_lig with explicit deterministic ordering
```

Validation:
- Order/mapping mismatch is `FXE-W90-003`.

## 5) Spin Mapping Rules (MUST)
MUST:
- `soc_mode = "with_soc"`:
  - local index includes orbital+spin composite index.
  - spin-flip terms are read from Wannier payload directly.
- `soc_mode = "without_soc"`:
  - local index includes orbital only.
  - spinful expansion uses:
    - up block = raw Wannier payload
    - down block = complex conjugate of up block
    - spin-flip blocks = zero

Code form:
```text
if soc_mode == "with_soc":
  use composite (orbital,spin) indices directly
else:
  h_upup = h_w90
  h_dndn = conj(h_w90)
  h_updn = 0
  h_dnup = 0
```

Validation:
- Spin-mode mismatch is `FXE-W90-002`.

## 6) Unit Metadata, No Normalization (MUST)
MUST:
- Input unit metadata must be declared (`energy_unit`).
- Parsed Hamiltonian values must be consumed and emitted as raw values in the
  declared unit. The parser must not convert between `eV`, `meV`, `Ha`, `Ry`,
  or any other energy unit.
- Users are responsible for keeping Wannier90-derived files and runtime scalar
  inputs in one consistent unit.

Code form:
```text
H_raw = H_input
```

Validation:
- Missing or empty unit metadata is `FXE-W90-003`.

## 7) Hermiticity and Order Checks (MUST)
MUST:
- Parsed onsite/hopping blocks must pass Hermiticity tolerance checks where applicable.
- Parsed axis orders must be emitted in metadata:
  - `atom_order_id`
  - `orbital_order_id`
  - `spin_order_id`
  - `ligand_order_id`

Code form:
```text
check_hermitian(H, eps_herm)
emit order_ids in meta
```

Validation:
- Check failure is `FXE-NUM-001` or `FXE-W90-003`.

## 8) Parser Output Contract (MUST)
MUST:
- Parser output payload must include:
  - raw onsite/hopping blocks in the declared `energy_unit`
  - explicit mapping tables (`map_f_i`, `map_f_j`, `map_lig`)
  - cell bindings and derived relative vectors (`f_site_i_cell`, `f_site_j_cell`,
    `ligand_cells`, `R_ij`, `R_io`, `R_jo`)
  - order ids
  - source file hashes

Code form:
```text
w90_parsed = {
  H_blocks_eV, map_f_i, map_f_j, map_lig,
  f_site_i_cell, f_site_j_cell, ligand_cells, R_ij, R_io, R_jo,
  order_ids, file_hashes
}
```

Validation:
- Missing mapping tables is `FXE-W90-002`.
