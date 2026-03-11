# 02-06-NON_KRAMERS_DOUBLET

This file defines the non-Kramers doublet contract for even-electron `f^n`
cases.
Kramers-doublet rules for odd `n` remain in
`./standards/en/02-models/02-05-KRAMERS_DOUBLET_G_TENSOR.md`.
Wannier90-derived input constraints are defined in
`./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`.

## 1) Scope (MUST)
MUST:
- Apply only to even-electron sectors (`n % 2 == 0`).
- Target is a CEF low-energy doublet (exact or quasi-degenerate).
- This file defines projection/gauge/output rules, not the CEF model formula.

Code form:
```text
require n % 2 == 0
inputs = {H_local, Jx, Jy, Jz, multipole_ops, symmetry_info}
```

Validation:
- If `n` is odd, use `./standards/en/02-models/02-05-KRAMERS_DOUBLET_G_TENSOR.md` instead.

## 2) Doublet Selection (MUST)
MUST:
- Select two lowest states in the target manifold and define splitting:

Math:
$$
\Delta_{\mathrm{nk}} \equiv E_2 - E_1.
$$

- Record whether it is exact/near-degenerate by threshold `eps_nk_split`.
  Default `eps_nk_split` is inherited from
  `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`.

Code form:
```text
is_quasi_doublet = (Delta_nk <= eps_nk_split)
```

Validation:
- `Delta_nk` and `eps_nk_split` must be recorded in metadata.

## 3) Projection to Pseudospin Space (MUST)
MUST:
- Define projector:

Math:
$$
P=\lvert \psi_1\rangle\langle\psi_1\rvert+\lvert \psi_2\rangle\langle\psi_2\rvert.
$$

- For any operator `O`, use:

Math:
$$
M_O \equiv P O P
= o_0 I + o_x \tau^x + o_y \tau^y + o_z \tau^z.
$$

Code form:
```text
M_O = Psi.conj().T @ O @ Psi
coeffs {o0,ox,oy,oz} from Pauli decomposition
```

Validation:
- All projected `M_O` must be Hermitian within `eps_herm`.

## 4) Magnetic vs Electric Channel Rule (MUST)
MUST:
- Define longitudinal axis (`c`) and in-plane axes (`a,b`).
- Magnetic channel is defined by projected dipole operators (`Jx/Jy/Jz`).
- Electric channel is defined by projected TR-even multipole operators
  (quadrupole/octupole/... from `multipole_ops` input set).

For the common case “magnetic on `c`, electric on `ab`”, enforce:

Math:
$$
\|P J_c P\|_F \gg \|P J_a P\|_F,\ \|P J_b P\|_F.
$$

Math:
$$
\mathrm{rank}\left(\{P Q_m P\}_{Q_m\in\mathcal Q_{ab}}\right)\ge 2.
$$

Code form:
```text
mag_ratio_a = ||PJaP||_F / max(||PJcP||_F, eps_zero)
mag_ratio_b = ||PJbP||_F / max(||PJcP||_F, eps_zero)
require mag_ratio_a <= eps_mag_ab and mag_ratio_b <= eps_mag_ab
require two independent in-plane electric channels
```

`eps_mag_ab` default is inherited from
`./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`.

Validation:
- If magnetic/electric channel conditions fail, mark model-tag mismatch.

## 5) Canonical Pseudospin Gauge (MUST)
MUST:
- Fix gauge deterministically so exported pseudospin is stable:
1. Choose `tau^z` from normalized `P J_c P`.
2. Choose `tau^x,tau^y` from two independent projected in-plane electric channels
   using deterministic orthonormalization order.
3. Fix residual sign/order ambiguity by deterministic tie-break
   (`z` first, then `x`, then `y`).

Code form:
```text
tau_z <- normalize(PJcP)
tau_x, tau_y <- orthonormalize(PQ1P, PQ2P) with fixed order
apply deterministic sign convention
```

Validation:
- Same input must yield identical pseudospin axis/sign convention.

## 6) Output Contract (MUST)
MUST output:
- `doublet_vectors` (`psi1`, `psi2`).
- `Delta_nk`.
- Projected dipole matrices: `M_Jx`, `M_Jy`, `M_Jz`.
- Selected projected electric-channel matrices: `M_Q1`, `M_Q2` (and labels).
- Pseudospin mapping coefficients for dipole and electric channels.
- `gauge_meta` and all thresholds used.
- `symmetry_meta` when irrep classification is enabled (see `02-07`).
  - Contract fields: `irrep_display`, `irrep_primary`, `irrep_aliases`,
    `mapping_unverified`, `allowed_multipoles`, `excited_irreps`.
  - For inversion-containing groups, parity is uniquely determined by `J`
    (no dual parity branches).

Code form:
```text
outputs = {
  doublet_vectors, Delta_nk,
  M_Jx, M_Jy, M_Jz,
  M_Q1, M_Q2, Q_labels,
  map_dipole, map_electric,
  gauge_meta, threshold_meta,
  symmetry_meta?
}
```

Validation:
- Output metadata must include `basis_id` and `orbital_order_id`.
