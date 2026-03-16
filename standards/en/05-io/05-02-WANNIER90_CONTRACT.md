# 05-02-WANNIER90_CONTRACT

This file defines the input contract when hopping/CEF/Kramers data are derived
from DFT+Wannier90 outputs.

## 1) Scope and Modes (MUST)
MUST:
- Two source modes are allowed:
  - `source_mode = literature_params`
  - `source_mode = wannier90`
- This file constrains `source_mode = wannier90`.
- Deterministic file parsing and atom/orbital/spin mapping are defined in
  `./standards/en/05-io/05-03-WANNIER90_PARSING.md`.

Code form:
```text
if source_mode != "wannier90": skip this file
```

Validation:
- Mixed source modes in one artifact are forbidden unless explicitly tagged.

## 2) Required Wannier90 Metadata (MUST)
MUST:
- Record:
  - `soc_mode in {"with_soc","without_soc"}`
  - `orbital_basis = real_harmonic_default_w90`
  - `orbital_order_id`
  - `energy_unit` (default `eV`)
  - `spin_completion_rule` (required when `soc_mode="without_soc"` and spinful tensors are constructed)
- Record atom-index bindings:
  - `f_site_i`, `f_site_j`
  - `f_site_i_cell`, `f_site_j_cell` (integer triplets)
  - `ligand_indices`
  - `ligand_cells` (integer-triplet list aligned with `ligand_indices`)
  - `all_wannier_atom_indices`

Code form:
```text
required = {
  soc_mode, orbital_basis, orbital_order_id, energy_unit,
  f_site_i, f_site_j, f_site_i_cell, f_site_j_cell,
  ligand_indices, ligand_cells, all_wannier_atom_indices
}
if soc_mode == "without_soc" and spinful_required:
  require spin_completion_rule
require len(f_site_i_cell) == 3 and len(f_site_j_cell) == 3
require len(ligand_cells) == len(ligand_indices)
for cell in ligand_cells: require len(cell) == 3
```

Validation:
- Missing any required field is a hard failure.

## 2.1) Crystal-Cell Translation Contract (MUST)
MUST:
- Site-image hopping must be indexed by relative cell translation, not by absolute
  choice of the `000` cell.
- Let integer lattice-cell vectors be:
  `T_i = f_site_i_cell`, `T_j = f_site_j_cell`,
  and for ligand `o`, `T_o = ligand_cells[o]`.
- Define relative vectors:

Math:
$$
\mathbf R_{ij} = T_j - T_i,\qquad
\mathbf R_{io} = T_o - T_i,\qquad
\mathbf R_{jo} = T_o - T_j.
$$

- Direct $f$-$f$ hopping must sample `wannier90_hr.dat` at $\mathbf R_{ij}$.
- Ligand-mediated terms must sample $i\to o$ at $\mathbf R_{io}$ and
  $j\to o$ at $\mathbf R_{jo}$.
- Onsite terms use $\mathbf R=(0,0,0)$ in the selected local site image.
- If an entry is missing at $\mathbf R$, implementation may recover it from
  Hermitian completion:

Math:
$$
H_{mn}(\mathbf R)=H_{nm}^{\ast}(-\mathbf R).
$$

- If neither direct nor Hermitian-completed entry exists, fail hard.
- Uniform common-cell shift invariance must hold:
  shifting all selected cells by the same integer vector leaves results unchanged.

Code form:
```text
R_ij = f_site_j_cell - f_site_i_cell
R_io[o] = ligand_cells[o] - f_site_i_cell
R_jo[o] = ligand_cells[o] - f_site_j_cell

H_mn_R = fetch_hr(m, n, R)
if missing(H_mn_R):
  H_mn_R = conj(fetch_hr(n, m, -R))
if still_missing: fail(FXE-W90-002)
```

Validation:
- Metadata must record `{f_site_i_cell, f_site_j_cell, ligand_cells, R_ij}`.
- Missing-cell binding or invalid triplet lengths is a hard failure.

## 3) Effective f-f Hopping from Ligand p (MUST)
MUST:
- Distinguish `without_soc` and `with_soc` formulas.
- Build effective hopping as:
  - direct `f-f` hopping term `t^{(0)}`
  - plus ligand-mediated second-order correction.
- Ligand reduction MUST be performed in raw Wannier basis first; basis transforms
  (real/complex/cubic) are applied only after reduced `h/t` are obtained.
- Hopping matrix elements in this section are sampled using the relative-cell
  rule in Section 2.1.
- The physically exact denominator is channel-resolved; a reduced global
  denominator is an optional approximation defined in Section 4.
- Use index symbols:
  - `i, i^{\prime}`: two target `f` sites.
  - `u, v`: local `f`-manifold basis indices on sites `i, i^{\prime}`.
  - `o`: ligand-site index.
  - `p`: ligand orbital (channel) index on site `o`.
- Symbol scope in this file is local:
  - here `p` means ligand orbital index only;
  - do not confuse with SOPT `p,p',q,q'` in `04-00/04-01/04-02`, where they denote
    site-bound `f`-orbital indices after reduction.

### 3.1) `without_soc` Formula (MUST)
MUST:
- In `without_soc`, there is no spin-flip term.
- In this case, `u/v/p` contain orbital indices only (no spin index).
- Hopping carries an explicit spin index: `t_{i u,o p,\sigma}`.
- Effective hopping is written as `\tilde t = t^{(0)} + \delta t`.

Math:
$$
\tilde t^{\mathrm{nsoc}}_{i u,\,i^{\prime} v,\sigma}
=
t^{(0)}_{i u,\,i^{\prime} v,\sigma}
+
\sum_{o,p}
\frac{
 t_{i u,\,o p,\sigma}\,
 t_{i^{\prime} v,\,o p,\sigma}^{*}
}{
\Delta_{p-uv}
}.
$$

Math:
$$
H_{\mathrm{hop},ii^{\prime}}^{(\mu),\mathrm{nsoc}}
=
\sum_{u,v,\sigma}
\left(
 \tilde t^{\mathrm{nsoc}}_{i u,\,i^{\prime} v,\sigma}\,
 c^\dagger_{i u \sigma}c_{i^{\prime} v \sigma}
 + \mathrm{h.c.}
\right).
$$

Code form:
```text
for i, i_prime in f_site_pairs:
  for u in f_basis[i]:
    for v in f_basis[i_prime]:
      for sigma in spins:
        t_tilde_nsoc[i,u,i_prime,v,sigma] = t0_nsoc[i,u,i_prime,v,sigma]
        for o in ligand_sites:
          for p in ligand_orbitals[o]:
            t_tilde_nsoc[i,u,i_prime,v,sigma] += (
              t_nsoc[i,u,o,p,sigma] * conj(t_nsoc[i_prime,v,o,p,sigma]) / Delta_puv[p,u,v]
            )
        H_hop_nsoc += t_tilde_nsoc[i,u,i_prime,v,sigma] * cdag(i,u,sigma) * c(i_prime,v,sigma) + h.c.
```

### 3.2) `with_soc` Formula (MUST)
MUST:
- In `with_soc`, `u/v/p` are all composite orbital-spin indices.
- Do not write a separate explicit `\sigma` sum.
- Hopping is written as `t_{i u,o p}`.
- Effective hopping is written as `\tilde t = t^{(0)} + \delta t`.

Math:
$$
\tilde t^{\mathrm{soc}}_{i u,\,i^{\prime} v}
=
t^{(0)}_{i u,\,i^{\prime} v}
+
\sum_{o,p}
\frac{
 t_{i u,\,o p}\,
 t_{i^{\prime} v,\,o p}^{*}
}{
\Delta_{p-uv}
}.
$$

Math:
$$
H_{\mathrm{hop},ii^{\prime}}^{(\mu),\mathrm{soc}}
=
\sum_{u,v}
\left(
\tilde t^{\mathrm{soc}}_{i u,\,i^{\prime} v}\,
c^\dagger_{i u}c_{i^{\prime} v}
+ \mathrm{h.c.}
\right).
$$

Code form:
```text
for i, i_prime in f_site_pairs:
  for u in f_basis_soc[i]:
    for v in f_basis_soc[i_prime]:
      t_tilde_soc[i,u,i_prime,v] = t0_soc[i,u,i_prime,v]
      for o in ligand_sites:
        for p in ligand_basis_soc[o]:
          t_tilde_soc[i,u,i_prime,v] += (
            t_soc[i,u,o,p] * conj(t_soc[i_prime,v,o,p]) / Delta_puv[p,u,v]
          )
      H_hop_soc += t_tilde_soc[i,u,i_prime,v] * cdag(i,u) * c(i_prime,v) + h.c.
```

### 3.3) Spin Completion Rule for `without_soc` Wannier90 (MUST)
MUST:
- If Wannier90 payload is `without_soc` and downstream requires spinful tensors:
  - up-spin block uses raw Wannier90 output,
  - down-spin block uses complex conjugate of up-spin block,
  - spin-flip blocks are zero.

Math:
$$
h^{\uparrow\uparrow}=h^{\mathrm{w90}},\qquad
h^{\downarrow\downarrow}=\left(h^{\mathrm{w90}}\right)^*,\qquad
h^{\uparrow\downarrow}=h^{\downarrow\uparrow}=0.
$$

Code form:
```text
if soc_mode == "without_soc" and spinful_required:
  h_upup = h_w90
  h_dndn = conj(h_w90)
  h_updn = 0
  h_dnup = 0
```

Validation:
- Resulting hopping matrix must satisfy Hermiticity within `eps_herm`.

## 4) Delta (`\Delta_{p-uv}`) Policy (MUST)
MUST:
- Allow two input modes:
  - `delta_mode = manual`
  - `delta_mode = from_onsite`
- Manual-mode contract:
  - `delta_manual_kind in {"channelwise","global_mean"}`.
  - if `delta_manual_kind = "global_mean"`:
    provide scalar `delta_manual_value` in unit `energy_unit`.
  - if `delta_manual_kind = "channelwise"`:
    provide `delta_manual_file` (NPZ) containing
    `Delta_puv[p,u,v]` with shape `(n_p,n_u,n_v)` in unit `energy_unit`.
  - `delta_reduction` must be consistent with manual kind:
    `global_mean <-> global_mean`, `channelwise <-> channelwise`.
- Denominator notation follows:
  - project/literature notation: `\Delta_{p-uv}`
- In Eq. (Section 3), denominator is written as `\Delta_{p-uv}` while the
  numerator is summed over `o,p`:
  - `\sum_{o,p} t_{i u,o p} t_{i^{\prime} v,o p}^{*} / \Delta_{p-uv}`.
- For `from_onsite`, first build ligand-site-resolved harmonic denominators:

Math:
$$
\Delta_{u}^{(o,p)}=\epsilon_{i u}-\epsilon_{o p},\quad
\Delta_{v}^{(o,p)}=\epsilon_{i^{\prime} v}-\epsilon_{o p},
$$

Math:
$$
\Delta_{p-uv}^{(o)}
=
\frac{2\Delta_u^{(o,p)}\Delta_v^{(o,p)}}{\Delta_u^{(o,p)}+\Delta_v^{(o,p)}}.
$$

Code form:
```text
for i, i_prime in f_site_pairs:
  for o in ligand_sites:
    for p in ligand_orbitals[o]:
      for u in f_basis[i]:
        for v in f_basis[i_prime]:
          du = eps_f[i,u] - eps_lig[o,p]
          dv = eps_f[i_prime,v] - eps_lig[o,p]
          Delta_puv_o[o,p,u,v] = 2 * du * dv / (du + dv)
```

For `with_soc`, the same formula is used, but `u/v/p` are composite
orbital-spin indices.

Then reduce ligand-site-resolved denominators to Eq. (Section 3) form:

Math:
$$
\Delta_{p-uv}
=
\mathrm{reduce}_{o}\!\left[\Delta_{p-uv}^{(o)}\right].
$$

Code form:
```text
for p,u,v in channel_indices:
  Delta_puv[p,u,v] = reduce_over_o(Delta_puv_o[:,p,u,v], mode=delta_reduction)
```

- Channel reduction policy:
  - `delta_reduction = channelwise`: keep `\Delta_{p-uv}` channel-resolved in `p, u, v`.
  - `delta_reduction = global_mean`: replace all channel denominators by one averaged value (approximation).

For `global_mean`:

Math:
$$
\bar\Delta_{p-uv}
=
\frac{1}{N_{\Delta}}
\sum_{u,v,p}\Delta_{p-uv}.
$$

Code form (global mean):
```text
delta_bar_puv = mean(Delta_puv[p,u,v] for p,u,v in channel_indices)
```

Code form (mode selection):
```text
if delta_mode == "manual" and delta_manual_kind == "global_mean":
  require delta_reduction == "global_mean"
  Delta_puv[p,u,v] = delta_manual_value
elif delta_mode == "manual" and delta_manual_kind == "channelwise":
  require delta_reduction == "channelwise"
  Delta_puv = load_npz(delta_manual_file)["Delta_puv"]    # shape (n_p,n_u,n_v)
elif delta_mode == "from_onsite" and delta_reduction == "channelwise":
  use per_channel_harmonic_deltas
elif delta_mode == "from_onsite" and delta_reduction == "global_mean":
  delta_puv = mean(per_channel_harmonic_deltas)
```

Validation:
- Must record `delta_mode`, `delta_reduction`, per-channel deltas, mean, and std in metadata.
- For manual mode, missing/invalid manual fields or shape mismatch is a hard failure.
- All denominators must be finite and satisfy `abs(Delta_puv) > eps_zero`.

## 5) Real-to-Complex Basis Transform (MUST)
MUST:
- Wannier90 default `f` orbitals are treated as real harmonics with fixed order.
- Basis order MUST follow this project convention:
  - real basis order: `m = [0, 1, -1, 2, -2, 3, -3]`
  - complex basis order: `m = [-3, -2, -1, 0, 1, 2, 3]`
- `U_r2c` MUST be selected by `orbital_order_id` (no runtime sign/phase guessing).
- Inputs of this section (`h_real`, `t_real`) MUST be the reduced outputs from
  Section 3/4, not the unreduced Wannier blocks.

Math:
$$
U_{\mathrm{r2c}}:
\text{ deterministic map from real-harmonic basis to complex-harmonic basis}.
$$

Math:
$$
h_{\mathrm{complex}} = U_{\mathrm{r2c}}^{T} h_{\mathrm{real}} U_{\mathrm{r2c}}^{*},\qquad
t_{\mathrm{complex}} = U_{\mathrm{r2c}}^{T} t_{\mathrm{real}} U_{\mathrm{r2c}}^{*}.
$$

Code form:
```text
U_r2c = build_U_r2c(orbital_order_id, spinor_flag)
U_c2r = inv(U_r2c)

require is_unitary(U_r2c, eps_unitary)

h_complex = U_r2c.T @ h_real @ U_r2c.conj()
t_complex = U_r2c.T @ t_real @ U_r2c.conj()
```

Code form (execution order):
```text
# 1) reduce in raw Wannier basis (Section 3/4)
h_real, t_real = reduce_ligand_channels(wannier_blocks, delta_policy)
# 2) then apply basis transforms (this section)
h_complex = U_r2c.T @ h_real @ U_r2c.conj()
t_complex = U_r2c.T @ t_real @ U_r2c.conj()
```

Code form (explicit non-spinor template):
```text
tmp = np.array([
                 [0, 0, 0, 1, 0, 0, 0],
  (1/sqrt(2))  * [0, 0, 1, 0,-1, 0, 0],
  (1j/sqrt(2)) * [0, 0, 1, 0, 1, 0, 0],
  (1/sqrt(2))  * [0, 1, 0, 0, 0, 1, 0],
  (1j/sqrt(2)) * [0, 1, 0, 0, 0,-1, 0],
  (1/sqrt(2))  * [1, 0, 0, 0, 0, 0,-1],
  (1j/sqrt(2)) * [1, 0, 0, 0, 0, 0, 1],
], dtype=complex)
U_r2c = tmp
if SPINOR:
  U_r2c = kron(tmp, I2)
```

MUST (complex <-> cubic):
- Define one deterministic complex->cubic transform `U_c2cub`.
- Cubic basis order MUST follow project convention:
  `[\xi, \eta, \zeta, A, \alpha, \beta, \gamma]`.
- `U_cub2c = inv(U_c2cub)`.

Math:
$$
h_{\mathrm{cubic}} = U_{\mathrm{c2cub}}^{T} h_{\mathrm{complex}} U_{\mathrm{c2cub}}^{*},\qquad
t_{\mathrm{cubic}} = U_{\mathrm{c2cub}}^{T} t_{\mathrm{complex}} U_{\mathrm{c2cub}}^{*}.
$$

Code form:
```text
U_c2cub = build_U_c2cub(orbital_order_id, spinor_flag)
U_cub2c = inv(U_c2cub)
require is_unitary(U_c2cub, eps_unitary)

h_cubic = U_c2cub.T @ h_complex @ U_c2cub.conj()
t_cubic = U_c2cub.T @ t_complex @ U_c2cub.conj()
```

Code form (explicit non-spinor template):
```text
sq3 = sqrt(3); sq5 = sqrt(5)
tmp_c2cub = np.array([
  (1/4)        * [-sq3, 0, -sq5, 0,  sq5, 0,  sq3],
  (1j/4)       * [-sq3, 0,  sq5, 0,  sq5, 0, -sq3],
  (1/sqrt(2))  * [   0, 1,    0, 0,    0, 1,    0],
  (1j/sqrt(2)) * [   0, 1,    0, 0,    0,-1,    0],
  (1/4)        * [ sq5, 0, -sq3, 0,  sq3, 0, -sq5],
  (1j/4)       * [-sq5, 0, -sq3, 0, -sq3, 0, -sq5],
                 [   0, 0,    0, 1,    0, 0,    0],
], dtype=complex)
U_c2cub = tmp_c2cub
if SPINOR:
  U_c2cub = kron(tmp_c2cub, I2)
```

Code form (derived maps):
```text
U_r2cub = U_r2c @ U_c2cub
U_cub2r = inv(U_r2cub)
```

Validation:
- `U_r2c` version/id must be recorded.
- `U_c2cub` version/id must be recorded.
- Input and transformed matrices must preserve trace and Hermiticity tolerance.

## 6) CEF-from-Wannier Fit Contract (MUST)
MUST:
- For CEF parameter extraction:
1. Build onsite `f` block from Wannier90.
2. Rotate real->complex.
3. Project to LSJM SOC-low target manifold.
4. Fit Stevens parameters under `./standards/en/02-hamiltonian/02-03-HCEF.md`.

Fit quality must be validated:

Math:
$$
r_{\mathrm{fit}}=
\frac{\|H_{\mathrm{proj}}-H_{\mathrm{Stevens}}\|_F}
{\max(\|H_{\mathrm{proj}}\|_F,\varepsilon_{\mathrm{zero}})}.
$$

Code form:
```text
require r_fit <= eps_map
```

Validation:
- Fit failure is hard failure; no silent fallback.

## 7) C3v Axis Convention (MUST)
MUST:
- For `C3v` in this project:
  - `c` axis is `z`
  - `a` axis is `y`
- Axis labels used in outputs must follow this convention.

Code form:
```text
if symmetry == "C3v":
  axis_c = "z"
  axis_a = "y"
```

Validation:
- Metadata must include explicit axis mapping for reproducibility.
