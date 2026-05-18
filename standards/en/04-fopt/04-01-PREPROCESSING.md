# 04-01-FOPT_PREPROCESSING

This file defines FOPT preprocessing Levels `L0`, `L1`, and `L2`.
Global FOPT scope is defined in
`./standards/en/04-fopt/04-00-FOPT_FORMALISM.md`.
Low-level determinant signs follow
`./standards/en/01-core/01-00-FOCK_SLATER.md` and
`./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`.

## 0) Variable Classes (Submodule Scope, MUST)
This file covers only FOPT `L0/L1/L2`.

Per-level definition:
- `L0`: input `{charge_sectors, n_f_orb, n_p_orb}`; output
  `{F_create_raw, P_create_raw}`.
- `L1`: input `{F_create_raw, P_create_raw, U_f, U_p, R_f, R_p}`;
  output `{F_create_rot, P_create_rot}`.
- `L2`: input `{F_create_rot, P_create_rot, t_r_lambda, charge_pairs}`;
  output `{V_plus}`.

Constraint:
- `L0/L1` must not consume hopping matrices.
- `L2` must not consume resolvents, path lists, or W/Kramers projectors.

Code form:
```text
FOPT_L0_L2_inputs_exclude = {resolvents, four_hop_paths, W, kramer_labels}
```

Validation:
- Any `L0/L1` entry point accepting `t_r_lambda` is invalid.
- Any `L2` entry point accepting a resolvent or path enumeration is invalid.

## 1) Level 0: Raw Local Primitives (MUST)
MUST:
- Build f-site creation primitives on canonical f-shell determinant bases.
- Build ligand creation primitives on canonical ligand p-shell determinant
  bases.
- Downstream code must consume the adjoint when an annihilation action is
  required; reverse primitives are not stored independently.
- Use matrix elements in bra-ket convention with state vectors as columns.
- Preserve determinant ordering, parity-below-index signs, dtype policy, and
  orbital index ordering inherited from the core standards.
- Do not apply site labels, ligand labels, local-frame rotations, Wannier
  rotations, hopping matrices, or local working-basis projections at `L0`.

Math:
$$
F_{\mathrm{raw}}^{\dagger,a}[N_f]_{\alpha\beta}
=
\langle \alpha^{N_f+1}|f_a^\dagger|\beta^{N_f}\rangle.
$$

Math:
$$
P_{\mathrm{raw}}^{\dagger,b}[N_p]_{\rho\delta}
=
\langle \rho^{N_p+1}|p_b^\dagger|\delta^{N_p}\rangle.
$$

Code form:
```text
F_create_raw[N_f][a].shape = (dim_f(N_f+1), dim_f(N_f))
P_create_raw[N_p][b].shape = (dim_p(N_p+1), dim_p(N_p))
```

Index:
- `a` is the canonical f spin-orbital index.
- `b` is the canonical ligand p spin-orbital index.
- `alpha,beta` are canonical f determinant-sector indices.
- `rho,delta` are canonical ligand determinant-sector indices.

Validation:
- `F_create_raw[N_f][a]` must fail for invalid `N_f` or `a`.
- `P_create_raw[N_p][b]` must fail for invalid `N_p` or `b`.
- Ligand annihilation from `N_p+1` to `N_p` must be consumed as the adjoint of
  `P_create_raw[N_p][b]`.
- Creation on an occupied orbital and annihilation on an empty orbital must
  produce zero matrix elements.

## 2) Level 1: Working-Basis and Frame Rotation (MUST)
MUST:
- Rotate/bind raw primitives into site-specific f working bases and
  ligand-specific p working bases.
- The f working basis is selected by `model.scheme`: LSJM adjacent-sector
  transforms for `RS`, and IONED adjacent-sector transforms for `ED`.
- The main-sector f leg remains the SOC-lowest LSJM subspace in both schemes.
- Apply physical one-particle frame rotations on the primitive orbital axis.
- Keep f-site labels `r` and ligand labels `lambda` explicit.
- Record all state-basis and one-particle-frame order ids in metadata.
- Do not multiply by hopping matrices at this level.

Math:
$$
F_{\mathrm{rot}}^{r,\alpha}[N_f]
=
\sum_a
R_f[r]_{a\alpha}\,
\left(U_f[r,N_f+1]\right)^\dagger
F_{\mathrm{raw}}^{\dagger,a}[N_f]
U_f[r,N_f].
$$

Math:
$$
P_{\mathrm{rot}}^{\lambda,\beta}[N_p]
=
\sum_b
R_p[\lambda]_{b\beta}\,
\left(U_p[\lambda,N_p-1]\right)^\dagger
P_{\mathrm{raw}}^{b}[N_p]
U_p[\lambda,N_p].
$$

Code form:
```text
F_create_rot[r][N_f][alpha] = sum_a R_f[r][a,alpha] * U_f_out^dag @ F_create_raw[N_f][a] @ U_f_in
P_create_rot[lambda][N_p][beta] = sum_b R_p[lambda][b,beta] * U_p_out^dag @ P_create_raw[N_p][b] @ U_p_in
```

Index:
- `U_f[r,N]` maps canonical f determinants in sector `N` to the selected
  f-site working basis for that sector. For adjacent sectors this is LSJM in
  `RS` and IONED in `ED`.
- `U_p[lambda,N]` maps canonical ligand determinants in sector `N` to the
  selected ligand working basis for that sector.
- `R_f[r]` maps physical f orbital labels used by hopping into canonical raw
  primitive labels.
- `R_p[lambda]` maps physical ligand orbital labels used by hopping into
  canonical raw primitive labels.

Validation:
- All `U_f` and `U_p` matrices must be column-orthonormal.
- `R_f` and `R_p` must have row counts matching raw primitive orbital axes.
- Output shapes must be checked explicitly for every sector and every site.
- Metadata must include `f_state_order_id`, `p_state_order_id`,
  `f_orbital_order_id`, `p_orbital_order_id`, and
  `active_pair_order_id`.

## 3) Level 2: Active-Pair Forward Blocks (MUST)
MUST:
- Build only `V_plus` blocks for p-to-f hopping.
- Use the active-pair tensor-product order `f < p`.
- Do not include any inter-block fermion embedding sign.
- Reserve all full-cluster embedding signs for future `L3`.
- Do not store reverse hopping blocks.

Math:
$$
V_{+}^{r\lambda}[N_f,N_p]
=
\sum_{\alpha\beta}
t_{r\lambda}^{\alpha\beta}
\left(
F_{\mathrm{rot}}^{r,\alpha}[N_f]
\otimes
P_{\mathrm{rot}}^{\lambda,\beta}[N_p]
\right).
$$

Code form:
```text
V_plus[r,lambda,N_f,N_p] = sum_alpha_beta t[alpha,beta] * kron(F_create_rot[r][N_f][alpha], P_create_rot[lambda][N_p-1][beta]^dagger)
```

Index:
- `alpha` is the physical f orbital index in `t_r_lambda`.
- `beta` is the physical ligand orbital index in `t_r_lambda`.
- Rows are ordered as `(f_out,p_out)`.
- Columns are ordered as `(f_in,p_in)`.

Validation:
- `V_plus.shape == (D_f[N_f+1] * D_p[N_p-1], D_f[N_f] * D_p[N_p])`.
- Linearity check:
  `V_plus(t1 + c*t2) == V_plus(t1) + c*V_plus(t2)`.
- Zero hopping check:
  `V_plus(0) == 0`.
- Adjoint consistency check:
  `dagger(V_plus[N_f-1,N_p+1])` has the shape required for the reverse
  current-sector hop from `(N_f,N_p)` to `(N_f-1,N_p+1)`.
- No key named `B`, `V_minus`, or `reverse` may be emitted by `L2`.

## 4) Charge-Pair Selection (MUST)
MUST:
- Accept an explicit iterable of charge pairs for deterministic construction.
- Provide a helper for the minimum low-sector-derived pair set.
- Filter or reject invalid pairs before allocating matrices.

Code form:
```text
required_fopt_pairs(n, n_p_full=6) = valid_sorted({(n,n_p_full), (n-1,n_p_full), (n,n_p_full-1)})
```

Validation:
- Returned charge pairs must be lexicographically sorted by `(N_f,N_p)`.
- Duplicates must be removed deterministically.
- A pair is valid only when `0 <= N_f < n_f_orb` and `1 <= N_p <= n_p_orb`.

## 5) Test Requirements (MUST)
MUST:
- Test raw f creation and ligand creation shapes.
- Test ligand annihilation by consuming the stored creation primitive adjoint.
- Test `L1` shape binding with identity transforms and with at least one
  nontrivial unitary rotation.
- Test `L2` zero hopping.
- Test `L2` linearity in `t_r_lambda`.
- Test active-pair shape on a toy system small enough for deterministic exact
  comparison.
- Test the reverse-hop adjoint relation without storing a reverse block.

Code form:
```text
pytest tests/test_fopt_l0.py tests/test_fopt_l1.py tests/test_fopt_l2.py -q
```

Validation:
- Tests must not require full fourth-order path enumeration.
- Tests must not require resolvent construction.
