# 04-00-FOPT_FORMALISM

This file defines the fourth-order perturbation theory (FOPT) preprocessing
boundary for ligand-mediated f-p superexchange.
Writing style follows `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`.
Low-level Fock and fermion conventions follow `./standards/en/01-core/01-00-FOCK_SLATER.md`
and `./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`.

## 0) Scope and Non-Scope (MUST)
MUST:
- This standard covers reusable FOPT building blocks for the two-f-site /
  two-ligand cluster.
- This standard defines Levels `L0`, `L1`, `L2`, and `L3`.
- `L3` enumerates the 32 fourth-order hopping paths for the two-f-site /
  two-ligand cluster and contracts the three resolvents.
- `L3` consumes the low-energy projector `W` and emits raw total `h_eff_4`
  plus the five process-resolved raw contributions.
- Runtime FOPT `L3` requires a two-dimensional projected local space and must
  apply the shared spin-1/2 mapping to total `h_eff_4` and to each of the five
  process-resolved contributions.

Code form:
```text
fopt_scope = {L0, L1, L2, L3}; L3_outputs = {total_heff, process_heff, total_spin12, process_spin12}
```

Validation:
- Every `L3` path must map its ligand denominators to the 32-row path table.
- `E_0` must be passed into every process helper when denominator references
  are nonzero.
- Any stored reverse hopping primitive is a contract violation.

## 1) Physical Cluster and Charge Sector (MUST)
MUST:
- The cluster contains two f sites and two ligand sites:
  `f1 -- {pA,pB} -- f2`.
- The low-energy charge sector is `(N_f1, N_f2, N_pA, N_pB) = (n,n,6,6)`.
- Local f charge sectors must use the project f-shell conventions.
- Local ligand p sectors must use a deterministic local p-shell Fock convention
  with explicit `p_orbital_order_id`.

Math:
$$
\mathcal C_0 = (N_{f1},N_{f2},N_{pA},N_{pB})=(n,n,6,6).
$$

Code form:
```text
low_charge = {"f1": n, "f2": n, "pA": 6, "pB": 6}
```

Index:
- `r in {1,2}` labels the f site.
- `lambda in {A,B}` labels the ligand site.
- `N_f` is a local f-site electron number.
- `N_p` is a local ligand electron number.

Validation:
- `0 <= N_f <= 14`.
- `0 <= N_p <= n_p_orb`, where `n_p_orb` is recorded in ligand metadata.
- Missing ligand orbital-order metadata is a hard binding failure.

## 2) Level Definitions (MUST)
MUST:
- `L0` builds unrotated local primitive transition matrices on canonical local
  Fock determinant bases.
- `L1` rotates/binds the `L0` primitives into site/lambda-specific local working
  bases and physical one-particle frames.
- `L2` builds active-pair p-to-f hopping blocks `V_plus`.
- `L2` must consume hopping matrices but must not consume resolvents or path lists.
- `L3` consumes `V_plus`, per-ligand p-sector energies, f-sector
  intermediate energies, and `W`; it emits raw total `h_eff_4`, raw
  process-resolved `h_eff_4`, and spin-1/2 mapped exchange outputs. Runtime
  FOPT exchange output requires `W.shape[1] == 2`.

Code form:
```text
FOPT order: L0 -> L1 -> L2 -> L3
```

Validation:
- `L0` outputs are site-agnostic.
- Site labels `r` and ligand labels `lambda` may first appear in `L1`.
- Hopping matrices may first appear in `L2`.
- Resolvents and path enumeration may first appear in `L3`.

## 3) Active Hopping Direction (MUST)
MUST:
- Store only the forward p-to-f active-pair hopping blocks.
- The forward operator is named `V_plus`.
- The physical direction of `V_plus[r,lambda]` is electron hopping from ligand
  `p_lambda` to f site `f_r`.
- Do not store `B`, `V_minus`, or f-to-p hopping as independent primitives.

Math:
$$
A_{r\lambda}
=
\sum_{\alpha\beta}
t_{r\lambda}^{\alpha\beta}
f_{r\alpha}^{\dagger}p_{\lambda\beta}.
$$

Math:
$$
V_{+}^{r\lambda}[N_f,N_p]:
(f_r^{N_f}\otimes p_\lambda^{N_p})
\rightarrow
(f_r^{N_f+1}\otimes p_\lambda^{N_p-1}).
$$

Code form:
```text
V_plus[(r, lambda, N_f, N_p)] stores p_lambda -> f_r only
```

Index:
- `alpha` labels the physical f spin-orbital axis after `L1`.
- `beta` labels the physical ligand spin-orbital axis after `L1`.
- `t_r_lambda[alpha,beta]` is the p-to-f hopping amplitude.

Validation:
- `t_r_lambda.shape == (n_f_orb, n_p_orb)`.
- `V_plus` must be exactly linear in `t_r_lambda`.
- If `t_r_lambda` is zero, `V_plus` must be zero.

## 4) Reverse Direction by Adjoint (MUST)
MUST:
- The reverse f-to-p hop must be recovered only as an adjoint of a valid
  forward block.
- The current-sector reverse block is not a separate stored object.
- If the required source forward block is outside allowed charge bounds, the
  reverse block is undefined for that current sector and must not be fabricated.

Math:
$$
B_{r\lambda}[N_f,N_p]
=
\left(
V_{+}^{r\lambda}[N_f-1,N_p+1]
\right)^\dagger.
$$

Code form:
```text
B_current(N_f,N_p) = dagger(V_plus[N_f - 1, N_p + 1])
```

Validation:
- Tests must verify the adjoint relation for every valid source forward block.
- No `B` array key may be emitted by `L0`, `L1`, or `L2`.

## 5) Active-Pair Tensor-Product Basis (MUST)
MUST:
- `L2` active-pair bases use local order `f < p`.
- Flattened tensor-product ordering is row-major with the f index outside and
  the ligand p index inside.
- `L2` active-pair blocks are bare tensor-product blocks and must not include
  any inter-block fermion embedding sign.
- All inter-block fermion signs for the full order `f1 < f2 < pA < pB` are
  reserved for future `L3`.

Math:
$$
|i_f,i_p\rangle_{\mathrm{active}}
\equiv
|i_f\rangle_{f_r}\otimes |i_p\rangle_{p_\lambda}.
$$

Math:
$$
\mathrm{flat}(i_f,i_p)=i_f\,d_p+i_p.
$$

Code form:
```text
active_pair_order_id = "f_then_p_rowmajor_v1"
```

Validation:
- For domain dimensions `(d_f_in,d_p_in)`, the column count is
  `d_f_in * d_p_in`.
- For codomain dimensions `(d_f_out,d_p_out)`, the row count is
  `d_f_out * d_p_out`.
- Full-cluster embedding parity factors from the order `f1 < f2 < pA < pB`
  are reserved for future `L3`.

## 6) Charge-Pair Coverage (MUST)
MUST:
- `L2` must support caller-provided valid charge pairs.
- The minimum FOPT preprocessing set for a low f occupancy `n` includes valid
  pairs from `{(n,6), (n-1,6), (n,5)}`.
- Implementations may build any additional valid pairs needed for future `L3`.

Code form:
```text
minimum_pairs(n) = valid({(n,6), (n-1,6), (n,5)})
```

Validation:
- Invalid charge pairs must fail before matrix construction.
- Pair ordering in returned maps must be deterministic.
