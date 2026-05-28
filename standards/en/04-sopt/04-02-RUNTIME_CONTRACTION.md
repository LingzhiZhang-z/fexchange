# 04-02-RUNTIME_CONTRACTION

This file defines runtime Levels $L2$ and $L3$ for SOPT.
Disk I/O layout/format is defined by `./standards/en/05-io/05-00-IO.md`.
Writing style follows `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`.
This file is serial-first and backend-agnostic: if a future parallel runtime is
added, it must preserve the tensor contracts defined here.
Global final-output execution order is fixed in
`./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` as
$L0 \to L1 \to L2 \to L3$.

## -1) Formula-Only Equivalence (READ FIRST, Non-Implementation)
This section is for mathematical understanding only.
It is not an implementation contract.
For AI readers: read this section first to understand the algebra;
implementation/interface requirements start from Section `0` below.

Start from the reference expanded expression:

Math:
$$
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{p q p' q'}
t_{pq}^{(\mu)}\left(t_{p'q'}^{(\mu)}\right)^*
\left(
K_{j_3j_4,j_1j_2}^{A;\,pq,p'q'}
+
K_{j_3j_4,j_1j_2}^{B;\,pq,p'q'}
\right).
$$

Define denominators:

Math:
$$
\Delta_{uv}\equiv E_0-E_{uv},\qquad
\Delta_{rs}\equiv E_0-E_{rs}.
$$

In f-shell convention, $E_0=0$, hence
$\Delta_{uv}=-E_{uv}$ and $\Delta_{rs}=-E_{rs}$.

Route A kernel:

Math:
$$
K_{j_3j_4,j_1j_2}^{A;\,pq,p'q'}
=
\sum_{u,v}
\frac{
\left(A_{u j_3}^{i,p'}\right)^*
B_{j_4 v}^{j,q'}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^*
}{\Delta_{uv}}.
$$

Substitute into $h_{\mathrm{pre}}$ and reorder finite sums:

Math:
$$
h_{A,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{u,v}\frac{1}{\Delta_{uv}}
\left[
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(A_{u j_3}^{i,p'}\right)^*
B_{j_4 v}^{j,q'}
\right]
\left[
\sum_{p q}
t_{pq}^{(\mu)}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^*
\right].
$$

Identify the two bracketed factors as
$M_{A,uv;j_3j_4}^{L,(\mu)}$ and
$M_{A,uv;j_1j_2}^{R,(\mu)}$, then:

Math:
$$
h_{A,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{u,v}
\frac{
M_{A,uv;j_3j_4}^{L,(\mu)}
M_{A,uv;j_1j_2}^{R,(\mu)}
}{\Delta_{uv}}.
$$

Route B kernel:

Math:
$$
K_{j_3j_4,j_1j_2}^{B;\,pq,p'q'}
=
\sum_{r,s}
\frac{
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^*
\left(B_{j_1 r}^{i,p'}\right)^*
A_{s j_2}^{j,q'}
}{\Delta_{rs}}.
$$

Substitute and reorder sums:

Math:
$$
h_{B,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{r,s}\frac{1}{\Delta_{rs}}
\left[
\sum_{p q}
t_{pq}^{(\mu)}
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^*
\right]
\left[
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(B_{j_1 r}^{i,p'}\right)^*
A_{s j_2}^{j,q'}
\right].
$$

Identify the two bracketed factors as
$M_{B,rs;j_3j_4}^{L,(\mu)}$ and
$M_{B,rs;j_1j_2}^{R,(\mu)}$, then:

Math:
$$
h_{B,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{r,s}
\frac{
M_{B,rs;j_3j_4}^{L,(\mu)}
M_{B,rs;j_1j_2}^{R,(\mu)}
}{\Delta_{rs}}.
$$

Final result:

Math:
$$
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
=
h_{A,j_3j_4,j_1j_2}^{(\mu)}
+
h_{B,j_3j_4,j_1j_2}^{(\mu)}.
$$

This transformation is exact (no approximation): it only changes summation order and factorization.

## 0) Variable Classes (Submodule Scope, MUST)
This file covers $L2/L3$ and uses three variable classes:
- Input variables: read from external interfaces or upstream level outputs.
- Intermediate variables: internal working variables only; not exposed as this level's outputs.
- Output variables: interface variables emitted by this level for downstream levels/callers.

Per-level definition:
- $L2$: input `{A, B, t_mu}`; intermediate `{workspace}`; output `{M_A, M_B}`.
- $L3$: input `{M_A, M_B, E_u, W, labels_abcd, labels_order_id}` or
  reference `{h_pre_j_mu, W, labels_abcd, labels_order_id}`; intermediate
  `{E_uv, E_rs, h_pre_mu}`; output `{h_mu_abcd, Heff_mu_abcd}` and, when the projected
  local space is `2 x 2`, optional `{J_mu, mapping_residual}`.

## 0.0.0) $E_u$ Intermediate-State Energy Source (MUST)
`E_u` is a runtime-derived denominator input, not an independent persistent artifact.
It is constructed at $L3$ entry from LSJM energy coefficient arrays (`E_terms.npz`)
of the adjacent sectors ($f^{n-1}$, $f^{n+1}$) using branch-resolved
`F^0/F^2/F^4/F^6/\zeta/offset` parameters.
The denominator reference is controlled by `[fsite].energy_reference`.

Math:
$$
E_u^{(m)}[u] = \mathrm{offset}^{(m)}
+ F^{0,(m)}\cdot\mathrm{coef\_F0}[u]
+ F^{2,(m)}\cdot\mathrm{coef\_F2}[u]
+ F^{4,(m)}\cdot\mathrm{coef\_F4}[u]
+ F^{6,(m)}\cdot\mathrm{coef\_F6}[u]
+ \zeta^{(m)}\cdot\mathrm{coef\_zeta}[u],
\quad m\in\{n+1,\,n-1\}.
$$

Main-sector reference:

Math:
$$
E_{\mathrm{ref}} =
\begin{cases}
0, & \texttt{energy\_reference = "zero"} \\
\mathrm{offset}^{(n)}
+ F^{0,(n)}\cdot\mathrm{coef\_F0}[u_0]
+ F^{2,(n)}\cdot\mathrm{coef\_F2}[u_0]
+ F^{4,(n)}\cdot\mathrm{coef\_F4}[u_0]
+ F^{6,(n)}\cdot\mathrm{coef\_F6}[u_0],
& \texttt{energy\_reference = "lsjm\_ground"}
\end{cases}
$$

where `u0` is the selected `f^n` reference state from LSJM.

Code form:
```text
E_u_np1 = E_lsjm_np1 - E_ref
E_u_nm1 = E_lsjm_nm1 - E_ref
```

If a side branch uses a target minimum gap instead of an explicit `offset`,
the branch-local offset is resolved implicitly at denominator construction time:

Math:
$$
E_u^{(n+1)}[u] =
U^+ + E_{\mathrm{raw}}^{(n+1)}[u]
- \min_v E_{\mathrm{raw}}^{(n+1)}[v],
$$

Math:
$$
E_u^{(n-1)}[u] =
U^- + E_{\mathrm{raw}}^{(n-1)}[u]
- \min_v E_{\mathrm{raw}}^{(n-1)}[v].
$$

Here `Uplus` is only valid in `[fsite_np1]`, `Uminus` is only valid in
`[fsite_nm1]`, and each is mutually exclusive with the corresponding explicit
`offset`. `E_raw` is reconstructed from the same branch-local
`F^0/F^2/F^4/F^6/\zeta` parameters with no branch offset applied.

Source artifacts:
- `E_terms.npz` from LSJM outputs in sectors $n-1$, $n+1$ (disk path per `./standards/en/05-io/05-00-IO.md`).
- Branch defaults come from main `[fsite]`; branch overrides come from
  `[fsite_nm1]/[fsite_np1]`.
- `offset` is branch-local and defaults to `0` when no target minimum gap is set.
- `E_ref` uses the main-sector (`f^n`) LSJM output when `energy_reference = "lsjm_ground"`.
- `E_u` is NOT persisted as a separate disk artifact; it is computed on-the-fly
  at $L3$ entry.

Validation:
- No mandatory sign constraint is imposed on branch energies in `E_u`.
- Near-zero denominators and non-finite numeric values in denominator assembly must be flagged with `FXE-NUM-002`.
- Construction formula must match `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` Section 0.2.2.

## 0.0.1) $W$ Projector Construction Contract (MUST)
The Kramers/CEF projector $W$ maps from the SOC-lowest LSJM subspace (dimension $n_j=2J_0+1$,
defined in `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` Section 0.2.1)
to the target low-energy CEF basis (dimension $n_k$).

Construction procedure:
1. Start with $U^{n,\mathrm{soc0}}$ columns (the SOC-lowest $J_0$ multiplet in Fock basis).
2. Build $H_{\mathrm{cef}}$ projected into this $J_0$ subspace:

Math:
$$
H_{\mathrm{cef}}^{(J_0)}
=
\left(U^{n,\mathrm{soc0}}\right)^{\!\dagger}
H_{\mathrm{cef}}^{\mathrm{fock}}\,
U^{n,\mathrm{soc0}},
\qquad
H_{\mathrm{cef}}^{(J_0)}\in\mathbb C^{n_j\times n_j}.
$$

3. Diagonalize:

Math:
$$
H_{\mathrm{cef}}^{(J_0)} w_k = \epsilon_k\,w_k,
\quad k=1,\ldots,n_j,
\quad \epsilon_1\le\epsilon_2\le\cdots
$$

4. Select the target low-energy subspace:
   - Kramers systems (odd $n$): select lowest doublet ($n_k=2$).
     Verify Kramers degeneracy $|\epsilon_1-\epsilon_2|\le\varepsilon_{\mathrm{eig\_cluster}}$.
     Apply gauge-fixing from `./standards/en/03-spectrum/03-02-KRAMERS_DOUBLET.md`.
   - Non-Kramers systems (even $n$): select lowest quasi-doublet ($n_k=2$).
     Apply gauge-fixing from `./standards/en/03-spectrum/03-03-NON_KRAMERS_DOUBLET.md`.
   - Larger target spaces ($n_k>2$): select the lowest $n_k$ states with explicit energy-gap criterion.
5. Assemble $W\in\mathbb C^{n_j\times n_k}$ whose columns are the selected eigenstates $w_k$.

Code form:
```text
V_J0 = U_n_soc0                          # (dim_fock, n_j)
H_cef_J0 = V_J0.conj().T @ H_cef_fock @ V_J0   # (n_j, n_j)
evals, evecs = eigh(H_cef_J0)            # sorted ascending
W = evecs[:, :n_k]                        # (n_j, n_k)
# apply Kramers / non-Kramers gauge-fixing
```

Validation:
- $W^\dagger W = I_{n_k}$ within `eps_orth`.
- For Kramers doublet: TR-pair and gauge checks from module 02-05.
- `W.shape = (n_j, n_k)` where `n_j` matches L1 output $j$-axis dimension.
- `kramer_name` must be recorded in metadata.

## 0.1) External Runtime Input Schema for $L2/L3$ (MUST)
MUST:
- Hopping (`t_mu`) and Kramer projector (`W`, `kramer_labels`) are external runtime inputs.
- Global header gate from `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md` is mandatory:
  `schema_version`, `standard_version`, `basis_id`, `orbital_order_id`, `unit`.
- One run computes one bond only; no `mu` axis is allowed in input hopping payload.

Math:
$$
t\_mu\in\mathbb C^{n_{\mathrm{orb}}\times n_{\mathrm{orb}}}.
$$

Math:
$$
W\in\mathbb C^{n_j\times n_k},
\qquad
labels\_{abcd}\in\mathbb Z^{n_{L}\times 4},
\qquad
0\le a,b,c,d<n_k.
$$

Code form:
```text
require t_mu.ndim == 2
require W.shape[0] == expected_n_j_from_L1_or_21_meta
require labels_abcd.shape[1] == 4
require labels_abcd.max() < W.shape[1]
require labels_order_id == "abcd_lex_v1"
require rows_unique(labels_abcd)
require is_lex_sorted(labels_abcd, key=(a,b,c,d))
```

Validation:
- Any schema/binding mismatch is a hard failure before contraction.
- `W` orthonormality check: `W^dag W = I` within `eps_orth`.
- In the reference projection path, enforce `W.shape[0] == h_pre_j_mu.shape[0]`
  before projection.
- `labels_abcd` ordering/bijective checks are hard failures.

## 1) Level 2: Route Factors $M_A/M_B$ (Phi Form, MUST)
MUST:
- This level computes route factors after site-binding and hopping contraction.
- Use symbols $M_A$ and $M_B$ (instead of $\Phi$).
- This level does not build/store explicit $K_{j_3 j_4,j_1 j_2}^{pq,p'q'}$.

Site binding:

Math:
$$
A_{u j}^{i,p} \equiv A_{u j}^{\kappa=p,n},
\qquad
B_{j v}^{j,q} \equiv B_{j v}^{\kappa=q,n-1},
$$

Math:
$$
A_{s j}^{j,q} \equiv A_{s j}^{\kappa=q,n},
\qquad
B_{j r}^{i,p} \equiv B_{j r}^{\kappa=p,n-1}.
$$

Denominator energies:

Math:
$$
E_{uv}=E_i^{n+1}(u)+E_j^{n-1}(v),
\qquad
E_{rs}=E_i^{n-1}(r)+E_j^{n+1}(s).
$$

Route-A factors:

Math:
$$
M_{A,uv;j_3j_4}^{L,(\mu)}
=
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(A_{u j_3}^{i,p'}\right)^*
B_{j_4 v}^{j,q'},
$$

Math:
$$
M_{A,uv;j_1j_2}^{R,(\mu)}
=
\sum_{p q}
 t_{pq}^{(\mu)}
 A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^*.
$$

Route-B factors:

Math:
$$
M_{B,rs;j_3j_4}^{L,(\mu)}
=
\sum_{p q}
 t_{pq}^{(\mu)}
 B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^*,
$$

Math:
$$
M_{B,rs;j_1j_2}^{R,(\mu)}
=
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(B_{j_1 r}^{i,p'}\right)^*
A_{s j_2}^{j,q'}.
$$

Hermitian-conjugation relation (same channel $\mu$, same index tuple):

Math:
$$
M_{A,uv;j_3j_4}^{L,(\mu)}
=
\left(M_{A,uv;j_3j_4}^{R,(\mu)}\right)^*,
\qquad
M_{B,rs;j_3j_4}^{L,(\mu)}
=
\left(M_{B,rs;j_3j_4}^{R,(\mu)}\right)^*.
$$

Persisted L2 outputs are defined as:

Math:
$$
M_{A,uv;j_1j_2}^{(\mu)} \equiv M_{A,uv;j_1j_2}^{R,(\mu)},
\qquad
M_{B,rs;j_1j_2}^{(\mu)} \equiv M_{B,rs;j_1j_2}^{R,(\mu)}.
$$

Persisted axis order is fixed:
- `M_A` axis order: `(u, v, j1, j2)` with `axis_order_id = "uvj1j2_v1"`.
- `M_B` axis order: `(r, s, j1, j2)` with `axis_order_id = "rsj1j2_v1"`.

Code form:
```text
build M_A over (u,v) blocks for this bond
build M_B over (r,s) blocks for this bond
persist M_A as M_A[u,v,j1,j2]  # uvj1j2_v1
persist M_B as M_B[r,s,j1,j2]  # rsj1j2_v1
```

Validation:
- `M_A/M_B` index order must be fixed and documented.
- Use blockwise streaming over `(u,v)` and `(r,s)`; full dense materialization is optional debug mode only.
- Persisted metadata must include `axis_order_id` for `M_A/M_B`.

## 2) Internal Denominator Summation to $h_{pre,j}^{(\mu)}$ (MUST)
MUST:
- This internal calculation sums intermediate states with denominators and constructs $h_{pre,j}^{(\mu)}$.
- It is algebraically equivalent to the old `$K$ then contract with $t$` route.
- It constructs $E_{uv}$ and $E_{rs}$ from `E_u`; these are not defined in $L2$.

Denominator definitions (inside $L3$):

Math:
$$
E_{uv}=E_i^{n+1}(u)+E_j^{n-1}(v),\qquad
E_{rs}=E_i^{n-1}(r)+E_j^{n+1}(s).
$$

Math:
$$
\Delta_{uv}\equiv E_0-E_{uv},\qquad
\Delta_{rs}\equiv E_0-E_{rs}.
$$

If implementation uses composite indices $m=(u,v)$ and $n=(r,s)$,
$\Delta_{uv}$ and $\Delta_{rs}$ can be represented as denominator vectors
`denom_A[m]` and `denom_B[n]`.

Math:
$$
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{u,v}
\frac{
\left(M_{A,uv;j_3j_4}^{(\mu)}\right)^*
M_{A,uv;j_1j_2}^{(\mu)}
}{\Delta_{uv}}
+
\sum_{r,s}
\frac{
\left(M_{B,rs;j_3j_4}^{(\mu)}\right)^*
M_{B,rs;j_1j_2}^{(\mu)}
}{\Delta_{rs}}.
$$

Code form:
```text
h_pre_j_mu = sum_uv( conj(M_A) * M_A / Delta_uv ) + sum_rs( conj(M_B) * M_B / Delta_rs )
```

Equivalent matrix contraction (recommended):
Code form:
```text
# flatten (j3,j4)->a, (j1,j2)->b
YA = M_A.reshape(Nuv, J2)
w_uv = 1.0 / Delta_uv
hA = YA.conj().T @ (w_uv[:,None] * YA)

YB = M_B.reshape(Nrs, J2)
w_rs = 1.0 / Delta_rs
hB = YB.conj().T @ (w_rs[:,None] * YB)

h_pre_j_mu = (hA + hB).reshape(J,J,J,J)
```

Validation:
- Denominator must follow $\Delta=E_0-E_{\mathrm{intermediate}}$.
- Zero-hop check: if `t=0`, then `h_pre_j_mu=0`.

Output:
- `h_pre_j_mu` is an internal/reference tensor, not a standalone runtime artifact.

## 3) Level 3: Fix Kramers Basis and Build Final Outputs (MUST)
MUST:
- Inside final $L3$, apply outer $W$ projection.
- Final public interface must use $a,b,c,d$ semantics.
- $W$ must map from the $f^n$ SOC-lowest LSJM subspace to the CEF/Kramers basis.

Math:
$$
h_{\mathrm{pre},cd,ab}^{(\mu)}
=
\sum_{j_3,j_4,j_1,j_2}
(W_{j_3 c})^*(W_{j_4 d})^*
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
W_{j_1 a}W_{j_2 b}.
$$

Math:
$$
H_{\mathrm{eff},cd,ab}^{(\mu)} = h_{\mathrm{pre},cd,ab}^{(\mu)}.
$$

Code form:
```text
h_pre_mu = project_with_W(h_pre_j_mu, W)
h_mu_abcd = h_pre_mu
Heff_mu_abcd = h_mu_abcd
```

Runtime path (MUST):
- The canonical runtime final-$L3$ path is the **fused** implementation
  (`build_L3` in the current code): it computes `outputs_L3` directly from `{M_A, M_B, E_u, W}`
  by projecting each route factor's external LSJM legs with `W` first, then
  performing the same denominator-weighted Gram contraction in the projected
  ($n_k$) space. The runtime MUST NOT materialize `h_pre_j_mu` on this path.
- The code form above (materialized `project_with_W(h_pre_j_mu, W)`, i.e.
  `build_L4_legacy(build_L3_legacy(...), W)`) is the **reference implementation**. It is off
  the runtime path and serves as the algebraic-equivalence oracle for the fused path.
- The fused path MUST be algebraically equal to the reference path, verified
  numerically within tolerance (pinned at `1e-12` in the test suite). The
  equality holds because $W$ acts only on the external $j$-legs and the
  denominator is a pure $(u,v)/(r,s)$ weight, so projection commutes with the
  intermediate-state sum.
Applicability (informative):
- The fused path is a large memory/compute saving when $n_k \ll n_j$ (the
  production case: $n_k=2$ lowest Kramers doublet vs $n_j=2J+1$), because the
  $n_j^4$ `h_pre_j_mu` tensor is never formed. When $n_k \approx n_j$ ($W$
  near-square) it is roughly neutral or slightly slower, since the projection
  is then applied per intermediate-state pair rather than once on the
  aggregate. Correctness is unaffected in either regime.
- Contrast with FOPT: both SOPT and FOPT consume $W$ inside $L3$; SOPT keeps a
  reference materialized path only to anchor the equivalence test.

Validation:
- Hermiticity check: $\mathrm{Heff}^{(\mu)}=\left(\mathrm{Heff}^{(\mu)}\right)^\dagger$ within tolerance.
- $W$ projection dimensions must match $h_{\mathrm{pre},j}^{(\mu)}$.

## 4) Parallel Execution and Root-Write Policy (MUST)
MUST:
- MPI/parallel layout is runtime environment, not input-file content.
- Worker ranks may compute disjoint shards of $L2/L3$ tensors.
- Before persistence, shards must be gathered/reduced to root rank.
- Root rank must assemble full tensors then write `data.npz` and `meta.json`.
- Non-root ranks must never write persistent stage artifacts.
- Runtime stdout and `meta.json` must record parallel summary fields:
  `parallel_backend`, `world_size`, `root_rank`, `local_rank`, `gather_policy`.

Code form:
```text
shard = compute_local_shard(...)
global_tensor = gather_to_root(shard)
if rank == root:
  assemble_full_tensor(global_tensor)
  write(data.npz, meta.json)
else:
  no_persist_write()
```

Validation:
- Concurrent multi-rank writes to one artifact path are hard failures.

## 5) Runtime I/O Summary
Code form:
```text
inputs_L2_L3   = {A, B, E_u, t_mu, W, labels_abcd, labels_order_id}
outputs_L2     = {M_A, M_B}
outputs_L3     = {h_mu_abcd, Heff_mu_abcd}
```
