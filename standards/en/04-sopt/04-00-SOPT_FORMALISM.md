# 04-00-SOPT_FORMALISM

This file defines the SOPT physical core and minimal programmable interface.
It does not define state-construction details or level-by-level implementation details (see `./standards/en/04-sopt/04-01-PRECOMPUTE_PIPELINE.md`, `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`, and `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`).
Disk I/O layout/format is defined by `./standards/en/05-io/05-00-IO.md`.
Writing style follows `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`.
Runtime tolerances, deterministic linear algebra, and global input gates follow
`./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`.

## 0) Level Definitions (MUST)
Level semantics:
- $L0$: Fock-basis primitive-transition level (construct transition elements on canonical Fock basis only; no site labels $i/j$; no external state-file dependency).
- $L1$: local transition-vertex construction (rotate intermediate $f^{n+1}/f^{n-1}$ legs and project the $f^n$ leg onto the lowest-SOC LSJM subspace).
- $L2$: route-factor construction level (compute $M_A/M_B$ from $A/B$, site-binding, and hopping contraction).
- $L3$: denominator-weighted intermediate-state summation (accumulate to $h_{\mathrm{pre},j}^{(\mu)}$).
- $L4$: fixed-Kramers-basis output (apply $W$ projection and produce single-channel $\mathrm{Heff}^{(\mu)}$).

Execution order:
- Default and required order is $L0 \to L1 \to L2 \to L3 \to L4$ (i.e., $L3 > L4$).

Ownership boundary:
- Formula details for $L0/L1$ are in `./standards/en/04-sopt/04-01-PRECOMPUTE_PIPELINE.md`.
- Formula details for $L2/L3/L4$ are in `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`.
- Post-SOPT pseudospin-$\tfrac{1}{2}$ mapping is in `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`.
- This file keeps only cross-level contracts and shared symbols.
- Site labels $i/j$ are introduced from $L2$; both $L0/L1$ remain site-agnostic.
- Transition-direction convention: compute/cache tensors in the unified low-occupancy $\to$ high-occupancy $f^\dagger$ form; reverse transitions are reconstructed by Hermitian conjugation.
- Runtime granularity: one run computes one bond only; $\mu$ is a bond label for that run (not an input tensor axis).

## 0.1) Naming Convention (MUST)
- Fock-basis indices: $\alpha,\beta,\gamma,\chi$.
- $f^n$ SOC-lowest LSJM-subspace indices: $j_1,j_2,j_3,j_4$ (not the site-$j$ label).
- Kramers low-energy indices: $a,b,c,d$.
- Single-site intermediate indices: $u,v,r,s$.
- Two-site intermediate composite indices: $m,n$, with Route A $m=(u,v)$ and Route B $n=(r,s)$.
- Generic single-site orbital index: $\kappa$ (before site binding).
- Site-bound orbital indices: $p,p'$ on site-$i$; $q,q'$ on site-$j$.
- Scope note: this `p,p',q,q'` convention is SOPT-internal (post-reduction).
  It is distinct from ligand index notation `(o,p)` used in
  `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`.

## 0.2) Core Symbols (Compact, MUST)
- Projectors/Hamiltonians: $P,Q,H_0,H_{\mathrm{hop}}^{(\mu)},H_{\mathrm{eff}}^{(2)}$.
- Energies/denominators: $E_0,E_n,E_{uv},E_{rs},\Delta_{uv},\Delta_{rs}$ where
  $\Delta_{uv}=E_0-E_{uv}$ and $\Delta_{rs}=E_0-E_{rs}$.
- Effective tensors: $h_{cd,ab}^{(\mu)}$ and $\mathrm{Heff}_{cd,ab}^{(\mu)}$ (single fixed $\mu$ channel).
- Fermion operators: $f,\hat f,\hat N_i$ with embeddings
  $\hat f_{ip}=f_{ip}\otimes I$ and $\hat f_{jq}=(-1)^{\hat N_i}\otimes f_{jq}$.
- Hopping/cache objects: $t_{pq}^{(\mu)},t_{p'q'}^{(\mu)\ast},M_A,M_B,W,U^{n,\mathrm{soc0}}$.
- Code-alias rule: `t_mu[p,q]` in code blocks is an implementation variable name and is semantically identical to $t_{pq}^{(\mu)}$.

## 0.2.1) SOC-Lowest Subspace Definition (MUST)
The "$f^n$ SOC-lowest LSJM subspace" (also "$f^n$ LSJM ground multiplet") referenced throughout SOPT is defined as follows:

1. Identify the ground LS term: the $(\alpha_0, L_0, S_0)$ term with the lowest
   Coulomb eigenvalue $\varepsilon_{\alpha_0}$ from LSMS (module 03-00).
   If multiple $\alpha$ values exist for the same $(L_0,S_0)$, pick the one with the
   smallest $\varepsilon_{\alpha}$; if numerically tied, apply deterministic
   tie-break from `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md`.
2. Identify the ground $J$: within the ground LS term, apply Hund's third rule:
   - $n \le 2\ell$ (less than half-filled, $\ell=3$): $J_0 = |L_0-S_0|$.
   - $n > 2\ell$ (more than half-filled): $J_0 = L_0+S_0$.
   - $n = 2\ell+1 = 7$ (exactly half-filled): $L_0=0$, so $J_0 = S_0$.
3. The SOC-lowest subspace consists of $2J_0+1$ states:

Math:
$$
\mathcal S_{\mathrm{soc0}}^{(n)}
= \bigl\{\lvert \alpha_0,L_0,S_0,J_0,M\rangle : M=-J_0,\ldots,J_0\bigr\}.
$$

4. $U^{n,\mathrm{soc0}}\in\mathbb C^{d_{\mathrm{fock}}\times(2J_0+1)}$ is the
   Fock-to-subspace transform whose columns are these states in canonical LSJM
   order inherited from `./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md`.

Code form:
```text
ground_term = argmin_alpha_L_S(E_coulomb[alpha,L,S])
(alpha0, L0, S0) = ground_term
if n <= 2*ell: J0 = abs(L0 - S0)
elif n > 2*ell: J0 = L0 + S0
n_j = 2*J0 + 1
U_n_soc0 = V_lsjm_fock[:, columns_for(alpha0,L0,S0,J0)]
U_n_soc0.shape = (dim_fock, n_j)
```

Validation:
- $n_j = 2J_0+1$ must be recorded in metadata and passed to downstream modules.
- $U^{n,\mathrm{soc0}\dagger}U^{n,\mathrm{soc0}}=I$ within `eps_orth`.
- Ground-term identification must be deterministic across runs.

## 0.2.2) Intermediate-State Energy Construction (MUST)
`E_u` intermediate-state energies are LSJM eigenvalues of $H_{\mathrm{int}}+H_{\mathrm{soc}}$
in the adjacent particle-number sectors ($f^{n+1}$ and $f^{n-1}$).
By project convention, the ground-state reference is fixed as $E_0=0$ and
runtime energies must already be expressed in this reference (no extra shift).

Math:
$$
E_u^{(n+1)}[u] = E^{\mathrm{LSJM}}_{n+1}(u),
$$

$$
E_v^{(n-1)}[v] = E^{\mathrm{LSJM}}_{n-1}(v),
$$

where:
- $E^{\mathrm{LSJM}}_{m}(u) = \sum_{k} F^k\cdot\mathrm{coef\_F_k}[u] + \zeta\cdot\mathrm{coef\_zeta}[u]$
  is the total LSJM energy of state $u$ in sector $f^m$.

Code form:
```text
E_u_np1   = E_lsjm_np1              # all states in f^{n+1}
E_u_nm1   = E_lsjm_nm1              # all states in f^{n-1}
```

Index:
- $u$: runs over all LSJM states in $f^{n+1}$.
- $v$: runs over all LSJM states in $f^{n-1}$.

Validation:
- No mandatory sign constraint is imposed on branch energies $E_u^{(n+1)}$ or $E_v^{(n-1)}$.
- Near-zero denominators ($|\Delta_{uv}|<\varepsilon_{\mathrm{zero}}$) or non-finite numeric values must be flagged with `FXE-NUM-002`.
- Construction requires $F^0,F^2,F^4,F^6,\zeta$ for sectors $n-1$ and $n+1$.

## 0.3) Variable Classes (MUST)
Definitions:
- Input variables: read from other module interfaces or passed in by the caller.
- Intermediate variables: temporary variables used inside a module only; not external interface artifacts.
- Output variables: variables exported by a module to downstream modules/callers.

Large module (full SOPT chain, $L0 \to L4$):
- Input variables: `E_u`, `U_np1`, `U_n_soc0`, `U_nm1`, `t_mu`, `W`, `labels_abcd`, `labels_order_id`.
- Intermediate variables: `X`, `Y`, `A`, `B`, `E_uv`, `E_rs`, `M_A`, `M_B`, `h_pre_j_mu`.
- Output variables: `h_mu_abcd`, `Heff_mu_abcd`.

Small modules (per level):
- $L0$: input `{}`; intermediate `{sign/workspace}`; output `{X, Y}`.
- $L1$: input `{X, Y, U_np1, U_n_soc0, U_nm1}`; intermediate `{workspace}`; output `{A, B}`.
- $L2$: input `{A, B, t_mu}`; intermediate `{workspace}`; output `{M_A, M_B}`.
- $L3$: input `{M_A, M_B, E_u}`; intermediate `{E_uv, E_rs, workspace}`; output `{h_pre_j_mu}`.
- $L4$: input `{h_pre_j_mu, W, labels_abcd, labels_order_id}`; intermediate `{h_pre_mu}`; output `{h_mu_abcd, Heff_mu_abcd}`.

## 1) Core SOPT Rule
MUST:
- Use second-order projected perturbation for effective interaction.
- Use one channel-resolved tensor $h^{(\mu)}_{cd,ab}$ per bond/channel $\mu$.

Math:
$$
H_{\mathrm{eff}}^{(2)}
= -P H_{\mathrm{hop}} Q (QH_0Q-E_0)^{-1} Q H_{\mathrm{hop}} P.
$$

Math:
$$
h^{(\mu)}_{cd,ab}
= \sum_n
\frac{
\langle c,d \mid H_{\mathrm{hop}}^{(\mu)} \mid n\rangle
\langle n \mid H_{\mathrm{hop}}^{(\mu)} \mid a,b\rangle
}{E_0-E_n}.
$$

Code form:
```text
h_mu[c,d,a,b] = sum_n hop_out[c,d,n] * hop_in[n,a,b] / (E0 - En[n])
```

Index:
- $a,b,c,d$: low-energy projected states on two sites.
- $n$: intermediate LSJM two-site state.
- $\mu$: bond/channel label.

Validation:
- `hop_out` and `hop_in` must align over the $n$ dimension.
- `h_mu` must map one-to-one to projected basis ordering.

## 2) Denominator and Reference-Energy Rule
MUST:
- Use denominator convention $E_0 - E_n$ only.
- In f-shell runs, set $E_0 = 0$.
- Do not mix alternative sign conventions in the same implementation.

Math:
$$
E_0 = 0,\qquad
\frac{1}{E_0-E_n} = -\frac{1}{E_n}.
$$

Code form:
```text
denom[n] = -En[n]
```

Index:
- $E_n$: intermediate-state energy under this SOPT convention.

Validation:
- Spot-check denominator sign against direct perturbative reference.

## 3) Intermediate-State and Route Rule
MUST:
- Include two virtual routes: addition/removal and removal/addition.
- Keep projected low-energy states in fixed $f^n$ sector, and retain only the lowest-SOC LSJM subspace (the $f^n$ LSJM ground multiplet).

Math:
$$
\lvert a,b\rangle \equiv \lvert a\rangle_i^{(n)} \otimes \lvert b\rangle_j^{(n)},
\qquad
\lvert c,d\rangle \equiv \lvert c\rangle_i^{(n)} \otimes \lvert d\rangle_j^{(n)}.
$$

Math:
$$
\text{Route A: }\lvert u,v\rangle=
\lvert u\rangle_i^{(n+1)}\otimes \lvert v\rangle_j^{(n-1)},
\quad
E_{uv}=E_i^{(n+1)}(u)+E_j^{(n-1)}(v).
$$

Math:
$$
\text{Route B: }\lvert r,s\rangle=
\lvert r\rangle_i^{(n-1)}\otimes \lvert s\rangle_j^{(n+1)},
\quad
E_{rs}=E_i^{(n-1)}(r)+E_j^{(n+1)}(s).
$$

Code form:
```text
routeA_state = (u,v)
routeB_state = (r,s)
```

Index:
- $u,v,r,s$: intermediate-state labels only.

Validation:
- Route-A and Route-B intermediate sets must be internally consistent with sector particle numbers.

## 4) Fermionic-Grading Rule
MUST:
- Fix site order as $i < j$.
- Use graded embedding for site-$j$ operators.

Math:
$$
\hat f_{ip}=f_{ip}\otimes I,\qquad
\hat f_{jq}=(-1)^{\hat N_i}\otimes f_{jq}.
$$

Math:
$$
\langle u,v\rvert \hat f_{ip}^{\dagger}\hat f_{jq}\lvert a,b\rangle
=(-1)^{n_a}
\langle u\rvert f_{ip}^{\dagger}\lvert a\rangle
\langle v\rvert f_{jq}\lvert b\rangle.
$$

Math:
$$
\langle c,d\rvert \hat f_{jr}^{\dagger}\hat f_{is}\lvert u,v\rangle
=(-1)^{n_c}
\langle c\rvert f_{is}\lvert u\rangle
\langle d\rvert f_{jr}^{\dagger}\lvert v\rangle.
$$

Code form:
```text
f_i(p) = kron(f_i_p, I_j)
f_j(q) = kron(parity_i, f_j_q)
```

Index:
- $n_a,n_c$: particle numbers on site-$i$ for low-energy basis states.

Validation:
- Check anti-commutation across sites after embedding.
- In fixed $f^n$ subspace, confirm total parity factors reduce to $+1$.

## 5) Hopping-Form Rule
MUST:
- $p,p'$ are orbital indices on site-$i$ only.
- $q,q'$ are orbital indices on site-$j$ only.
- $p,q$ denote $j \to i$, and $p',q'$ denote $i \to j$.
- This section defines hopping form only; route-factor and contraction details belong to `04-01`/`04-02`.

Math:
$$
H_{\mathrm{hop}}^{(\mu)}
= \sum_{p q} t_{pq}^{(\mu)}\, f_{ip}^{\dagger} f_{jq}
+ \sum_{p' q'} t_{p'q'}^{(\mu)\ast}\, f_{jq'}^{\dagger} f_{ip'}.
$$

Code form:
```text
H_hop_mu = sum_{p,q} t_mu[p,q] * f_i_dag[p] * f_j[q]
         + sum_{p2,q2} conj(t_mu[p2,q2]) * f_j_dag[q2] * f_i[p2]
```

Index:
- $t_{pq}^{(\mu)}$: hopping amplitude from site-$j$ to site-$i$.
- $t_{p'q'}^{(\mu)\ast}$: conjugate amplitude for reverse direction.

Validation:
- $H_{\mathrm{hop}}^{(\mu)}$ must be Hermitian.
- No cross-site index misuse ($p/q$ swap) is allowed.

## 6) Minimal I/O and Runtime Checks
MUST:
- The execution order is fixed as $L0 \to L1 \to L2 \to L3 \to L4$ (i.e., $L3 > L4$).
- This file defines SOPT-local levels only (`L0..L4`).
- Global runtime window (`LMSM..L4`) is defined in
  `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md` and
  `./standards/en/05-io/05-00-IO.md`.
- $L0$ must be generated in-code at runtime and requires no external input artifact.
- $L0$ output must be a single site-agnostic local-transition tensor set; site-specific differences are forbidden at $L0$.
- Interface caches must not store both directions of one transition pair; only low $\to$ high tensors are stored and reverse direction is recovered by conjugation.
- Large-module inputs must include: `E_u`, `U_np1`, `U_n_soc0`, `U_nm1`, `t_mu`, `W`, `labels_abcd`, `labels_order_id`.
- Large-module outputs must include: `h_mu_abcd`, `Heff_mu_abcd`.
- `Heff_mu_abcd` may be consumed by module 04-03 as post-processing input for
  spin-$\tfrac{1}{2}$ mapping.
- Large-module intermediate variables (`X/Y/A/B/E_uv/E_rs/M_A/M_B/h_pre_j_mu`) must not be treated as final public outputs.
- In split-level execution, upstream level outputs may be used as downstream inputs; they remain intermediate variables in large-module semantics.
- $E_{uv}/E_{rs}$ are internal $L3$ composite energies derived from `E_u`, and must not be persisted as external outputs.
- External-facing outputs must use $a,b,c,d$ index semantics; $u,v,r,s$ remain internal intermediate indices only.
- `labels_abcd` canonical ordering is fixed as lexicographic `(a,b,c,d)` with
  nested-loop realization:
  `for a in 0..n_k-1, for b in 0..n_k-1, for c in 0..n_k-1, for d in 0..n_k-1`.
  This ordering is identified by `labels_order_id = "abcd_lex_v1"`.
- Any subset/truncation of labels must preserve this relative order and include
  explicit `labels_abcd` rows plus `labels_order_id` in metadata.

Math:
$$
\text{Order: } L0 \rightarrow L1 \rightarrow L2 \rightarrow L3 \rightarrow L4.
$$

Math:
$$
H_{\mathrm{eff},\,cd,ab}^{(\mu)}=h_{cd,ab}^{(\mu)}.
$$

Code form:
```text
module_inputs       = {E_u, U_np1, U_n_soc0, U_nm1, t_mu, W, labels_abcd, labels_order_id}
module_internal     = {X, Y, A, B, E_uv, E_rs, M_A, M_B, h_pre_j_mu}
module_outputs      = {h_mu_abcd, Heff_mu_abcd}
submodule_handoff   = {L0: X/Y, L1: A/B, L2: M_A&M_B, L3: h_pre_j_mu}
labels_order_id     = "abcd_lex_v1"
```

Index:
- $u,v,r,s$: internal intermediate indices for $L1/L2/L3$ only.
- $a,b,c,d$: final low-energy output indices and public interface.

Validation:
- Definitions in `04-01` and `04-02` must remain consistent with Sections 1-5 in this file.
- Hermiticity check: $\mathrm{Heff}^{(\mu)}=\left(\mathrm{Heff}^{(\mu)}\right)^\dagger$ within tolerance.
- Zero-hop sanity check: $t=0 \Rightarrow \mathrm{Heff}^{(\mu)}=0$.
- Shape/dtype compatibility must hold across all contractions.
