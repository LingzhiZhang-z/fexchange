# 04-REF-DERIVATION

This file is the archived derivation reference for the previous 04-02
contraction specification.
Normative rules are in `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`.
This English file is the authoritative reference copy after the standards-tree
restructure.

This file defines the legacy runtime Levels $L2$, $L3$, and $L4$ for SOPT.
Disk I/O layout/format is defined by `./standards/en/05-io/05-00-IO.md`.
Writing style follows `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`.
This archived derivation keeps the previous $L3/L4$ split only as a reference;
the current runtime order is defined in
`./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`.

## 0) Variable Classes (Submodule Scope, MUST)
This file covers $L2/L3/L4$ and uses three variable classes:
- Input variables: read from external interfaces or upstream level outputs.
- Intermediate variables: internal working variables only; not exposed as this level's outputs.
- Output variables: interface variables emitted by this level for downstream levels/callers.

Per-level definition:
- $L2$: input `{A, B, E_u}`; intermediate `{E_uv, E_rs, KA, KB}`; output `{K}`.
- $L3$: input `{K, t_mu}`; intermediate `{workspace}`; output `{h_pre_j_mu}`.
- $L4$: input `{h_pre_j_mu, W, labels_abcd}`; intermediate `{h_pre_mu}`; output `{h_mu_abcd, Heff_mu_abcd}`.

## 0.1) External Runtime Input Schema for $L3/L4$ (MUST)
MUST:
- Hopping (`t_mu`, `mu_labels`) and Kramer projector (`W`, `kramer_labels`) are external runtime inputs.
- Global header gate from `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md` is mandatory:
  `schema_version`, `standard_version`, `basis_id`, `orbital_order_id`, `unit`.
- Channel order in outputs must preserve input `mu_labels` order exactly.

Math:
$$
t\_mu\in\mathbb C^{n_\mu\times n_{\mathrm{orb}}\times n_{\mathrm{orb}}}
\ \text{or}\ 
\mathbb C^{n_{\mathrm{orb}}\times n_{\mathrm{orb}}}
\ (\text{promote to }n_\mu=1).
$$

Math:
$$
W\in\mathbb C^{n_j\times n_k},
\qquad
labels\_{abcd}\in\mathbb Z^{n_{\mathrm{out}}\times 4},
\qquad
0\le a,b,c,d<n_k.
$$

Code form:
```text
if t_mu.ndim == 2: t_mu = t_mu[None, :, :]
require len(mu_labels) == t_mu.shape[0]
require W.shape[0] == h_pre_j_mu.shape[0]     # j-axis binding
require labels_abcd.shape[1] == 4
require labels_abcd.max() < W.shape[1]
```

Index:
- `mu_labels[k]` is the canonical label of channel axis `k`.
- `j` axis of `W` must match the LSJM low-SOC subspace order inherited from module `03-01`.

Validation:
- Any schema/binding mismatch is a hard failure before contraction.
- `W` orthonormality check: `W^dag W = I` within `eps_orth`.

NPZ key contract (when file payloads are used):
Code form:
```text
hopping npz required keys:
  t_mu, mu_labels, n_orb, hopping_name,
  schema_version, standard_version, basis_id, orbital_order_id, unit

kramer npz required keys:
  W, kramer_labels, n_j,
  schema_version, standard_version, basis_id, orbital_order_id, unit

labels npz/json required keys:
  labels_abcd, labels_order_id
```

## 1) Legacy Level 2: Intermediate-State Summation and Bare Kernel $K$ (Frozen Definition)
MUST:
- This level constructs the bare kernel $K$ with no Kramers indices.
- This level performs site binding: bind the $L1$ index $\kappa$ to concrete
  site-orbital indices.
- Explicit summation uses route-A indices $(u,v)$ and route-B indices $(r,s)$.
- This level only performs intermediate-state summation and denominator
  assembly; $W$ must not appear in the definition of $K$.
- The input energy must be the single-site form `E_u` (provided separately for
  the $n+1$ and $n-1$ sectors); $E_{uv}$ and $E_{rs}$ are composed internally.
- For the LSJM intermediate-state path, `E_u` must be reconstructed from energy
  coefficients:
  $E_u = F0\,c_{u,F0}+F2\,c_{u,F2}+F4\,c_{u,F4}+F6\,c_{u,F6}+\zeta\,c_{u,\zeta}$.
- The coefficients $c_{u,F0/F2/F4/F6/\zeta}$ must come from the 03-01 module's
  energy-component outputs (operator expectation-value coefficients); this
  level must not replace them with hard-coded SOC formulas.
- To ensure $K = K_A + K_B$ can be used directly downstream with the unified
  $t_{pq}t_{p'q'}^\ast$ contraction, this section adopts the route-B
  "aligned-slot" definition, equivalent to swapping $(pq)\leftrightarrow(p'q')$
  in the natural route-B kernel.
- If a two-site composite intermediate-state index is used, write
  $m=(u,v)$ for route A and $n=(r,s)$ for route B.

Math:
$$
A_{u j}^{i,p} \equiv A_{u j}^{\kappa=p,n},
\qquad
B_{j v}^{j,q} \equiv B_{j v}^{\kappa=q,n-1}.
$$

Math:
$$
E_i^{n+1}(u)
\equiv
F0\,c_{u,F0}^{n+1}
+ F2\,c_{u,F2}^{n+1}
+ F4\,c_{u,F4}^{n+1}
+ F6\,c_{u,F6}^{n+1}
+ \zeta\,c_{u,\zeta}^{n+1}.
$$

Math:
$$
E_j^{n-1}(v)
\equiv
F0\,c_{v,F0}^{n-1}
+ F2\,c_{v,F2}^{n-1}
+ F4\,c_{v,F4}^{n-1}
+ F6\,c_{v,F6}^{n-1}
+ \zeta\,c_{v,\zeta}^{n-1}.
$$

Math:
$$
E_{uv} \equiv E_i^{n+1}(u)+E_j^{n-1}(v).
$$

Math:
$$
K_{j_3 j_4,\,j_1 j_2}^{A;\,pq,p'q'}
= \sum_{u,v}
\frac{
\left(A_{u j_3}^{i,p'}\right)^{\ast}
B_{j_4 v}^{j,q'}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^{\ast}
}{-E_{uv}}.
$$

Math:
$$
E_i^{n-1}(r)
\equiv
F0\,c_{r,F0}^{n-1}
+ F2\,c_{r,F2}^{n-1}
+ F4\,c_{r,F4}^{n-1}
+ F6\,c_{r,F6}^{n-1}
+ \zeta\,c_{r,\zeta}^{n-1}.
$$

Math:
$$
E_j^{n+1}(s)
\equiv
F0\,c_{s,F0}^{n+1}
+ F2\,c_{s,F2}^{n+1}
+ F4\,c_{s,F4}^{n+1}
+ F6\,c_{s,F6}^{n+1}
+ \zeta\,c_{s,\zeta}^{n+1}.
$$

Math:
$$
E_{rs} \equiv E_i^{n-1}(r)+E_j^{n+1}(s).
$$

Math:
$$
A_{s j}^{j,q} \equiv A_{s j}^{\kappa=q,n},
\qquad
B_{j r}^{i,p} \equiv B_{j r}^{\kappa=p,n-1}.
$$

Math:
$$
K_{j_3 j_4,\,j_1 j_2}^{B;\,pq,p'q'}
= \sum_{r,s}
\frac{
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^{\ast}
\left(B_{j_1 r}^{i,p'}\right)^{\ast}
A_{s j_2}^{j,q'}
}{-E_{rs}}.
$$

Math:
$$
K^{pq,p'q'}_{j_3 j_4,\,j_1 j_2}
=
K^{A;\,pq,p'q'}_{j_3 j_4,\,j_1 j_2}
+
K^{B;\,pq,p'q'}_{j_3 j_4,\,j_1 j_2}.
$$

From the effective Hamiltonian to the $A/B$ bare-kernel form:

Math:
$$
\left(h_{\mathrm{pre}}^{(\mu)}\right)_{j_3 j_4,\,j_1 j_2}
=
\sum_n
\frac{
\langle j_3,j_4|H_{\mathrm{hop}}^{(\mu)}|n\rangle
\langle n|H_{\mathrm{hop}}^{(\mu)}|j_1,j_2\rangle
}{-E_n},
$$

Math:
$$
\left(h_{\mathrm{pre}}^{(\mu)}\right)_{j_3 j_4,\,j_1 j_2}
=
\sum_{p q p' q'}
t_{pq}^{(\mu)}\,t_{p'q'}^{(\mu)\ast}\,
K_{j_3 j_4,\,j_1 j_2}^{pq,p'q'}.
$$

Route-A numerator factorization:

Math:
$$
\langle j_3,j_4|f_{jq'}^{\dagger}f_{ip'}|u,v\rangle
\langle u,v|f_{ip}^{\dagger}f_{jq}|j_1,j_2\rangle
=
\left(A_{u j_3}^{i,p'}\right)^{\ast}
B_{j_4 v}^{j,q'}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^{\ast}.
$$

Route-B numerator factorization:

Math:
$$
\langle j_3,j_4|f_{ip}^{\dagger}f_{jq}|r,s\rangle
\langle r,s|f_{jq'}^{\dagger}f_{ip'}|j_1,j_2\rangle
=
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^{\ast}
\left(B_{j_1 r}^{i,p'}\right)^{\ast}
A_{s j_2}^{j,q'}.
$$

Code form:
```text
KA[j3,j4,j1,j2,p,q,p2,q2] = sum_{u,v} conj(A_i_p2[u,j3]) * B_j_q2[j4,v] * A_i_p[u,j1] * conj(B_j_q[j2,v]) / (-Euv[u,v])
KB[j3,j4,j1,j2,p,q,p2,q2] = sum_{r,s} B_i_p[j3,r] * conj(A_j_q[s,j4]) * conj(B_i_p2[j1,r]) * A_j_q2[s,j2] / (-Ers[r,s])
K = KA + KB
```

Index:
- $K[...,p,q,p',q']$ is the bare kernel with no Kramers indices (the $L2$
  output).
- The superscripts `$i$/$j$` on $A/B$ are site labels only, not LSJM subspace
  indices `$j_1,j_2,j_3,j_4$`.

Validation:
- The $L2$ output kernel index order must be fixed as
  $[j_3,j_4,j_1,j_2,p,q,p',q']$ (or an explicitly documented equivalent order).
- $W$ must not enter the definition of $K$.
- The denominator sign must remain consistent with the $E_0=0$ convention from
  `04-00`.

Output (MUST):
- This level must emit $K$ independently, together with any required
  shape/label metadata.
- This level must not persist $E_{uv}$ or $E_{rs}$ as public outputs.
- This level's output is the direct input of
  `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`.

## 2) Level 3: Fix Hopping and Contract
MUST:
- Contract hopping with bare kernel $K$.
- Index conventions must remain fully consistent with `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`.

Math:
$$
h_{\mathrm{pre},\,j_3 j_4,\,j_1 j_2}^{(\mu)}
= \sum_{p q p' q'}
t_{pq}^{(\mu)}\,t_{p'q'}^{(\mu)\ast}
K_{j_3 j_4,\,j_1 j_2}^{pq,p'q'}.
$$

Code form:
```text
h_pre_j_mu[j3,j4,j1,j2] = sum_{p,q,p2,q2} t_mu[p,q] * conj(t_mu[p2,q2]) * K[j3,j4,j1,j2,p,q,p2,q2]
```

Index:
- $h_{\mathrm{pre},j}^{(\mu)}$: post-contraction, pre-Kramers-projection channel tensor (defined on the $f^n$ SOC-lowest LSJM subspace).

Validation:
- Index order between $t$ and $K$ must be consistent.
- Zero-hop check: if $t=0$, then $h_{\mathrm{pre},j}^{(\mu)}=0$.

Output (MUST):
- This level must emit $h_{\mathrm{pre},j}^{(\mu)}$ (code-equivalent name: `h_pre_j_mu`) as an independent output.
- This output is the direct input of $L4$.

## 3) Level 4: Fix Kramers Basis and Build Final Outputs
MUST:
- After $L3$, apply outer $W$ projection.
- Final public interface must use $a,b,c,d$ semantics.
- $W$ must map from the $f^n$ SOC-lowest LSJM subspace to the CEF/Kramers basis.

Math:
$$
h_{\mathrm{pre},\,cd,ab}^{(\mu)}
= \sum_{j_3,j_4,j_1,j_2}
(W_{j_3 c})^{\ast}(W_{j_4 d})^{\ast}
h_{\mathrm{pre},\,j_3 j_4,\,j_1 j_2}^{(\mu)}
W_{j_1 a}W_{j_2 b}.
$$

Math:
$$
H_{\mathrm{eff},\,cd,ab}^{(\mu)}
= h_{\mathrm{pre},\,cd,ab}^{(\mu)}.
$$

Code form:
```text
h_pre_mu = project_with_W(h_pre_j_mu, W)
h_mu_abcd = h_pre_mu
Heff_mu_abcd = h_mu_abcd
```

Index:
- $h_{\mathrm{pre},cd,ab}^{(\mu)}$: channel tensor after $W$ projection.
- $h_{\mu,abcd}$: final channel tensor in target Kramers basis.
- $\mathrm{Heff}_{abcd}^{(\mu)}$: effective Hamiltonian for a fixed single $\mu$-bond channel.

Validation:
- Hermiticity check: $\mathrm{Heff}^{(\mu)}=\left(\mathrm{Heff}^{(\mu)}\right)^{\dagger}$ within tolerance.
- $W$ projection dimensions must match $h_{\mathrm{pre},j}^{(\mu)}$.

Output (MUST):
- This level must emit $h_{\mu,abcd}$ and $\mathrm{Heff}_{abcd}^{(\mu)}$ as independent outputs.
- When no additional post-rotation is introduced by this spec, the two are numerically identical.
- $\mathrm{Heff}_{abcd}^{(\mu)}$ is the direct input of
  `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md` when spin-model export is enabled.

## 4) Runtime I/O (Level Summary)
MUST:
- Inputs must include $\{A,B,E_u,K,t^{(\mu)},W,\mathrm{labels}_{abcd}\}$.

Code form:
```text
inputs_L2_L4  = {A, B, E_u, K, t_mu, W, labels_abcd}
outputs_L2    = {K}
outputs_L3    = {h_pre_j_mu}
outputs_L4    = {h_mu_abcd, Heff_mu_abcd}
```
