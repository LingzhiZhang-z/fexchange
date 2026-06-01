# 04-01-PRECOMPUTE

This file defines SOPT precompute Levels $L0$ and $L1$.
Conventions follow `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`: in f-shell runs, $E_0=0$, and denominator handling is performed downstream in module `04-02`.
Disk I/O layout/format is defined by `./standards/en/05-io/05-00-IO.md`.
Writing style follows `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`.
Execution order is defined in `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`.
Naming conventions are inherited from Section 0.1 of `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`.

## 0) Variable Classes (Submodule Scope, MUST)
This file covers $L0/L1$ and uses three variable classes:
- Input variables: read from external interfaces or upstream level outputs.
- Intermediate variables: internal working variables only; not exposed as this level's outputs.
- Output variables: interface variables emitted by this level for downstream levels.

Per-level definition:
- $L0$: input `{}`; intermediate `{sign/workspace}`; output `{X, Y}`.
- $L1$: input `{X, Y, U_np1, U_n_low, U_nm1}`; intermediate `{workspace}`; output `{A, B}`.

Constraint:
- Runtime hopping/Kramer objects are not part of this module.
- Denominator assembly and route summation are handled in `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`.

## 0.1) Boundary with External Hopping/Kramer Inputs (MUST)
MUST:
- Levels `L0/L1` must not consume external hopping inputs.
- Levels `L0/L1` must not consume Kramer/projector inputs in the default
  `runtime.kramer_source = "stevens"` route.
- In the explicit `runtime.kramer_source = "manual"` route, L1 consumes the
  run-input `f^n` Fock-basis Kramers state file as its main low-energy
  subspace and the L1 artifact must be run-scoped.
- Hopping/Kramer schemas are enforced only in module `04-02`.
- `L1` outputs must carry deterministic axis metadata required by downstream runtime binding:
  LSJM-subspace order id for `j` axes and orbital-order identity for `kappa/p/q` mapping.

Code form:
```text
L0_L1_inputs_exclude = {t_mu, W, labels_abcd, E_u}
L1_meta_required = {j_order_id, orbital_order_id, vertex_axis_order_id}
```

Validation:
- If `L0/L1` runtime receives hopping/Kramer payloads, they must be ignored or rejected by contract mode.
- `L1` artifacts without downstream-binding metadata are invalid.

## 1) Level 0: Fock-Basis Primitive Transitions
MUST:
- This level constructs transition elements on canonical Fock basis only, with no external state-file dependency and no site labels $i/j$.
- Fock basis ordering and fermionic-sign rules must be fixed (inherit from `01-physics` conventions).
- This level uses a unified low $\to$ high storage convention and stores $f^\dagger$ direction only.

Math:
$$
X^{\kappa,n}_{\alpha\beta}
\equiv
\langle \alpha^{n+1} \rvert f_{\kappa}^{\dagger} \lvert \beta^{n} \rangle.
$$

Math:
$$
Y^{\kappa,n-1}_{\beta \gamma}
\equiv
\langle \beta^{n} \rvert f_{\kappa}^{\dagger} \lvert \gamma^{n-1} \rangle.
$$

Math:
$$
\langle \beta^{n} \rvert f_{\kappa} \lvert \alpha^{n+1} \rangle
=
\left(X^{\kappa,n}_{\alpha\beta}\right)^{\ast},
\qquad
\langle \gamma^{n-1} \rvert f_{\kappa} \lvert \beta^{n} \rangle
=
\left(Y^{\kappa,n-1}_{\beta \gamma}\right)^{\ast}.
$$

Code form:
```text
build X_n[kappa] and Y_nm1[kappa] with f_dag only
recover reverse direction by conjugation
```

Index:
- $\alpha,\beta,\gamma$: Fock-basis labels used in this level.
- $\kappa$: generic local single-site orbital index (site-agnostic).

Validation:
- Sector mapping must be correct ($n \to n+1$ or $n \to n-1$).
- Fermionic signs must match the agreed bit-ordering convention.
- All sites must reuse the same $X/Y$ tensors; no site-specific variant is allowed at $L0$.
- Conjugate reverse matrices must not be cached as separate persistent artifacts.

Output (MUST):
- This level must emit $X^{\kappa,n}$ and $Y^{\kappa,n-1}$ (or an equivalent recoverable representation).
- This output is the direct input of $L1$.

## 2) Level 1: Local Transition Vertices (RS/ED)
MUST:
- This level defines generic single-site vertices without introducing site labels $i/j$.
- This level applies basis rotation on intermediate sectors ($f^{n+1}$ and
  $f^{n-1}$) according to `model.scheme`.
- In `model.scheme = "RS"`, intermediate-sector transforms are LSJM transforms.
- In `model.scheme = "ED"`, intermediate-sector transforms are IONED transforms
  from module `03-05`.
- In `runtime.kramer_source = "stevens"` (default), this level projects the
  $f^n$ leg from Fock basis to the SOC-lowest LSJM subspace (the $f^n$ LSJM
  ground multiplet).
- In `runtime.kramer_source = "manual"`, this level projects the $f^n$ leg
  from Fock basis to the user-supplied orthonormal Kramers low-energy basis.
- Do not introduce Kramers labels $a,b,c,d$ at this level.
- $U^{(m)}$ is the selected column-wise Fock-to-working-basis transform matrix
  in sector $f^m$: LSJM for `RS`, IONED for `ED`.
- $U^{n,\mathrm{low}}$ is the column-wise transform from $f^n$ Fock basis to
  the selected low-energy subspace:
  `U_n_soc0` for `stevens`, and `K_fock` for `manual`.
- Intermediate-state indices use only $u,v,r,s$.

Math:
$$
A^{\kappa,n}_{u j}
=
\sum_{\alpha,\beta}
\left(U^{n+1}_{\alpha u}\right)^{\ast}
X^{\kappa,n}_{\alpha\beta}
\left(U^{n,\mathrm{low}}_{\beta j}\right),
$$

Math:
$$
B^{\kappa,n-1}_{j v}
=
\sum_{\beta,\gamma}
\left(U^{n,\mathrm{low}}_{\beta j}\right)^{\ast}
Y^{\kappa,n-1}_{\beta\gamma}
\left(U^{n-1}_{\gamma v}\right).
$$

Math:
$$
\langle u^{n+1} \rvert f_{\kappa}^{\dagger} \lvert j^{n,\mathrm{low}}\rangle
=
A^{\kappa,n}_{u j},
\qquad
\langle j^{n,\mathrm{low}} \rvert f_{\kappa} \lvert u^{n+1}\rangle
=
\left(A^{\kappa,n}_{u j}\right)^{\ast}.
$$

Math:
$$
\langle j^{n,\mathrm{low}} \rvert f_{\kappa}^{\dagger} \lvert v^{n-1}\rangle
=
B^{\kappa,n-1}_{j v},
\qquad
\langle v^{n-1} \rvert f_{\kappa} \lvert j^{n,\mathrm{low}}\rangle
=
\left(B^{\kappa,n-1}_{j v}\right)^{\ast}.
$$

Code form:
```text
build generic {A_kappa_n[u,j], B_kappa_nm1[j,v]} without site labels
rotate (n+1)/(n-1) legs with U_np1, U_nm1
select U_np1/U_nm1 from LSJM when scheme=RS, from IONED when scheme=ED
project n-leg to selected low subspace with U_n_low
recover reverse direction by complex conjugation
```

Validation:
- Vertex tensor dimensions must match sector dimensions.
- Operator direction ($\dagger$ / non-$\dagger$) must match index semantics.
- In `stevens`, this level must use $U^{n,\mathrm{soc0}}$, and its column
  space must cover only the SOC-lowest LSJM subspace.
- In `manual`, this level must use the user-supplied orthonormal `K_fock`
  columns, and its `j` axis is the manual low-energy/Kramers state axis.
- `ED` must not replace the main-sector low-energy LSJM projector; it replaces
  only the adjacent intermediate-sector transforms and energies.
- This level must not contain Kramers indices $a,b,c,d$ or projector $W$.

Output (MUST):
- This level must emit vertex tensors $A^{\kappa,n}_{u j}$ and $B^{\kappa,n-1}_{j v}$.
- This output is the direct input of module 04-02 $L2$.
