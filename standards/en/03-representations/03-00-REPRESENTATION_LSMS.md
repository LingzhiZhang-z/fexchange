# 03-00-REPRESENTATION_LSMS

This file defines the LSMS representation contract.

## 1) State Definition
LSMS basis states are

Math:
$$
\lvert \alpha L M_L S M_S \rangle,
\qquad M = M_L + M_S.
$$

Equivalent spin notation in outputs uses $twoS = 2S$.

## 2) Construction Rule
LSMS states are obtained from Coulomb-only Hamiltonian ($H_{\mathrm{int}}$):

Math:
$$
H_{\mathrm{int}}\lvert \psi_a^{\mathrm{LSMS}} \rangle
= E^{\mathrm{int}}_a\lvert \psi_a^{\mathrm{LSMS}} \rangle.
$$

No SOC or CEF term is included at this stage.

### 2.1 Preferred Runtime Generation (Null-Space Route)
The preferred LSMS generator is the runtime/null-space route (not CFP-first).
This is the normative path.

### 2.2 CFP Route (Allowed, Secondary)
CFP-based construction is allowed as a secondary implementation for:
- cross-checking runtime results,
- bootstrap/testing,
- compatibility with existing tables.

When CFP and runtime/null-space outputs disagree (beyond tolerance), runtime
route plus operator-eigenvalue checks ($L^2$, $S^2$, $H_{\mathrm{int}}$) is authoritative.

## 3) Runtime Generation Rules (MUST)
This section is the mandatory generation recipe for the runtime/null-space path.

### 3.1 Mathematical Rule (Highest-Weight via Raising-Operator Kernels)
1. Build $H_{\mathrm{int}}$, $L^2$, $S^2$, $L_z$, $S_z$, $L_+$, $S_+$, $L_-$, $S_-$ in the
   fixed $n_{\mathrm{ele}}$ Fock sector (`basis_id` from
   `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`; operator definitions from
   `./standards/en/02-models/02-04-ANGULAR_MOMENTUM_OPERATORS.md`).
2. Build ML/MS subspaces:

Math:
$$
\mathcal V_{M_L,M_S}
= \ker(L_z-M_L I)\cap\ker(S_z-M_S I).
$$

3. For each target $(L,S)$, define inter-subspace raising maps:

Math:
$$
A_L^{(L,S)} = P_{L+1,S}\,L_+\,P_{L,S},\qquad
A_S^{(L,S)} = P_{L,S+1}\,S_+\,P_{L,S},
$$

where $P_{M_L,M_S}$ projects onto $\mathcal V_{M_L,M_S}$.
4. Define highest-weight candidate space:

Math:
$$
\mathcal H_{L,S}^{\mathrm{hw}}
= \ker\!\left(A_L^{(L,S)}\right)\cap\ker\!\left(A_S^{(L,S)}\right)
\subseteq \mathcal V_{L,S}.
$$

This is the primary runtime null-space definition.

### 3.2 Alpha Fixing Rule (When dim ker > 1)
If $\dim(\mathcal H_{L,S}^{hw}) = r > 1$, $\alpha$ MUST be fixed by diagonalizing
$H_{\mathrm{int}}$ inside this highest-weight subspace:

1. Let $B_{L,S}$ be an orthonormal basis matrix of $\mathcal H_{L,S}^{hw}$.
2. Build projected matrix:

Math:
$$
H^{\mathrm{hw}}_{L,S} = B_{L,S}^{\dagger} H_{\mathrm{int}} B_{L,S}.
$$

3. Diagonalize:

Math:
$$
H^{\mathrm{hw}}_{L,S} u_\alpha = \varepsilon_\alpha u_\alpha.
$$

4. Define canonical highest-weight states:

Math:
$$
\lvert hw_{\alpha,L,S}\rangle = B_{L,S} u_\alpha.
$$

5. Assign $\alpha$ by ascending $\varepsilon_\alpha$; if energies are numerically
   tied, apply deterministic tie-break + phase convention.

This is the required rule for repeated terms.

### 3.2.1 Fixed Coulomb Reference Scale for Alpha Fixing (MUST)
For LSMS internal alpha-fixing (Sections 3.2/3.4), implementations MUST use a
scale-normalized Coulomb reference that does not depend on absolute
$U=F^0$.

Use:
- $F^0=0$,
- $J_H=1$,
- $r_{42}=F^4/F^2$, $r_{62}=F^6/F^2$.

Given

Math:
$$
J_H=\frac{286F^2+195F^4+250F^6}{6435},
\qquad
F^4=r_{42}F^2,\quad F^6=r_{62}F^2,
$$

solve

Math:
$$
F^2=\frac{6435}{286+195r_{42}+250r_{62}},\quad
F^4=r_{42}F^2,\quad
F^6=r_{62}F^2.
$$

Then build the LSMS internal reference operator:

Math:
$$
H_{\mathrm{int}}^{\mathrm{ref}}
= F^2\hat O_2+F^4\hat O_4+F^6\hat O_6.
$$

Notes:
- Any nonzero global scaling of $H_{\mathrm{int}}^{\mathrm{ref}}$ gives the same
  eigenvectors/order; this rule fixes one deterministic scale.
- This rule is for LSMS internal basis construction and ordering only.

### 3.3 Multiplet Generation + Implementation Steps
1. From each $\lvert hw_{\alpha,L,S}\rangle$, generate full multiplet:

Math:
$$
\lvert \alpha L M_L S M_S\rangle
\propto (L_-)^{L-M_L}(S_-)^{S-M_S}\lvert hw_{\alpha,L,S}\rangle.
$$

2. Normalize after each lowering step.
3. Implementation workflow MUST be:
- build subspace bases ($V_{M_L,M_S}$),
- construct $A_L^{(L,S)}$, $A_S^{(L,S)}$,
- solve joint null space,
- if nullity > 1, diagonalize projected $H_{\mathrm{int}}$ to fix $\alpha$,
- generate all `(M_L,M_S)` states by lowering,
- run residual checks for $L^2$, $S^2$, $L_z$, $S_z$, and consistency check with
  $H_{\mathrm{int}}$.

Important: runtime LSMS states MUST NOT be defined by
$\ker(H_{\mathrm{int}}-EI)$ alone.

### 3.4 Runtime Implementation Contract
The runtime implementation SHALL follow this matrix workflow:

1. Sector indexing:
- Build $sector[(M_L,M_S)] \to indices$ in Fock basis.
- For each sector, define embedding matrix $R_{M_L,M_S}$ with shape
  `(dim_fock, dim_sector)`.

2. Raising-kernel solve:
- Build
  $A_L = R_{L+1,S}^\dagger L_+ R_{L,S}$,
  $A_S = R_{L,S+1}^\dagger S_+ R_{L,S}$.
- Solve null space of stacked matrix
  $A = \begin{bmatrix}A_L\\A_S\end{bmatrix}$ to get $B_{L,S}$ (columns are orthonormal hw candidates in
  sector coordinates).
- Lift to full space with $HW_{\mathrm{full}} = R_{L,S} B_{L,S}$.

3. Alpha fixing by projected Coulomb:
- Build $H_{\mathrm{sub}} = R_{L,S}^\dagger H_{\mathrm{int}} R_{L,S}$.
- Project: $H_{\mathrm{hw}} = B_{L,S}^\dagger H_{\mathrm{sub}} B_{L,S}$.
- Diagonalize $H_{\mathrm{hw}}$; rotate basis $B_{L,S} \leftarrow B_{L,S}U$.
- Final highest-weight vectors:
  $\lvert hw_{\alpha,L,S}\rangle = R_{L,S} B_{L,S}(:,\alpha)$.

4. Ladder generation (normalized recursion):
- Orbital lowering:
  $\lvert L,M_L-1;S,S\rangle = [L_- \lvert L,M_L;S,S\rangle] / \sqrt{L(L+1)-M_L(M_L-1)}$.
- Spin lowering:
  $\lvert L,M_L;S,M_S-1\rangle = [S_- \lvert L,M_L;S,M_S\rangle] / \sqrt{S(S+1)-M_S(M_S-1)}$.
- Normalize after each step and apply fixed phase convention.

5. Column assembly:
- Column order MUST follow Section 6.
- Each column stores one $\lvert \alpha L M_L S M_S\rangle$ in `V_fock`.

Code form:
```text
build sector embeddings R[ML,MS]
AL = R[L+1,S]^dag @ L_plus @ R[L,S]
AS = R[L,S+1]^dag @ S_plus @ R[L,S]
B_hw = nullspace(vstack([AL, AS]))
if dim(B_hw) > 1: diagonalize(B_hw^dag @ H_int_sub @ B_hw) to fix alpha
generate multiplets with L_minus and S_minus recursion
```

### 3.5 Boundary Conditions (MUST)
1. If target sector $\mathcal V_{L+1,S}$ is empty, treat $A_L$ as an empty-row
   matrix; the $L_+$ constraint is then automatically satisfied.
2. If target sector $\mathcal V_{L,S+1}$ is empty, treat $A_S$ as an empty-row
   matrix; the $S_+$ constraint is then automatically satisfied.
3. Null-space solve uses the stacked matrix
   $A=\begin{bmatrix}A_L\\A_S\end{bmatrix}$ (vertical stack). If one block is empty, solve with the
   remaining block only.
4. Even at boundaries, if highest-weight candidate dimension is greater than 1,
   $\alpha$ MUST still be fixed by Section 3.2 (projected $H_{\mathrm{int}}$ diagonalization).

## 4) Fock-Basis Expansion Contract
Each LSMS state is represented in Fock basis from
`./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`:

Math:
$$
\lvert \psi_a^{\mathrm{LSMS}} \rangle
= \sum_{\mu} V^{\mathrm{LSMS}}_{\mu a}\,\lvert \mu \rangle_{\mathrm{fock}}.
$$

Storage contract:
- `V_fock.shape = (dim_fock, n_states)`
- column `a` is one LSMS state.

## 5) Label Contract (MUST)
Per state column, labels MUST include:
- `alpha`
- `L`
- `twoS`
- `ML`
- `MS`

Optional text label (`label`/term symbol) is allowed but not sufficient without the fields above.

## 6) Canonical Ordering (MUST)
LSMS column order is deterministic:
1. group by `(alpha, L, twoS)` with deterministic `alpha` rule from runtime
   construction (if CFP is used, it must map to the same order),
2. within each term: `ML` ascending,
3. then `MS` ascending.

Any other ordering must be explicitly declared in metadata with a reversible
map.

## 7) Energy Contract
LSMS energy output MUST be term-decomposed and aligned one-to-one with LSMS columns.

Required per-column coefficients:
- `coef_F0`
- `coef_F2`
- `coef_F4`
- `coef_F6`

### 7.1 Operator-Expectation Rule (MUST)
Define Coulomb decomposition:

Math:
$$
H_{\mathrm{int}} = \sum_{k\in\{0,2,4,6\}} F^k\,\hat O_k.
$$

For LSMS state column $a$ with vector $\lvert \psi_a^{\mathrm{LSMS}}\rangle$:

Math:
$$
\mathrm{coef\_F_k}[a]
=
\langle \psi_a^{\mathrm{LSMS}} \rvert \hat O_k \lvert \psi_a^{\mathrm{LSMS}} \rangle,
\quad k\in\{0,2,4,6\}.
$$

For LSMS state column `a`, total interaction energy is reconstructed by

Math:
$$
E^{\mathrm{LSMS}}_a
=
F0\cdot \mathrm{coef\_F0}[a]
+ F2\cdot \mathrm{coef\_F2}[a]
+ F4\cdot \mathrm{coef\_F4}[a]
+ F6\cdot \mathrm{coef\_F6}[a].
$$

Notes:
- This stage contains Coulomb contributions only (no SOC/CEF term).
- If an implementation also stores a total-energy array, it is a derived field and
  MUST be numerically consistent with the formula above.

### 7.2 Diagonal-Subspace Check for $H_{\mathrm{int}}$ (MUST)
Use LSMS output states to build projected matrix

Math:
$$
\left(H_{\mathrm{int}}^{\mathrm{LSMS}}\right)_{ab}
=
\langle \psi_a^{\mathrm{LSMS}} \rvert H_{\mathrm{int}} \lvert \psi_b^{\mathrm{LSMS}} \rangle.
$$

The off-diagonal part must be numerically small in the LSMS output subspace:

Math:
$$
\max_{a\neq b}
\left|
\left(H_{\mathrm{int}}^{\mathrm{LSMS}}\right)_{ab}
\right|
\le \varepsilon_{\mathrm{diag}}.
$$

Optional diagnostic (recommended): also report off-diagonal norms of each
$\hat O_k$ projection to monitor basis quality.

## 8) Validation
- Orthonormality check: $V_{fock}^\dagger V_{fock} = I$ (within tolerance).
- Dimension check: number of columns equals number of LSMS labels.
- Basis check: `basis_id` must match Fock-basis contract.
- `H_int` projected off-diagonal check must satisfy Section 7.2.

### 8.1 Downstream Interface Metadata (MUST)
MUST:
- LSMS outputs must expose `basis_id` and `orbital_order_id` for downstream binding.
- Metadata must retain `n_orb` and `n_ele` of the working sector.

Code form:
```text
lsms_meta_required = {basis_id, orbital_order_id, n_orb, n_ele, state_order_id}
```

Validation:
- Missing required metadata is an interface failure for downstream `03-01/04-01/04-02`.

## 9) Interface to LSJM (Stage 03-01)
LSMS output is the direct input of LSJM construction defined in the subsequent LSJM standard.

Required interface meaning:
1. `V^{\mathrm{LSMS}}` must contain complete multiplets
   $\lvert \alpha L M_L S M_S\rangle$ with deterministic ordering.
2. LSJM is constructed by block CG transform:

Math:
$$
V^{\mathrm{LSJM}} = V^{\mathrm{LSMS}} U_{\mathrm{LSMS}\to\mathrm{LSJM}}.
$$

3. The LSMS interaction energy part $E^{\mathrm{int}}(\alpha,L,S)$ is passed to
   LSJM as coefficient arrays (`coef_F0/F2/F4/F6`) rather than a single
   pre-combined scalar array.
