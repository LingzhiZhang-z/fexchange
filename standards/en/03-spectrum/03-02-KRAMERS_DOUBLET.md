# 03-02-KRAMERS_DOUBLET

This file defines the Kramers-doublet construction contract and the projected
Pauli/g-tensor contract.
Model formulas of `H_int/H_soc/H_cef` remain in `02-00/02-01/02-02/02-03`.
Wannier90-derived input constraints are defined in
`./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`.

## 1) Scope and Inputs (MUST)
MUST:
- This file applies to odd-electron `f^n` cases where a Kramers doublet is required.
- Input Hamiltonian must be the local model from `02-00/02-01/02-02/02-03`.
- Angular-momentum operators `Jx/Jy/Jz` must follow `./standards/en/01-core/01-06-ANGULAR_MOMENTUM.md`.
- Default target is the lowest CEF-split doublet in the SOC-low LSJM manifold.

Code form:
```text
inputs = {H_local, Jx, Jy, Jz, n, symmetry_branch, cef_params}
require n % 2 == 1
```

Validation:
- If `n` is even, this contract is not applicable.

## 2) Kramers-Pair Condition (MUST)
MUST:
- A selected doublet must satisfy Kramers degeneracy and time-reversal pairing.
- Let `Theta` be the anti-unitary time-reversal operator:

Math:
$$
\Theta \lvert k_1\rangle = \lvert k_2\rangle,\qquad
\Theta \lvert k_2\rangle = -\lvert k_1\rangle.
$$

Code form:
```text
check |E_k1 - E_k2| <= eps_eig_cluster
check TR_pair_residual <= eps_norm
```

Validation:
- Failure of degeneracy or TR pairing is a hard failure.

### 2.1) Time-Reversal Operator Implementation (MUST)
The anti-unitary time-reversal operator on the f-shell single-particle basis is
$\Theta=U_T K$, where $K$ is complex conjugation.

For a single spin-orbital $\lvert m,\sigma\rangle$ with $\ell=3$:

Math:
$$
\Theta\lvert m,\sigma\rangle
=(-1)^{\ell-m+\frac{1}{2}-\sigma}\lvert{-m},{-\sigma}\rangle.
$$

$U_T$ is a $14\times14$ monomial matrix (permutation with phases).
It maps orbital $p(m,\sigma)$ to orbital $\bar p(-m,-\sigma)$, so for $m\neq0$
the mapping **crosses** $m$-blocks; only the $m=0$ block maps to itself.

Math:
$$
(U_T)_{\bar p,\,p}
=(-1)^{\ell-m_p+\frac{1}{2}-\sigma_p},
\qquad
\bar p = p(-m_p,-\sigma_p),
$$

all other entries are zero.

In the project orbital order ($p=0\ldots13$, $m=-3\ldots3$,
$\sigma=-\tfrac{1}{2},+\tfrac{1}{2}$ for each $m$):

Code form:
```text
# build 14x14 single-particle U_T
U_T = np.zeros((14,14), dtype=complex)
for p in range(14):
    m_p, sigma_p = orbital_map(p)            # decode (m, sigma)
    p_bar = orbital_index(-m_p, -sigma_p)    # target orbital
    phase = (-1)**(3 - m_p + 0.5 - sigma_p)
    U_T[p_bar, p] = phase

# many-body action on state vector psi (n-electron Fock sector)
Theta_psi = U_T_n @ psi.conj()
```

For the many-body $n$-electron Fock sector:
1. For each Slater determinant, apply $U_T$ to every occupied orbital,
   collect single-particle phases, and include the sign from reordering
   the time-reversed set into canonical bit order.
2. For a state vector $\lvert\psi\rangle=\sum_\alpha c_\alpha\lvert\alpha\rangle$:

Math:
$$
\Theta\lvert\psi\rangle
= \sum_\alpha c_\alpha^{\ast}\bigl(U_T^{(n)}\lvert\alpha\rangle\bigr),
$$

where $U_T^{(n)}$ is the many-body time-reversal unitary in the $n$-electron sector.

Validation:
- $\Theta^2=(-1)^n$ on $n$-particle states (for $\ell=3$, single-particle $\Theta^2=-1$).
- For odd $n$: Kramers theorem guarantees at least two-fold degeneracy.
- TR-pair residual: $\|\Theta\lvert k_1\rangle-\lvert k_2\rangle\|\le\varepsilon_{\mathrm{norm}}$.

## 3) Projected Pauli-Map Contract (MUST)
MUST:
- Define projector onto the chosen doublet:

Math:
$$
P=\lvert k_1\rangle\langle k_1\rvert+\lvert k_2\rangle\langle k_2\rvert.
$$

- Project angular momentum:

Math:
$$
M_\alpha \equiv P J_\alpha P,\qquad \alpha\in\{x,y,z\}.
$$

- In doublet space, use Pauli expansion:

Math:
$$
M_\alpha = \frac{1}{2}\sum_{\beta\in\{x,y,z\}}\Lambda_{\alpha\beta}\sigma_\beta.
$$

- Define g-tensor by one explicit module constant:

Math:
$$
g_{\alpha\beta}=c_g\,\Lambda_{\alpha\beta}.
$$

Code form:
```text
M_alpha = K.conj().T @ J_alpha @ K
Lambda[alpha,beta] from Pauli decomposition of M_alpha
g_tensor = c_g * Lambda
```

Validation:
- `c_g` must be recorded in metadata.
- `M_alpha` must be Hermitian within `eps_herm`.

## 4) SU(2) Gauge Freedom and Invariants (MUST)
MUST:
- Any internal basis change in the doublet must be SU(2):

Math:
$$
\lvert k_a'\rangle=\sum_b \lvert k_b\rangle U_{ba},\qquad U\in SU(2).
$$

Math:
$$
M_\alpha' = U^\dagger M_\alpha U,\qquad
\Lambda'=\Lambda R(U),\ R(U)\in SO(3).
$$

- Physical comparison must use gauge-invariant quantities (for example, singular values of `g_tensor` or eigenvalues of `g_tensor g_tensor^T`).

Validation:
- Cross-run comparisons must not rely on raw gauge-dependent column phases/orders.

## 5) Deterministic Gauge-Fixing (MUST)
MUST:
- After selecting a doublet, fix gauge in this order:
1. Diagonalize `M_Jz`.
2. Apply residual U(1) phase so off-diagonal of `M_Jx` is real.
3. Enforce TR pair convention
   (`Theta|k1>=|k2>`, `Theta|k2>=-|k1>`).
4. Apply deterministic sign harmonization to make `g` signs as consistent as possible.

Code form:
```text
step1: U_z = eigh(M_Jz).evecs
step2: U_phase from phase(M_Jx[0,1])
step3: U_tr enforce TR pair convention
step4: choose among allowed SU(2) discrete transforms with deterministic tie-break
```

Sign harmonization rule (MUST):
- Objective: maximize sign consistency among `(g_x, g_y, g_z)`.
- If multiple candidates tie, apply tie-break in fixed order:
  `g_z >= 0`, then `g_y >= 0`, then `g_x >= 0`.

Validation:
- Repeated runs on identical inputs must produce identical `(k1,k2)` ordering and `g` sign convention.

## 6) Output Contract (MUST)
MUST output:
- `kramer_vectors`: two columns `[k1, k2]` in canonical gauge.
- `M_Jx`, `M_Jy`, `M_Jz`: projected `2x2` matrices.
- `Lambda` and `g_tensor`.
- `g_components` if axis form is used.
- `gauge_meta`: `{unitary_total, tr_residual, pauli_residual, sign_rule}`.
- `symmetry_meta` when irrep classification is enabled (see `02-07`).
  - Contract fields: `irrep_display`, `irrep_primary`, `irrep_aliases`,
    `mapping_unverified`, `allowed_multipoles`, `excited_irreps`.
  - For inversion-containing groups, parity is uniquely determined by `J`
    (no dual parity branches).

Code form:
```text
outputs = {
  kramer_vectors, M_Jx, M_Jy, M_Jz, Lambda, g_tensor, gauge_meta,
  symmetry_meta?
}
```

Validation:
- All outputs must share one `basis_id` and `orbital_order_id`.

## 7) Runtime Checks (MUST)
MUST:
- Hermiticity check for projected matrices.
- TR-pair residual check.
- Pauli-map residual check:

Math:
$$
r_{\mathrm{pauli}}=
\max_{\alpha}
\frac{
\left\|M_\alpha-\frac{1}{2}\sum_\beta\Lambda_{\alpha\beta}\sigma_\beta\right\|_F
}{
\max\left(\|M_\alpha\|_F,\varepsilon_{\mathrm{zero}}\right)
}.
$$

Code form:
```text
require r_pauli <= eps_norm
```

Validation:
- Any check failure is a hard failure.
