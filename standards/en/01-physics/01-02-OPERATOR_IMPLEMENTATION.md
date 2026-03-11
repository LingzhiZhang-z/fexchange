# 01-02-OPERATOR_IMPLEMENTATION

This file defines how operators are represented and implemented in code, consistent with
`./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
and state-vector conventions.

## 1) Scope (MUST)
- This file defines implementation contracts, not new physics.
- Physics-level definitions of $L/S/J$ operators are specified in later model-layer standards.
- Any operator implementation must be traceable to `basis_id_from`/`basis_id_to` and bit rules from `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`.

## 2) Three-Level Operator Structure (MUST)
Use a strict three-level structure:
1. Primitive op: one fermionic operator (`cdag(p)` or `c(p)`).
2. Monomial: complex coefficient times an ordered primitive-op list.
3. Operator: finite sum of monomials plus metadata.

Math:
$$
\hat O = \sum_t g_t \prod_{r=1}^{m_t} a_{t,r},\qquad
a_{t,r}\in\{c_p^\dagger,\;c_p\}.
$$

Code form:
```text
PrimitiveOp  = {kind: "cdag"|"c", p: int}
Monomial     = {coef: complex, ops: list[PrimitiveOp]}    # ordered
Operator     = {terms: list[Monomial], metadata: {...}}
```

Index:
- $t$: monomial index.
- $r$: primitive-op position in one monomial.
- $p$: orbital index (`0..n_orb-1`).
- $g_t$: scalar coefficient of monomial $t$ (not a basis index).

Validation:
- Primitive `p` must be within basis orbital range.
- Monomial operator list order must be explicit and preserved.

## 3) Canonical Ordering and Simplification (MUST)
- Internal canonical form is normal-ordered:
  all creation ops on the left, annihilation ops on the right.
- Reordering must use fermionic anti-commutation:
  $c_p c_q^\dagger = \delta_{pq} - c_q^\dagger c_p$.
- Duplicate creators on same index or duplicate annihilators on same index in one monomial give zero.
- After normalization, equal monomials must be merged by coefficient summation.

Math:
$$
\{c_p,c_q^\dagger\}=\delta_{pq},\qquad
\{c_p,c_q\}=0,\qquad
\{c_p^\dagger,c_q^\dagger\}=0.
$$

Code form:
```text
normalize(monomial) -> list[monomial]     # may branch due to delta terms
combine_like_terms(operator) -> operator
```

Validation:
- Canonicalization must be deterministic.
- Zero-coefficient terms must be dropped using fixed tolerance.

## 3.1) Two-Body Index Canonical Rule (MUST)
For quartic fermion monomials, use one canonical index order:

Math:
$$
\hat O^{(2)}_{ijkl}=c_i^\dagger c_j^\dagger c_k c_l,
\qquad i<j,\ k<l.
$$

Code form:
```text
two_body_key = (i, j, k, l) with i<j and k<l
term        = coef * cdag(i) cdag(j) c(k) c(l)
```

Rule:
- If input has `i>j`, swap creators and multiply coefficient by `-1`.
- If input has `k>l`, swap annihilators and multiply coefficient by `-1`.
- If any repeated creator index (`i=j`) or repeated annihilator index (`k=l`) appears, term is zero.
- In this subsection, `i,j,k,l` are orbital indices, not site labels.

## 4) Action on Fock Determinants (MUST)
- Bitstring action follows `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`.
- Primitive action sign uses parity below index:
  $(-1)^{N_{<p}}$.
- Monomial action applies primitive ops from right to left.

Math:
$$
c_p^\dagger\lvert det\rangle=
\begin{cases}
0,& n_p=1\\
(-1)^{N_{<p}(det)}\lvert det\cup\{p\}\rangle,& n_p=0
\end{cases}
$$

Math:
$$
c_p\lvert det\rangle=
\begin{cases}
(-1)^{N_{<p}(det)}\lvert det\setminus\{p\}\rangle,& n_p=1\\
0,& n_p=0
\end{cases}
$$

Code form:
```text
apply_c_dag(det, p) -> (phase, det_new) | None
apply_c(det, p)     -> (phase, det_new) | None
apply_monomial(det, monomial) -> (amp, det_new) | None
```

Validation:
- Primitive results must obey zero-action rules.
- Sign must match parity from bit ordering.

## 4.1) Action on StateVec/StateSet (MUST)
- Operator action on states must follow column-vector convention.
- Matrix-element convention:
  $O_{\beta\alpha}=\langle\beta|\hat O|\alpha\rangle$.
- `StateSet` action is right-multiplication in column convention.

Math:
$$
d_\beta = \sum_{\alpha} O_{\beta\alpha} c_\alpha,
\qquad
V_{\mathrm{out}} = O\,V_{\mathrm{in}}.
$$

Code form:
```text
apply_operator_to_statevec(operator, statevec_in) -> statevec_out
apply_operator_to_stateset(operator, stateset_in) -> stateset_out
```

Validation:
- `stateset_in` must match `operator.basis_id_from` and `operator.sector_from`.
- `stateset_out` must match `operator.basis_id_to` and `operator.sector_to`.
- Column order of `stateset_in` must be preserved in `stateset_out`.
- `stateset_out.state_order_id` must equal `stateset_in.state_order_id`.

## 5) Input/Intermediate/Output for Operator Construction (MUST)
- Input variables:
  coefficient tensors (`O_pq`, `V_pqrs`, ...),
  basis metadata (`basis_id_from`, `basis_id_to`, `n_orb`, sector info).
- Intermediate variables:
  expanded monomial list, normalized terms, temporary hash maps.
- Output variables:
  canonical `Operator` object; optional matrix/cache form derived from it.

Code form:
```text
build_operator(inputs) -> Operator
operator_to_matrix(operator, sector_basis) -> sparse_matrix (optional)
```

Validation:
- `basis_id_from` and `basis_id_to` must be explicit.
- Input-state `basis_id` mismatch with `basis_id_from` must fail.
- `sector_from/sector_to` must match actual operator rank and action.

## 6) Serialization Contract (MUST)
- Primary exchange format: term-list JSON.
- Optional acceleration format: sparse matrix cache (`npz`) with mandatory metadata.

Code form:
```text
JSON fields:
  schema, name, basis_id_from, basis_id_to, orbital_order_id, n_orb,
  sector_from, sector_to, terms=[{coef:[re,im], ops:[[kind,p],...]}]

NPZ fields (optional):
  row, col, data_re, data_im, shape,
  basis_id_from, basis_id_to, orbital_order_id, sector_from, sector_to, name
```

Validation:
- JSON term ordering must be stable after canonicalization.
- Matrix cache without matching metadata must be rejected.

## 7) Minimal Test Set (MUST)
1. Anti-commutation identities on random determinants.
2. Hermitian consistency:
   `dagger(dagger(O)) == O`.
3. Ladder checks:
   $L_- = L_+^\dagger$, $S_- = S_+^\dagger$.
4. Number-operator commutators:
   $[N,c_p^\dagger]=c_p^\dagger$, $[N,c_p]=-c_p$.
5. Deterministic canonicalization:
   same input terms always produce identical sorted output terms.
