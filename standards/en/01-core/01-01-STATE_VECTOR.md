# 01-01-STATE_VECTOR

This file defines mandatory state-vector conventions for all modules.

## 1) Scope (MUST)
- This file defines state objects and gauge/canonical rules.
- This file does not define model Hamiltonians or perturbation formulas.

## 2) State Objects (MUST)
Use three object levels:
1. `BasisDet`: one Slater determinant in Fock basis.
2. `StateVec`: one physical state as a linear combination of `BasisDet`.
3. `StateSet`: a set of `StateVec` stored as a matrix.

Rule:
- `StateVec` is allowed for internal single-state calculations.
- Public interface exchange may use either `StateSet` or a stage-result dict that
  carries equivalent fields (`V_fock`, `labels`, basis metadata).
- A single exported state may be encoded as `StateSet` with `n_states = 1`, or
  as a one-column `V_fock` payload inside a result dict.

Math:
$$
\lvert \psi_j \rangle = \sum_{\alpha=1}^{D_n} c_{\alpha j}\,\lvert \alpha \rangle,
\qquad
V_{\alpha j} = c_{\alpha j}.
$$

Code form:
```text
BasisDet  = int det
StateVec  = {basis_id, n_ele, coeffs[alpha]}
StateSet  = {basis_id, n_ele, state_order_id, V_fock[alpha, j], labels[j], meta?}
```

Index:
- $\alpha$: determinant index in one fixed Fock sector basis.
- $j$: state index.
- $D_n$: sector dimension.

Symbol convention:
- Determinant-basis indices use Greek letters (`alpha`, `beta`, `gamma`, `eta`) in code form.
- State indices use Latin letters (`j`, `k_state`, ...).
- Do not reuse determinant-index symbols as scalar coefficient names in nearby formulas.

Validation:
- `StateVec`/`StateSet` must bind to exactly one `basis_id` and one `n_ele`.
- Coefficients must be complex-valued.

## 3) Basis Binding and Ordering (MUST)
- Determinant ordering is inherited from `./standards/en/01-core/01-00-FOCK_SLATER.md`.
- `basis_id` mismatch across files/modules must fail immediately.
- Any basis transform must preserve traceability to source/target `basis_id`.

Validation:
- `basis_id`, `n_ele`, and determinant order tag must match before algebra.
- `n_orb` may be stored explicitly or derived from the bound basis definition.

## 4) Column Convention for StateSet (MUST)
- State vectors are columns.
- Shape rule:
  - `V_fock.shape = (dim_fock, n_states)`.
- Energy arrays and labels are aligned by column index.

Math:
$$
\Psi = V_{\mathrm{fock}},\qquad
\Psi^\dagger \Psi = I\ \text{(for orthonormal set)}.
$$

Validation:
- Column norm and overlap checks must be deterministic and tolerance-controlled.

## 5) Normalization and Phase Gauge (MUST)
- Every exported `StateVec` must be normalized.
- Global phase must be fixed deterministically.

Math:
$$
\sum_{\alpha} \lvert c_\alpha \rvert^2 = 1.
$$

Gauge rule:
1. Find pivot index $\alpha_\star = \arg\max_\alpha |c_\alpha|$.
2. If tie, choose smallest index.
3. Multiply state by phase so that $c_{\alpha_\star}\in\mathbb{R}_{\ge 0}$.

Validation:
- Same numeric input must produce identical phase-fixed output.

## 6) Truncation and Canonicalization (MUST)
- If sparse truncation is used, the rule must be explicit and deterministic.
- Standard rule:
  - drop terms with `abs(c_alpha) < eps_drop`;
  - renormalize;
  - reapply phase gauge from Section 5.

Validation:
- Canonicalization must be idempotent:
  applying it twice gives identical output.

## 7) Degenerate Subspace Rule (MUST)
- If multiple vectors span a degenerate subspace, basis orientation must be fixed.
- First, diagonalize one declared tie-break operator in that subspace.
- If still degenerate, apply deterministic lexicographic pivot rule column-by-column.

Validation:
- Repeated runs with same inputs must produce same subspace basis order and phases.

## 8) Serialization Contract (MUST)
- This project does not require one universal on-disk `StateSet` container.
- If `StateSet` is serialized, the minimum required metadata is:
  `basis_id`, `n_ele`, `state_order_id`, `labels`.
- `meta`, `n_orb`, `unit`, `schema`, and `energies` are optional/project-specific
  extensions.
- `state_order_id` must be explicit and stable when present.

Code form:
```text
StateSet NPZ:
  V_fock, labels, basis_id, n_ele, state_order_id, meta(optional)
```

Validation:
- Missing minimum binding metadata must fail loading.

## 9) Operator Interface Alignment (MUST)
- When `StateSet` is used as operator input/output, basis metadata must match operator endpoint metadata exactly.
- For number-conserving operators, input and output `basis_id` are identical.
- For number-changing operators, input and output must use different sector `basis_id` values, each consistent with its own `n_ele`.

Code form:
```text
apply_operator(operator, stateset_in) -> stateset_out
require:
  stateset_in.basis_id == operator.basis_id_from
  stateset_out.basis_id == operator.basis_id_to
  stateset_in.n_ele     == operator.sector_from
  stateset_out.n_ele    == operator.sector_to
```
