# 06-00-RUNTIME_NUMERICS

This file defines runtime numerical defaults and global input-gate rules.
It is normative for downstream standards and implementations.

## 1) Scope (MUST)
MUST:
- This file defines numeric tolerances, deterministic linear-algebra behavior, and global input-binding gates.
- Physics formulas remain in `02-00/03-00/04-00` standards.

Code form:
```text
apply_to_modules = {02-00,02-01,02-02,02-03,02-04,02-05,02-06,03-00,03-01,04-00,04-01,04-02,04-03,05-00,05-02,05-03}
```

Validation:
- Any override must be explicitly recorded in runtime metadata.

## 2) Floating-Point and Dtype Policy (MUST)
MUST:
- Default real dtype: `float64`.
- Default complex dtype: `complex128`.
- Mixed dtypes in one contraction chain are forbidden unless explicitly cast.

Code form:
```text
dtype_real_default    = float64
dtype_complex_default = complex128
```

Validation:
- Export metadata must include effective dtype.

## 3) Global Tolerance Table (MUST)
MUST:
- Use one shared tolerance table unless a module declares stricter values.

Math:
$$
\varepsilon_{\mathrm{zero}}=10^{-12},\quad
\varepsilon_{\mathrm{norm}}=10^{-10},\quad
\varepsilon_{\mathrm{orth}}=10^{-10},\quad
\varepsilon_{\mathrm{diag}}=10^{-9},
$$

Math:
$$
\varepsilon_{\mathrm{eig\_cluster}}=10^{-10},\quad
\varepsilon_{\mathrm{svd\_rel}}=10^{-12},\quad
\varepsilon_{\mathrm{map}}=10^{-8},\quad
\varepsilon_{\mathrm{herm}}=10^{-10}.
$$

Math:
$$
\varepsilon_{\mathrm{unitary}}=10^{-10},\quad
\varepsilon_{\mathrm{nk\_split}}=10^{-6},\quad
\varepsilon_{\mathrm{mag\_ab}}=10^{-1}.
$$

Code form:
```text
eps_zero        = 1e-12
eps_norm        = 1e-10
eps_orth        = 1e-10
eps_diag        = 1e-9
eps_eig_cluster = 1e-10
eps_svd_rel     = 1e-12
eps_map         = 1e-8
eps_herm        = 1e-10
eps_unitary     = 1e-10
eps_nk_split    = 1e-6
eps_mag_ab      = 1e-1
```

Index:
- `eps_diag`: projected off-diagonal leakage checks in `03-00/03-01`.
- `eps_map`: reconstruction residual in module `04-03`.
- `eps_svd_rel`: null-space cutoff ratio against largest singular value.
- `eps_unitary`: unitarity checks for explicit basis transforms (`05-02`).
- `eps_nk_split`: non-Kramers quasi-doublet splitting threshold (`02-06`), in internal energy unit.
- `eps_mag_ab`: non-Kramers in-plane magnetic-leakage ratio threshold (`02-06`).

Validation:
- Runtime metadata must report active `eps_*`.

## 4) Deterministic Eigen/SVD Rules (MUST)
MUST:
- Hermitian diagonalization uses Hermitian solver (`eigh`-class).
- Eigenpairs are sorted by ascending eigenvalue.
- Degenerate clusters are detected by `eps_eig_cluster`.
- Degenerate-cluster gauge must be fixed by deterministic pivot-phase rule from `./standards/en/01-core/01-01-STATE_VECTOR.md`.

Math:
$$
|\lambda_i-\lambda_j|\le \varepsilon_{\mathrm{eig\_cluster}}
\Rightarrow i,j\text{ in one degenerate cluster}.
$$

Math:
$$
\sigma_r \le \varepsilon_{\mathrm{svd\_rel}}\,\sigma_{\max}
\Rightarrow \sigma_r\text{ belongs to null space}.
$$

Code form:
```text
evals, evecs = eigh(H)
sort evals ascending
cluster by |eval_i - eval_j| <= eps_eig_cluster
fix cluster gauge deterministically

U,S,Vh = svd(A, full_matrices=False)
null_mask = (S <= eps_svd_rel * S.max())
```

Validation:
- Same input must yield identical column order and fixed phase across runs.

## 5) Canonical Phase and Tie-Break (MUST)
MUST:
- State phase follows `./standards/en/01-core/01-01-STATE_VECTOR.md` pivot rule.
- Ties choose smallest determinant index.
- Residual machine-precision ties use lexicographic real-imag sequence rule.

Code form:
```text
pivot = argmax(abs(v))
if tie: choose smallest index
phase-fix so v[pivot] is real and >= 0
```

Validation:
- Canonicalization must be idempotent.

## 6) Hermiticity and Orthonormality Checks (MUST)
MUST:
- Hermiticity check uses normalized Frobenius residual.
- Orthonormality check uses `||V^dag V - I||_F`.

Math:
$$
r_{\mathrm{herm}} = \frac{\|H-H^\dagger\|_F}{\max(\|H\|_F,\varepsilon_{\mathrm{zero}})}
\le \varepsilon_{\mathrm{herm}}.
$$

Math:
$$
r_{\mathrm{orth}} = \|V^\dagger V-I\|_F \le \varepsilon_{\mathrm{orth}}.
$$

Code form:
```text
r_herm = norm(H - H.conj().T, 'fro') / max(norm(H,'fro'), eps_zero)
r_orth = norm(V.conj().T @ V - I, 'fro')
```

Validation:
- Exceeding threshold is a hard failure.

## 7) Runtime Path Defaults (MUST)
MUST:
- Use dense path by default.
- Switch to sparse COO only when estimated density is below threshold.
- Large contractions must support chunking to cap peak memory.

Code form:
```text
density_sparse_switch = 0.15
peak_memory_budget_gb = 8.0
chunk_policy = "auto-by-memory"
```

Validation:
- If runtime actually switches among dense/sparse/chunked paths, metadata must
  record the selected path and chunk policy.

## 8) Global Input Header Gate (MUST)
MUST:
- Every external runtime payload must include:
  `schema_version`, `standard_version`, `basis_id`, `orbital_order_id`, `unit`.

Code form:
```text
required_header = {schema_version, standard_version, basis_id, orbital_order_id, unit}
```

Validation:
- Missing header fields are fatal input errors.

## 9) Global Channel/Binding Gate (MUST)
MUST:
- One run computes one bond only.
- Runtime hopping input is a single matrix (no `mu` axis in input payload).
- `\mu` in formulas is a bond label for this run, not an array axis.
- Cross-interface bindings must be checked before contraction.

Code form:
```text
require t.shape == (n_orb, n_orb)
optional bond_label: string

require input.basis_id == core.basis_id
require input.orbital_order_id == core.orbital_order_id
```

Validation:
- Any illegal extra channel axis or binding mismatch is a hard failure.

## 10) Error Policy (MUST)
MUST:
- Input schema/binding failures are hard failures.
- No silent fallback to output snapshots.

Code form:
```text
if schema_check_fail:  raise InputSchemaError
if binding_check_fail: raise InputBindingError
```

Validation:
- Error message must include field name and expected vs actual shape/dtype/value domain.

## 11) Runtime Metadata Contract (MUST)
MUST:
- Every persisted stage output must include enough numerical metadata to
  identify the active dtype/tolerance profile.
- Runtime-path fields are required only when multiple paths are actually in use.

Code form:
```text
numerics_meta = {
  dtype_real, dtype_complex,
  eps_zero, eps_norm, eps_orth, eps_diag,
  eps_eig_cluster, eps_svd_rel, eps_map, eps_herm,
  eps_unitary, eps_nk_split, eps_mag_ab,
  density_sparse_switch, peak_memory_budget_gb,
  chunk_policy(optional), selected_path(optional)
}
```

Validation:
- Missing essential dtype/tolerance metadata is a contract violation.
