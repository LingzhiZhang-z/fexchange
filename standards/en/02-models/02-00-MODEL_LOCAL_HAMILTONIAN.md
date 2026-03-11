# 02-00-MODEL_LOCAL_HAMILTONIAN

This file defines the local single-ion model decomposition and cross-term contracts.
Reference source (single entry for model-layer files):
- [Nature Computational Science (2024), s43246-024-00634-w](https://www.nature.com/articles/s43246-024-00634-w)

## 1) Local-Hamiltonian Decomposition (MUST)
MUST:
- Use the additive model with three terms only.
- Keep term semantics consistent with files `02-01/02-02/02-03`.

Math:
$$
H_{\mathrm{local}} = H_{\mathrm{int}} + H_{\mathrm{soc}} + H_{\mathrm{cef}}.
$$

Code form:
```text
H_local = H_int + H_soc + H_cef
```

Index:
- `int`: Coulomb multiplet term.
- `soc`: atomic spin-orbit term.
- `cef`: crystal electric-field term.

Validation:
- Any extra local term requires a new model scheme/version.

## 2) Coulomb-Term Contract (MUST)
MUST:
- Use Slater-Condon parameters `F0/F2/F4/F6`.
- Retain the `F0` sector shift in absolute energies.

Math:
$$
H_{\mathrm{int}} = F^0\,\frac{n(n-1)}{2} + F^2\hat O_2 + F^4\hat O_4 + F^6\hat O_6.
$$

Code form:
```text
H_int = F0 * n*(n-1)/2 + F2*O2 + F4*O4 + F6*O6
```

Index:
- `n`: electron number in one fixed sector.
- `O2/O4/O6`: rank-resolved Coulomb operators (defined in `./standards/en/02-models/02-01-HINT_FORM.md`).

Validation:
- `F0` contributes equally to all states in the same `n` sector.

## 3) SOC-Term Contract (MUST)
MUST:
- Use one SOC strength parameter `zeta`.
- Adopt the operator definition from `./standards/en/02-models/02-02-HSOC_FORM.md`.

Math:
$$
H_{\mathrm{soc}} = \zeta\sum_i \mathbf l_i\cdot\mathbf s_i.
$$

Code form:
```text
H_soc = zeta * sum_i (l_i · s_i)
```

Index:
- `zeta`: SOC strength.

Validation:
- SOC matrix must be Hermitian.

## 4) CEF-Term Contract (MUST)
MUST:
- Use Stevens-operator form.
- Allow only symmetry branches defined in `./standards/en/02-models/02-03-HCEF_FORM.md`.

Math:
$$
H_{\mathrm{cef}} = \sum_{k,q} B_k^q O_k^q.
$$

Code form:
```text
H_cef = sum_{k,q} B[k,q] * O[k,q]
```

Index:
- `B_k^q`: CEF parameters in one fixed energy unit.
- `O_k^q`: Stevens operators.

Validation:
- `B_k^q`, `F^k`, and `zeta` must share one energy unit system.

## 5) Hierarchy and Solve Order (MUST)
MUST:
- Use RS-style hierarchy
  $|H_{\mathrm{int}}| > |H_{\mathrm{soc}}| \gg |H_{\mathrm{cef}}|$.
- Solve in strict order: `Hint -> Hsoc -> Hcef`.

Code form:
```text
step1: diagonalize(H_int) -> LSMS
step2: diagonalize(H_soc, within fixed LS blocks) -> LSJM
step3: apply(H_cef, on selected SOC manifold)
```

Validation:
- If implementation mixes different `LS` blocks in step2, it must declare a new scheme/version.

## 6) Energy-Reference Rule (MUST)
MUST:
- Keep additive absolute-energy bookkeeping.
- If a stage reports relative energies, explicitly record the reference in metadata.

Math:
$$
E_{\mathrm{total}} = E_{\mathrm{int}} + E_{\mathrm{soc}} + E_{\mathrm{cef}}.
$$

Code form:
```text
E_total = E_int + E_soc + E_cef
```

Validation:
- Cross-file comparisons require identical absolute-energy convention.

## 7) SOPT Interface Note (MUST)
MUST:
- For SOPT, intermediate-state energies come from `H_int + H_soc`.
- Kramers-pair zero-reference and effective `U/J` conventions belong to SOPT standards (`04-00/04-01/04-02`), not this file.
