# 01-03-STEVENS_OPERATORS

This file defines the Stevens operator contract in the $|J,M\rangle$ basis.
Scope is restricted to the CEF-compatible subset and multipole operators used in this project.

## 1) Scope (MUST)
MUST:
- Support CEF-relevant ranks: `k` ∈ {2, 3, 4, 6}.
- Support CEF-relevant components per symmetry (Oh and C3v).
- Use basis order `M = -J, ..., J` (ascending).
- Produce `(2J+1) x (2J+1)` complex matrices.

## 2) Convention (MUST)
MUST:
- Use Hutchings-style Stevens operators with Condon-Shortley phase.
- Use tesseral output for runtime (`cos`/`sin`).

Runtime modes:
- `mode='cos'`: cosine tesseral component.
- `mode='sin'`: sine tesseral component.

## 3) CEF-Compatible Subset (MUST)
The following operators MUST be implemented and match `02-03` Section 6 definitions exactly:
- Diagonal: `O20`, `O40`, `O60`
- Oh off-diagonal: `O44c`, `O64c`
- C3v off-diagonal: `O43c`, `O43s`, `O63c`, `O63s`, `O66`

## 4) Multipole Operators (MUST)
MUST provide a multipole operator builder for:
- `"magnetic_dipole"`: Jx, Jy, Jz (rank-1, equivalent to O₁⁰, O₁¹ᶜ, O₁¹ˢ).
- `"electric_quadrupole"`: rank-2 tesseral operators.
- `"magnetic_octupole"`: rank-3 tesseral operators.

## 5) API Contract (MUST)

Code form:
```text
build_cef_stevens_operators(J, symmetry='Oh'|'C3v', mode_q3='cos'|'sin') -> dict[str, NDArray]
build_multipole_operators(J, multipole_type) -> dict[str, NDArray]
```

Rules:
- Invalid symmetry, mode, or multipole_type must fail fast.
- Returned matrices must be deterministic.

## 6) Validation (MUST)
MUST check:
- Hermiticity of tesseral operators.
- `q=0` operators are diagonal where mandated by closed-form definitions.
- CEF subset consistency with `02-03` Section 6.
