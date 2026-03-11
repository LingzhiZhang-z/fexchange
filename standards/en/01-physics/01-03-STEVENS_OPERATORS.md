# 01-03-STEVENS_OPERATORS

This file defines the general Stevens operator contract in the \(|J,M\rangle\) basis.
It extends `02-models/02-03-HCEF_FORM.md` Section 6 from a CEF subset to all ranks/components used in this project.

## 1) Scope (MUST)
MUST:
- Support ranks `0 <= k <= 6`.
- Support components `-k <= q <= k`.
- Use basis order `M = -J, ..., J` (ascending).
- Produce `(2J+1) x (2J+1)` complex matrices.

## 2) Convention (MUST)
MUST:
- Use Hutchings-style Stevens operators with Condon-Shortley phase.
- Use tesseral output for runtime (`cos`/`sin`) and complex output for conversion.

Runtime modes:
- `mode='cos'`: cosine tesseral component.
- `mode='sin'`: sine tesseral component.
- `mode='complex'`: complex spherical component.

## 3) Rank-1 Contract (MUST)
MUST satisfy:

Math:
$$
O_1^0 = J_z,\qquad
O_1^{1,c} = J_x,\qquad
O_1^{1,s} = J_y.
$$

## 4) CEF-Compatible Subset (MUST)
The following subset MUST match `02-03` Section 6 definitions exactly:
- `O20`, `O40`, `O60`
- `O44c`, `O64c`
- `O43c`, `O43s`, `O63c`, `O63s`, `O66`

This subset is used by `build_cef_stevens_operators(...)` for backward-compatible CEF behavior.

## 5) Generalized Construction (MUST)
For all `(k,q)` with `k <= 6`, implementation MUST provide a valid operator matrix via:
- direct Hutchings polynomial form where defined, or
- equivalent tensor conversion through `01-04` and `01-05`.

For `q > 0`, tesseral reconstruction is:

Math:
$$
O_k^{q,c} = \frac{1}{\sqrt2}\left[(-1)^q O_k^q + O_k^{-q}\right],
\qquad
O_k^{q,s} = \frac{1}{i\sqrt2}\left[(-1)^q O_k^q - O_k^{-q}\right].
$$

## 6) API Contract (MUST)

Code form:
```text
build_stevens_operator(J, k, q, *, mode='cos') -> NDArray
build_stevens_set(J, k, *, modes='tesseral'|'complex') -> dict[str, NDArray]
build_cef_stevens_operators(J, symmetry='Oh'|'C3v', mode_q3='cos'|'sin') -> dict[str, NDArray]
```

Rules:
- `mode` is ignored for `q=0`.
- Invalid `k`, `q`, or mode must fail fast.
- Returned matrices must be deterministic.

## 7) Validation (MUST)
MUST check:
- Hermiticity of tesseral operators.
- `q=0` operators are diagonal where mandated by closed-form definitions.
- CEF subset equality with legacy implementation.
