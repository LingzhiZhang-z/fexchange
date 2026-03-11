# 01-05-STEVENS_TENSOR_CONVERSION

This file defines the conversion contract between Stevens operators and spherical tensor operators.

## 1) Scope (MUST)
MUST provide bidirectional mapping between:
- Stevens tesseral operators \(O_k^{q,c}\), \(O_k^{q,s}\)
- Spherical tensor operators \(T_k^q\)

for `k <= 6`.

## 2) Conversion Formulas (MUST)
For `q=0`:

Math:
$$
O_k^0 = \alpha_{k0}(J)\, T_k^0.
$$

For `q>0`:

Math:
$$
O_k^{q,c} = \frac{\alpha_{kq}(J)}{\sqrt2}\left[(-1)^q T_k^q + T_k^{-q}\right],
$$

Math:
$$
O_k^{q,s} = \frac{\alpha_{kq}(J)}{i\sqrt2}\left[(-1)^q T_k^q - T_k^{-q}\right].
$$

Note: under the unit-RME convention of `01-04`, conversion coefficients are generally `J`-dependent and MUST be computed numerically.

## 3) Numerical Coefficient Policy (MUST)
MUST compute coefficients by matrix comparison at runtime:

Code form:
```text
alpha = <B, O> / <B, B>
```
where:
- `O` is the Stevens matrix,
- `B` is the matching tensor basis combination,
- `<A,B> = Tr(A† B)`.

Hardcoded literature tables are optional; runtime computation is normative.

## 4) Inverse Mapping (MUST)
MUST provide inverse mapping from tensor components back to Stevens tesseral channels using the same coefficient convention.

## 5) API Contract (MUST)

Code form:
```text
stevens_to_tensor_coefficient(J, k, q, mode='cos') -> complex
convert_stevens_to_tensors(J, k, q, mode='cos') -> dict[int, complex]
convert_tensor_to_stevens(J, k, q, mode='cos') -> dict[str, complex]
```

## 6) Validation (MUST)
MUST validate round-trip numerically for integer and half-integer manifolds.
Recommended coverage: `J = 2, 5/2, 4, 7/2`.

Validation criterion:

Math:
$$
\|O - O_{\text{reconstructed}}\|_F < \varepsilon,
\qquad
\|T - T_{\text{reconstructed}}\|_F < \varepsilon.
$$
