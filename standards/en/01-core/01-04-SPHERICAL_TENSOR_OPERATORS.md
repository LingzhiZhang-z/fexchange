# 01-04-SPHERICAL_TENSOR_OPERATORS

This file defines irreducible spherical tensor operators \(T_k^q\) in one fixed-\(J\) manifold.

## 1) Scope (MUST)
MUST:
- Build \(T_k^q\) in \(|J,M\rangle\) basis with `M=-J,...,J` ascending.
- Support integer and half-integer `J`.
- Support ranks/components with `k >= 0`, `-k <= q <= k`.

## 2) Matrix Elements (MUST)
Use `sympy.physics.wigner.wigner_3j` and unit reduced matrix element convention:

Math:
$$
\langle J,M'|T_k^q|J,M\rangle
= (-1)^{J-M'}\,\begin{pmatrix}J & k & J\\-M' & q & M\end{pmatrix}.
$$

Implementation form:
```text
T[M', M] = (-1)^(J-M') * float(wigner_3j(J, k, J, -M', q, M))
```

## 3) Selection Rules (MUST)
MUST enforce:
- `q = M' - M`
- Triangle condition with fixed manifold: `k <= 2J`
- Entries violating selection rules evaluate to zero numerically.

## 4) Algebraic Properties (MUST)
MUST satisfy (within tolerance):

Math:
$$
(T_k^q)^\dagger = (-1)^q T_k^{-q}.
$$

Orthogonality (Frobenius inner product):

Math:
$$
\mathrm{Tr}\left[(T_k^q)^\dagger T_{k'}^{q'}\right] \propto \delta_{kk'}\delta_{qq'}.
$$

## 5) Multipole Classification (MUST)
By rank parity:
- Odd `k` (`1,3,5`): magnetic multipoles (time-reversal odd).
- Even `k` (`2,4,6`): electric multipoles (time-reversal even).

Naming in API/meta:
- `k=1`: `magnetic_dipole`
- `k=2`: `electric_quadrupole`
- `k=3`: `magnetic_octupole`

## 6) API Contract (MUST)

Code form:
```text
either low-level tensor API:
  build_spherical_tensor(J, k, q) -> NDArray
  build_multipole_set(J, k) -> dict[int, NDArray]
  multipole_type(k) -> 'magnetic' | 'electric'

or higher-level family API:
  build_multipole_operators(J, multipole_type) -> dict[str, NDArray]
```

Exact public function names are not fixed by this standard; the requirement is
that one of the two API styles above be available.

## 7) Validation (MUST)
MUST test across integer and half-integer `J` values:
- Hermiticity-conjugation relation.
- Orthogonality of different `(k,q)` channels.
- Selection-rule sparsity pattern.
