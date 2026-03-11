# 02-04-ANGULAR_MOMENTUM_OPERATORS

This file defines the angular-momentum operator conventions used by representation standards (`03-00/03-01`).

## 1) Operator Set (MUST)
- Orbital operators: $L_x,L_y,L_z,L_\pm$.
- Spin operators: $S_x,S_y,S_z,S_\pm$.
- Total operators: $J_x,J_y,J_z,J_\pm$.

Definitions:
Math:
$$
L_\pm = L_x \pm iL_y,\qquad
S_\pm = S_x \pm iS_y,\qquad
J = L+S,\quad J_\pm=L_\pm+S_\pm,\quad J_z=L_z+S_z.
$$

## 2) Second-Quantized Form on f Shell (MUST)
Use $\ell=3$, $m=-3,\dots,3$, $\sigma=\pm \frac{1}{2}$:
Math:
$$
L_z=\sum_{m,\sigma} m\,c^\dagger_{m\sigma}c_{m\sigma},
\qquad
S_z=\sum_{m,\sigma}\sigma\,c^\dagger_{m\sigma}c_{m\sigma}.
$$

Math:
$$
L_+ = \sum_{m=-\ell}^{\ell-1}\sum_\sigma
\sqrt{\ell(\ell+1)-m(m+1)}\;
c^\dagger_{m+1,\sigma}c_{m,\sigma},
\qquad
L_- = L_+^\dagger.
$$

Math:
$$
S_+ = \sum_m c^\dagger_{m,+1/2}c_{m,-1/2},
\qquad
S_- = S_+^\dagger.
$$

## 3) Ladder Action in LS Basis (MUST)
Math:
$$
L_\pm\lvert L,M_L\rangle
=
\sqrt{L(L+1)-M_L(M_L\pm1)}
\lvert L,M_L\pm1\rangle,
$$

Math:
$$
S_\pm\lvert S,M_S\rangle
=
\sqrt{S(S+1)-M_S(M_S\pm1)}
\lvert S,M_S\pm1\rangle.
$$

Therefore:
Math:
$$
J_\pm\lvert J,M\rangle
=
\sqrt{J(J+1)-M(M\pm1)}
\lvert J,M\pm1\rangle.
$$

## 4) Contract for 03-00/03-01
- `./standards/en/03-representations/03-00-REPRESENTATION_LSMS.md` MUST use this file for $L_\pm,S_\pm,L_z,S_z$ definitions.
- `./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md` MUST use this file for $J_\pm,J_z$ consistency.
- Any implementation changing these operator definitions must declare a new scheme/version.

## 5) Implementation Contract (MUST)
Use this order:
1. Build one-body matrices for $L_x,L_y,L_z,S_x,S_y,S_z$ in the fixed orbital order.
2. Build ladder operators by definition:
   $L_\pm=L_x\pm iL_y$, $S_\pm=S_x\pm iS_y$, $J_\pm=L_\pm+S_\pm$.
3. Lift one-body matrices to many-body operators through second quantization.
4. Export using `01-02-OPERATOR_IMPLEMENTATION` one-body operator interface.

Code form:
```text
build one_body {Lx,Ly,Lz,Sx,Sy,Sz}
Lpm = Lx ± i*Ly
Spm = Sx ± i*Sy
Jpm = Lpm + Spm
lift to many-body via second quantization
```

Validation:
- $L_- = L_+^\dagger$, $S_- = S_+^\dagger$, $J_- = J_+^\dagger$.
- Commutators must satisfy angular-momentum algebra within tolerance.
