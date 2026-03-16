# 04-03-SPIN12_MAPPING

This file defines post-SOPT mapping from `04-02-RUNTIME_CONTRACTION` outputs to a
pseudospin-$\tfrac{1}{2}$ model.
It reads $\mathrm{Heff}_{cd,ab}^{(\mu)}$ and projects it to spin couplings.
Disk I/O layout/format is defined by `./standards/en/05-io/05-00-IO.md`.
Writing style follows `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`.

## 0) Scope (MUST)
- Input comes from level-$L4$ outputs in `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`.
- This module is a post-processing map; it does not modify `L0..L4`.
- Applicable only when each site low-energy space is two-dimensional
  (Kramers pseudospin-$\tfrac{1}{2}$ doublet).

## 1) Input Contract (MUST)
For each bond/channel $\mu$:
- matrix elements
  $\left(\mathrm{Heff}^{(\mu)}\right)_{cd,ab}
  =\langle c,d \rvert \mathrm{Heff}^{(\mu)} \lvert a,b \rangle$,
- fixed local Kramers basis order on each site (`+,-` or equivalent),
- basis/gauge metadata from upstream mapping.

If local dimension is not `2 x 2`, this module must fail.

## 2) Operator Basis and Projection (MUST)
Define
$\sigma^0=I_2,\sigma^x,\sigma^y,\sigma^z$ and
$S^\alpha=\frac{1}{2}\sigma^\alpha$ ($\alpha=x,y,z$).

For one $\mu$ channel, reshape $\mathrm{Heff}^{(\mu)}$ as a $4\times4$ matrix on
$\lvert a,b\rangle$ basis and expand:

Math:
$$
\mathrm{Heff}^{(\mu)}
=
\sum_{\eta,\nu\in\{0,x,y,z\}}
C_{\eta\nu}^{(\mu)}\,
\sigma_i^\eta \otimes \sigma_j^\nu,
$$

Math:
$$
C_{\eta\nu}^{(\mu)}
=
\frac{1}{4}\,\mathrm{Tr}
\!\left[
\left(\sigma_i^\eta \otimes \sigma_j^\nu\right)
\mathrm{Heff}^{(\mu)}
\right].
$$

## 3) Spin-$\tfrac{1}{2}$ Exchange Form (MUST)
Primary exported model must be

Math:
$$
H_{\mathrm{spin}}^{(\mu)}
\equiv
\sum_{\alpha,\beta\in\{x,y,z\}}
J_{\alpha\beta}^{(\mu)} S_i^\alpha S_j^\beta.
$$

Coefficient map:

Math:
$$
J_{\alpha\beta}^{(\mu)} = 4C_{\alpha\beta}^{(\mu)}.
$$

Equivalent trace form:

Math:
$$
J_{\alpha\beta}^{(\mu)}
=
\mathrm{Tr}\!\left[
\left(\sigma_i^\alpha\otimes \sigma_j^\beta\right)\mathrm{Heff}^{(\mu)}
\right].
$$

### 3.1 Raw Exchange-Coefficient Naming (MUST)
Primary exchange output is the full matrix $J_{\alpha\beta}^{(\mu)}$.
Component naming is:

Math:
$$
J^{(\mu)}=
\begin{pmatrix}
J_x^{(\mu)} & J_{xy}^{(\mu)} & J_{xz}^{(\mu)}\\
J_{yx}^{(\mu)} & J_y^{(\mu)} & J_{yz}^{(\mu)}\\
J_{zx}^{(\mu)} & J_{zy}^{(\mu)} & J_z^{(\mu)}
\end{pmatrix},
$$

with
$J_x\equiv J_{xx}$, $J_y\equiv J_{yy}$, $J_z\equiv J_{zz}$.
Diagonal elements are kept as-is (no extra remapping).

### 3.2 Derived Decomposition (MUST, from $J$)
In addition to raw $J$, module 04-03 must export the derived set from the raw
$J$ matrix:
- isotropic:
  $J_{\mathrm{iso}}=\frac{1}{3}(J_x+J_y+J_z)$,
- bond-default Kitaev parameter:
  for default $z$-bond, define
  $K^{(z\text{-bond})}=J_z-J_{\mathrm{iso}}$,
- DM vector:
  $D_x=\frac{1}{2}(J_{yz}-J_{zy})$,
  $D_y=\frac{1}{2}(J_{zx}-J_{xz})$,
  $D_z=\frac{1}{2}(J_{xy}-J_{yx})$,
- symmetric anisotropy:
  $\Gamma=\frac{1}{2}(J+J^\mathsf{T})-J_{\mathrm{diag}}$ with
  $J_{\mathrm{diag}}=\mathrm{diag}(J_x,J_y,J_z)$.

Note:
- `K_mu` in this module means Kitaev-type exchange parameter, not the level-$L2$
  kernel tensor `K`.

### 3.3 Non-Exchange Terms (MUST define, optional in model fit)
Identity and one-site terms are not part of the exchange-only model
$\sum_{\alpha\beta}J_{\alpha\beta}S_i^\alpha S_j^\beta$.
They are defined by

Math:
$$
\mathrm{const}^{(\mu)} = C_{00}^{(\mu)},\quad
h_{i,\alpha}^{(\mu)} = 2C_{\alpha 0}^{(\mu)},\quad
h_{j,\alpha}^{(\mu)} = 2C_{0\alpha}^{(\mu)}.
$$

## 4) Gauge/Basis Rule (MUST)
- Couplings depend on local Kramers gauge.
- For reproducibility/comparison, mapping must record the gauge/basis id.
- Any local basis rotation used before projection must be explicit and logged.

## 5) Validation (MUST)
Per $\mu$:
- Hermiticity of input $\mathrm{Heff}^{(\mu)}$.
- Reconstruct
  $\widetilde H^{(\mu)}$ from exported
  outputs $(J,J_{\mathrm{iso}},K,\mathbf D,\Gamma,\mathrm{const},\mathbf h_i,\mathbf h_j)$ and check

Math:
$$
r_\mu
=
\frac{
\left\|\widetilde H^{(\mu)}-\mathrm{Heff}^{(\mu)}\right\|_F
}{
\left\|\mathrm{Heff}^{(\mu)}\right\|_F
}
\le \varepsilon_{\mathrm{map}}.
$$

- Imaginary part leakage of exported real couplings must be below tolerance.
- If `K` is exported for default $z$-bond, check
  $K = J_z - J_{\mathrm{iso}}$.

## 6) Runtime I/O (Summary)
Code form:
```text
inputs_33  = {Heff_mu_abcd, labels_abcd, kramer_basis_id}
outputs_33 = {J_mu[3,3], J_iso_mu, K_mu, D_mu[3], Gamma_mu[3,3], const_mu, h_i_mu[3], h_j_mu[3], residual_mu}
```
