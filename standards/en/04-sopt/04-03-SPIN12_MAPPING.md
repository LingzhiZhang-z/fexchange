# 04-03-SPIN12_MAPPING

This file defines the shared mapping from projected effective-Hamiltonian
outputs to a pseudospin-$\tfrac{1}{2}$ model.
It reads $\mathrm{Heff}_{cd,ab}$ blocks and projects them to spin couplings.
Disk I/O layout/format is defined by `./standards/en/05-io/05-00-IO.md`.
Writing style follows `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`.

## 0) Scope (MUST)
- Input comes from SOPT final-$L3$ outputs in
  `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`, or from FOPT `L3`
  total/process projected `h_eff_4` outputs in
  `./standards/en/04-fopt/04-00-FOPT_FORMALISM.md`.
- This module is a post-processing map; it does not modify `L0..L3`.
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

### 3.2 Derived Decomposition (OUT OF SCOPE FOR CURRENT RUNTIME)
The current runtime exports only the raw exchange matrix $J_{\alpha\beta}^{(\mu)}$
plus the mapping residual from Section 5.
Any further decomposition (`J_iso`, `K`, `D`, `Gamma`, local fields, constants)
is outside the current runtime output contract.

## 4) Gauge/Basis Rule (MUST)
- Couplings depend on local Kramers gauge.
- For reproducibility/comparison, mapping must record the gauge/basis id.
- Any local basis rotation used before projection must be explicit and logged.

## 5) Validation (MUST)
Per $\mu$:
- Hermiticity of input $\mathrm{Heff}^{(\mu)}$.
- Reconstruct
  $\widetilde H^{(\mu)}$ from exported
  output $J$ and record

Math:
$$
r_\mu
=
\frac{
\left\|\widetilde H^{(\mu)}-\mathrm{Heff}^{(\mu)}\right\|_F
}{
\left\|\mathrm{Heff}^{(\mu)}\right\|_F
}.
$$

- The scalar constant $C_{00}^{(\mu)} I\otimes I$ is not exported and may be
  retained only for this residual check.
- Local-field terms $C_{0\alpha}^{(\mu)}$, $C_{\alpha0}^{(\mu)}$ and any other
  non-exchange leakage are not exported as exchange. They must remain visible
  through a failed `mapping_residual` check; implementations must not fold them
  into $J_{\alpha\beta}^{(\mu)}$.
- `mapping_residual <= eps_map` is required. It means the projected Hamiltonian
  is exchange-only up to scalar shift.
- Imaginary part leakage of exported real couplings must be below tolerance.

## 6) Runtime I/O (Summary)
Code form:
```text
inputs_33  = {Heff_mu_abcd, labels_abcd, kramer_basis_id}
outputs_33 = {J_mu[3,3], mapping_residual}
```
