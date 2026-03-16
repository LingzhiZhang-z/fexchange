# 03-04-IRREP_CLASSIFICATION

This standard defines the API/data contract for irrep classification of CEF
states and multipole-compatibility reporting.

Related standards:
- CEF construction: `02-03-HCEF`
- Angular-momentum operators: `01-06-ANGULAR_MOMENTUM`
- Kramers/non-Kramers projector outputs: `02-05`, `02-06`

Reference convention:
- Koster/Dresselhaus-style double-group convention.

## 1) Scope, Inputs, and Branching (MUST)

### 1.1 Scope

MUST:
- Work on single-site CEF eigenvectors in the `|J,M>` basis.
- Support point groups `Oh`, `D3d`, `C3v`.
- Support integer and half-integer `J` in one unified projector pipeline.
- Never pre-filter by "integer-only irreps" or "half-integer-only irreps"
  before projector evaluation.

### 1.2 Inputs

```text
inputs = {J, evecs, point_group, n_f?}
require evecs.shape == (int(2*J + 1), n_states)
require point_group in {"Oh", "D3d", "C3v"}
```

Parity factor for inversion-containing groups:

```text
p = (-1)**(2J)
```

Optional consistency check when `n_f` is provided:

```text
require (-1)**n_f == (-1)**(2J)
```

### 1.3 Parity labeling policy

MUST:
- For `Oh`, `D3d`: return exactly one parity-resolved branch.
- For `C3v`: return one parity-agnostic branch (no `g/u` tags).
- Never return dual parity branches in this standard.

## 2) Irrep Naming Contract (MUST)

Each classified state MUST expose all three name layers:
- `irrep_display`: readable label (`Γ...` form).
- `irrep_primary`: implementation key (`Gamma...` form).
- `irrep_aliases`: optional verified aliases (for example `A1g`, `T2u`).

Also MUST expose:
- `mapping_unverified: bool`

Rules:
- If alias mapping is not strictly verified, do not force an alias.
- In that case set `irrep_aliases = []` and `mapping_unverified = true`.

## 3) Static Group Data Contract (MUST)

### 3.1 O* rotational core (order 48)

Class keys and sizes:

```text
E, R, 3C2, 3RC2, C4, RC4, 6C2p, 6RC2p, C3, RC3
1, 1,   3,    3,   6,   6,    6,     6,   8,   8
```

### 3.2 C3v* rotational core (order 12)

Class keys and sizes:

```text
E, R, 2C3, 2RC3, 3sigma_v, 3Rsigma_v
1, 1,   2,    2,        3,         3
```

### 3.3 Character-table storage requirements

MUST:
- Keep character tables as static data.
- Store class sizes together with each table.
- Support complex characters.
- Use Appendix A/B values exactly for this standard.

Validation MUST hold for every irrep row:
- O*: `sum_c n_c |chi(c)|^2 = 48`
- C3v*: `sum_c n_c |chi(c)|^2 = 12`

### 3.4 Inversion extension

MUST:
- Build `Oh*` from `O* x C_i`.
- Build `D3d*` from `C3v* x C_i`.

For inversion-containing groups:

```text
chi(i*g) = p * chi(g),  p in {+1, -1}
```

Parity for `Oh`/`D3d` is determined by `J`:

```text
p = (-1)**(2J)
```

If `n_f` is provided, it is a consistency-check input and MUST satisfy the
same parity (`(-1)^n_f == (-1)^(2J)`).

## 4) Representation Matrices (MUST)

Construct representatives in `|J,M>` basis by Wigner-D:

$$
D^J(\alpha,\beta,\gamma)
= e^{-i\alpha J_z} e^{-i\beta J_y} e^{-i\gamma J_z}.
$$

MUST:
- Construct class-representative matrices for active classes.
- Satisfy `D(E) = I` and unitarity.

Double-group central rotation:

```text
D(R) = (-1)**(2J) * I
```

This MUST come from the same matrix-construction route (no manual override
branch).

For `Oh`, `D3d`:

```text
D(i) = p * I
D(i*g) = D(i) @ D(g)
```

with `p = (-1)**(2J)` (and optional `n_f` consistency check as above).

For `C3v`: no inversion classes.

## 5) Projector Classification (MUST)

Use projection operators:

$$
P_\Gamma = \frac{d_\Gamma}{|G|}
\sum_c \chi_\Gamma(c)^*\,S_c,
\quad S_c = \sum_{g \in c} D^J(g).
$$

Here $S_c$ is the class-summed operator (sum of $D^J(g)$ over all group elements
in conjugacy class $c$). This is equivalent to $\sum_{g \in G} \chi_\Gamma(g)^* D^J(g) \cdot d_\Gamma / |G|$.

MUST NOT use $n_c \cdot D^J(g_c)$ with a single class representative $g_c$;
representation matrices are not class functions and this does not yield
idempotent projectors.

Classification rule:

```text
for each state psi:
    score(Gamma) = ||P_Gamma @ psi||
    label = argmax score
    if max(score) < eps_proj:
        label = "unknown"
```

MUST:
- Evaluate all irreps in the active table.
- Never prune irrep candidates manually before projection.

## 6) Multipole Compatibility Contract (MUST)

Provide static lookup:

```text
allowed_multipoles(irrep, point_group)
```

Unknown irrep/group MUST raise `KeyError`.

Token contract for `allowed_multipoles` values:
- `magnetic_dipole`
- `electric_quadrupole`
- `magnetic_octupole`

### 6.1 Oh single-valued sector (`R = +1`)

| Primary | Display | Alias (parity-resolved) | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|---------|---------|-------------------------|-----------------|---------------------|-------------------|
| `Gamma1` | `Γ1` | `A1g` / `A1u` | no | no | yes |
| `Gamma2` | `Γ2` | `A2g` / `A2u` | no | no | yes |
| `Gamma3` | `Γ3` | `Eg` / `Eu` | no | yes | yes |
| `Gamma4` | `Γ4` | `T1g` / `T1u` | yes | no | yes |
| `Gamma5` | `Γ5` | `T2g` / `T2u` | no | yes | yes |

**Single-valued / double-valued split**

### 6.2 Oh double-valued sector (`R = -1`)

| Primary | Display | Alias | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|---------|---------|-------|-----------------|---------------------|-------------------|
| `Gamma6` | `Γ6` | none (universal) | yes | no | yes |
| `Gamma7` | `Γ7` | none (universal) | yes | no | yes |
| `Gamma8` | `Γ8` | none (universal) | yes | yes | yes |

### 6.3 D3d single-valued sector (`R = +1`)

| Primary | Display | Alias | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|---------|---------|-------|-----------------|---------------------|-------------------|
| `Gamma1+` | `Γ1+` | `A1g` | no | no | no |
| `Gamma2+` | `Γ2+` | `A2g` | yes (`z`) | no | yes |
| `Gamma3+` | `Γ3+` | `Eg` | yes (`x,y`) | yes | yes |
| `Gamma1-` | `Γ1-` | `A1u` | no | no | no |
| `Gamma2-` | `Γ2-` | `A2u` | no | no | yes |
| `Gamma3-` | `Γ3-` | `Eu` | no | yes | yes |

**Single-valued / double-valued split**

### 6.4 D3d/C3v spinor sector (`R = -1`)

| Family | Display | Alias | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|--------|---------|-------|-----------------|---------------------|-------------------|
| `Gamma4` family | `Γ4` | none (universal) | yes (`z`) | no | yes |
| `Gamma5` family | `Γ5` | none (universal) | yes (`x,y`) | yes | yes |
| `Gamma6` family | `Γ6` | none (universal) | yes (`x,y`) | yes | yes |

Rules:
- For `D3d`, use `+/-` parity tags from `p = (-1)**(2J)`.
- If `n_f` is provided, require parity consistency with `J`.
- For `C3v`, use the same spinor-channel rules without parity tags.

## 7) Output Schema and Usage Modes (MUST)

### 7.1 Pipeline mode

When eigenvectors are available in `load_or_build_W`, append symmetry metadata.

Single-branch shape:

```text
{
  "irrep_display": str,
  "irrep_primary": str,
  "irrep_aliases": list[str],
  "mapping_unverified": bool,
  "allowed_multipoles": list[str],
  "excited_irreps": [
    {
      "energy_index": int,
      "irrep_display": str,
      "irrep_primary": str,
      "irrep_aliases": list[str],
      "mapping_unverified": bool
    }, ...
  ]
}
```

### 7.2 Standalone mode

```text
analyze_cef_symmetry(J, point_group, *, B_params=None, evecs=None, n_f=None)
```

MUST:
- Require exactly one of `B_params` and `evecs`.
- Return branch semantics identical to pipeline mode.
- For `C3v`, return one parity-agnostic branch (no inversion parity tags).

## 8) Runtime Validation and Failure Policy (MUST)

MUST validate numerically:
- projector idempotency
- projector completeness
- character-table orthogonality
- reference decomposition regression in the selected convention

```text
eps_proj = 1e-6
assert projector_checks_pass
assert table_orthogonality_pass
```

Failure policy:
- Unknown lookup key -> `KeyError`.
- Numerical validation failure -> hard failure with diagnostic residuals.
- Label as `"unknown"` only when all projection norms are below threshold.

## Appendix A) O* Character Table (normative data)

| Sector | Display | Primary | E | R | 3C2 | 3RC2 | C4 | RC4 | 6C2p | 6RC2p | C3 | RC3 |
|--------|---------|---------|---|---|-----|------|----|-----|------|-------|----|-----|
| single | `Γ1` | `Gamma1` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| single | `Γ2` | `Gamma2` | 1 | 1 | 1 | 1 | -1 | -1 | -1 | -1 | 1 | 1 |
| single | `Γ3` | `Gamma3` | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | -1 | -1 |
| single | `Γ4` | `Gamma4` | 3 | 3 | -1 | -1 | 1 | 1 | -1 | -1 | 0 | 0 |
| single | `Γ5` | `Gamma5` | 3 | 3 | -1 | -1 | -1 | -1 | 1 | 1 | 0 | 0 |
| double | `Γ6` | `Gamma6` | 2 | -2 | 0 | 0 | sqrt(2) | -sqrt(2) | 0 | 0 | 1 | -1 |
| double | `Γ7` | `Gamma7` | 2 | -2 | 0 | 0 | -sqrt(2) | sqrt(2) | 0 | 0 | 1 | -1 |
| double | `Γ8` | `Gamma8` | 4 | -4 | 0 | 0 | 0 | 0 | 0 | 0 | -1 | 1 |

## Appendix B) C3v* Character Table (normative data)

| Sector | Display | Primary | E | R | 2C3 | 2RC3 | 3sigma_v | 3Rsigma_v |
|--------|---------|---------|---|---|-----|------|----------|-----------|
| single | `Γ1` | `Gamma1` | 1 | 1 | 1 | 1 | 1 | 1 |
| single | `Γ2` | `Gamma2` | 1 | 1 | 1 | 1 | -1 | -1 |
| single | `Γ3` | `Gamma3` | 2 | 2 | -1 | -1 | 0 | 0 |
| double | `Γ4` | `Gamma4` | 2 | -2 | 1 | -1 | 0 | 0 |
| double | `Γ5` | `Gamma5` | 1 | -1 | -1 | 1 | i | -i |
| double | `Γ6` | `Gamma6` | 1 | -1 | -1 | 1 | -i | i |

## Appendix C) Alias Mapping (normative, verified part)

`Oh` single-valued aliases:
- `Gamma1 -> A1g` (`p=+1`), `Gamma1 -> A1u` (`p=-1`)
- `Gamma2 -> A2g` (`p=+1`), `Gamma2 -> A2u` (`p=-1`)
- `Gamma3 -> Eg` (`p=+1`), `Gamma3 -> Eu` (`p=-1`)
- `Gamma4 -> T1g` (`p=+1`), `Gamma4 -> T1u` (`p=-1`)
- `Gamma5 -> T2g` (`p=+1`), `Gamma5 -> T2u` (`p=-1`)

`D3d` single-valued aliases:
- `Gamma1+ -> A1g`, `Gamma2+ -> A2g`, `Gamma3+ -> Eg`
- `Gamma1- -> A1u`, `Gamma2- -> A2u`, `Gamma3- -> Eu`

Spinor irreps (`Gamma6/7/8` in cubic core, `Gamma4/5/6` in trigonal core):
- No universal Schoenflies alias is mandated by this standard.
