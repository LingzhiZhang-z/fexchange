# 05-04-RUN_INPUT

This file defines the single-file runtime input contract.
Goal: one human-readable file is the only external control surface.

## 1) Scope (MUST)
MUST:
- Runtime inputs must come from one file only.
- Default format is TOML.
- No hidden defaults outside this file are allowed in production mode.
- MPI process-layout controls are not part of this input file contract.
- Runtime backend layout is implementation-defined and remains outside this
  input-file contract.

Code form:
```text
input_source = ./run_input.toml
```

Validation:
- Missing input file is `FXE-INPUT-001`.

## 2) File Identity (MUST)
MUST:
- Canonical filename: `run_input.toml`.
- Required top-level keys:
  - `schema_version`
  - `standard_version`
  - `run_id`
  - `title`

Code form:
```text
required_top = {schema_version, standard_version, run_id, title}
```

Validation:
- Missing key is `FXE-INPUT-002`.

## 3) Required Sections (MUST)
MUST:
- Required tables:
  - `[paths]`
  - `[runtime]`
  - `[checks]`
- No unsupported extra top-level sections are allowed.
- Optional tables:
  - `[units]`
  - `[physics]`
  - `[physics_nm1]`
  - `[physics_np1]`
  - `[model]`
  - `[sopt]`
  - `[sopt_nm1]`
  - `[sopt_np1]`
  - `[inputs]`
  - `[sources]`
- Conditionally required tables (by execution window `LMSM ... end_level`):
  - `[inputs]` (required if window includes `L2` or `L3` or `L4`)
  - `[sopt]` (required if window includes `L3` or `L4`)
- `[sources]` may be omitted in the input file; the loader may materialize and
  normalize it from labels or input filename stems before validating canonical
  source names.
- `[physics]` and `[model]` may be omitted at input time; downstream runtime may
  resolve core parameters from unique upstream artifacts on disk instead.

Code form:
```text
required_core_sections = {paths, runtime, checks}
require_conditional_sections(end_level)
normalize_sources(inputs, sources)
```

Validation:
- Missing required section is `FXE-INPUT-002`.
- Unsupported extra top-level section is `FXE-INPUT-003`.

## 4) Field Contract (MUST)
MUST:
- `[units]`:
  - `energy` (string, optional; `meV` or `eV`; default `meV`)
  - runtime loader MUST normalize all energy-like inputs to one internal unit system
    before downstream stages consume them
- `[physics]`:
  - `n_ele` (int, `1..13`)
  - `RE` (string, optional; `auto` or one of `Ce/Pr/Nd/Pm/Sm/Eu/Gd/Tb/Dy/Ho/Er/Tm/Yb`)
  - `F2_ratio` / `F4_ratio` / `F6_ratio` (float; optional explicit ratio-source triple; all-or-none)
  - internal-derived only: `r42 = F4_ratio/F2_ratio`, `r62 = F6_ratio/F2_ratio`
  - constraint: `F2_ratio != 0`
  - explicit input `F2_ratio/F4_ratio/F6_ratio` define only the ratios `r42/r62`; their common scale carries no independent runtime meaning
  - runtime absolute `F2/F4/F6` used in LSJM ordering and L3/L4 denominator reconstruction MUST be reconstructed from `sopt.Jh` and the derived ratios per `02-01-HINT`
  - if `RE != "auto"` and no explicit ratio triple is present, implementations may derive default ratio-source values from `RE`
- `[physics_nm1]`, `[physics_np1]`:
  - optional branch overrides for adjacent sectors `f^(n-1)` and `f^(n+1)`
  - allowed fields: `RE`, `F2_ratio`, `F4_ratio`, `F6_ratio`
  - explicit branch ratio keys are all-or-none
  - if branch explicit ratio keys are absent and branch `RE != "auto"`, implementations may derive branch ratio-source defaults from branch `RE`
  - otherwise branch ratios fall back to the resolved main-sector ratios
- `[model]`:
  - `scheme` (string, optional when section exists; currently only `RS` is supported)
- `[sopt]`:
  - `U` (float; physically $U = F^0$, the zeroth Slater-Condon parameter)
  - `Jh` (float)
  - `zeta` (float; required unless derived from `physics.RE`)
  - `offset` (float, optional; default `0`)
  - `energy_reference` (string, optional; `lsjm_ground` or `zero`; default `lsjm_ground`)
- `[sopt_nm1]`, `[sopt_np1]`:
  - optional branch overrides for adjacent sectors `f^(n-1)` and `f^(n+1)`
  - allowed fields: `U`, `Jh`, `zeta`, `offset`
  - missing fields fall back field-by-field to resolved main `[sopt]`
  - `offset` defaults to `0`
  - if branch `RE != "auto"` and branch `zeta` is omitted, implementations may
    derive branch `zeta` from the branch `RE` preset instead of inheriting the
    main-sector `zeta`
- `[sopt]` / `[physics]` cross-reference note (MUST):
  - `F^0` is NOT a field in `[physics]` because within a fixed `n` sector
    it contributes a constant $F^0 n(n-1)/2$ to all states and does not
    affect LMSM/LSJM relative energies.
  - For SOPT energy denominators (cross-sector differences $f^{n\pm1}$ vs $f^n$),
    $F^0 = U$ is taken from `[sopt].U`.
  - Implementations MUST use `sopt.U` as $F^0$ when reconstructing absolute
    LSJM energies across sectors for intermediate-state denominators.
  - Implementations MUST reconstruct absolute branch `F2/F4/F6` from branch
    `Jh` and branch ratios:
    `F2 = 6435*Jh / (286 + 195*r42 + 250*r62)`, `F4 = r42*F2`, `F6 = r62*F2`.
  - If branch overrides exist, implementations MUST use branch-resolved
    ratios from `physics_nm1/physics_np1` together with branch `Jh` from
    `sopt_nm1/sopt_np1` for `f^(n-1)` and `f^(n+1)` denominator construction,
    while the main `[physics]` / `[sopt]` remain the default source for omitted
    branch fields.
  - `sopt.energy_reference` controls the main-sector denominator reference:
    - `zero`: use $E_\mathrm{ref}=0$
    - `lsjm_ground`: use the selected `f^n` reference state `u0` from LSJM and
      reconstruct
      $E_\mathrm{ref}=\mathrm{offset}^{(n)} + F^{0,(n)}c_{F0}(u0) + F^{2,(n)}c_{F2}(u0) + F^{4,(n)}c_{F4}(u0) + F^{6,(n)}c_{F6}(u0)$
      (without the `zeta * coef_zeta` term)
- `[sources]`:
  - table is optional at input time
  - `hopping_label` (string; user-facing label for `L2/L3/L4`)
  - `projection_label` (string; user-facing label for `L4`)
  - internal-canonical only:
    - `hopping_name` (stable cache/key token normalized from the first non-empty
      source among `sources.hopping_name`, `sources.hopping_label`,
      `inputs.hopping_label`, `stem(inputs.hopping_file)`)
    - `kramer_name` (stable cache/key token normalized from the first non-empty
      source among `sources.kramer_name`, `sources.projection_label`,
      `inputs.projection_label`, `stem(inputs.projector_file)`)
- `[paths]`:
  - `output_root` (string, MUST equal `"./outputs"` in this standard version)
- `[inputs]`:
  - `hopping_file`
  - `projector_file` (required if window includes `L4`)
- `[runtime]`:
  - `end_level` (string: `LMSM`, `LSJM`, `L0`, `L1`, `L2`, `L3`, or `L4`)
- `[checks]`:
  - `strict_mode` (bool)
  - `eps_profile` (string, e.g. `default`)

Validation:
- Type/value-domain mismatch is `FXE-INPUT-003`.

Code form:
```text
r42_input = F4_ratio_input / F2_ratio_input
r62_input = F6_ratio_input / F2_ratio_input
F2 = 6435 * Jh / (286 + 195*r42_input + 250*r62_input)
F4 = r42_input * F2
F6 = r62_input * F2
level_order = {LMSM:1, LSJM:2, L0:3, L1:4, L2:5, L3:6, L4:7}
branch_nm1 = resolve_branch(main=physics/sopt, override=physics_nm1/sopt_nm1)
branch_np1 = resolve_branch(main=physics/sopt, override=physics_np1/sopt_np1)
```

## 5) Referenced File Contract (MUST)
MUST:
- `[inputs]` carries resolved runtime file paths only.
- `inputs.hopping_file` is the external hopping input consumed by `L2/L3/L4`.
- `inputs.projector_file` is the external projector/Kramers input consumed by `L4`.
- The current run-input contract does not include additional source-specific
  top-level tables such as `[wannier90]`.
- Source-specific parsing and physical interpretation of referenced files remain
  governed by their own standards and runtime stages.  For Wannier90-related
  semantics, see:
  - `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`
  - `./standards/en/05-io/05-03-WANNIER90_PARSING.md`

Validation:
- Missing required referenced file path is `FXE-INPUT-002/003`.

## 6) Deterministic Expansion Rule (MUST)
MUST:
- Runtime is single-point in `U/Jh/zeta` (no sweep/cartesian product in input contract).
- One input file corresponds to one deterministic SOPT parameter tuple
  `(U, Jh, zeta)`.
- Optional branch overrides do not change the single-point nature of the run;
  they only alter how adjacent-sector `f^(n-1)` / `f^(n+1)` physics and
  denominator parameters are resolved.
- Runtime execution window is `LMSM ... runtime.end_level`.
  Execution always starts from the first level; each level checks disk cache
  first and reads if a matching artifact exists, otherwise computes.
- Main-sector core parameters may come either from explicit `[physics]` / `[model]`
  input or from a unique upstream core token resolved from disk artifacts.
- Input gate by window (minimum):
  - if window includes `L2`: require `[inputs]` and canonical `sources.hopping_name`
  - if window includes `L3`: require `[inputs]`, canonical `sources.hopping_name`, and `[sopt]`
  - if window includes `L4`: require `[inputs]`, canonical `sources.hopping_name`, canonical `sources.kramer_name`, and `[sopt]`
- Levels above `end_level` are skipped.
- Precompute-only mode is expressed by `end_level <= L2`.
- `paths.output_root` must be exactly `./outputs` for contract compatibility with `05-00-IO`.

Code form:
```text
if window_includes(L3):
  run_point = (sopt.U, sopt.Jh, sopt.zeta)
```

Validation:
- Missing/invalid `sopt.U/Jh/zeta` (when window includes `L3` or `L4`) is `FXE-INPUT-003`.
- Missing window-required input field is `FXE-INPUT-002/003`.
- Missing/invalid canonical `sources.hopping_name`/`sources.kramer_name` (as required by window) is `FXE-INPUT-003`.
- `paths.output_root != "./outputs"` is `FXE-INPUT-003`.

## 7) Minimal Readable Example (MUST)
Code form:
```toml
schema_version   = "fxe.run_input.v1"
standard_version = "2026-02"
run_id           = "n6_demo"
title            = "N=6 single-point test"

[physics]
n_ele = 6
RE    = "Eu"

[physics_nm1]
RE = "Sm"

[model]
scheme = "RS"

[units]
energy = "meV"

[sopt]
U                = 3.000000000000
Jh               = 4.000000000000
offset           = 0.000000000000
energy_reference = "lsjm_ground"

[sopt_nm1]
U      = 0.000000000000
offset = 12.000000000000

[sources]
hopping_label    = "w90_demo_bond0"
projection_label = "cef_lowest_doublet_v1"

[inputs]
hopping_file   = "./data/wannier90/w90_t_mu.npz"
projector_file = "./data/cef/kramer.npz"

[paths]
output_root = "./outputs"

[runtime]
end_level = "L1"

[checks]
strict_mode = true
eps_profile = "default"
```

Validation:
- The example must pass schema/type checks without implicit defaults.

## 8) Error Mapping (MUST)
MUST:
- Failures in this file use fixed codes from:
  `./standards/en/06-utils/06-01-ERROR_CODES.md`.

Code form:
```text
input_error -> FXE-INPUT-* / FXE-SCHEMA-*
```

Validation:
- Uncoded input failure is invalid.
