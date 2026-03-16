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
- Conditionally required tables (by execution window):
  - `[physics]` (required if window includes `LMSM` or `L0`)
  - `[model]` (required if window includes `LMSM`)
  - `[sopt]` (required if window includes `L3` or `L4`)
  - `[sources]` (required if window includes `L2` or `L3` or `L4`)
  - `[wannier90]` (required if `sources.hopping_source = "wannier90"` and window includes `L2` or `L3` or `L4`)
  - `[kramer_input]` (required if `sources.kramer_source = "file"` and window includes `L4`)

Code form:
```text
required_core_sections = {paths, runtime, checks}
require_conditional_sections(start_level, end_level, sources)
```

Validation:
- Missing required section is `FXE-INPUT-002`.
- Unsupported extra top-level section is `FXE-INPUT-003`.

## 4) Field Contract (MUST)
MUST:
- `[physics]`:
  - `n_ele` (int, `1..13`)
  - `F2` (float)
  - `F4` (float)
  - `F6` (float)
  - internal-derived only: `r42 = F4/F2`, `r62 = F6/F2`
  - constraint: `F2 != 0`
- `[model]`:
  - `scheme` (string, currently `RS`)
  - `symmetry` (string: `Oh` or `C3v`)
  - `c3v_mode_q3` (string: `cos` or `sin`)
- `[sopt]`:
  - `U` (float; physically $U = F^0$, the zeroth Slater-Condon parameter)
  - `Jh` (float)
  - `zeta` (float)
- `[sopt]` / `[physics]` cross-reference note (MUST):
  - `F^0` is NOT a field in `[physics]` because within a fixed `n` sector
    it contributes a constant $F^0 n(n-1)/2$ to all states and does not
    affect LMSM/LSJM relative energies.
  - For SOPT energy denominators (cross-sector differences $f^{n\pm1}$ vs $f^n$),
    $F^0 = U$ is taken from `[sopt].U`.
  - Implementations MUST use `sopt.U` as $F^0$ when reconstructing absolute
    LSJM energies across sectors for intermediate-state denominators.
- `[sources]`:
  - `hopping_source` (`wannier90` or `file`)
  - `kramer_source` (`cef` or `file`)
  - `hopping_name` (string, stable cache/key token for `L2/L3/L4`)
  - `kramer_name` (string, required if window includes `L4`; stable cache/key token)
- `[paths]`:
  - `output_root` (string, MUST equal `"./outputs"` in this standard version)
  - if `hopping_source=file`: `hopping_file`
  - if `kramer_source=file`: `kramer_file`
- `[runtime]`:
  - `start_level` (string: `LMSM`, `LSJM`, `L0`, `L1`, `L2`, `L3`, or `L4`)
  - `end_level` (string: `LMSM`, `LSJM`, `L0`, `L1`, `L2`, `L3`, or `L4`)
  - `on_missing_upstream` (string: `fail`)
  - `read_first` (bool, must be `true`)
- `[checks]`:
  - `strict_mode` (bool)
  - `eps_profile` (string, e.g. `default`)

Validation:
- Type/value-domain mismatch is `FXE-INPUT-003`.
- Invalid level window (`start_level > end_level`) is `FXE-INPUT-003`.

Code form:
```text
r42 = F4 / F2
r62 = F6 / F2
level_order = {LMSM:1, LSJM:2, L0:3, L1:4, L2:5, L3:6, L4:7}
require level_order[start_level] <= level_order[end_level]
```

## 5) Wannier90 Sub-Contract (MUST)
MUST:
- If `hopping_source = "wannier90"`, section `[wannier90]` is mandatory.
- Required `[wannier90]` fields:
  - `soc_mode` (`with_soc` or `without_soc`)
  - `hr_path`
  - `win_path`
  - `orbital_basis` (`real_harmonic_default_w90`)
  - `orbital_order_id`
  - `energy_unit` (`eV` by default)
  - `f_site_i`, `f_site_j` (int)
  - `f_site_i_cell`, `f_site_j_cell` (array[int], each length = 3)
  - `ligand_indices` (array[int], length >= 0; empty means direct `f-f` only)
  - `ligand_cells` (array[array[int]], same length as `ligand_indices`, each length = 3)
  - `all_wannier_atom_indices` (array[int], length >= 1)
  - `delta_mode` (`manual` or `from_onsite`)
  - `delta_reduction` (`channelwise` or `global_mean`)
  - if `delta_mode = "manual"`:
    - `delta_manual_kind` (`channelwise` or `global_mean`)
    - if `delta_manual_kind = "global_mean"`: `delta_manual_value` (float, unit=`energy_unit`)
    - if `delta_manual_kind = "channelwise"`: `delta_manual_file` (path to NPZ containing `Delta_puv[p,u,v]` in unit=`energy_unit`)
- Parsing and mapping details follow:
  `./standards/en/05-io/05-03-WANNIER90_PARSING.md`.
- Physical hopping/CEF contract follows:
  `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`.

Validation:
- Missing required Wannier90 field is `FXE-W90-001/002`.

## 6) Deterministic Expansion Rule (MUST)
MUST:
- Runtime is single-point in `U/Jh/zeta` (no sweep/cartesian product in input contract).
- One input file corresponds to one deterministic SOPT parameter tuple
  `(U, Jh, zeta)`.
- Runtime execution window is contiguous:
  `runtime.start_level ... runtime.end_level`.
- Before execution from `runtime.start_level`, implementation must preflight-scan
  required upstream artifacts on disk and verify window-required input fields.
- If any required upstream artifact is missing or invalid, execution must fail.
- Input gate by window (minimum):
  - if window includes `LMSM`: require `[physics]` + `[model]`
  - if window includes `LSJM`: require upstream `LMSM` artifact
  - if window includes `L0`: require `[physics]`
  - if window includes `L1`: require upstream `LSJM` and `L0` artifacts
  - if window includes `L2`: require `[sources]`, `sources.hopping_name`, and hopping input source
  - if window includes `L3`: require `[sources]`, `sources.hopping_name`, and `[sopt]`
  - if window includes `L4`: require `[sources]`, `sources.hopping_name`, `sources.kramer_name`, `[sopt]`, and Kramer input source
- `start_level = end_level` means compute that single level only;
  all upstream levels must already exist/validate, downstream levels are skipped.
- Precompute-only mode is expressed by `end_level <= L2`.
- `paths.output_root` must be exactly `./outputs` for contract compatibility with `05-00-IO`.

Code form:
```text
if window_includes(L3):
  run_point = (sopt.U, sopt.Jh, sopt.zeta)
```

Validation:
- Missing/invalid `sopt.U/Jh/zeta` (when window includes `L3` or `L4`) is `FXE-INPUT-003`.
- Missing/invalid upstream artifact at preflight is `FXE-IO-001/002`.
- Missing window-required input field is `FXE-INPUT-002/003`.
- Missing/invalid `sources.hopping_name`/`sources.kramer_name` (as required by window) is `FXE-INPUT-003`.
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
F2    = 1.000000000000
F4    = 1.000000000000
F6    = 2.000000000000

[model]
scheme       = "RS"
symmetry     = "C3v"
c3v_mode_q3  = "sin"

[sopt]
U    = 3.000000000000
Jh   = 4.000000000000
zeta = 5.000000000000

[sources]
hopping_source = "wannier90"
kramer_source  = "cef"
hopping_name   = "w90_demo_bond0"
kramer_name    = "cef_lowest_doublet_v1"

[paths]
output_root = "./outputs"

[wannier90]
soc_mode                  = "with_soc"
hr_path                   = "./data/wannier90/wannier_hr.dat"
win_path                  = "./data/wannier90/wannier.win"
orbital_basis             = "real_harmonic_default_w90"
orbital_order_id          = "w90_f_default_v1"
energy_unit               = "eV"
f_site_i                  = 0
f_site_j                  = 1
f_site_i_cell             = [0, 0, 0]
f_site_j_cell             = [1, 0, 0]
ligand_indices            = [2, 3, 4, 5]
ligand_cells              = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]
all_wannier_atom_indices  = [0, 1, 2, 3, 4, 5]
delta_mode                = "from_onsite"
delta_reduction           = "channelwise"

[runtime]
start_level         = "L1"
end_level           = "L1"
on_missing_upstream = "fail"
read_first          = true

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
input_error -> FXE-INPUT-* / FXE-W90-* / FXE-SCHEMA-*
```

Validation:
- Uncoded input failure is invalid.
