# 05-04-RUN_INPUT

This file defines the single-file runtime input contract.

## 1) Scope (MUST)
MUST:
- Runtime inputs must come from one TOML file.
- Required top-level keys are `schema_version` and `standard_version`.
- Optional top-level provenance keys are `run_id` and `title`.
- Required tables are `[paths]`, `[runtime]`, and `[fsite]`.
- `[checks]` is optional runtime metadata; implementations may ignore it unless
  a later standard assigns a specific runtime gate to a field inside it.
- `[model]` is optional and defaults to `scheme = "RS"`.
- `model.scheme` must be `RS` or `ED`.
- `[inputs]` is required for `end_level >= L2`.
- `[ligand.1]` and `[ligand.2]` are required for `branch = "fopt"`.
- `[units]` is optional metadata only. Runtime numeric fields and hopping
  files are consumed as raw values with no internal unit conversion; users must
  keep all energy-like inputs in one consistent unit.
- `[sweep]` is reserved for the `fexchange sweep` front-end
  (`./standards/en/05-io/05-05-SWEEP_INPUT.md`). It is not a valid top-level
  section for `fexchange run`; the sweep front-end strips it before validating
  each materialized single-run input against this file.

Validation:
- Missing input file is `FXE-INPUT-001`.
- Missing required key or table is `FXE-INPUT-002`.
- Unsupported top-level sections are `FXE-INPUT-003`.

## 2) Runtime Table (MUST)
MUST:
- `runtime.branch` is required and must be `sopt` or `fopt`.
- `runtime.end_level` is required and must be one of
  `LMSM`, `LSJM`, `L0`, `L1`, `L2`, `L3`.
- `runtime.run_name` is required when `end_level >= L1`, because `L1/F` and all
  downstream artifacts are run-scoped for every scheme and Kramers route (RS and
  ED, stevens and manual).
- `runtime.kramer_source` is optional and defaults to `stevens`.
  Accepted normalized values are `stevens` and `manual`; runtime loaders may
  accept legacy aliases `steven` and `mannual`.
- `runtime.kramer_name` is no longer a runtime contract field. If a legacy input
  carries it, implementations may keep it as inert provenance, but it must not
  affect validation, artifact paths, keys, or computation.
- Both branches terminate at `L3`. FOPT `L3` includes total/process raw
  `h_eff_4` outputs and total/process spin-1/2 exchange outputs. Runtime FOPT
  exchange output requires a two-dimensional projected local space.

Code form:
```toml
[runtime]
branch = "fopt"
end_level = "L3"
run_name = "lab_A"
kramer_source = "stevens"
```

## 3) f-Site Table (MUST)
MUST:
- `[fsite]` combines the former f-shell ratio fields and denominator fields.
- Required fields:
  - `n_ele`
  - `U`
  - `Jh`
  - either `RE` preset or all of `F2_ratio`, `F4_ratio`, `F6_ratio`
- Optional fields:
  - `RE`
  - `zeta`
  - `offset`
  - `energy_reference` (`lsjm_ground` or `zero`)
- `[fsite_nm1]` and `[fsite_np1]` may override any subset of:
  `RE`, `F2_ratio`, `F4_ratio`, `F6_ratio`, `U`, `Jh`, `zeta`, `offset`.
  Adjacent sectors may therefore carry their own Slater ratios (e.g.
  neighboring-element presets).
- `[fsite_np1]` may alternatively set `Uplus`, the target minimum energy gap
  from the main `f^n` reference to the `f^{n+1}` sector. `Uplus` is mutually
  exclusive with `fsite_np1.offset`.
- `[fsite_nm1]` may alternatively set `Uminus`, the target minimum energy gap
  from the main `f^n` reference to the `f^{n-1}` sector. `Uminus` is mutually
  exclusive with `fsite_nm1.offset`.
- `Uplus` and `Uminus` are input conveniences for automatic branch-local
  offset construction. Runtime denominator contraction consumes the resolved
  intermediate energies, not an additional Hamiltonian term.

Code form:
```toml
[fsite]
n_ele = 1
RE = "Ce"
U = 4.0
Jh = 0.85
zeta = 0.05
offset = 0.0
energy_reference = "lsjm_ground"

[fsite_np1]
Uplus = 5.0

[fsite_nm1]
Uminus = 5.0
```

Validation:
- `n_ele` must be an integer in `1..13`.
- `F2_ratio` must be nonzero when explicit ratios are used.
- `F2/F4/F6` for denominator reconstruction are derived from `Jh` and
  `r42 = F4_ratio/F2_ratio`, `r62 = F6_ratio/F2_ratio`.

## 4) Ligand Tables (MUST)
MUST:
- FOPT requires both `[ligand.1]` and `[ligand.2]`.
- Required ligand fields are `Delta` and `U_p`.
- `lambda_p` is optional and defaults to `0.0`.
- `lambda_p = 0.0` selects the no-SOC ligand cache; nonzero selects the SOC
  ligand cache.

Code form:
```toml
[ligand.1]
Delta = 4.0
U_p = 6.0
lambda_p = 0.05

[ligand.2]
Delta = 4.5
U_p = 5.5
lambda_p = 0.0
```

## 5) Inputs Table (MUST)
MUST:
- `[inputs].hopping_file` is required for `end_level >= L2`.
- `[inputs].kramer_file` is the unified doublet input path.
- `[inputs].kramer_file` is required for `end_level >= L2` when
  `runtime.kramer_source = "stevens"`; in this mode it carries the projector
  `W` from the SOC-lowest LSJM subspace into the target doublet/quasi-doublet.
- `[inputs].kramer_file` is required for `end_level >= L1` when
  `runtime.kramer_source = "manual"`; in this mode it carries the external
  Kramers basis in Fock-determinant form.
- `[inputs].hcef_file` is optional. When present with `model.scheme = "ED"`,
  it is used as a one-body CEF matrix in adjacent-sector IONED.
- Runtime matrix text files use multi-block format with `[key]` headers.
- SOPT hopping must contain block `[t_mu]` with shape `(14, 14)`.
- FOPT hopping must contain blocks `[t_f1_lig1]`, `[t_f1_lig2]`,
  `[t_f2_lig1]`, `[t_f2_lig2]`, each shape `(14, 6)`.
- Stevens-mode `kramer_file` text input (`.txt` / `.dat`) must use one block per doublet
  state, named `[W_state_0]`, `[W_state_1]`, … `[W_state_{n_k-1}]`. Each
  block holds exactly `n_j` rows of `real imag` (one column of `W`).
  State indices must be contiguous starting at 0; columns are stacked in
  numeric order to form `W` with shape `(n_j, n_k)`.
- Stevens-mode `kramer_file` binary input (`.npy` / `.npz`) is a rank-2 array (key `W` for
  `.npz`) with `shape[0] == n_j`.
- Manual Kramers text input starts with `fn <n>` and then uses exactly two
  blocks, `[K_state_0]` and `[K_state_1]`. Each data row has exactly 16 fields:
  `real imag occ_0 ... occ_13`. The occupation fields are `0/1` values in the
  canonical f spin-orbital order, and each row must have exactly `n` occupied
  orbitals. Blocks are stacked in numeric order to form
  `K_fock.shape = (dim_fock(n), n_k)`.
- `hcef_file` is a Hermitian `14 x 14` one-body matrix in the canonical f
  spin-orbital order. Text input may use block `[hcef]` with `14*14` complex
  rows or a plain matrix format accepted by the runtime matrix loader.

Code form:
```toml
[inputs]
hopping_file = "data/hopping/wan_v1.txt"
kramer_file = "data/projector/kr_a.txt"      # stevens projector W
hcef_file = "data/hcef/hcef_14x14.txt"
# or, in manual mode:
# kramer_file = "data/kramer/manual_kramer.txt"
```

## 6) Paths Table (MUST)
MUST:
- `[paths].output_root` is required and anchors core artifacts and global index
  files.
- `[paths].output_run` is optional and is interpreted as the base directory for
  run-scoped artifacts. When omitted, the base is `output_root`; when set, that
  value replaces `output_root` as the base. The resolved run anchor is always
  `<base>/<runtime.run_name>`.
- Run-scoped artifacts (`IONED`, `L1/F`, `L2`, `L3`, `source.txt`, `run.log`)
  are written under the resolved run anchor.

Code form:
```toml
[paths]
output_root = "./outputs"
output_run = "./outputs/custom_base"  # resolved run anchor adds runtime.run_name
```

## 7) ED Scheme Example (MUST)
Code form:
```toml
[model]
scheme = "ED"

[runtime]
branch = "sopt"
end_level = "L1"
run_name = "ed_demo"
```

## 8) Minimal FOPT Example (MUST)
Code form:
```toml
schema_version = "fxe.run_input.v1"
standard_version = "2026-02"
run_id = "fopt_demo"
title = "FOPT single point"

[fsite]
n_ele = 1
F2_ratio = 1.0
F4_ratio = 0.6
F6_ratio = 0.4
U = 4.0
Jh = 0.85
zeta = 0.05

[model]
scheme = "RS"

[inputs]
hopping_file = "data/hopping/fopt.txt"
kramer_file = "data/projector/W.txt"

[ligand.1]
Delta = 4.0
U_p = 6.0
lambda_p = 0.05

[ligand.2]
Delta = 4.5
U_p = 5.5
lambda_p = 0.0

[paths]
output_root = "./outputs"
# output_run = "./outputs/base"  # optional base; run_name is appended

[runtime]
branch = "fopt"
end_level = "L3"
run_name = "demo"

[checks]
strict_mode = true
eps_profile = "default"
```
