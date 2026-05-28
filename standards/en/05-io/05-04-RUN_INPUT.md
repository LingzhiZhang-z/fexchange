# 05-04-RUN_INPUT

This file defines the single-file runtime input contract.

## 1) Scope (MUST)
MUST:
- Runtime inputs must come from one TOML file.
- Required top-level keys are `schema_version`, `standard_version`, `run_id`,
  and `title`.
- Required tables are `[paths]`, `[runtime]`, `[checks]`, and `[fsite]`.
- `[model]` is optional and defaults to `scheme = "RS"`.
- `model.scheme` must be `RS` or `ED`.
- `[inputs]` is required for `end_level >= L2`.
- `[ligand.1]` and `[ligand.2]` are required for `branch = "fopt"`.
- `[units]` is optional metadata only. Runtime numeric fields and hopping
  files are consumed as raw values with no internal unit conversion; users must
  keep all energy-like inputs in one consistent unit.

Validation:
- Missing input file is `FXE-INPUT-001`.
- Missing required key or table is `FXE-INPUT-002`.
- Unsupported top-level sections are `FXE-INPUT-003`.

## 2) Runtime Table (MUST)
MUST:
- `runtime.branch` is required and must be `sopt` or `fopt`.
- `runtime.end_level` is required and must be one of
  `LMSM`, `LSJM`, `L0`, `L1`, `L2`, `L3`.
- `runtime.run_name` is required when `end_level >= L2`.
- `runtime.run_name` is also required when `model.scheme = "ED"` and
  `end_level >= L1`, because ED intermediate artifacts are run-scoped.
- `runtime.kramer_name` is required for `end_level >= L2`, because L2 consumes
  the projector and writes projector-dependent projected factors.
- Both branches terminate at `L3`. FOPT `L3` includes total/process raw
  `h_eff_4` outputs and total/process spin-1/2 exchange outputs. Runtime FOPT
  exchange output requires a two-dimensional projected local space.

Code form:
```toml
[runtime]
branch = "fopt"
end_level = "L3"
run_name = "lab_A"
kramer_name = "proj_a"
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
- `[inputs].projector_file` is required for `end_level >= L2`.
- Runtime matrix text files use multi-block format with `[key]` headers.
- SOPT hopping must contain block `[t_mu]` with shape `(14, 14)`.
- FOPT hopping must contain blocks `[t_f1_lig1]`, `[t_f1_lig2]`,
  `[t_f2_lig1]`, `[t_f2_lig2]`, each shape `(14, 6)`.
- Projector text input (`.txt` / `.dat`) must use one block per doublet
  state, named `[W_state_0]`, `[W_state_1]`, … `[W_state_{n_k-1}]`. Each
  block holds exactly `n_j` rows of `real imag` (one column of `W`).
  State indices must be contiguous starting at 0; columns are stacked in
  numeric order to form `W` with shape `(n_j, n_k)`.
- Projector binary input (`.npy` / `.npz`) is a rank-2 array (key `W` for
  `.npz`) with `shape[0] == n_j`.

Code form:
```toml
[inputs]
hopping_file = "data/hopping/wan_v1.txt"
projector_file = "data/projector/kr_a.txt"
```

## 6) ED Scheme Example (MUST)
Code form:
```toml
[model]
scheme = "ED"

[runtime]
branch = "sopt"
end_level = "L1"
run_name = "ed_demo"
```

## 7) Minimal FOPT Example (MUST)
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
projector_file = "data/projector/W.txt"

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

[runtime]
branch = "fopt"
end_level = "L3"
run_name = "demo"
kramer_name = "proj_a"

[checks]
strict_mode = true
eps_profile = "default"
```
