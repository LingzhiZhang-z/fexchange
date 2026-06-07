# 05-05-SWEEP_INPUT

This file defines the parameter-sweep input contract for `fexchange sweep`.
It extends the single-run contract in `./standards/en/05-io/05-04-RUN_INPUT.md`
without changing the per-case runtime schema.

## 1) Scope (MUST)
MUST:
- A sweep input is a normal run-input TOML plus one additional top-level
  `[sweep]` table.
- `fexchange sweep <base.toml>` is the only CLI entry point that consumes
  `[sweep]`.
- `fexchange run <input.toml>` must reject `[sweep]` as an unsupported top-level
  section.
- The sweep front-end must strip `[sweep]` before validating each materialized
  case against `05-04-RUN_INPUT`.
- Cases are explicitly enumerated rows. No Cartesian-product expansion is
  implied or allowed by this contract.

Validation:
- Missing `[sweep]` or `[sweep].table` is `FXE-INPUT-002`.
- Invalid sweep table syntax or unsupported columns are `FXE-INPUT-003`.

## 2) Table Grammar (MUST)
MUST:
- `[sweep].table` is a multi-line string.
- The first nonblank line is an integer case count `N`, with `N >= 1`.
- The second nonblank line is a whitespace-separated header of parameter paths.
- The header must include `runtime.run_name`.
- Each remaining nonblank line is one case row.
- The number of case rows must equal `N`.
- Each row must have exactly the same number of cells as the header.
- Header names must be unique.
- Cells are whitespace-delimited; paths and labels must not contain whitespace.

Code form:
```toml
[sweep]
table = """
2
fsite.U   fsite.Jh   runtime.run_name
2000.0    0.0        run_u2_jh0
3000.0    300.0      run_u3_jh01
"""
```

## 3) Value Typing (MUST)
MUST:
- Cells in `runtime.run_name` and any `inputs.*_file` column remain strings.
- Other cells are parsed as `float` when possible; otherwise they remain strings
  and are later validated by the materialized single-run schema.

Validation:
- A materialized case whose numeric field receives a nonnumeric string fails
  through `05-04-RUN_INPUT` validation.

## 4) Parameter Paths (MUST)
MUST:
- A qualified path contains `.` and sets exactly one leaf.
- A bare name contains no `.` and broadcasts to all allowed target sections.
- Qualified paths override bare names when both target the same leaf.
- Missing intermediate tables may be created by the sweep expander.

Allowed bare names and targets:

```text
U, Jh, zeta, offset -> fsite, fsite_nm1, fsite_np1
Uplus               -> fsite_np1
Uminus              -> fsite_nm1
Delta, U_p, lambda_p -> ligand.1, ligand.2
```

Allowed qualified leaves:

```text
fsite.{U,Jh,zeta,offset}
fsite_nm1.{U,Jh,zeta,offset,Uminus}
fsite_np1.{U,Jh,zeta,offset,Uplus}
ligand.{1,2}.{Delta,U_p,lambda_p}
inputs.{kramer_file,hopping_file,hcef_file}
runtime.run_name
```

Validation:
- Unknown bare names or qualified paths are `FXE-INPUT-003`.
- Duplicate columns are `FXE-INPUT-003`.

## 5) Forbidden Sweep Fields (MUST)
MUST:
- The following fields are not sweepable:

```text
n_ele
F2_ratio
F4_ratio
F6_ratio
RE
energy_reference
model.scheme
runtime.kramer_source
```

- Any qualified path targeting a forbidden leaf is also forbidden, for example
  `fsite.n_ele` and `fsite.F2_ratio`.
- `runtime.kramer_name` is not a contract field and is not sweepable.

Rationale:
- `n_ele`, Slater ratios, `RE`, and `model.scheme` affect shared core artifacts
  and are fixed within one sweep batch.
- `runtime.kramer_source` fixes the meaning of `inputs.kramer_file` for the
  whole sweep.

## 6) Uplus/Uminus Offset Rule (MUST)
MUST:
- Setting `Uplus` removes `fsite_np1.offset` if present.
- Setting `Uminus` removes `fsite_nm1.offset` if present.
- This preserves the mutual exclusion required by `05-04-RUN_INPUT`.

## 7) Run Names and Output Anchors (MUST)
MUST:
- Every materialized case must have a unique `runtime.run_name`.
- The sweep table must contain a `runtime.run_name` column. Those values are
  used as the per-case run names and must be unique and nonempty.
- Each case uses the `paths.output_run` normalization defined by
  `05-04-RUN_INPUT`: the resolved run anchor is
  `<paths.output_run or paths.output_root>/<runtime.run_name>`.

Validation:
- Missing, duplicate, or empty `runtime.run_name` values are `FXE-INPUT-003`.

## 8) Execution Semantics (MUST)
MUST:
- Each case is validated in memory with the same loader used by `fexchange run`.
- Each case is executed through the same runtime pipeline used by
  `fexchange run`.
- Serial execution runs every case in one process.
- MPI execution uses `mpi4py` when available and communicator size is greater
  than one.
- Under MPI, rank 0 expands and validates all cases, broadcasts the case table,
  pre-warms shared core artifacts once per distinct core key, and all ranks
  execute static slices `cases[rank::size]`.
- If launched under an MPI launcher with size greater than one but `mpi4py` is
  unavailable, the run must fail instead of letting every rank fall back to the
  serial path.

## 9) Progress Output (MUST)
MUST:
- The sweep runner writes a plain-text progress file:

```text
<output_root>/sweep_<base_stem>.txt
```

- The progress file contains a header, one line per finished case, and a final
  `# done: k/N, f failed` line.
- Each case line records status, rank (or serial marker), elapsed time, and
  `runtime.run_name`.
- Case failures are recorded as failed case lines; one failed case does not
  require stopping the remaining cases in the same rank.
