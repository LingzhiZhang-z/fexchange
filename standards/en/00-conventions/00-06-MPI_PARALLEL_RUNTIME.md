# 00-06-MPI_PARALLEL_RUNTIME

This file defines normative MPI/parallel runtime behavior.
It applies to execution windows over `LMSM, LSJM, L0, L1, L2, L3, L4` (and optional `04-03` / spin12 post-map).

## 1) Scope (MUST)
MUST:
- Define runtime parallel behavior only.
- Not redefine physics formulas or disk key layout.
- Be consistent with:
  - `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`
  - `./standards/en/05-io/05-00-IO.md`

Code form:
```text
parallel_contract_scope = {runtime_roles, sharding, comm, root_write, failure_propagation}
```

Validation:
- Any implementation-level parallel policy conflicting with this file is invalid.

## 2) Runtime Source Rule (MUST)
MUST:
- MPI process-layout controls are runtime-environment settings, not run-input fields.
- Input file may select level window, but not `world_size/rank/topology`.

Code form:
```text
runtime_parallel = detect_from_runtime_env()
run_input_excludes = {use_mpi, mpi_world, threads_per_rank}
```

Validation:
- If process-layout is injected through run-input fields, reject with `FXE-INPUT-003`.

## 3) Required Runtime Parallel Metadata (MUST)
MUST:
- For each run, record at least:
  - `parallel_enabled` (bool)
  - `parallel_backend` (`serial` or `mpi4py`)
  - `world_size` (int >= 1)
  - `rank` (int, `0 <= rank < world_size`)
  - `root_rank` (int, default `0`)
  - `local_rank` (int, when available)
  - `gather_policy` (`gather_to_root` or `reduce_to_root`)

Code form:
```text
parallel_meta = {parallel_enabled, parallel_backend, world_size, rank, root_rank, local_rank, gather_policy}
```

Validation:
- Missing required parallel metadata is `FXE-RUNTIME-001`.

## 4) Role and Responsibility (MUST)
MUST:
- `root_rank` is the only rank allowed to persist stage artifacts.
- Non-root ranks may compute shards and communicate results, but must not write final artifacts.
- In serial mode, rank `0` is both worker and root.

Code form:
```text
if rank == root_rank: can_persist = true
else: can_persist = false
```

Validation:
- Any non-root write to final artifact path is `FXE-IO-003`.

## 5) Preflight and Start Barrier (MUST)
MUST:
- Root performs preflight checks first:
  - level window validity (`start_level <= end_level`)
  - required upstream artifacts
  - required window-input fields
- Root broadcasts preflight verdict and execution plan to all ranks.
- If preflight fails, all ranks stop; no rank computes.

Code form:
```text
if rank == root_rank:
  plan = preflight(...)
plan = bcast(plan, root=root_rank)
if not plan.ok: abort_all_ranks()
barrier()
```

Validation:
- Computing before successful preflight broadcast is `FXE-RUNTIME-001`.

## 6) Sharding Rule (MUST)
MUST:
- Work partition must be disjoint and complete over the active level.
- Each global output element must be assigned to exactly one rank.
- Sharding rule and index ranges must be deterministic and recorded in metadata.

Code form:
```text
shards = deterministic_partition(global_index_space, world_size)
local_shard = shards[rank]
```

Validation:
- Overlapping or missing shard ranges are `FXE-BIND-003`.

## 6.1) Default Level-Wise Sharding Profile (MUST)
MUST:
- Use the following default sharding axes unless an explicit override policy is documented.
- Override is allowed only if it preserves determinism and disjoint+complete coverage.

| Level | Default shard domain | Output ownership |
|---|---|---|
| `LMSM` | block list of target term tasks `(L,S)` (or `(alpha,L,S)` after alpha-fixing) | one rank owns one term-task block |
| `LSJM` | block list of term-coupling tasks `(alpha,L,S)` | one rank owns one term-task block |
| `L0` | orbital index `kappa` chunks | one rank owns one `kappa` range |
| `L1` | intermediate-leg chunks (`u` for `A`, `v` for `B`) | one rank owns one leg range |
| `L2` | route-A pair chunks `(u,v)` and route-B pair chunks `(r,s)` | one rank owns one pair-range per route |
| `L3` | denominator pair chunks (`m=(u,v)` and `n=(r,s)`) | one rank owns one denominator-range contribution |
| `L4` | default serial on root; optional `(a,b,c,d)` label chunks only for very large label sets | one rank owns one output-label range (or root-only serial) |
| `04-03` | serial-only on root (no MPI sharding) | root owns full output |

Code form:
```text
default_shard_axis = {
  LMSM: term_blocks_L_S_or_alpha_L_S,
  LSJM: term_blocks_alpha_L_S,
  L0:   kappa,
  L1:   {A:u, B:v},
  L2:   {A:(u,v), B:(r,s)},
  L3:   {A:m=(u,v), B:n=(r,s)},
  L4:   root_serial_default_or_labels_abcd_if_large,
  04-03: root_serial_only
}
```

Validation:
- Active run metadata must record effective `shard_axis` and `shard_ranges`.
- If level `04-03` is run with MPI sharding, reject with `FXE-RUNTIME-001`.
- For `LSJM`, shard metadata must indicate `shard_axis = (alpha,L,S)`.

## 7) Communication and Assembly Rule (MUST)
MUST:
- Worker ranks return local shards to root via configured gather/reduce policy.
- Root assembles the full tensor/matrix with deterministic index ordering.
- If numerical reduction is used, reduction order policy must be fixed and documented.

Code form:
```text
local = compute(local_shard)
global_parts = gather_or_reduce_to_root(local, policy=gather_policy)
if rank == root_rank:
  global_tensor = deterministic_assemble(global_parts)
```

Validation:
- Non-deterministic assembly order across identical runs is `FXE-RUNTIME-001`.

## 8) Persistence Rule (MUST)
MUST:
- Only root writes final `data.npz`/`meta.json`.
- Write must be atomic.
- On success, root records parallel metadata in both `meta.json` and stdout.

Code form:
```text
if rank == root_rank:
  atomic_write(data_npz, meta_json)
  emit_parallel_summary_stdout(parallel_meta)
```

Validation:
- Multi-rank concurrent writes to one artifact path are forbidden (`FXE-IO-003`).

## 9) Failure Propagation Rule (MUST)
MUST:
- Any hard failure on any rank must trigger global failure for the run.
- Failure payload must include rank context:
  `rank`, `world_size`, `root_rank`, `level`, `op`.

Code form:
```text
if local_error:
  notify_root_and_abort_all(payload_with_rank_context)
```

Validation:
- Rank-local silent failure is invalid (`FXE-RUNTIME-001`).
