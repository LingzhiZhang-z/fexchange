# 00-06-MPI_PARALLEL_RUNTIME

本文件定义 MPI/并行运行时行为的规范约束。
适用于 `LMSM, LSJM, L0, L1, L2, L3, L4`（及可选 `04-03` 后处理）窗口执行。

## 1) 作用范围（MUST）
MUST:
- 只定义并行运行时行为。
- 不重定义物理公式与磁盘 key 布局。
- 必须与以下规范一致：
  - `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`
  - `./standards/en/05-io/05-00-IO.md`

Code form:
```text
parallel_contract_scope = {runtime_roles, sharding, comm, root_write, failure_propagation}
```

Validation:
- 与本文件冲突的实现级并行策略视为无效。

## 2) 运行时来源规则（MUST）
MUST:
- MPI 进程布局控制来自运行时环境，不来自输入文件字段。
- 输入文件可选择执行窗口，但不负责 `world_size/rank/topology`。

Code form:
```text
runtime_parallel = detect_from_runtime_env()
run_input_excludes = {use_mpi, mpi_world, threads_per_rank}
```

Validation:
- 若通过输入文件注入进程布局字段，使用 `FXE-INPUT-003` 失败。

## 3) 并行元数据必填项（MUST）
MUST:
- 每次运行至少记录：
  - `parallel_enabled`（bool）
  - `parallel_backend`（`serial` 或 `mpi4py`）
  - `world_size`（int >= 1）
  - `rank`（int，`0 <= rank < world_size`）
  - `root_rank`（int，默认 `0`）
  - `local_rank`（int，可用时提供）
  - `gather_policy`（`gather_to_root` 或 `reduce_to_root`）

Code form:
```text
parallel_meta = {parallel_enabled, parallel_backend, world_size, rank, root_rank, local_rank, gather_policy}
```

Validation:
- 缺失并行元数据使用 `FXE-RUNTIME-001`。

## 4) 角色与职责（MUST）
MUST:
- 只有 `root_rank` 允许持久化阶段工件。
- 非 root rank 允许计算分片与通信，但禁止写最终工件。
- 串行模式下，`rank=0` 同时是 worker 与 root。

Code form:
```text
if rank == root_rank: can_persist = true
else: can_persist = false
```

Validation:
- 非 root 写最终工件路径使用 `FXE-IO-003`。

## 5) 预扫描与起始同步（MUST）
MUST:
- 由 root 先执行预扫描：
  - 窗口合法性（`start_level <= end_level`）
  - 上游工件完备性
  - 窗口必需输入字段
- root 向所有 rank 广播预扫描结论与执行计划。
- 若预扫描失败，所有 rank 必须停止，不得进入计算。

Code form:
```text
if rank == root_rank:
  plan = preflight(...)
plan = bcast(plan, root=root_rank)
if not plan.ok: abort_all_ranks()
barrier()
```

Validation:
- 未完成预扫描广播就开始计算，使用 `FXE-RUNTIME-001`。

## 6) 分片规则（MUST）
MUST:
- 活跃层级上的任务分片必须“互斥且完备”。
- 每个全局输出元素只能由一个 rank 负责。
- 分片规则与索引范围必须确定性，并写入元数据。

Code form:
```text
shards = deterministic_partition(global_index_space, world_size)
local_shard = shards[rank]
```

Validation:
- 分片重叠或遗漏使用 `FXE-BIND-003`。

## 6.1) 分层默认分片配置（MUST）
MUST:
- 除非文档中明确声明覆盖策略，否则使用下表默认分片轴。
- 覆盖策略仅在保持“确定性 + 互斥且完备”时允许。

| 层级 | 默认分片域 | 输出归属 |
|---|---|---|
| `LMSM` | 目标项任务块 `(L,S)`（或固定 `alpha` 后的 `(alpha,L,S)`） | 每个 rank 负责一个项任务块 |
| `LSJM` | 项耦合任务块 `(alpha,L,S)` | 每个 rank 负责一个项任务块 |
| `L0` | 轨道指标 `kappa` 分块 | 每个 rank 负责一个 `kappa` 范围 |
| `L1` | 中间腿分块（`A` 用 `u`，`B` 用 `v`） | 每个 rank 负责一个腿指标范围 |
| `L2` | 路线 A 的 `(u,v)` 对分块 + 路线 B 的 `(r,s)` 对分块 | 每个 rank 每条路线负责一个 pair 范围 |
| `L3` | 分母对分块（`m=(u,v)` 与 `n=(r,s)`） | 每个 rank 负责一个分母范围贡献 |
| `L4` | 默认 root 串行；仅在标签规模很大时允许 `(a,b,c,d)` 标签分块 | 每个 rank 负责一个输出标签范围（或 root 串行） |
| `04-03` | 仅 root 串行（禁止 MPI 分片） | root 负责完整输出 |

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
  04-03:   root_serial_only
}
```

Validation:
- 运行元数据必须记录实际生效的 `shard_axis` 与 `shard_ranges`。
- 若 `04-03` 采用 MPI 分片执行，使用 `FXE-RUNTIME-001` 失败。
- 对 `LSJM`，分片元数据必须声明 `shard_axis = (alpha,L,S)`。

## 7) 通信与组装规则（MUST）
MUST:
- worker rank 按 `gather_policy` 把本地分片返回 root。
- root 以确定性索引顺序组装完整矩阵/张量。
- 若采用数值归约，归约顺序策略必须固定并文档化。

Code form:
```text
local = compute(local_shard)
global_parts = gather_or_reduce_to_root(local, policy=gather_policy)
if rank == root_rank:
  global_tensor = deterministic_assemble(global_parts)
```

Validation:
- 相同输入下组装顺序不确定，使用 `FXE-RUNTIME-001`。

## 8) 持久化规则（MUST）
MUST:
- 仅 root 写最终 `data.npz`/`meta.json`。
- 写入必须原子化。
- 成功后 root 必须在 `meta.json` 与 stdout 同步输出并行元数据摘要。

Code form:
```text
if rank == root_rank:
  atomic_write(data_npz, meta_json)
  emit_parallel_summary_stdout(parallel_meta)
```

Validation:
- 多 rank 并发写同一路径工件属于禁止行为（`FXE-IO-003`）。

## 9) 失败传播规则（MUST）
MUST:
- 任一 rank 的硬失败都必须升级为全局失败。
- 失败载荷必须包含 rank 上下文：
  `rank`, `world_size`, `root_rank`, `level`, `op`。

Code form:
```text
if local_error:
  notify_root_and_abort_all(payload_with_rank_context)
```

Validation:
- rank 本地静默失败属于无效行为（`FXE-RUNTIME-001`）。
