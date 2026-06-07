# 05-05-SWEEP_INPUT

本文件定义 `fexchange sweep` 的参数扫描输入契约。
它扩展 `./standards/en/05-io/05-04-RUN_INPUT.md` 中的单点运行契约，
但不改变每个 case 的 runtime schema。

## 1) 适用范围（MUST）
MUST:
- sweep input 是普通 run-input TOML 加一个额外顶层 `[sweep]` 表。
- `fexchange sweep <base.toml>` 是唯一消费 `[sweep]` 的 CLI entry point。
- `fexchange run <input.toml>` 必须把 `[sweep]` 当作 unsupported top-level section 拒绝。
- sweep front-end 必须在按 `05-04-RUN_INPUT` 校验每个 materialized case 前移除 `[sweep]`。
- Cases 由行显式枚举。本契约不隐含也不允许 Cartesian-product expansion。

Validation:
- 缺失 `[sweep]` 或 `[sweep].table` 使用 `FXE-INPUT-002`。
- sweep table syntax 非法或 columns 不受支持使用 `FXE-INPUT-003`。

## 2) 表语法（MUST）
MUST:
- `[sweep].table` 是 multi-line string。
- 第一条非空行为整数 case count `N`，且 `N >= 1`。
- 第二条非空行为 whitespace-separated parameter paths header。
- 后续每条非空行为一个 case row。
- case rows 数量必须等于 `N`。
- 每行 cell 数必须与 header 完全一致。
- Header names 必须唯一。
- Cells 使用 whitespace 分隔；paths 和 labels 不得包含 whitespace。

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

## 3) 值类型（MUST）
MUST:
- `runtime.run_name` 和任何 `inputs.*_file` column 的 cells 保持 string。
- 其他 cells 尽可能解析为 `float`；否则保留为 string，并交给 materialized
  single-run schema 后续校验。

Validation:
- 若 materialized case 的 numeric field 收到 nonnumeric string，将通过
  `05-04-RUN_INPUT` validation 失败。

## 4) 参数路径（MUST）
MUST:
- Qualified path 含 `.`，并设置恰好一个 leaf。
- Bare name 不含 `.`，并广播到所有允许的 target sections。
- 当 bare name 和 qualified path 同时指向同一 leaf 时，qualified path 覆盖 bare name。
- sweep expander 可以创建缺失的 intermediate tables。

允许的 bare names 与 targets：

```text
U, Jh, zeta, offset -> fsite, fsite_nm1, fsite_np1
Uplus               -> fsite_np1
Uminus              -> fsite_nm1
Delta, U_p, lambda_p -> ligand.1, ligand.2
```

允许的 qualified leaves：

```text
fsite.{U,Jh,zeta,offset}
fsite_nm1.{U,Jh,zeta,offset,Uminus}
fsite_np1.{U,Jh,zeta,offset,Uplus}
ligand.{1,2}.{Delta,U_p,lambda_p}
inputs.{kramer_file,hopping_file,hcef_file}
runtime.run_name
```

Validation:
- 未知 bare name 或 qualified path 使用 `FXE-INPUT-003`。
- 重复 columns 使用 `FXE-INPUT-003`。

## 5) 禁止 sweep 的字段（MUST）
MUST:
- 以下字段不可 sweep：

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

- 任何指向 forbidden leaf 的 qualified path 也被禁止，例如
  `fsite.n_ele` 和 `fsite.F2_ratio`。
- `runtime.kramer_name` 不是 contract field，也不可 sweep。

Rationale:
- `n_ele`、Slater ratios、`RE` 和 `model.scheme` 影响共享 core artifacts，
  在一个 sweep batch 内固定。
- `runtime.kramer_source` 决定整个 sweep 中 `inputs.kramer_file` 的含义。

## 6) Uplus/Uminus Offset 规则（MUST）
MUST:
- 设置 `Uplus` 时，如存在 `fsite_np1.offset`，必须移除它。
- 设置 `Uminus` 时，如存在 `fsite_nm1.offset`，必须移除它。
- 这保持 `05-04-RUN_INPUT` 要求的互斥关系。

## 7) Run Names 与 Output Anchors（MUST）
MUST:
- 每个 materialized case 必须有唯一的 `runtime.run_name`。
- 如果 sweep table 包含 `runtime.run_name`，使用这些值，且必须唯一、非空。
- 如果 sweep table 不包含 `runtime.run_name`，expander 生成
  `<base_run_name>_sNNN`，case index 采用 zero padding。
- 每个 case 使用 `05-04-RUN_INPUT` 定义的 `paths.output_run` normalization：
  解析后的 run anchor 为
  `<paths.output_run or paths.output_root>/<runtime.run_name>`。

Validation:
- 重复或空的 `runtime.run_name` 使用 `FXE-INPUT-003`。

## 8) 执行语义（MUST）
MUST:
- 每个 case 都在内存中使用与 `fexchange run` 相同的 loader 校验。
- 每个 case 都通过与 `fexchange run` 相同的 runtime pipeline 执行。
- 串行执行在一个进程内运行所有 case。
- MPI 执行在可用且 communicator size 大于一时使用 `mpi4py`。
- 在 MPI 下，rank 0 展开并校验所有 cases，广播 case table，对每个 distinct core key
  只预热 shared core artifacts 一次，然后所有 ranks 执行静态切片
  `cases[rank::size]`。
- 如果通过 MPI launcher 启动且 size 大于一，但 `mpi4py` 不可用，运行必须失败，
  不得让每个 rank 回退到 serial path。

## 9) Progress Output（MUST）
MUST:
- sweep runner 写入 plain-text progress file：

```text
<output_root>/sweep_<base_stem>.txt
```

- progress file 包含 header、每个完成 case 一行，以及最后的
  `# done: k/N, f failed` 行。
- 每个 case line 记录 status、rank（或 serial marker）、elapsed time 和
  `runtime.run_name`。
- case failure 记录为 failed case line；一个 case 失败不要求停止同一 rank 中的其余 cases。
