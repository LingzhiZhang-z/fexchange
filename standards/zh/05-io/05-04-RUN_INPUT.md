# 05-04-RUN_INPUT

本文件定义单文件运行输入契约。

## 1) 适用范围（MUST）
MUST:
- 运行输入必须来自一个 TOML 文件。
- 顶层必填键为 `schema_version`、`standard_version`、`run_id` 和 `title`。
- 必填表为 `[paths]`、`[runtime]`、`[checks]` 和 `[fsite]`。
- `[model]` 可选，默认 `scheme = "RS"`。
- `model.scheme` 必须是 `RS` 或 `ED`。
- 当 `end_level >= L2` 时，`[inputs]` 必填。
- 当 `branch = "fopt"` 时，`[ligand.1]` 和 `[ligand.2]` 必填。
- `[units]` 只是可选 metadata。Runtime numeric fields 和 hopping files 以 raw
  values 消费，不做内部单位转换；用户必须保证所有能量类输入使用一致单位。
- `[sweep]` 保留给 `fexchange sweep` front-end
  (`./standards/en/05-io/05-05-SWEEP_INPUT.md`)。它不是 `fexchange run` 的合法
  顶层 section；sweep front-end 在按本文件校验每个 materialized single-run
  input 之前会移除 `[sweep]`。

Validation:
- 输入文件缺失使用 `FXE-INPUT-001`。
- 缺失必需键或表使用 `FXE-INPUT-002`。
- 不支持的顶层 section 使用 `FXE-INPUT-003`。

## 2) Runtime 表（MUST）
MUST:
- `runtime.branch` 必填，且必须是 `sopt` 或 `fopt`。
- `runtime.end_level` 必填，且必须是
  `LMSM`、`LSJM`、`L0`、`L1`、`L2`、`L3` 之一。
- 当 `end_level >= L1` 时，`runtime.run_name` 必填，因为 `L1/F` 和所有下游
  工件对每种 scheme 和 Kramers route（RS/ED、stevens/manual）都是 run-scoped。
- `runtime.kramer_source` 可选，默认 `stevens`。
  接受的 normalized values 是 `stevens` 和 `manual`；runtime loaders 可以接受
  legacy aliases `steven` 和 `mannual`。
- `runtime.kramer_name` 不再是 runtime contract field。如果 legacy input 携带它，
  实现可以把它保留为 inert provenance，但它不得影响 validation、artifact paths、
  keys 或 computation。
- 两个 branch 都终止于 `L3`。FOPT `L3` 包含 total/process raw `h_eff_4` 输出和
  total/process spin-1/2 exchange 输出。Runtime FOPT exchange 输出要求 projected
  local space 为二维。

Code form:
```toml
[runtime]
branch = "fopt"
end_level = "L3"
run_name = "lab_A"
kramer_source = "stevens"
```

## 3) f-Site 表（MUST）
MUST:
- `[fsite]` 合并了原 f-shell ratio fields 和 denominator fields。
- 必填字段：
  - `n_ele`
  - `U`
  - `Jh`
  - `RE` preset 或 `F2_ratio`, `F4_ratio`, `F6_ratio` 全部显式给出
- 可选字段：
  - `RE`
  - `zeta`
  - `offset`
  - `energy_reference`（`lsjm_ground` 或 `zero`）
- `[fsite_nm1]` 和 `[fsite_np1]` 可覆盖以下任意子集：
  `RE`, `F2_ratio`, `F4_ratio`, `F6_ratio`, `U`, `Jh`, `zeta`, `offset`。
  因此相邻扇区可以携带自己的 Slater ratio（例如相邻元素 preset）。
- `[fsite_np1]` 也可以设置 `Uplus`，表示从主 `f^n` reference 到 `f^{n+1}` 扇区的
  target minimum energy gap。`Uplus` 与 `fsite_np1.offset` 互斥。
- `[fsite_nm1]` 也可以设置 `Uminus`，表示从主 `f^n` reference 到 `f^{n-1}` 扇区的
  target minimum energy gap。`Uminus` 与 `fsite_nm1.offset` 互斥。
- `Uplus` 和 `Uminus` 是自动构造 branch-local offset 的输入便利项。
  Runtime denominator contraction 消费的是解析后的 intermediate energies，
  而不是额外的 Hamiltonian term。

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
- `n_ele` 必须是 `1..13` 的整数。
- 使用显式 ratio 时，`F2_ratio` 必须非零。
- denominator reconstruction 中的 `F2/F4/F6` 由 `Jh` 和
  `r42 = F4_ratio/F2_ratio`, `r62 = F6_ratio/F2_ratio` 派生。

## 4) Ligand 表（MUST）
MUST:
- FOPT 需要 `[ligand.1]` 和 `[ligand.2]`。
- 必填 ligand fields 是 `Delta` 和 `U_p`。
- `lambda_p` 可选，默认 `0.0`。
- `lambda_p = 0.0` 选择 no-SOC ligand cache；非零选择 SOC ligand cache。

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

## 5) Inputs 表（MUST）
MUST:
- 当 `end_level >= L2` 时，`[inputs].hopping_file` 必填。
- `[inputs].kramer_file` 是统一的 doublet input path。
- 当 `runtime.kramer_source = "stevens"` 且 `end_level >= L2` 时，
  `[inputs].kramer_file` 必填；此模式下它携带从 SOC-lowest LSJM subspace 到目标
  doublet/quasi-doublet 的 projector `W`。
- 当 `runtime.kramer_source = "manual"` 且 `end_level >= L1` 时，
  `[inputs].kramer_file` 必填；此模式下它携带 Fock-determinant form 的外部
  Kramers basis。
- `[inputs].hcef_file` 可选。当它和 `model.scheme = "ED"` 同时存在时，
  它作为 adjacent-sector IONED 的 one-body CEF matrix 使用。
- Runtime matrix text files 使用带 `[key]` header 的 multi-block format。
- SOPT hopping 必须包含 block `[t_mu]`，shape `(14, 14)`。
- FOPT hopping 必须包含 blocks `[t_f1_lig1]`, `[t_f1_lig2]`,
  `[t_f2_lig1]`, `[t_f2_lig2]`，每个 shape `(14, 6)`。
- Stevens-mode `kramer_file` text input（`.txt` / `.dat`）必须每个 doublet state
  一个 block，命名为 `[W_state_0]`, `[W_state_1]`, ... `[W_state_{n_k-1}]`。
  每个 block 恰好包含 `n_j` 行 `real imag`（`W` 的一列）。
  State indices 必须从 0 开始连续；列按数字顺序堆叠为 shape `(n_j, n_k)` 的 `W`。
- Stevens-mode `kramer_file` binary input（`.npy` / `.npz`）是 rank-2 array
  （`.npz` key 为 `W`），且 `shape[0] == n_j`。
- Manual Kramers text input 以 `fn <n>` 开头，然后使用恰好两个 block：
  `[K_state_0]` 和 `[K_state_1]`。每行恰好 16 个字段：
  `real imag occ_0 ... occ_13`。Occupation fields 是 canonical f spin-orbital
  order 下的 `0/1` 值，且每行必须恰好有 `n` 个 occupied orbitals。
  Blocks 按数字顺序堆叠为 `K_fock.shape = (dim_fock(n), n_k)`。
- `hcef_file` 是 canonical f spin-orbital order 下的 Hermitian `14 x 14`
  one-body matrix。Text input 可以使用带 `14*14` 个 complex rows 的 block
  `[hcef]`，或 runtime matrix loader 接受的 plain matrix format。

Code form:
```toml
[inputs]
hopping_file = "data/hopping/wan_v1.txt"
kramer_file = "data/projector/kr_a.txt"      # stevens projector W
hcef_file = "data/hcef/hcef_14x14.txt"
# or, in manual mode:
# kramer_file = "data/kramer/manual_kramer.txt"
```

## 6) Paths 表（MUST）
MUST:
- `[paths].output_root` 必填，并锚定 core artifacts 和 global index files。
- `[paths].output_run` 可选，解释为 run-scoped artifacts 的 base directory。
  省略时 base 是 `output_root`；设置时该值替代 `output_root` 作为 base。
  解析后的 run anchor 始终是 `<base>/<runtime.run_name>`。
- Run-scoped artifacts（`IONED`、`L1/F`、`L2`、`L3`、`source.txt`、`run.log`）
  写入解析后的 run anchor 下。

Code form:
```toml
[paths]
output_root = "./outputs"
output_run = "./outputs/custom_base"  # resolved run anchor adds runtime.run_name
```

## 7) ED Scheme 示例（MUST）
Code form:
```toml
[model]
scheme = "ED"

[runtime]
branch = "sopt"
end_level = "L1"
run_name = "ed_demo"
```

## 8) Minimal FOPT 示例（MUST）
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
