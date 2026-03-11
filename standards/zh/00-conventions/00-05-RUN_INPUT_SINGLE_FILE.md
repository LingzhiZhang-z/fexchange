# 00-05-RUN_INPUT_SINGLE_FILE

本文件定义单文件运行输入契约。
目标：使用一个可读文件作为唯一外部控制面。

## 1) 作用范围（MUST）
MUST:
- 运行输入必须只来自一个文件。
- 默认格式使用 TOML。
- 生产模式下禁止使用文件外隐式默认值。
- MPI 进程布局控制不属于本输入文件契约。
- MPI 运行时行为由 `./standards/en/00-conventions/00-06-MPI_PARALLEL_RUNTIME.md` 定义。

Code form:
```text
input_source = ./run_input.toml
```

Validation:
- 输入文件缺失使用 `FXE-INPUT-001`。

## 2) 文件身份（MUST）
MUST:
- 规范文件名：`run_input.toml`。
- 顶层必填键：
  - `schema_version`
  - `standard_version`
  - `run_id`
  - `title`

Code form:
```text
required_top = {schema_version, standard_version, run_id, title}
```

Validation:
- 缺失键使用 `FXE-INPUT-002`。

## 3) 必需分节（MUST）
MUST:
- 必需表：
  - `[paths]`
  - `[runtime]`
  - `[checks]`
- 条件必需表（由执行窗口决定）：
  - `[physics]`（窗口包含 `LMSM` 或 `L0` 时必需）
  - `[model]`（窗口包含 `LMSM` 时必需）
  - `[sopt]`（窗口包含 `L3` 或 `L4` 时必需）
  - `[sources]`（窗口包含 `L2` 或 `L3` 或 `L4` 时必需）
  - `[wannier90]`（`sources.hopping_source = "wannier90"` 且窗口包含 `L2` 或 `L3` 或 `L4` 时必需）
  - `[kramer_input]`（`sources.kramer_source = "file"` 且窗口包含 `L4` 时必需）

Code form:
```text
required_core_sections = {paths, runtime, checks}
require_conditional_sections(start_level, end_level, sources)
```

Validation:
- 缺失必需分节使用 `FXE-INPUT-002`。

## 4) 字段契约（MUST）
MUST:
- `[physics]`：
  - `n_ele`（int，`1..13`）
  - `F2`（float）
  - `F4`（float）
  - `F6`（float）
  - 仅内部派生：`r42 = F4/F2`，`r62 = F6/F2`
  - 约束：`F2 != 0`
- `[model]`：
  - `scheme`（string，目前为 `RS`）
  - `symmetry`（string：`Oh` 或 `C3v`）
  - `c3v_mode_q3`（string：`cos` 或 `sin`）
- `[sopt]`：
  - `U`（float；物理上 $U = F^0$，即第零 Slater-Condon 参数）
  - `Jh`（float）
  - `zeta`（float）
- `[sopt]` / `[physics]` 交叉引用说明（MUST）：
  - `F^0` 不是 `[physics]` 中的字段，因为在固定 `n` 子空间内它对所有态贡献常数 $F^0 n(n-1)/2$，不影响 LMSM/LSJM 相对能量。
  - 对于 SOPT 能量分母（跨子空间差异 $f^{n\pm1}$ 与 $f^n$），$F^0 = U$ 取自 `[sopt].U`。
  - 实现 MUST 在跨子空间重构绝对 LSJM 能量用于中间态分母时使用 `sopt.U` 作为 $F^0$。
- `[sources]`：
  - `hopping_source`（`wannier90` 或 `file`）
  - `kramer_source`（`cef` 或 `file`）
  - `hopping_name`（string，`L2/L3/L4` 的稳定缓存/key token）
  - `kramer_name`（string，窗口包含 `L4` 时必填；稳定缓存/key token）
- `[paths]`：
  - `output_root`（string；本标准版本必须等于 `"./outputs"`）
  - 若 `hopping_source=file`：`hopping_file`
  - 若 `kramer_source=file`：`kramer_file`
- `[runtime]`：
  - `start_level`（string：`LMSM`、`LSJM`、`L0`、`L1`、`L2`、`L3` 或 `L4`）
  - `end_level`（string：`LMSM`、`LSJM`、`L0`、`L1`、`L2`、`L3` 或 `L4`）
  - `on_missing_upstream`（string：`fail`）
  - `read_first`（bool，必须为 `true`）
- `[checks]`：
  - `strict_mode`（bool）
  - `eps_profile`（string，如 `default`）

Validation:
- 类型/取值域不合法使用 `FXE-INPUT-003`。
- level 窗口非法（`start_level > end_level`）使用 `FXE-INPUT-003`。

Code form:
```text
r42 = F4 / F2
r62 = F6 / F2
level_order = {LMSM:1, LSJM:2, L0:3, L1:4, L2:5, L3:6, L4:7}
require level_order[start_level] <= level_order[end_level]
```

## 5) Wannier90 子契约（MUST）
MUST:
- 若 `hopping_source = "wannier90"`，则 `[wannier90]` 必填。
- `[wannier90]` 必填字段：
  - `soc_mode`（`with_soc` 或 `without_soc`）
  - `hr_path`
  - `win_path`
  - `orbital_basis`（`real_harmonic_default_w90`）
  - `orbital_order_id`
  - `energy_unit`（默认 `eV`）
  - `f_site_i`, `f_site_j`（int）
  - `f_site_i_cell`, `f_site_j_cell`（array[int]，各长度为 3）
  - `ligand_indices`（array[int]，长度 >= 0；为空表示只用直接 `f-f` 项）
  - `ligand_cells`（array[array[int]]，长度与 `ligand_indices` 相同，每个元素长度为 3）
  - `all_wannier_atom_indices`（array[int]，长度 >= 1）
  - `delta_mode`（`manual` 或 `from_onsite`）
  - `delta_reduction`（`channelwise` 或 `global_mean`）
  - 若 `delta_mode = "manual"`：
    - `delta_manual_kind`（`channelwise` 或 `global_mean`）
    - 若 `delta_manual_kind = "global_mean"`：`delta_manual_value`（float，单位=`energy_unit`）
    - 若 `delta_manual_kind = "channelwise"`：`delta_manual_file`（路径，NPZ 中包含 `Delta_puv[p,u,v]`，单位=`energy_unit`）
- 解析与映射细则遵循：
  `./standards/en/05-io/05-03-WANNIER90_PARSING_RULES.md`。
- 物理 hopping/CEF 契约遵循：
  `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`。

Validation:
- Wannier90 必填字段缺失使用 `FXE-W90-001/002`。

## 6) 确定性展开规则（MUST）
MUST:
- 运行输入中的 `U/Jh/zeta` 采用单点模式（输入契约中不做 sweep/笛卡尔积）。
- 单个输入文件对应一个确定性的 SOPT 参数元组
  `(U, Jh, zeta)`。
- 运行窗口必须是连续区间：
  `runtime.start_level ... runtime.end_level`。
- 运行从 `runtime.start_level` 启动前，必须先扫描磁盘并校验该层所需上游工件，同时校验窗口需要的输入字段。
- 任一上游工件缺失或校验失败时，必须直接失败。
- 窗口输入闸门（最小要求）：
  - 窗口包含 `LMSM`：要求 `[physics]` + `[model]`
  - 窗口包含 `LSJM`：要求上游 `LMSM` 工件
  - 窗口包含 `L0`：要求 `[physics]`
  - 窗口包含 `L1`：要求上游 `LSJM` 与 `L0` 工件
  - 窗口包含 `L2`：要求 `[sources]`、`sources.hopping_name` 与 hopping 输入源
  - 窗口包含 `L3`：要求 `[sources]`、`sources.hopping_name` 与 `[sopt]`
  - 窗口包含 `L4`：要求 `[sources]`、`sources.hopping_name`、`sources.kramer_name`、`[sopt]` 与 Kramer 输入源
- `start_level = end_level` 表示只计算该层；
  所有上游层必须已存在且通过校验，下游层全部跳过。
- 仅预计算模式可用 `end_level <= L2` 表示。
- 为兼容 `05-00-IO` 规范，`paths.output_root` 必须严格等于 `./outputs`。

Code form:
```text
if window_includes(L3):
  run_point = (sopt.U, sopt.Jh, sopt.zeta)
```

Validation:
- 当窗口包含 `L3` 或 `L4` 时，`sopt.U/Jh/zeta` 缺失或非法使用 `FXE-INPUT-003`。
- 预扫描发现上游工件缺失/无效使用 `FXE-IO-001/002`。
- 窗口必需输入字段缺失使用 `FXE-INPUT-002/003`。
- 按窗口要求缺失/非法 `sources.hopping_name` 或 `sources.kramer_name` 使用 `FXE-INPUT-003`。
- `paths.output_root != "./outputs"` 使用 `FXE-INPUT-003`。

## 7) 最小可读示例（MUST）
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
- 示例必须在无隐式默认值条件下通过 schema/type 校验。

## 8) 错误码映射（MUST）
MUST:
- 本文件失败必须使用固定错误码，来源于：
  `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`。

Code form:
```text
input_error -> FXE-INPUT-* / FXE-W90-* / FXE-SCHEMA-*
```

Validation:
- 无错误码输入失败报告无效。
