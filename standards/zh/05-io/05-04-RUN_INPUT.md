# 05-04-RUN_INPUT

本文件定义单文件运行输入契约。
目标：使用一个可读文件作为唯一外部控制面。

## 1) 作用范围（MUST）
MUST:
- 运行输入必须只来自一个文件。
- 默认格式使用 TOML。
- 生产模式下禁止使用文件外隐式默认值。
- MPI 进程布局控制不属于本输入文件契约。
- 运行后端布局由实现决定，不属于本输入文件契约。

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
- 不允许出现未支持的额外顶层分节。
- 可选表：
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
- 条件必需表（由执行窗口 `LMSM ... end_level` 决定）：
  - `[inputs]`（窗口包含 `L2` 或 `L3` 或 `L4` 时必需）
  - `[sopt]`（窗口包含 `L3` 或 `L4` 时必需）
- `[sources]` 可以在输入文件中省略；loader 可以先根据 label 或输入文件名 stem
  生成并规范化该分节，再校验规范化后的 source name。
- `[physics]` 和 `[model]` 可以在输入时省略；下游运行时可以从磁盘上的唯一上游工件
  推断核心参数。

Code form:
```text
required_core_sections = {paths, runtime, checks}
require_conditional_sections(end_level)
normalize_sources(inputs, sources)
```

Validation:
- 缺失必需分节使用 `FXE-INPUT-002`。
- 出现未支持的顶层分节使用 `FXE-INPUT-003`。

## 4) 字段契约（MUST）
MUST:
- `[units]`：
  - `energy`（string，可选；`meV` 或 `eV`；默认 `meV`）
  - 运行时 loader MUST 在下游消费前把所有能量类输入规范化到统一内部单位
- `[physics]`：
  - `n_ele`（int，`1..13`）
  - `RE`（string，可选；`auto` 或 `Ce/Pr/Nd/Pm/Sm/Eu/Gd/Tb/Dy/Ho/Er/Tm/Yb` 之一）
  - `F2_ratio` / `F4_ratio` / `F6_ratio`（float，可选的显式 ratio-source 三元组；必须全给或全不给）
  - 仅内部派生：`r42 = F4_ratio/F2_ratio`，`r62 = F6_ratio/F2_ratio`
  - 约束：`F2_ratio != 0`
  - 显式输入 `F2_ratio/F4_ratio/F6_ratio` 只定义 `r42/r62`；它们的公共尺度本身不携带独立运行时物理意义
  - 运行时在 LSJM 排序和 `L3/L4` 分母重建中使用的绝对 `F2/F4/F6` MUST 由 `sopt.Jh`
    与派生 ratio 按 `02-01-HINT` 重建
  - 若 `RE != "auto"` 且没有显式 ratio 三元组，实现可以从 `RE` 预设导出默认 ratio-source 值
- `[physics_nm1]`、`[physics_np1]`：
  - `f^(n-1)` 与 `f^(n+1)` 相邻扇区的可选 branch override
  - 允许字段：`RE`、`F2_ratio`、`F4_ratio`、`F6_ratio`
  - 显式 branch ratio 键必须全给或全不给
  - 如果 branch 显式 ratio 键缺失且 branch `RE != "auto"`，实现可以从 branch `RE` 导出默认 ratio-source
  - 否则 branch ratio 回退到主扇区已解析的 ratio
- `[model]`：
  - `scheme`（string；若该 section 存在则可选；当前仅支持 `RS`）
- `[sopt]`：
  - `U`（float；物理上 $U = F^0$，即第零 Slater-Condon 参数）
  - `Jh`（float）
  - `zeta`（float；若未由 `physics.RE` 导出则必填）
  - `offset`（float，可选；默认 `0`）
  - `energy_reference`（string，可选；`lsjm_ground` 或 `zero`；默认 `lsjm_ground`）
- `[sopt_nm1]`、`[sopt_np1]`：
  - `f^(n-1)` 与 `f^(n+1)` 相邻扇区的可选 branch override
  - 允许字段：`U`、`Jh`、`zeta`、`offset`
  - 缺失字段按字段级回退到已解析的主 `[sopt]`
  - `offset` 默认 `0`
  - 如果 branch `RE != "auto"` 且 branch `zeta` 缺失，实现可以从 branch `RE`
    预设导出 branch `zeta`，而不是继承主扇区 `zeta`
- `[sopt]` / `[physics]` 交叉引用说明（MUST）：
  - `F^0` 不是 `[physics]` 中的字段，因为在固定 `n` 子空间内它对所有态贡献常数 $F^0 n(n-1)/2$，不影响 LMSM/LSJM 相对能量。
  - 对于 SOPT 能量分母（跨子空间差异 $f^{n\pm1}$ 与 $f^n$），$F^0 = U$ 取自 `[sopt].U`。
  - 实现 MUST 在跨子空间重构绝对 LSJM 能量用于中间态分母时使用 `sopt.U` 作为 $F^0$。
  - 实现 MUST 用 branch `Jh` 与 branch ratio 重构绝对 branch `F2/F4/F6`：
    `F2 = 6435*Jh / (286 + 195*r42 + 250*r62)`，`F4 = r42*F2`，`F6 = r62*F2`。
  - 若存在 branch override，实现 MUST 对 `f^(n-1)` 和 `f^(n+1)` 使用
    `physics_nm1/physics_np1` 解析出的 branch ratio 与 `sopt_nm1/sopt_np1`
    中的 branch `Jh` 构造分母，而主 `[physics]` / `[sopt]` 仍作为缺失 branch 字段的默认来源。
  - `sopt.energy_reference` 控制主扇区分母参考能：
    - `zero`：使用 $E_\mathrm{ref}=0$
    - `lsjm_ground`：使用 LSJM 中选定的 `f^n` 参考态 `u0`，并重构
      $E_\mathrm{ref}=\mathrm{offset}^{(n)} + F^{0,(n)}c_{F0}(u0) + F^{2,(n)}c_{F2}(u0) + F^{4,(n)}c_{F4}(u0) + F^{6,(n)}c_{F6}(u0)$
      （不包含 `zeta * coef_zeta` 项）
- `[sources]`：
  - 该表在输入时可省略
  - `hopping_label`（string；`L2/L3/L4` 的用户可读 label）
  - `projection_label`（string；`L4` 的用户可读 label）
  - 仅内部规范字段：
    - `hopping_name`（稳定 cache/key token；从以下第一个非空来源规范化得到：
      `sources.hopping_name`、`sources.hopping_label`、`inputs.hopping_label`、
      `stem(inputs.hopping_file)`）
    - `kramer_name`（稳定 cache/key token；从以下第一个非空来源规范化得到：
      `sources.kramer_name`、`sources.projection_label`、`inputs.projection_label`、
      `stem(inputs.projector_file)`）
- `[paths]`：
  - `output_root`（string；本标准版本 MUST 等于 `"./outputs"`）
- `[inputs]`：
  - `hopping_file`
  - `projector_file`（窗口包含 `L4` 时必需）
- `[runtime]`：
  - `end_level`（string：`LMSM`、`LSJM`、`L0`、`L1`、`L2`、`L3` 或 `L4`）
- `[checks]`：
  - `strict_mode`（bool）
  - `eps_profile`（string，例如 `default`）

Validation:
- 类型或取值域不合法使用 `FXE-INPUT-003`。

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

## 5) 引用文件契约（MUST）
MUST:
- `[inputs]` 只承载已经解析好的运行时文件路径。
- `inputs.hopping_file` 是 `L2/L3/L4` 使用的外部 hopping 输入。
- `inputs.projector_file` 是 `L4` 使用的外部 projector/Kramers 输入。
- 当前 run-input 契约不再包含额外的 source-specific 顶层表，例如 `[wannier90]`。
- 被引用文件的 source-specific 解析规则和物理解释仍由各自标准与运行阶段负责。
  对于 Wannier90 相关语义，见：
  - `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`
  - `./standards/en/05-io/05-03-WANNIER90_PARSING.md`

Validation:
- 缺失必需的引用文件路径使用 `FXE-INPUT-002/003`。

## 6) 确定性展开规则（MUST）
MUST:
- 运行输入中的 `U/Jh/zeta` 采用单点模式（输入契约中不做 sweep/笛卡尔积）。
- 单个输入文件对应一个确定性的 SOPT 参数元组
  `(U, Jh, zeta)`。
- 可选 branch override 不改变单点运行本质；
  它们只改变相邻扇区 `f^(n-1)` / `f^(n+1)` 的物理参数和分母参数如何解析。
- 运行窗口是 `LMSM ... runtime.end_level`。
  执行总是从最前面的 level 开始；每个 level 先检查磁盘缓存，若存在匹配工件则读取，否则计算。
- 主扇区核心参数可以来自显式 `[physics]` / `[model]` 输入，
  也可以来自磁盘上唯一的上游 core token。
- 窗口输入闸门（最小要求）：
  - 窗口包含 `L2`：要求 `[inputs]` 与规范化后的 `sources.hopping_name`
  - 窗口包含 `L3`：要求 `[inputs]`、规范化后的 `sources.hopping_name` 与 `[sopt]`
  - 窗口包含 `L4`：要求 `[inputs]`、规范化后的 `sources.hopping_name`、规范化后的 `sources.kramer_name` 与 `[sopt]`
- 高于 `end_level` 的 level 会被跳过。
- 仅预计算模式可用 `end_level <= L2` 表示。
- 为兼容 `05-00-IO`，`paths.output_root` 必须严格等于 `./outputs`。

Code form:
```text
if window_includes(L3):
  run_point = (sopt.U, sopt.Jh, sopt.zeta)
```

Validation:
- 当窗口包含 `L3` 或 `L4` 时，`sopt.U/Jh/zeta` 缺失或非法使用 `FXE-INPUT-003`。
- 窗口必需输入字段缺失使用 `FXE-INPUT-002/003`。
- 按窗口要求缺失或非法的 `sources.hopping_name` / `sources.kramer_name` 使用 `FXE-INPUT-003`。
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
- 示例必须在不依赖隐式默认值的前提下通过 schema/type 校验。

## 8) 错误码映射（MUST）
MUST:
- 本文件失败必须使用固定错误码，来源于：
  `./standards/en/06-utils/06-01-ERROR_CODES.md`。

Code form:
```text
input_error -> FXE-INPUT-* / FXE-SCHEMA-*
```

Validation:
- 无错误码的输入失败报告无效。
