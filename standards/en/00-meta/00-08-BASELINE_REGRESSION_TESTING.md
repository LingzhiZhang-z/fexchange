# Baseline Regression Testing

This file records the current baseline regression-testing prompt/contract for
fresh end-to-end validation runs.

# fexchange 测试 Prompt

把下面整段内容直接发给另一个 AI，让它按这个要求在当前仓库里做测试。

---

你现在位于仓库根目录：

`/Users/lingzhi/Documents/Code/fexchange`

你的任务是对当前代码做一次“以真实运行结果为准”的回归验证。请不要先假设代码是对的，也不要只看单元测试；要实际调用主程序和 `fexchange/tools/` 下的脚本做端到端验证。

## 严格约束

1. **不得查看或复用 `tests/` 和 `scripts/` 下的任何现有测试脚本。** 你必须从零编写所有测试逻辑。
2. **纯 shell 实现。** 整个测试链路只用 bash + awk + gnuplot，不需要额外的 Python 辅助脚本来读取结果。CLI 的输出是纯文本，直接用 shell 工具处理。
3. **输出目录必须是随机唯一的。** 例如 `outputs/validation_$(date +%s)_$(openssl rand -hex 4)`，不得使用固定目录名，避免旧结果干扰。
4. **`w90_decompose_h_local.py` 的 projector/CEF 路线必须使用 direct-local。** 也就是：凡是需要从 `h_local` 生成最终用于测试的 CEF/projector 参数时，必须走 `Scheme B` 的 direct-local 路线（CLI 上等价于 `--direct_local_fit`）；不能把 `Scheme A` 的 orbital `h_cef` 拟合结果当作最终 projector 来源。`Scheme A` 只能作为中间诊断输出，或用于读取 `zeta`。

## 目标

先做一轮快速回归，分两部分：

1. YbOX 的 6 个例子
2. PRB 曲线生成检查

### A. YbOX 快速回归

YbOX 是 f13 系统（`n_ele = 13`）。

范围固定为 YbOX 的 6 个例子：

- `YbOBr_soc_1st_bond1`
- `YbOBr_soc_2nd_bond1`
- `YbOCl_soc_1st_bond1`
- `YbOCl_soc_2nd_bond1`
- `YbOF_soc_1st_bond1`
- `YbOF_soc_2nd_bond1`

限制条件：

- 只测 `bond1`
- 只测 `U = 6 eV`
- `Jh/U` 用各自 legacy 输入文件里给出的完整 ratio 列表
- `energy_reference` 取 `lsjm_ground`
- `offset_n = 0`
- `nm1` 和 `np1` 不单独写 branch 覆盖，默认共享主 branch 已解析好的参数

YbOX 这 6 个 case 的 `SLATER_RATIOS` 都相同：

- `F2/Jh = 12.327`
- `F4/Jh = 7.757`
- `F6/Jh = 5.587`

这里的 `SLATER_RATIOS` 在当前仓库标准里**只作为 ratio-source 三元组使用**。
也就是：

- 它们用于定义 `r42 = F4_ratio/F2_ratio`、`r62 = F6_ratio/F2_ratio`
- 最终运行时实际使用的绝对 `F2/F4/F6` 必须由当前 `run_input` 语义从 `Jh` 反推
- 不要把它理解成旧式的“直接写死绝对 `F2 = 12.327 * Jh`”语义

因此对任意一个测试点：

- `Jh_meV = 1000 * U_eV * (Jh/U)`
- `F2_ratio = 12.327`
- `F4_ratio = 7.757`
- `F6_ratio = 5.587`

若需要在日志或辅助输出中给出当前标准下的绝对 `F2/F4/F6`，应按项目标准公式计算：

- `r42 = F4_ratio / F2_ratio`
- `r62 = F6_ratio / F2_ratio`
- `F2 = 6435 * Jh / (286 + 195*r42 + 250*r62)`
- `F4 = r42 * F2`
- `F6 = r62 * F2`


YbOX 的 `zeta` 不要手写死常数，应从 `w90_decompose_h_local.py` 的输出读取。当前这 6 个 case 的参考值是：

- `YbOBr_soc_1st_bond1`: `391.738000 meV`
- `YbOBr_soc_2nd_bond1`: `391.739778 meV`
- `YbOCl_soc_1st_bond1`: `393.334444 meV`
- `YbOCl_soc_2nd_bond1`: `393.334444 meV`
- `YbOF_soc_1st_bond1`: `393.643611 meV`
- `YbOF_soc_2nd_bond1`: `393.643778 meV`

### B. PRB 测试

PRB 是 f1 系统（`n_ele = 1`）。

PRB 部分至少要覆盖当前仓库已有的这两类测试：

1. 标准 PRB 曲线生成
2. `Oh, J=5/2` projector 变体

PRB 的基本要求：

- 参考图片在 `data/test/PRB/Ref_prb.png`
- 目标不是必须做自动图片比对，但至少要把曲线数据和最终图片生成出来
- 材料是：
  - `K`
  - `Rb`
  - `Cs`
- `U` 扫：
  - `2 eV`
  - `3 eV`
  - `4 eV`
  - `6 eV`
- `Jh/U` 扫：
  - `0.00` 到 `0.20`
  - 步长 `0.01`
- `zeta`：
  - `K = 120 meV`
  - `Rb = 110 meV`
  - `Cs = 110 meV`

PRB 的 `SLATER_RATIOS` 固定为：

- `F2/Jh = 12.980`
- `F4/Jh = 8.163`
- `F6/Jh = 5.878`

这里同样必须按**当前标准语义**理解：

- 这三个数只作为 `F2_ratio/F4_ratio/F6_ratio` 输入
- 不要回到历史口径去把它们直接当成绝对 `F2/F4/F6 = const * Jh`

因此对任意一个 PRB 测试点：

- `Jh_meV = 1000 * U_eV * (Jh/U)`
- `F2_ratio = 12.980`
- `F4_ratio = 8.163`
- `F6_ratio = 5.878`

如需显式写出当前标准下的绝对 `F2/F4/F6`，同样必须按项目标准公式由 `Jh` 反推，而不是使用历史近似口径。

标准 PRB 测试：

- hopping 用 `fexchange/tools/slater_koster_pf.py literature`
- projector 用当前仓库测试链路里的手工 `Gamma7` 投影
- 输出每个材料、每个 `U` 的曲线数据文件
- 最后用 `gnuplot` 生成总图

PRB 变体测试：

- hopping 仍用 `fexchange/tools/slater_koster_pf.py literature`
- projector 不再手写，而是用：
  - `fexchange/tools/cef_states.py`
  - `--point-group Oh`
  - `--J 2.5`
- 同样生成曲线数据和图片

注意：

- 这个变体测试的目标是验证“另一条 projector 生成路线能否走通并产出合法结果”
- 不要求该变体曲线必须与标准 `Gamma7` 曲线不同
- 如果两条路线在当前参数选择下数值相同，这是允许的；应如实记录，不应当判为失败

## 输入与参考数据

测试数据在：

- `data/test/YbOBr`
- `data/test/YbOCl`
- `data/test/YbOF`

每个材料用：

- `data/test/<material>/input/soc_1st_bond1.dat`
- `data/test/<material>/input/soc_2nd_bond1.dat`

参考结果用：

- `data/test/<material>/<case>/result_HOLE_U6.00.dat`

PRB 参考数据：

- `data/test/PRB/Ref_prb.png`

legacy 输入文件里需要读取的信息包括：

- `FILE_HR`
- `SLATER_RATIOS`
- `ratios`
- `ATOMS`
- `SPINOR`
- `NDIMS`
- `<RS ... RS>` 里的 cell 信息

## 运行原则

1. 必须以主程序真实输出作为验证对象。
2. 要优先调用：
   - `fexchange/cli.py`
   - `fexchange/tools/w90_extract.py`
   - `fexchange/tools/w90_decompose_h_local.py`
   - `fexchange/tools/cef_states.py`
3. 不要把 `tests/` 下的 Python 脚本当成被验证程序本身。
4. 可以写临时 shell 脚本来编排测试。
5. 如果为了数据读取和对比必须写小的辅助脚本，它们只能做数据处理，不能替代主程序计算。
6. 结论必须基于 fresh 运行结果，不要只复述旧的 summary。

## 推荐测试流程

### A. YbOX

对每个 case：

1. 从 legacy 输入文件解析材料、bond、Wannier90 路径、atom/cell、`SLATER_RATIOS`、`ratios`。
2. 运行 `w90_extract.py` 生成：
   - `w90_t_mu.dat`
   - `w90_h_local.dat`
3. 对每个材料的同一 family（`1st` 或 `2nd`）：
   - 用第一组 case 的 `w90_h_local.dat` 调 `w90_decompose_h_local.py`
   - 这里必须显式使用 direct-local 路线；如果需要生成 CEF 配置或 projector，必须传 `--direct_local_fit`
   - 允许从该命令输出中读取 `zeta`，但最终用于 projector 的参数来源必须是 direct-local 的 `Scheme B`
   - 再用 `cef_states.py` 生成 projector
4. 对每个 ratio：
   - 写 `run_input.toml`
   - 调 `python fexchange/cli.py run run_input.toml`
   - 从 CLI 在 `outputs/` 下生成的 txt 结果文件直接读取 `J_mu`（见下方"CLI 输出格式"）
   - 和 `result_HOLE_U6.00.dat` 里对应 ratio 的参考值比较
5. 比较时允许做 gauge 对齐；推荐先用 `ratio = 0.10` 找 family gauge，再应用到该 family 全部 ratio。

### B. PRB

对 PRB 标准测试：

1. 对每个材料目录创建独立输出目录。
2. 用 `slater_koster_pf.py literature` 生成 hopping。
3. 准备 projector：
   - 标准 PRB 测试用手工 `Gamma7` projector
   - 变体测试用 `cef_states.py --point-group Oh --J 2.5`
   - 如果该 projector 需要从 `h_local` 间接构造或校准，同样必须基于 `w90_decompose_h_local.py --direct_local_fit` 的结果，而不是 `Scheme A`
4. 对每个 `U` 和每个 ratio：
   - 写 `run_input.toml`
   - 调 `python fexchange/cli.py run run_input.toml`
   - 从 CLI 在 `outputs/` 下生成的 txt 结果文件直接读取 `J_mu`（见下方"CLI 输出格式"）
   - 整理成 PRB 曲线数据文件
5. 把 `J_mu` 整理成 PRB 曲线时，至少输出：
   - `J`
   - `K`
   - `Gamma`
   - `GammaPrime`
   - `mapping_residual`
6. 用 `gnuplot` 输出最终 PNG 图片。

PRB 绘图缩放规则（必须和参考图一致）：

- `U = 2 eV` 曲线按原值绘制（乘 `1`）
- `U = 3 eV` 曲线绘图前乘 `1.5`
- `U = 4 eV` 曲线绘图前乘 `2`
- `U = 6 eV` 曲线绘图前乘 `3`

也就是说，图例应体现：

- `U = 2 eV`
- `(x1.5) U = 3 eV`
- `(x2) U = 4 eV`
- `(x3) U = 6 eV`

推荐至少产出：

- `prb_curves_U2.00.dat`
- `prb_curves_U3.00.dat`
- `prb_curves_U4.00.dat`
- `prb_curves_U6.00.dat`
- `prb_particle_plot.png`

## run_input 要求

`run_input.toml` 至少应包含：

- `schema_version = "fxe.run_input.v1"`
- `standard_version = "2026-02"`
- `[units] energy = "meV"`
- `[physics]` 中显式写：
  - `n_ele`
  - `F2_ratio`
  - `F4_ratio`
  - `F6_ratio`
- `[model] scheme = "RS"`
- `[sopt]` 中显式写：
  - `U`
  - `Jh`
  - `zeta`
  - `offset = 0`
  - `energy_reference = "lsjm_ground"`
- `[runtime]` 中显式写：
  - `kramer_name`
- `[inputs]` 中显式写：
  - `hopping_file`
  - `projector_file`
- `[paths] output_root = "./outputs"`
- `[runtime] start_level = "LMSM", end_level = "L3", on_missing_upstream = "fail", read_first = true`
- `[checks] strict_mode = true, eps_profile = "default"`

补充说明：

- 本 prompt 以当前仓库的 `run_input` 标准为准。
- `F2_ratio/F4_ratio/F6_ratio` 只定义比值，不直接携带绝对能量尺度。
- 运行时使用的绝对 `F2/F4/F6` 由 `[sopt].Jh` 和 ratio-source 三元组按项目标准自动确定。
- 不要引入历史 PRB 口径，也不要在这里重新定义另一种 `Jh` 含义。

## CLI 输出格式

每次 `fexchange run` 成功后，最终交换结果写在 run-scoped final artifact 目录：

```
<output_root>/<run_name>/L3/kramer-<kramer_name>/exchange.txt
```

`exchange.txt` 是 human-readable sidecar；机器契约仍然是同目录的 `data.npz`。
SOPT 写 total `J_mu`；FOPT 写 total 加 `P1..P5` process-resolved `J_mu`。

```
# label mapping_residual Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz
```

其中 `Jxx ... Jzz` 就是 3×3 的 `J_mu` 交换张量，单位沿用输入 raw 单位。

注意：

- `r42` 和 `r62` 应由最终使用的 `F2/F4/F6` 直接决定。
- 如果不写 `physics_nm1/physics_np1`，`nm1/np1` 应共享主 branch 的最终 `F2/F4/F6`。
- 如果不写 `sopt_nm1/sopt_np1`，`nm1/np1` 应共享主 branch 的 `U/Jh/zeta/offset`，但这里本轮测试不单独覆盖 branch。

## 判定标准

### A. YbOX

你需要给出每个 case 的：

- 材料
- case 名称
- family
- gauge
- `zeta_meV`
- `hlocal_text_match`
- `max_aligned_err`
- `pass`

推荐生成一个汇总表，例如：

`outputs/<your_validation_dir>/summary.tsv`

并且以：

- `pass = 1` 表示该 case 全部 ratio 的对齐后最大误差 `<= 1.0e-3`
- `pass = 0` 表示失败

### B. PRB

PRB 至少需要确认：

- 三个材料的曲线数据文件都生成成功
- 四个 `U` 的曲线文件都存在
- `gnuplot` 最终图片生成成功
- 标准 projector 和 `Oh, J=5/2` projector 变体都能独立跑通

如果你想做额外检查，可以再补：

- 曲线文件行数是否正确
- 关键耦合通道是否有限
- 图片是否与 `Ref_prb.png` 大致同趋势

## 你还需要顺手检查的两类问题

除了跑 6 个例子，请额外检查下面两点，并明确写进结论：

1. `run_input` 对 branch 覆盖段是否做了严格字段校验  
   例如 `physics_nm1.foo = 1` 这种非法字段会不会被静默放过。

2. 浅层 `__cfg-*` 结果文件名的签名，是否混入了不影响物理结果的元数据  
   例如只改 `run_id` 或 `title`，签名是否变化。

## 输出要求

最终回答请只包含下面四部分：

1. `Changed Files`
2. `Behavior Change`
3. `Tests`
4. `Residual Risk`

如果没有改代码，就在 `Changed Files` 里明确写“无”。

如果发现问题，优先按“代码审查”方式报告，重点写：

- 问题是什么
- 为什么会影响结果或接口稳定性
- 证据是什么
- 对应文件和行号

不要写空泛总结，不要只说“看起来没问题”。

---

如果你只需要一个最短执行目标，那么先完成这两件事：

1. “在仓库根目录 fresh 跑 YbOBr/YbOCl/YbOF 的 `bond1`、`U=6 eV` 六个例子，`energy_reference = lsjm_ground`，输出 `summary.tsv`，并检查 branch 非法字段是否会被 loader 静默接受。”
2. “对 PRB 跑一轮标准 projector 和 `Oh, J=5/2` projector 变体，生成曲线数据和 PNG 图片，并确认三套材料、四个 `U` 都落盘成功。”
