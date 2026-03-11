# 01-01-STATE_VECTOR_CONVENTION

本文件定义所有模块必须遵守的态矢表示规范。

## 1) 适用范围（MUST）
- 本文件只定义态对象与规范化/规范相位规则。
- 本文件不定义模型哈密顿量与微扰公式。

## 2) 态对象分层（MUST）
统一使用三层对象：
1. `BasisDet`：Fock 基中的单个 Slater 行列式。
2. `StateVec`：由多个 `BasisDet` 线性组合得到的物理态。
3. `StateSet`：一组 `StateVec`，以矩阵形式存储。

规则：
- `StateVec` 可用于模块内部的单态计算。
- 对外接口交换必须统一使用 `StateSet`。
- 若只导出一个态，也必须编码为 `n_states = 1` 的 `StateSet`。

Math:
$$
\lvert \psi_j \rangle = \sum_{\alpha=1}^{D_n} c_{\alpha j}\,\lvert \alpha \rangle,
\qquad
V_{\alpha j} = c_{\alpha j}.
$$

Code form:
```text
BasisDet  = int det
StateVec  = {basis_id, n_ele, coeffs[alpha]}
StateSet  = {basis_id, n_ele, state_order_id, V_fock[alpha, j], labels[j], meta}
```

Index:
- $\alpha$：固定扇区 Fock 基中的行列式索引。
- $j$：态索引。
- $D_n$：该电子数扇区维度。

符号约定：
- 行列式基底索引在 code form 中使用希腊字母名（`alpha`, `beta`, `gamma`, `eta`）。
- 态索引使用拉丁字母（`j`, `k_state` 等）。
- 不要在相邻公式中把行列式索引符号重复用作标量系数名。

Validation:
- `StateVec`/`StateSet` 必须唯一绑定一个 `basis_id` 和一个 `n_ele`。
- 系数必须是复数。

## 3) 基底绑定与排序（MUST）
- 行列式排序继承 `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`。
- 跨文件/跨模块 `basis_id` 不一致必须立即失败。
- 任意基底变换都必须保留源/目标 `basis_id` 的可追溯性。

Validation:
- 做线性代数前，`basis_id`、`n_orb`、`n_ele`、det 顺序标签必须一致。

## 4) StateSet 列向量约定（MUST）
- 每个态是矩阵的一列。
- 形状规则：
  - `V_fock.shape = (dim_fock, n_states)`。
- 能量数组与 label 必须按列索引对齐。

Math:
$$
\Psi = V_{\mathrm{fock}},\qquad
\Psi^\dagger \Psi = I\ \text{（正交归一态集）}.
$$

Validation:
- 列范数与重叠检查必须可重复、且容差固定。

## 5) 归一化与相位规范（MUST）
- 导出的每个 `StateVec` 必须归一化。
- 全局相位必须按确定性规则固定。

Math:
$$
\sum_{\alpha} \lvert c_\alpha \rvert^2 = 1.
$$

相位规则：
1. 取主元索引 $\alpha_\star = \arg\max_\alpha |c_\alpha|$。
2. 若并列，取最小索引。
3. 乘上整体相位，使 $c_{\alpha_\star}\in\mathbb{R}_{\ge 0}$。

Validation:
- 相同输入必须得到相同的相位固定结果。

## 6) 截断与规范化（MUST）
- 若采用稀疏截断，规则必须显式且确定。
- 标准规则：
  - 删除 `abs(c_alpha) < eps_drop` 的分量；
  - 重新归一化；
  - 再执行第 5 节相位固定。

Validation:
- 规范化过程必须幂等：重复执行两次结果相同。

## 7) 简并子空间规则（MUST）
- 若出现简并子空间，必须固定其基矢取向。
- 先在该子空间对角化一个已声明的 tie-break 算符。
- 若仍简并，再按字典序主元规则逐列固定。

Validation:
- 相同输入重复运行，必须得到同样的子空间基矢顺序和相位。

## 8) 序列化契约（MUST）
- 必需元数据：`schema`, `basis_id`, `n_orb`, `n_ele`, `state_order_id`, `unit`, `labels`, `meta`。
- `state_order_id` 必须显式且稳定。

Code form:
```text
StateSet NPZ:
  V_fock, labels, energies(optional), basis_id, n_orb, n_ele, state_order_id, meta
```

Validation:
- 缺少必需元数据时必须拒绝加载。

## 9) 与算符接口对齐（MUST）
- 当 `StateSet` 作为算符输入/输出时，其基底元数据必须与算符两端元数据严格一致。
- 对于守粒子数算符，输入和输出 `basis_id` 相同。
- 对于改变量子数算符，输入和输出必须使用不同扇区的 `basis_id`，并各自与本端 `n_ele` 一致。

Code form:
```text
apply_operator(operator, stateset_in) -> stateset_out
require:
  stateset_in.basis_id == operator.basis_id_from
  stateset_out.basis_id == operator.basis_id_to
  stateset_in.n_ele     == operator.sector_from
  stateset_out.n_ele    == operator.sector_to
```
