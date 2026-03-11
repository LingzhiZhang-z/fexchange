# 00-02-RUNTIME_NUMERICS_AND_INPUT_GATES

本文件定义运行时数值默认值与全局输入闸门规则。
本文件对后续规范与实现具有强约束效力。

## 1) 适用范围（MUST）
MUST:
- 本文件定义数值阈值、确定性线性代数行为与全局输入绑定闸门。
- 物理公式仍由 `02-00/03-00/04-00` 层定义。

Code form:
```text
apply_to_modules = {02-00,02-01,02-02,02-03,02-04,02-05,02-06,03-00,03-01,04-00,04-01,04-02,04-03,05-00,05-02,05-03}
```

Validation:
- 任何覆盖默认值的实现必须在运行时元数据中显式记录。

## 2) 浮点与 dtype 策略（MUST）
MUST:
- 默认实数 dtype：`float64`。
- 默认复数 dtype：`complex128`。
- 同一收缩链中禁止隐式混合 dtype（除非显式转换）。

Code form:
```text
dtype_real_default    = float64
dtype_complex_default = complex128
```

Validation:
- 导出元数据必须记录实际 dtype。

## 3) 全局阈值表（MUST）
MUST:
- 默认使用统一阈值表；若某模块更严格，必须显式声明。

Math:
$$
\varepsilon_{\mathrm{zero}}=10^{-12},\quad
\varepsilon_{\mathrm{norm}}=10^{-10},\quad
\varepsilon_{\mathrm{orth}}=10^{-10},\quad
\varepsilon_{\mathrm{diag}}=10^{-9},
$$

Math:
$$
\varepsilon_{\mathrm{eig\_cluster}}=10^{-10},\quad
\varepsilon_{\mathrm{svd\_rel}}=10^{-12},\quad
\varepsilon_{\mathrm{map}}=10^{-8},\quad
\varepsilon_{\mathrm{herm}}=10^{-10}.
$$

Math:
$$
\varepsilon_{\mathrm{unitary}}=10^{-10},\quad
\varepsilon_{\mathrm{nk\_split}}=10^{-6},\quad
\varepsilon_{\mathrm{mag\_ab}}=10^{-1}.
$$

Code form:
```text
eps_zero        = 1e-12
eps_norm        = 1e-10
eps_orth        = 1e-10
eps_diag        = 1e-9
eps_eig_cluster = 1e-10
eps_svd_rel     = 1e-12
eps_map         = 1e-8
eps_herm        = 1e-10
eps_unitary     = 1e-10
eps_nk_split    = 1e-6
eps_mag_ab      = 1e-1
```

Index:
- `eps_diag`：`03-00/03-01` 子空间非对角泄漏检查。
- `eps_map`：`04-03` 模块重构残差检查。
- `eps_svd_rel`：null-space 奇异值相对截断阈值。
- `eps_unitary`：显式基变换（`05-02`）的幺正性检查阈值。
- `eps_nk_split`：non-Kramers 准双重态劈裂阈值（`02-06`），单位为内部能量单位。
- `eps_mag_ab`：non-Kramers 面内磁泄漏比阈值（`02-06`）。

Validation:
- 运行时元数据必须记录所有激活的 `eps_*`。

## 4) 确定性本征/SVD 规则（MUST）
MUST:
- 厄米对角化必须使用 Hermitian 求解器（`eigh` 类）。
- 本征对按本征值升序排序。
- 简并簇使用 `eps_eig_cluster` 判定。
- 简并簇内规范固定必须使用 `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md` 的 pivot-phase 规则。

Math:
$$
|\lambda_i-\lambda_j|\le \varepsilon_{\mathrm{eig\_cluster}}
\Rightarrow i,j\text{ 属于同一简并簇}.
$$

Math:
$$
\sigma_r \le \varepsilon_{\mathrm{svd\_rel}}\,\sigma_{\max}
\Rightarrow \sigma_r\text{ 归入零空间}.
$$

Code form:
```text
evals, evecs = eigh(H)
sort evals ascending
cluster by |eval_i - eval_j| <= eps_eig_cluster
fix cluster gauge deterministically

U,S,Vh = svd(A, full_matrices=False)
null_mask = (S <= eps_svd_rel * S.max())
```

Validation:
- 同一输入重复运行必须得到相同列顺序与相位规范化结果。

## 5) 相位与 tie-break 规则（MUST）
MUST:
- 态相位固定遵循 `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md`。
- 并列 pivot 选最小行列式索引。
- 机器精度下仍并列时，按复系数序列字典序最小规则。

Code form:
```text
pivot = argmax(abs(v))
if tie: choose smallest index
phase-fix so v[pivot] is real and >= 0
```

Validation:
- 规范化必须幂等。

## 6) 厄米性与正交性检查（MUST）
MUST:
- 厄米性检查使用归一化 Frobenius 残差。
- 正交性检查使用 `||V^dag V - I||_F`。

Math:
$$
r_{\mathrm{herm}} = \frac{\|H-H^\dagger\|_F}{\max(\|H\|_F,\varepsilon_{\mathrm{zero}})}
\le \varepsilon_{\mathrm{herm}}.
$$

Math:
$$
r_{\mathrm{orth}} = \|V^\dagger V-I\|_F \le \varepsilon_{\mathrm{orth}}.
$$

Code form:
```text
r_herm = norm(H - H.conj().T, 'fro') / max(norm(H,'fro'), eps_zero)
r_orth = norm(V.conj().T @ V - I, 'fro')
```

Validation:
- 超阈值必须硬失败。

## 7) 运行路径默认值（MUST）
MUST:
- 默认 dense 路径。
- 仅在估计密度低于阈值时切换 sparse COO。
- 大型收缩必须支持分块，控制峰值内存。

Code form:
```text
density_sparse_switch = 0.15
peak_memory_budget_gb = 8.0
chunk_policy = "auto-by-memory"
```

Validation:
- 元数据必须记录路径与分块策略。

## 8) 全局输入头闸门（MUST）
MUST:
- 每个外部运行时载荷必须包含：
  `schema_version`, `standard_version`, `basis_id`, `orbital_order_id`, `unit`。

Code form:
```text
required_header = {schema_version, standard_version, basis_id, orbital_order_id, unit}
```

Validation:
- 缺失头字段属于致命输入错误。

## 9) 全局通道/绑定闸门（MUST）
MUST:
- 单次运行只计算一个 bond。
- 运行时 hopping 输入是单个矩阵（输入载荷中不包含 `mu` 轴）。
- 公式中的 `\mu` 是本次运行的 bond 标签，不是数组轴。
- 收缩前必须完成跨接口绑定检查。

Code form:
```text
require t.shape == (n_orb, n_orb)
optional bond_label: string

require input.basis_id == core.basis_id
require input.orbital_order_id == core.orbital_order_id
```

Validation:
- 非法额外通道轴或绑定不一致必须硬失败。

## 10) 错误策略（MUST）
MUST:
- 输入 schema/绑定失败必须硬失败。
- 禁止静默回退到输出快照。

Code form:
```text
if schema_check_fail:  raise InputSchemaError
if binding_check_fail: raise InputBindingError
```

Validation:
- 错误信息必须包含字段名与期望/实际 shape、dtype、取值域。

## 11) 运行时元数据契约（MUST）
MUST:
- 每个阶段输出必须包含数值配置快照。

Code form:
```text
numerics_meta = {
  dtype_real, dtype_complex,
  eps_zero, eps_norm, eps_orth, eps_diag,
  eps_eig_cluster, eps_svd_rel, eps_map, eps_herm,
  eps_unitary, eps_nk_split, eps_mag_ab,
  density_sparse_switch, peak_memory_budget_gb
}
```

Validation:
- 缺少 `numerics_meta` 视为契约违反。
