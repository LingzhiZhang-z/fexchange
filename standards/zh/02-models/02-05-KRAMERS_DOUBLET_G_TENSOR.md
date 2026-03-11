# 02-05-KRAMERS_DOUBLET_G_TENSOR

本文件定义 Kramers 双重态构造契约，以及投影到泡利矩阵/g-tensor 的契约。
`H_int/H_soc/H_cef` 的模型形式仍由 `02-00/02-01/02-02/02-03` 定义。
Wannier90 来源输入约束由 `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md` 定义。

## 1) 适用范围与输入（MUST）
MUST:
- 本文件适用于需要 Kramers 双重态的奇电子 `f^n` 情形。
- 输入哈密顿量必须来自 `02-00/02-01/02-02/02-03` 的局域模型。
- `Jx/Jy/Jz` 必须遵循 `./standards/en/02-models/02-04-ANGULAR_MOMENTUM_OPERATORS.md`。
- 默认目标子空间是 SOC 低能 LSJM 流形上，经 CEF 劈裂后的最低双重态。

Code form:
```text
inputs = {H_local, Jx, Jy, Jz, n, symmetry_branch, cef_params}
require n % 2 == 1
```

Validation:
- 若 `n` 为偶数，本契约不适用。

## 2) Kramers 配对条件（MUST）
MUST:
- 被选双重态必须满足 Kramers 简并与时间反演配对。
- 设 `Theta` 为反幺正时间反演算符：

Math:
$$
\Theta \lvert k_1\rangle = \lvert k_2\rangle,\qquad
\Theta \lvert k_2\rangle = -\lvert k_1\rangle.
$$

Code form:
```text
check |E_k1 - E_k2| <= eps_eig_cluster
check TR_pair_residual <= eps_norm
```

Validation:
- 若简并或 TR 配对失败，必须硬失败。

### 2.1) 时间反演算符实现（MUST）
f 壳层单粒子基上的反幺正时间反演算符为
$\Theta=U_T K$，其中 $K$ 为复共轭操作。

对单自旋轨道 $\lvert m,\sigma\rangle$（$\ell=3$）：

Math:
$$
\Theta\lvert m,\sigma\rangle
=(-1)^{\ell-m+\frac{1}{2}-\sigma}\lvert{-m},{-\sigma}\rangle.
$$

$U_T$ 是 $14\times14$ 的单项式矩阵（带相位的置换矩阵）。
它将轨道 $p(m,\sigma)$ 映射到轨道 $\bar p(-m,-\sigma)$，
因此对 $m\neq0$ 时映射**跨越**不同 $m$ 块；只有 $m=0$ 块映射到自身。

Math:
$$
(U_T)_{\bar p,\,p}
=(-1)^{\ell-m_p+\frac{1}{2}-\sigma_p},
\qquad
\bar p = p(-m_p,-\sigma_p),
$$

其余元素为零。

在本项目轨道序下（$p=0\ldots13$，$m=-3\ldots3$，
每个 $m$ 内 $\sigma=-\tfrac{1}{2},+\tfrac{1}{2}$）：

Code form:
```text
# 构建 14x14 单粒子 U_T
U_T = np.zeros((14,14), dtype=complex)
for p in range(14):
    m_p, sigma_p = orbital_map(p)            # 解码 (m, sigma)
    p_bar = orbital_index(-m_p, -sigma_p)    # 目标轨道
    phase = (-1)**(3 - m_p + 0.5 - sigma_p)
    U_T[p_bar, p] = phase

# 多体作用于态矢量 psi（n 电子 Fock 扇区）
Theta_psi = U_T_n @ psi.conj()
```

对多体 $n$ 电子 Fock 扇区：
1. 对每个 Slater 行列式，将 $U_T$ 施加于每个占据轨道，
   收集单粒子相位，并计算时间反演后轨道集重排到正则位序的符号。
2. 对态矢量 $\lvert\psi\rangle=\sum_\alpha c_\alpha\lvert\alpha\rangle$：

Math:
$$
\Theta\lvert\psi\rangle
= \sum_\alpha c_\alpha^{\ast}\bigl(U_T^{(n)}\lvert\alpha\rangle\bigr),
$$

其中 $U_T^{(n)}$ 是 $n$ 电子扇区的多体时间反演酉矩阵。

Validation:
- $\Theta^2=(-1)^n$（$\ell=3$ 下单粒子 $\Theta^2=-1$）。
- 对奇数 $n$：Kramers 定理保证至少二重简并。
- TR 配对残差：$\|\Theta\lvert k_1\rangle-\lvert k_2\rangle\|\le\varepsilon_{\mathrm{norm}}$。

## 3) 泡利投影契约（MUST）
MUST:
- 在双重态上定义投影算符：

Math:
$$
P=\lvert k_1\rangle\langle k_1\rvert+\lvert k_2\rangle\langle k_2\rvert.
$$

- 投影角动量算符：

Math:
$$
M_\alpha \equiv P J_\alpha P,\qquad \alpha\in\{x,y,z\}.
$$

- 在双重态空间中采用泡利展开：

Math:
$$
M_\alpha = \frac{1}{2}\sum_{\beta\in\{x,y,z\}}\Lambda_{\alpha\beta}\sigma_\beta.
$$

- 用一个显式常数给出 g-tensor 关系：

Math:
$$
g_{\alpha\beta}=c_g\,\Lambda_{\alpha\beta}.
$$

Code form:
```text
M_alpha = K.conj().T @ J_alpha @ K
Lambda[alpha,beta] from Pauli decomposition of M_alpha
g_tensor = c_g * Lambda
```

Validation:
- `c_g` 必须写入 metadata。
- `M_alpha` 必须在 `eps_herm` 阈值内厄米。

## 4) SU(2) 规范自由与不变量（MUST）
MUST:
- 双重态子空间内的基变换必须是 SU(2)：

Math:
$$
\lvert k_a'\rangle=\sum_b \lvert k_b\rangle U_{ba},\qquad U\in SU(2).
$$

Math:
$$
M_\alpha' = U^\dagger M_\alpha U,\qquad
\Lambda'=\Lambda R(U),\ R(U)\in SO(3).
$$

- 跨运行对比必须使用规范不变量（例如 `g_tensor` 的奇异值，或 `g_tensor g_tensor^T` 的本征值）。

Validation:
- 禁止直接用未规范化的列相位/列顺序做物理对比。

## 5) 确定性规范固定（MUST）
MUST:
- 选定双重态后，按以下顺序固定规范：
1. 对角化 `M_Jz`。
2. 用残余 U(1) 相位使 `M_Jx` 的非对角元为实数。
3. 强制 TR 配对约定
   （`Theta|k1>=|k2>`，`Theta|k2>=-|k1>`）。
4. 执行确定性符号统一，使 `g` 分量符号尽可能一致。

Code form:
```text
step1: U_z = eigh(M_Jz).evecs
step2: U_phase from phase(M_Jx[0,1])
step3: U_tr enforce TR pair convention
step4: choose among allowed SU(2) discrete transforms with deterministic tie-break
```

符号统一规则（MUST）：
- 目标：最大化 `(g_x, g_y, g_z)` 的符号一致性。
- 若多个候选并列，按固定次序打破并列：
  `g_z >= 0`，再 `g_y >= 0`，再 `g_x >= 0`。

Validation:
- 相同输入重复运行，必须得到相同的 `(k1,k2)` 顺序与 `g` 符号约定。

## 6) 输出契约（MUST）
MUST 输出：
- `kramer_vectors`：规范化后的两列 `[k1, k2]`。
- `M_Jx`, `M_Jy`, `M_Jz`：投影后的 `2x2` 矩阵。
- `Lambda` 与 `g_tensor`。
- 若使用轴向表示，输出 `g_components`。
- `gauge_meta`：`{unitary_total, tr_residual, pauli_residual, sign_rule}`。
- 启用 irrep 分类时输出 `symmetry_meta`（见 `02-07`）。
  - 契约字段：`irrep_display`、`irrep_primary`、`irrep_aliases`、
    `mapping_unverified`、`allowed_multipoles`、`excited_irreps`。
  - 对含反演点群，宇称由 `J` 唯一定义（不允许宇称双分支）。

Code form:
```text
outputs = {
  kramer_vectors, M_Jx, M_Jy, M_Jz, Lambda, g_tensor, gauge_meta,
  symmetry_meta?
}
```

Validation:
- 所有输出必须共享同一 `basis_id` 与 `orbital_order_id`。

## 7) 运行时检查（MUST）
MUST:
- 投影矩阵厄米性检查。
- TR 配对残差检查。
- 泡利映射残差检查：

Math:
$$
r_{\mathrm{pauli}}=
\max_{\alpha}
\frac{
\left\|M_\alpha-\frac{1}{2}\sum_\beta\Lambda_{\alpha\beta}\sigma_\beta\right\|_F
}{
\max\left(\|M_\alpha\|_F,\varepsilon_{\mathrm{zero}}\right)
}.
$$

Code form:
```text
require r_pauli <= eps_norm
```

Validation:
- 任一检查失败都必须硬失败。
