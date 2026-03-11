# 02-06-NON_KRAMERS_DOUBLET

本文件定义偶电子 `f^n` 情形下的 non-Kramers 双重态契约。
奇电子 `n` 的 Kramers 双重态规则仍由
`./standards/en/02-models/02-05-KRAMERS_DOUBLET_G_TENSOR.md` 定义。
Wannier90 来源输入约束由 `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md` 定义。

## 1) 适用范围（MUST）
MUST:
- 仅适用于偶电子扇区（`n % 2 == 0`）。
- 目标是 CEF 低能双重态（严格简并或近简并）。
- 本文件定义投影/规范化/输出规则，不定义 CEF 模型公式本身。

Code form:
```text
require n % 2 == 0
inputs = {H_local, Jx, Jy, Jz, multipole_ops, symmetry_info}
```

Validation:
- 若 `n` 为奇数，应使用 `./standards/en/02-models/02-05-KRAMERS_DOUBLET_G_TENSOR.md`。

## 2) 双重态选择（MUST）
MUST:
- 在目标流形中选取最低两个态，并定义劈裂：

Math:
$$
\Delta_{\mathrm{nk}} \equiv E_2 - E_1.
$$

- 用阈值 `eps_nk_split` 标记是否近简并。
  `eps_nk_split` 默认值继承自
  `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`。

Code form:
```text
is_quasi_doublet = (Delta_nk <= eps_nk_split)
```

Validation:
- `Delta_nk` 与 `eps_nk_split` 必须写入 metadata。

## 3) 投影到赝自旋空间（MUST）
MUST:
- 定义投影算符：

Math:
$$
P=\lvert \psi_1\rangle\langle\psi_1\rvert+\lvert \psi_2\rangle\langle\psi_2\rvert.
$$

- 对任意算符 `O`，采用：

Math:
$$
M_O \equiv P O P
= o_0 I + o_x \tau^x + o_y \tau^y + o_z \tau^z.
$$

Code form:
```text
M_O = Psi.conj().T @ O @ Psi
coeffs {o0,ox,oy,oz} from Pauli decomposition
```

Validation:
- 所有投影矩阵 `M_O` 必须在 `eps_herm` 内厄米。

## 4) 磁通道与电通道规则（MUST）
MUST:
- 明确纵向轴（`c`）与面内轴（`a,b`）。
- 磁通道由偶极算符投影（`Jx/Jy/Jz`）定义。
- 电通道由 TR-even 多极算符投影定义
  （由 `multipole_ops` 输入集合提供：四极/八极等）。

对常见情形“`c` 方向磁、`ab` 平面电”，约束为：

Math:
$$
\|P J_c P\|_F \gg \|P J_a P\|_F,\ \|P J_b P\|_F.
$$

Math:
$$
\mathrm{rank}\left(\{P Q_m P\}_{Q_m\in\mathcal Q_{ab}}\right)\ge 2.
$$

Code form:
```text
mag_ratio_a = ||PJaP||_F / max(||PJcP||_F, eps_zero)
mag_ratio_b = ||PJbP||_F / max(||PJcP||_F, eps_zero)
require mag_ratio_a <= eps_mag_ab and mag_ratio_b <= eps_mag_ab
require two independent in-plane electric channels
```

`eps_mag_ab` 默认值继承自
`./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`。

Validation:
- 若磁/电通道条件不满足，必须标记为模型标签不匹配。

## 5) 确定性赝自旋规范（MUST）
MUST:
- 用确定性流程固定规范，保证输出稳定：
1. 由归一化后的 `P J_c P` 定义 `tau^z`。
2. 由两条独立面内电通道的投影，按固定顺序正交化得到
   `tau^x,tau^y`。
3. 用固定 tie-break 规则消除剩余符号/顺序歧义
   （先 `z`，再 `x`，再 `y`）。

Code form:
```text
tau_z <- normalize(PJcP)
tau_x, tau_y <- orthonormalize(PQ1P, PQ2P) with fixed order
apply deterministic sign convention
```

Validation:
- 相同输入必须得到一致的赝自旋轴与符号约定。

## 6) 输出契约（MUST）
MUST 输出：
- `doublet_vectors`（`psi1`, `psi2`）。
- `Delta_nk`。
- 偶极投影矩阵：`M_Jx`, `M_Jy`, `M_Jz`。
- 电通道投影矩阵：`M_Q1`, `M_Q2`（及其标签）。
- 偶极/电通道到赝自旋的映射系数。
- `gauge_meta` 与所用阈值。
- 启用 irrep 分类时输出 `symmetry_meta`（见 `02-07`）。
  - 契约字段：`irrep_display`、`irrep_primary`、`irrep_aliases`、
    `mapping_unverified`、`allowed_multipoles`、`excited_irreps`。
  - 对含反演点群，宇称由 `J` 唯一定义（不允许宇称双分支）。

Code form:
```text
outputs = {
  doublet_vectors, Delta_nk,
  M_Jx, M_Jy, M_Jz,
  M_Q1, M_Q2, Q_labels,
  map_dipole, map_electric,
  gauge_meta, threshold_meta,
  symmetry_meta?
}
```

Validation:
- 输出 metadata 必须包含 `basis_id` 与 `orbital_order_id`。
