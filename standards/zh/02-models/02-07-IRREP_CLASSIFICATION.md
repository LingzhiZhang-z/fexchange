# 02-07-IRREP_CLASSIFICATION

本标准定义 CEF 态不可约表示分类与多极子兼容性报告的 API/数据契约。

相关标准：
- CEF 构建：`02-03-HCEF_FORM`
- 角动量算符：`02-04-ANGULAR_MOMENTUM_OPERATORS`
- Kramers/non-Kramers 投影输出：`02-05`、`02-06`

参考约定：
- Koster/Dresselhaus 风格双群约定。

## 1) 适用范围、输入与分支（MUST）

### 1.1 适用范围

MUST：
- 面向 `|J,M>` 基底下的单格点 CEF 本征向量。
- 支持点群 `Oh`、`D3d`、`C3v`。
- 在统一投影流程中同时支持整数与半整数 `J`。
- 禁止在投影前按“仅整数 irrep”或“仅半整数 irrep”做预筛选。

### 1.2 输入

```text
inputs = {J, evecs, point_group, n_f?}
require evecs.shape == (int(2*J + 1), n_states)
require point_group in {"Oh", "D3d", "C3v"}
```

含反演点群的宇称因子：

```text
p = (-1)**(2J)
```

给定 `n_f` 时的可选一致性检查：

```text
require (-1)**n_f == (-1)**(2J)
```

### 1.3 宇称标记策略

MUST：
- 对 `Oh`、`D3d`：仅返回一条已定宇称分支。
- 对 `C3v`：返回单一无宇称分支（不带 `g/u`）。
- 本标准禁止返回宇称双分支。

## 2) Irrep 命名契约（MUST）

每个分类态都必须同时给出三层命名：
- `irrep_display`：可读标签（`Γ...` 形式）。
- `irrep_primary`：实现主键（`Gamma...` 形式）。
- `irrep_aliases`：可选已验证别名（如 `A1g`、`T2u`）。

还必须给出：
- `mapping_unverified: bool`

规则：
- 若别名映射未被严格验证，不得强行输出别名。
- 此时必须设置 `irrep_aliases = []` 且 `mapping_unverified = true`。

## 3) 静态群数据契约（MUST）

### 3.1 O* 旋转核（阶 48）

类键与类大小：

```text
E, R, 3C2, 3RC2, C4, RC4, 6C2p, 6RC2p, C3, RC3
1, 1,   3,    3,   6,   6,    6,     6,   8,   8
```

### 3.2 C3v* 旋转核（阶 12）

类键与类大小：

```text
E, R, 2C3, 2RC3, 3sigma_v, 3Rsigma_v
1, 1,   2,    2,        3,         3
```

### 3.3 特征标表存储要求

MUST：
- 特征标表必须为静态数据。
- 每张表必须和类大小一起存储。
- 必须支持复数特征标。
- 本标准使用附录 A/B 的数值作为规范值。

每条 irrep 必须满足：
- O*：`sum_c n_c |chi(c)|^2 = 48`
- C3v*：`sum_c n_c |chi(c)|^2 = 12`

### 3.4 反演扩展

MUST：
- `Oh*` 由 `O* x C_i` 构造。
- `D3d*` 由 `C3v* x C_i` 构造。

对含反演点群：

```text
chi(i*g) = p * chi(g),  p in {+1, -1}
```

`Oh`/`D3d` 的宇称由 `J` 唯一确定：

```text
p = (-1)**(2J)
```

若给定 `n_f`，它仅用于一致性检查，且必须满足
`(-1)^n_f == (-1)^(2J)`。

## 4) 表示矩阵（MUST）

在 `|J,M>` 基底用 Wigner-D 构造代表矩阵：

$$
D^J(\alpha,\beta,\gamma)
= e^{-i\alpha J_z} e^{-i\beta J_y} e^{-i\gamma J_z}.
$$

MUST：
- 对活跃类构造代表元矩阵。
- 满足 `D(E) = I` 与幺正性。

双群中心旋转元：

```text
D(R) = (-1)**(2J) * I
```

该结果必须来自同一矩阵构造路径（禁止手写覆盖分支）。

对 `Oh`、`D3d`：

```text
D(i) = p * I
D(i*g) = D(i) @ D(g)
```

其中 `p = (-1)**(2J)`（并可做上述 `n_f` 一致性检查）。

对 `C3v`：不使用反演类。

## 5) 投影算符分类（MUST）

使用投影算符：

$$
P_\Gamma = \frac{d_\Gamma}{|G|}
\sum_c \chi_\Gamma(c)^*\,S_c,
\quad S_c = \sum_{g \in c} D^J(g).
$$

其中 $S_c$ 为类求和算符（对共轭类 $c$ 内所有群元素的 $D^J(g)$ 求和）。
等价于 $\sum_{g \in G} \chi_\Gamma(g)^* D^J(g) \cdot d_\Gamma / |G|$。

禁止使用 $n_c \cdot D^J(g_c)$（单一类代表元乘以类大小）；
表示矩阵不是类函数，该形式无法给出幂等投影算符。

分类规则：

```text
for each state psi:
    score(Gamma) = ||P_Gamma @ psi||
    label = argmax score
    if max(score) < eps_proj:
        label = "unknown"
```

MUST：
- 对活跃表中的全部 irrep 做评估。
- 禁止在投影前手工删减候选 irrep 集。

## 6) 多极子兼容性契约（MUST）

提供静态查表：

```text
allowed_multipoles(irrep, point_group)
```

未知 irrep/点群必须抛出 `KeyError`。

`allowed_multipoles` 返回值的令牌约定：
- `magnetic_dipole`（磁偶极）
- `electric_quadrupole`（电四极）
- `magnetic_octupole`（磁八极）

### 6.1 Oh 单值扇区（`R = +1`）

| 主键 | 显示名 | 别名（按宇称展开） | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|------|--------|--------------------|-----------------|---------------------|-------------------|
| `Gamma1` | `Γ1` | `A1g` / `A1u` | 否 | 否 | 是 |
| `Gamma2` | `Γ2` | `A2g` / `A2u` | 否 | 否 | 是 |
| `Gamma3` | `Γ3` | `Eg` / `Eu` | 否 | 是 | 是 |
| `Gamma4` | `Γ4` | `T1g` / `T1u` | 是 | 否 | 是 |
| `Gamma5` | `Γ5` | `T2g` / `T2u` | 否 | 是 | 是 |

**单值 / 双值分隔线**

### 6.2 Oh 双值扇区（`R = -1`）

| 主键 | 显示名 | 别名 | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|------|--------|------|-----------------|---------------------|-------------------|
| `Gamma6` | `Γ6` | 无统一别名 | 是 | 否 | 是 |
| `Gamma7` | `Γ7` | 无统一别名 | 是 | 否 | 是 |
| `Gamma8` | `Γ8` | 无统一别名 | 是 | 是 | 是 |

### 6.3 D3d 单值扇区（`R = +1`）

| 主键 | 显示名 | 别名 | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|------|--------|------|-----------------|---------------------|-------------------|
| `Gamma1+` | `Γ1+` | `A1g` | 否 | 否 | 否 |
| `Gamma2+` | `Γ2+` | `A2g` | 是（`z`） | 否 | 是 |
| `Gamma3+` | `Γ3+` | `Eg` | 是（`x,y`） | 是 | 是 |
| `Gamma1-` | `Γ1-` | `A1u` | 否 | 否 | 否 |
| `Gamma2-` | `Γ2-` | `A2u` | 否 | 否 | 是 |
| `Gamma3-` | `Γ3-` | `Eu` | 否 | 是 | 是 |

**单值 / 双值分隔线**

### 6.4 D3d/C3v 旋量扇区（`R = -1`）

| 家族 | 显示名 | 别名 | magnetic_dipole | electric_quadrupole | magnetic_octupole |
|------|--------|------|-----------------|---------------------|-------------------|
| `Gamma4` 家族 | `Γ4` | 无统一别名 | 是（`z`） | 否 | 是 |
| `Gamma5` 家族 | `Γ5` | 无统一别名 | 是（`x,y`） | 是 | 是 |
| `Gamma6` 家族 | `Γ6` | 无统一别名 | 是（`x,y`） | 是 | 是 |

规则：
- 对 `D3d`，按 `p = (-1)**(2J)` 使用 `+/-` 宇称标记。
- 若给定 `n_f`，必须与 `J` 的宇称判定一致。
- 对 `C3v`，采用同一旋量通道规则但不带宇称标记。

## 7) 输出结构与使用模式（MUST）

### 7.1 Pipeline 模式

当 `load_or_build_W` 可提供本征向量时，附加 symmetry metadata。

单分支结构：

```text
{
  "irrep_display": str,
  "irrep_primary": str,
  "irrep_aliases": list[str],
  "mapping_unverified": bool,
  "allowed_multipoles": list[str],
  "excited_irreps": [
    {
      "energy_index": int,
      "irrep_display": str,
      "irrep_primary": str,
      "irrep_aliases": list[str],
      "mapping_unverified": bool
    }, ...
  ]
}
```

### 7.2 独立分析模式

```text
analyze_cef_symmetry(J, point_group, *, B_params=None, evecs=None, n_f=None)
```

MUST：
- `B_params` 与 `evecs` 必须二选一。
- 分支语义与 pipeline 模式一致。
- 对 `C3v`，返回单一无宇称分支（不带反演宇称标签）。

## 8) 运行时校验与失败策略（MUST）

必须进行数值校验：
- 投影算符幂等性
- 投影算符完备性
- 特征标表正交性
- 参考约定下的分解回归

```text
eps_proj = 1e-6
assert projector_checks_pass
assert table_orthogonality_pass
```

失败策略：
- 未知查表键 -> `KeyError`。
- 数值校验失败 -> 硬失败并返回残差诊断。
- 仅当全部投影范数低于阈值时才标记为 `"unknown"`。

## 附录 A) O* 特征标表（规范数据）

| 扇区 | 显示名 | 主键 | E | R | 3C2 | 3RC2 | C4 | RC4 | 6C2p | 6RC2p | C3 | RC3 |
|------|--------|------|---|---|-----|------|----|-----|------|-------|----|-----|
| 单值 | `Γ1` | `Gamma1` | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 单值 | `Γ2` | `Gamma2` | 1 | 1 | 1 | 1 | -1 | -1 | -1 | -1 | 1 | 1 |
| 单值 | `Γ3` | `Gamma3` | 2 | 2 | 2 | 2 | 0 | 0 | 0 | 0 | -1 | -1 |
| 单值 | `Γ4` | `Gamma4` | 3 | 3 | -1 | -1 | 1 | 1 | -1 | -1 | 0 | 0 |
| 单值 | `Γ5` | `Gamma5` | 3 | 3 | -1 | -1 | -1 | -1 | 1 | 1 | 0 | 0 |
| 双值 | `Γ6` | `Gamma6` | 2 | -2 | 0 | 0 | sqrt(2) | -sqrt(2) | 0 | 0 | 1 | -1 |
| 双值 | `Γ7` | `Gamma7` | 2 | -2 | 0 | 0 | -sqrt(2) | sqrt(2) | 0 | 0 | 1 | -1 |
| 双值 | `Γ8` | `Gamma8` | 4 | -4 | 0 | 0 | 0 | 0 | 0 | 0 | -1 | 1 |

## 附录 B) C3v* 特征标表（规范数据）

| 扇区 | 显示名 | 主键 | E | R | 2C3 | 2RC3 | 3sigma_v | 3Rsigma_v |
|------|--------|------|---|---|-----|------|----------|-----------|
| 单值 | `Γ1` | `Gamma1` | 1 | 1 | 1 | 1 | 1 | 1 |
| 单值 | `Γ2` | `Gamma2` | 1 | 1 | 1 | 1 | -1 | -1 |
| 单值 | `Γ3` | `Gamma3` | 2 | 2 | -1 | -1 | 0 | 0 |
| 双值 | `Γ4` | `Gamma4` | 2 | -2 | 1 | -1 | 0 | 0 |
| 双值 | `Γ5` | `Gamma5` | 1 | -1 | -1 | 1 | i | -i |
| 双值 | `Γ6` | `Gamma6` | 1 | -1 | -1 | 1 | -i | i |

## 附录 C) 别名映射（规范中已验证部分）

`Oh` 单值别名：
- `Gamma1 -> A1g`（`p=+1`），`Gamma1 -> A1u`（`p=-1`）
- `Gamma2 -> A2g`（`p=+1`），`Gamma2 -> A2u`（`p=-1`）
- `Gamma3 -> Eg`（`p=+1`），`Gamma3 -> Eu`（`p=-1`）
- `Gamma4 -> T1g`（`p=+1`），`Gamma4 -> T1u`（`p=-1`）
- `Gamma5 -> T2g`（`p=+1`），`Gamma5 -> T2u`（`p=-1`）

`D3d` 单值别名：
- `Gamma1+ -> A1g`，`Gamma2+ -> A2g`，`Gamma3+ -> Eg`
- `Gamma1- -> A1u`，`Gamma2- -> A2u`，`Gamma3- -> Eu`

旋量表示（立方核 `Gamma6/7/8`，三方核 `Gamma4/5/6`）：
- 本标准不强制统一 Schoenflies 别名。
