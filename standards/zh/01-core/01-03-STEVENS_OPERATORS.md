# 01-03-STEVENS_OPERATORS

本文件定义 \(|J,M\rangle\) 基中的通用 Stevens 算符契约。
它将 `02-hamiltonian/02-03-HCEF.md` 第 6 节从 CEF 子集扩展到本项目使用的全部秩/分量。

## 1) 范围（MUST）
MUST：
- 支持秩 `0 <= k <= 6`。
- 支持分量 `-k <= q <= k`。
- 使用基序 `M = -J, ..., J`（升序）。
- 输出 `(2J+1) x (2J+1)` 复矩阵。

## 2) 约定（MUST）
MUST：
- 使用 Hutchings 风格 Stevens 算符与 Condon-Shortley 相位。
- 运行时默认提供实 tesseral 形式（`cos`/`sin`），并提供复分量形式用于转换。

运行模式：
- `mode='cos'`：余弦型 tesseral 分量。
- `mode='sin'`：正弦型 tesseral 分量。
- `mode='complex'`：复球张量分量。

## 3) 一阶算符契约（MUST）
必须满足：

Math:
$$
O_1^0 = J_z,\qquad
O_1^{1,c} = J_x,\qquad
O_1^{1,s} = J_y.
$$

## 4) CEF 兼容子集（MUST）
以下子集必须与 `02-03` 第 6 节定义完全一致：
- `O20`, `O40`, `O60`
- `O44c`, `O64c`
- `O43c`, `O43s`, `O63c`, `O63s`, `O66`

该子集由 `build_cef_stevens_operators(...)` 使用，以保证 CEF 行为向后兼容。

## 5) 广义构造（MUST）
对全部 `(k,q)`（`k <= 6`），实现必须给出有效矩阵，可通过：
- 已定义的 Hutchings 多项式闭式；或
- 基于 `01-04` 与 `01-05` 的等价张量转换。

当 `q > 0`，tesseral 重构为：

Math:
$$
O_k^{q,c} = \frac{1}{\sqrt2}\left[(-1)^q O_k^q + O_k^{-q}\right],
\qquad
O_k^{q,s} = \frac{1}{i\sqrt2}\left[(-1)^q O_k^q - O_k^{-q}\right].
$$

## 6) API 契约（MUST）

Code form:
```text
build_stevens_operator(J, k, q, *, mode='cos') -> NDArray
build_stevens_set(J, k, *, modes='tesseral'|'complex') -> dict[str, NDArray]
build_cef_stevens_operators(J, symmetry='Oh'|'C3v', mode_q3='cos'|'sin') -> dict[str, NDArray]
```

规则：
- `q=0` 时忽略 `mode`。
- 非法 `k`、`q` 或 mode 必须快速失败。
- 返回矩阵必须确定性。

## 7) 验证（MUST）
必须检查：
- tesseral 算符厄米性。
- 被闭式定义约束的 `q=0` 算符对角性。
- CEF 子集与旧实现逐项一致。
