# 01-04-SPHERICAL_TENSOR_OPERATORS

本文件定义固定 \(J\) 流形中的不可约球张量算符 \(T_k^q\)。

## 1) 范围（MUST）
MUST：
- 在 \(|J,M\rangle\) 基（`M=-J,...,J` 升序）构造 \(T_k^q\)。
- 支持整数与半整数 `J`。
- 支持 `k >= 0` 且 `-k <= q <= k`。

## 2) 矩阵元定义（MUST）
使用 `sympy.physics.wigner.wigner_3j`，并采用约化矩阵元为 1 的约定：

Math:
$$
\langle J,M'|T_k^q|J,M\rangle
= (-1)^{J-M'}\,\begin{pmatrix}J & k & J\\-M' & q & M\end{pmatrix}.
$$

实现形式：
```text
T[M', M] = (-1)^(J-M') * float(wigner_3j(J, k, J, -M', q, M))
```

## 3) 选择定则（MUST）
必须满足：
- `q = M' - M`
- 固定流形三角条件：`k <= 2J`
- 违反选择定则的矩阵元数值上为零。

## 4) 代数性质（MUST）
必须满足（在容差内）：

Math:
$$
(T_k^q)^\dagger = (-1)^q T_k^{-q}.
$$

正交性（Frobenius 内积）：

Math:
$$
\mathrm{Tr}\left[(T_k^q)^\dagger T_{k'}^{q'}\right] \propto \delta_{kk'}\delta_{qq'}.
$$

## 5) 多极子分类（MUST）
按秩奇偶：
- 奇数 `k`（`1,3,5`）：磁多极子（时间反演奇）。
- 偶数 `k`（`2,4,6`）：电多极子（时间反演偶）。

API/元数据命名：
- `k=1`: `magnetic_dipole`
- `k=2`: `electric_quadrupole`
- `k=3`: `magnetic_octupole`

## 6) API 契约（MUST）

Code form:
```text
build_spherical_tensor(J, k, q) -> NDArray
build_multipole_set(J, k) -> dict[int, NDArray]
multipole_type(k) -> 'magnetic' | 'electric'
```

## 7) 验证（MUST）
必须在整数与半整数 `J` 上测试：
- 厄米共轭关系。
- 不同 `(k,q)` 通道正交性。
- 选择定则稀疏结构。
