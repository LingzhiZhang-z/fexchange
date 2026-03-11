# 02-01-HINT_FORM

本文件定义库仑相互作用项 $H_{\mathrm{int}}$。
参考链接仅在 `./standards/en/02-models/02-00-MODEL_LOCAL_HAMILTONIAN.md` 开头统一给出。

## 1) 规范主形式（与实现一致）

Math:
$$
H_{\mathrm{int}}
= \frac{1}{2}\sum_i
\sum_{m_1,m_2,m_3,m_4}
\sum_{\sigma_1,\sigma_2}
\delta_{m_1+m_2,\,m_3+m_4}
\sum_{k=0,2,4,6}
F^k\,C^{(k)}(m_1,m_4)\,C^{(k)}(m_3,m_2)
\,c^{\dagger}_{i m_1 \sigma_1}
\,c^{\dagger}_{i m_2 \sigma_2}
\,c_{i m_3 \sigma_2}
\,c_{i m_4 \sigma_1}.
$$

本项目规范（MUST）：
- 前因子 $\tfrac{1}{2}$ 为必需；
- 第二个 $C^{(k)}$ 因子使用 $(m_3,m_2)$。

与这两点不一致的文献写法，以本规范为准。

## 2) Slater 积分与 $U/J_H$（文献约定）

Math:
$$
U = F^0,
$$

Math:
$$
J_H = \frac{1}{6435}\left(286F^2 + 195F^4 + 250F^6\right).
$$

## 3) 说明
- $F^0, F^2, F^4, F^6$ 为 Slater-Condon 参数。
- $C^{(k)}$ 在该记号中为 Gaunt 系数相关因子。

## 4) $C^{(k)}$ 的定义（MUST）
固定 f 壳层 $l=3$：

Math:
$$
q = m_a - m_b,
\qquad
C^{(k)}(m_a,m_b)
=
(-1)^{m_a}
\sqrt{\frac{4\pi}{2k+1}}
\int d\Omega\,
Y_{l,-m_a}(\Omega)\,
Y_{k,q}(\Omega)\,
Y_{l,m_b}(\Omega).
$$

使用球谐三重积恒等式：

Math:
$$
\int d\Omega\,
Y_{l,-m_a}(\Omega)\,
Y_{k,q}(\Omega)\,
Y_{l,m_b}(\Omega)
=
\sqrt{\frac{(2l+1)(2k+1)(2l+1)}{4\pi}}
\begin{pmatrix} l & k & l \\ 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} l & k & l \\ -m_a & q & m_b \end{pmatrix}.
$$

将该恒等式代回上式，得到：

Math:
$$
C^{(k)}(m_a,m_b)
=
(-1)^{m_a}
(2l+1)
\begin{pmatrix} l & k & l \\ 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} l & k & l \\ -m_a & q & m_b \end{pmatrix}.
$$

说明：
- 本项目固定采用上述归一化与相位约定。
- 文献中可能存在相位/归一化不同的等价写法。

选择定则（MUST）：
- $q=m_a-m_b$。
- $|q|\le k$。
- 二体矩阵元非零时必须满足 $m_1+m_2=m_3+m_4$。

## 5) 实现契约（MUST）
实现顺序固定如下：
1. 对 $k=0,2,4,6$ 按上述 $C^{(k)}$ 定义生成 rank 分量系数。
2. 在轨道-自旋指标上构造未反对称化二体系数。
3. 对二体系数做反对称化。
4. 按 $H_{\mathrm{int}}=\sum_k F^k H^{(k)}$ 累加。
5. 输出时遵循 `01-02-OPERATOR_IMPLEMENTATION` 的四费米规范序
   （`i<j`, `k<l`）。

Code form:
```text
hint = sum_{k in {0,2,4,6}} F[k] * H_rank[k]
H_rank[k] <- antisymmetrized(Ck products with spin/orbital constraints)
```

## 6) Wigner 3j / CG 系数实现指引（MUST）
第 4 节中的 $C^{(k)}$ 系数需要 Wigner 3j 符号。
模块 03-01（LSJM 构造，验证路径）也使用 CG 系数。

MUST:
- 全程使用 Condon-Shortley 相位约定。
- 使用 `sympy.physics.wigner` 作为唯一的 3j/CG 实现。
  该库提供精确有理运算并保证相位约定一致。
- `scipy` 不含 3j/CG 功能；不应假定可用。

CG 与 3j 的关系（MUST 保持一致）：

Math:
$$
\langle j_1 m_1;\,j_2 m_2 \mid j_3 m_3\rangle
=
(-1)^{-j_1+j_2-m_3}\sqrt{2j_3+1}
\begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & -m_3 \end{pmatrix}.
$$

Code form:
```text
from sympy.physics.wigner import wigner_3j, clebsch_gordan
```

Validation:
- 对称性检查：列置换相位须成立。
- 选择定则检查：$m_1+m_2+m_3\neq0$ 或三角不等式不满足时须为零。
- 与小 $j$（如 $j=3$）已知列表值交叉验证。
