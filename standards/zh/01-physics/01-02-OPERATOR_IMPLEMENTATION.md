# 01-02-OPERATOR_IMPLEMENTATION

本文件定义算符在代码中的表示与实现契约，并与
`./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
及态矢规范一致。

## 1) 适用范围（MUST）
- 本文件定义实现契约，不引入新物理模型。
- $L/S/J$ 的物理定义由后续模型层规范给出。
- 任意算符实现都必须可追溯到 `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md` 的 `basis_id_from`/`basis_id_to` 与 bit 规则。

## 2) 三层算符结构（MUST）
统一使用三层结构：
1. Primitive：单个费米原语算符（`cdag(p)` 或 `c(p)`）。
2. Monomial：复系数乘以“有序原语列表”。
3. Operator：若干 monomial 的有限和，加上元数据。

Math:
$$
\hat O = \sum_t g_t \prod_{r=1}^{m_t} a_{t,r},\qquad
a_{t,r}\in\{c_p^\dagger,\;c_p\}.
$$

Code form:
```text
PrimitiveOp  = {kind: "cdag"|"c", p: int}
Monomial     = {coef: complex, ops: list[PrimitiveOp]}    # ordered
Operator     = {terms: list[Monomial], metadata: {...}}
```

Index:
- $t$：monomial 编号。
- $r$：单个 monomial 内原语算符位置。
- $p$：轨道索引（`0..n_orb-1`）。
- $g_t$：第 $t$ 项 monomial 的标量系数（不是基底索引）。

Validation:
- Primitive 的 `p` 必须在基底轨道范围内。
- Monomial 的算符顺序必须显式存储，不可隐式推断。

## 3) 规范序与化简（MUST）
- 内部规范序采用 normal-order：
  产生算符在左，湮灭算符在右。
- 重排必须使用费米反对易关系：
  $c_p c_q^\dagger = \delta_{pq} - c_q^\dagger c_p$。
- 单个 monomial 内，同一索引重复产生或重复湮灭视为零项。
- 规范序后必须把同类 monomial 合并（系数求和）。

Math:
$$
\{c_p,c_q^\dagger\}=\delta_{pq},\qquad
\{c_p,c_q\}=0,\qquad
\{c_p^\dagger,c_q^\dagger\}=0.
$$

Code form:
```text
normalize(monomial) -> list[monomial]     # 可能因 delta 项分叉
combine_like_terms(operator) -> operator
```

Validation:
- 规范化过程必须是确定性的。
- 小于固定容差的零项必须删除。

## 3.1) 二体索引规范序规则（MUST）
对四费米 monomial 统一采用以下索引规范：

Math:
$$
\hat O^{(2)}_{ijkl}=c_i^\dagger c_j^\dagger c_k c_l,
\qquad i<j,\ k<l.
$$

Code form:
```text
two_body_key = (i, j, k, l) with i<j and k<l
term        = coef * cdag(i) cdag(j) c(k) c(l)
```

规则：
- 若输入 `i>j`，交换两个产生算符并令系数乘以 `-1`。
- 若输入 `k>l`，交换两个湮灭算符并令系数乘以 `-1`。
- 若出现重复产生索引（`i=j`）或重复湮灭索引（`k=l`），该项为零。
- 本小节中的 `i,j,k,l` 均为轨道索引，不是 site 索引。

## 4) 在 Fock 行列式上的作用（MUST）
- bitstring 作用规则继承 `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`。
- 原语作用符号由 $(-1)^{N_{<p}}$ 给出。
- monomial 作用顺序为从右到左依次施加原语算符。

Math:
$$
c_p^\dagger\lvert det\rangle=
\begin{cases}
0,& n_p=1\\
(-1)^{N_{<p}(det)}\lvert det\cup\{p\}\rangle,& n_p=0
\end{cases}
$$

Math:
$$
c_p\lvert det\rangle=
\begin{cases}
(-1)^{N_{<p}(det)}\lvert det\setminus\{p\}\rangle,& n_p=1\\
0,& n_p=0
\end{cases}
$$

Code form:
```text
apply_c_dag(det, p) -> (phase, det_new) | None
apply_c(det, p)     -> (phase, det_new) | None
apply_monomial(det, monomial) -> (amp, det_new) | None
```

Validation:
- 原语作用结果必须满足零作用规则。
- 符号必须与 bit 排序下的奇偶规则一致。

## 4.1) 在 StateVec/StateSet 上的作用（MUST）
- 算符作用在态上的规则必须采用列向量约定。
- 矩阵元约定：
  $O_{\beta\alpha}=\langle\beta|\hat O|\alpha\rangle$。
- `StateSet` 的作用采用列约定下的左乘形式。

Math:
$$
d_\beta = \sum_{\alpha} O_{\beta\alpha} c_\alpha,
\qquad
V_{\mathrm{out}} = O\,V_{\mathrm{in}}.
$$

Code form:
```text
apply_operator_to_statevec(operator, statevec_in) -> statevec_out
apply_operator_to_stateset(operator, stateset_in) -> stateset_out
```

Validation:
- `stateset_in` 必须匹配 `operator.basis_id_from` 与 `operator.sector_from`。
- `stateset_out` 必须匹配 `operator.basis_id_to` 与 `operator.sector_to`。
- `stateset_in` 的列顺序必须在 `stateset_out` 中保持不变。
- `stateset_out.state_order_id` 必须等于 `stateset_in.state_order_id`。

## 5) 算符构建的输入/中间/输出（MUST）
- 输入变量：
  系数张量（如 `O_pq`, `V_pqrs`）与基底元数据（`basis_id_from`, `basis_id_to`, `n_orb`, 扇区信息）。
- 中间变量：
  展开 monomial 列表、规范序结果、临时哈希表。
- 输出变量：
  规范化后的 `Operator` 对象；可选地导出其矩阵/缓存形式。

Code form:
```text
build_operator(inputs) -> Operator
operator_to_matrix(operator, sector_basis) -> sparse_matrix (optional)
```

Validation:
- `basis_id_from` 与 `basis_id_to` 必须显式给出。
- 输入态的 `basis_id` 与 `basis_id_from` 不一致时必须直接失败。
- `sector_from/sector_to` 必须与算符阶数和真实作用一致。

## 6) 序列化契约（MUST）
- 主交换格式：term-list JSON。
- 可选加速格式：稀疏矩阵缓存（`npz`），但必须附完整元数据。

Code form:
```text
JSON fields:
  schema, name, basis_id_from, basis_id_to, orbital_order_id, n_orb,
  sector_from, sector_to, terms=[{coef:[re,im], ops:[[kind,p],...]}]

NPZ fields (optional):
  row, col, data_re, data_im, shape,
  basis_id_from, basis_id_to, orbital_order_id, sector_from, sector_to, name
```

Validation:
- JSON 中 term 顺序在规范化后必须稳定。
- 元数据不匹配的矩阵缓存必须拒绝读取。

## 7) 最小测试集（MUST）
1. 在随机行列式上验证反对易关系。
2. 厄米一致性：
   `dagger(dagger(O)) == O`。
3. 升降算符关系：
   $L_- = L_+^\dagger$，$S_- = S_+^\dagger$。
4. 数算符对易关系：
   $[N,c_p^\dagger]=c_p^\dagger$，$[N,c_p]=-c_p$。
5. 规范序确定性：
   同一输入项集总能得到同一排序输出。
