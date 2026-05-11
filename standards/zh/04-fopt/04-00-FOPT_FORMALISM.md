# 04-00-FOPT_FORMALISM

本文件定义配体介导 f-p 超交换中四阶微扰论（FOPT）的预处理边界。
写作形式遵循 `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`。
底层 Fock 与费米子约定遵循
`./standards/en/01-core/01-00-FOCK_SLATER.md` 与
`./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`。

## 0) 范围与非范围（MUST）
MUST:
- 本规范覆盖两 f 站点 / 两配体 cluster 的 FOPT 可复用构件。
- 本规范只定义 `L0`、`L1` 与 `L2`。
- 不在本合同中实现 FOPT `L3`。
- 不在本合同中枚举四阶 hopping 路径。
- 不在本合同中实现完整的投影收缩
  `P V R V R V R V P`。
- 不在本合同中引入 W 投影、Kramers 投影或最终 exchange 拟合。

Code form:
```text
fopt_scope = {L0, L1, L2}; reserved_future = {L3, path_enumeration, resolvents, W_projection}
```

Validation:
- `fopt` 中任何枚举四跳路径或消费 resolvent 的函数都超出本规范。
- 任何被存储的反向 hopping primitive 都违反合同。

## 1) 物理 cluster 与电荷扇区（MUST）
MUST:
- cluster 包含两个 f 站点与两个配体站点：
  `f1 -- {pA,pB} -- f2`。
- 低能电荷扇区为 `(N_f1, N_f2, N_pA, N_pB) = (n,n,6,6)`。
- 局域 f 电荷扇区必须使用项目 f-shell 约定。
- 局域配体 p 扇区必须使用确定性的局域 p-shell Fock 约定，并显式记录
  `p_orbital_order_id`。

Math:
$$
\mathcal C_0 = (N_{f1},N_{f2},N_{pA},N_{pB})=(n,n,6,6).
$$

Code form:
```text
low_charge = {"f1": n, "f2": n, "pA": 6, "pB": 6}
```

Index:
- `r in {1,2}` 标记 f 站点。
- `lambda in {A,B}` 标记配体站点。
- `N_f` 是局域 f 站点电子数。
- `N_p` 是局域配体电子数。

Validation:
- `0 <= N_f <= 14`。
- `0 <= N_p <= n_p_orb`，其中 `n_p_orb` 必须记录在配体元数据中。
- 缺少配体轨道顺序元数据是硬绑定失败。

## 2) 分层定义（MUST）
MUST:
- `L0` 在正则局域 Fock determinant 基上构造未旋转的局域 primitive
  transition matrices。
- `L1` 将 `L0` primitives 旋转/绑定到具体 site/lambda 的局域工作基与物理单粒子坐标系。
- `L2` 构造 active-pair p-to-f hopping blocks `V_plus`。
- `L2` 必须消费 hopping 矩阵，但不得消费 resolvent 或路径列表。
- `L3` 以及后续层保留给未来规范。

Code form:
```text
FOPT order: L0 -> L1 -> L2 -> future L3
```

Validation:
- `L0` 输出是 site-agnostic。
- site 标签 `r` 与配体标签 `lambda` 最早可在 `L1` 出现。
- hopping 矩阵最早可在 `L2` 出现。

## 3) Active Hopping 方向（MUST）
MUST:
- 只存储正向 p-to-f active-pair hopping blocks。
- 正向算符命名为 `V_plus`。
- `V_plus[r,lambda]` 的物理方向是电子从配体 `p_lambda` 跳到 f 站点 `f_r`。
- 不得将 `B`、`V_minus` 或 f-to-p hopping 作为独立 primitive 存储。

Math:
$$
A_{r\lambda}
=
\sum_{\alpha\beta}
t_{r\lambda}^{\alpha\beta}
f_{r\alpha}^{\dagger}p_{\lambda\beta}.
$$

Math:
$$
V_{+}^{r\lambda}[N_f,N_p]:
(f_r^{N_f}\otimes p_\lambda^{N_p})
\rightarrow
(f_r^{N_f+1}\otimes p_\lambda^{N_p-1}).
$$

Code form:
```text
V_plus[(r, lambda, N_f, N_p)] stores p_lambda -> f_r only
```

Index:
- `alpha` 标记经过 `L1` 后的物理 f spin-orbital 轴。
- `beta` 标记经过 `L1` 后的物理配体 spin-orbital 轴。
- `t_r_lambda[alpha,beta]` 是 p-to-f hopping 振幅。

Validation:
- `t_r_lambda.shape == (n_f_orb, n_p_orb)`。
- `V_plus` 必须严格线性依赖 `t_r_lambda`。
- 若 `t_r_lambda` 为零，`V_plus` 必须为零。

## 4) 反向方向由伴随得到（MUST）
MUST:
- 反向 f-to-p hop 只能作为某个有效正向 block 的伴随来恢复。
- 当前扇区的反向 block 不是独立存储对象。
- 若所需源正向 block 超出允许电荷边界，则该当前扇区的反向 block 未定义，
  不得人为构造。

Math:
$$
B_{r\lambda}[N_f,N_p]
=
\left(
V_{+}^{r\lambda}[N_f-1,N_p+1]
\right)^\dagger.
$$

Code form:
```text
B_current(N_f,N_p) = dagger(V_plus[N_f - 1, N_p + 1])
```

Validation:
- 测试必须对每个有效源正向 block 验证伴随关系。
- `L0`、`L1` 或 `L2` 不得输出任何 `B` 数组 key。

## 5) Active-Pair 张量积基（MUST）
MUST:
- `L2` active-pair 基采用局域顺序 `f < p`。
- flatten 后的张量积顺序采用 row-major：f 指标在外，配体 p 指标在内。
- `L2` active-pair blocks 是裸张量积 blocks，不得包含任何块间费米嵌入符号。
- 完整顺序 `f1 < f2 < pA < pB` 下的所有块间费米符号都保留给未来 `L3`。

Math:
$$
|i_f,i_p\rangle_{\mathrm{active}}
\equiv
|i_f\rangle_{f_r}\otimes |i_p\rangle_{p_\lambda}.
$$

Math:
$$
\mathrm{flat}(i_f,i_p)=i_f\,d_p+i_p.
$$

Code form:
```text
active_pair_order_id = "f_then_p_rowmajor_v1"
```

Validation:
- 对 domain dimensions `(d_f_in,d_p_in)`，列数为 `d_f_in * d_p_in`。
- 对 codomain dimensions `(d_f_out,d_p_out)`，行数为 `d_f_out * d_p_out`。
- 来自完整顺序 `f1 < f2 < pA < pB` 的 full-cluster embedding parity factors
  保留给未来 `L3`。

## 6) 电荷对覆盖范围（MUST）
MUST:
- `L2` 必须支持调用方给出的有效 charge pairs。
- 对低能 f 占据数 `n`，最小 FOPT 预处理集合包含
  `{(n,6), (n-1,6), (n,5)}` 中的有效 pairs。
- 实现可以构造未来 `L3` 所需的任何额外有效 pairs。

Code form:
```text
minimum_pairs(n) = valid({(n,6), (n-1,6), (n,5)})
```

Validation:
- 无效 charge pairs 必须在矩阵构造前失败。
- 返回 map 中 pair 顺序必须确定。
