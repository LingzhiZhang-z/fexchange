# 03-00-REPRESENTATION_LSMS

本文件定义 LSMS 表示的契约。

## 1) 态定义
LSMS 基底态定义为

Math:
$$
\lvert \alpha L M_L S M_S \rangle,
\qquad M = M_L + M_S.
$$

输出中自旋统一用 $twoS = 2S$ 表示。

## 2) 构建规则
LSMS 态由纯库仑哈密顿量 $H_{\mathrm{int}}$ 得到：

Math:
$$
H_{\mathrm{int}}\lvert \psi_a^{\mathrm{LSMS}} \rangle
= E^{\mathrm{int}}_a\lvert \psi_a^{\mathrm{LSMS}} \rangle.
$$

本阶段不包含 SOC 和 CEF。

### 2.1 首选实时生成路径（Null-Space）
LSMS 的首选实现是实时/null-space 路径（不是 CFP 优先）。
该路径是规范性主路径。

### 2.2 CFP 路径（允许，但次级）
允许 CFP 构建作为次级实现，用于：
- 与实时路径交叉校验，
- 引导/测试，
- 兼容已有表格数据。

当 CFP 与实时/null-space 结果在容差外不一致时，以实时路径并结合
$L^2$、$S^2$、$H_{\mathrm{int}}$ 的本征校验为准。

## 3) 实时生成规则（MUST）
本节是实时/null-space 路径的强制生成流程。

### 3.1 数学规则（由升算符核定义最高权态）
1. 在固定 $n_{\mathrm{ele}}$ 的 Fock 扇区内构造 $H_{\mathrm{int}}$、$L^2$、$S^2$、$L_z$、$S_z$、
   $L_+$、$S_+$、$L_-$、$S_-$（`basis_id` 见
   `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`；算符定义见
   `./standards/en/02-models/02-04-ANGULAR_MOMENTUM_OPERATORS.md`）。
2. 用 `Lz` 与 `Sz` 构造 `ML/MS` 子空间：

Math:
$$
\mathcal V_{M_L,M_S}
= \ker(L_z-M_L I)\cap\ker(S_z-M_S I)
$$

3. 对每个目标 $(L,S)$，定义跨子空间升算符映射：

Math:
$$
A_L^{(L,S)} = P_{L+1,S}\,L_+\,P_{L,S},\qquad
A_S^{(L,S)} = P_{L,S+1}\,S_+\,P_{L,S},
$$

其中 $P_{M_L,M_S}$ 是到 $\mathcal V_{M_L,M_S}$ 的投影。
4. 最高权态候选空间定义为：

Math:
$$
\mathcal H_{L,S}^{\mathrm{hw}}
= \ker\!\left(A_L^{(L,S)}\right)\cap\ker\!\left(A_S^{(L,S)}\right)
\subseteq \mathcal V_{L,S}
$$

这是实时 LSMS 的主 null-space 定义。

### 3.2 $\alpha$ 固定规则（当 ker 维数大于 1）
若 $\dim(\mathcal H_{L,S}^{hw}) = r > 1$，必须在该最高权子空间内对角化
$H_{\mathrm{int}}$ 来固定 $\alpha$：

1. 设 $B_{L,S}$ 为 $\mathcal H_{L,S}^{hw}$ 的一组正交归一基矩阵。
2. 构造投影后的 $H_{\mathrm{int}}$：

Math:
$$
H^{\mathrm{hw}}_{L,S} = B_{L,S}^{\dagger} H_{\mathrm{int}} B_{L,S}
$$

3. 对角化：

Math:
$$
H^{\mathrm{hw}}_{L,S} u_\alpha = \varepsilon_\alpha u_\alpha
$$

4. 定义规范最高权态：

Math:
$$
\lvert hw_{\alpha,L,S}\rangle = B_{L,S} u_\alpha
$$

5. 按 $\varepsilon_\alpha$ 升序赋予 $\alpha$；若能量数值简并，必须使用确定性的
   次级排序与相位规范。

这是重复项（同一 `L,S` 多个态）下固定 `alpha` 的强制规则。

### 3.2.1 用于 Alpha 固定的库仑参考标度（MUST）
在 LSMS 内部的 alpha 固定（3.2/3.4）中，必须使用与绝对
$U=F^0$ 无关的“标度归一化”库仑参考算符。

采用：
- $F^0=0$；
- $J_H=1$；
- $r_{42}=F^4/F^2$，$r_{62}=F^6/F^2$。

由

Math:
$$
J_H=\frac{286F^2+195F^4+250F^6}{6435},
\qquad
F^4=r_{42}F^2,\quad F^6=r_{62}F^2
$$

解得

Math:
$$
F^2=\frac{6435}{286+195r_{42}+250r_{62}},\quad
F^4=r_{42}F^2,\quad
F^6=r_{62}F^2
$$

并据此构造 LSMS 内部参考算符：

Math:
$$
H_{\mathrm{int}}^{\mathrm{ref}}
= F^2\hat O_2+F^4\hat O_4+F^6\hat O_6
$$

说明：
- 对 $H_{\mathrm{int}}^{\mathrm{ref}}$ 施加任意非零全局缩放，不会改变本征矢与排序；
  该规则只是固定一个确定性的标度选择。
- 本规则仅用于 LSMS 内部基构造与排序。

### 3.3 多重态生成与实现步骤
1. 从每个 $\lvert hw_{\alpha,L,S}\rangle$ 生成完整多重态：

Math:
$$
\lvert \alpha L M_L S M_S\rangle
\propto (L_-)^{L-M_L}(S_-)^{S-M_S}\lvert hw_{\alpha,L,S}\rangle
$$

2. 每一步降算符后都归一化。
3. 实现流程必须按以下顺序：
- 构造各 $V_{M_L,M_S}$ 子空间基；
- 构造 $A_L^{(L,S)}$ 与 $A_S^{(L,S)}$；
- 求联合核空间；
- 若核空间维数大于 1，在该子空间内对角化 $H_{\mathrm{int}}$ 固定 $\alpha$；
- 用降算符生成全部 `(M_L,M_S)`；
- 做 $L^2$、$S^2$、$L_z$、$S_z$ 残差校验，并做 $H_{\mathrm{int}}$ 一致性校验。

重要：实时 LSMS 态不能只由 $\ker(H_{\mathrm{int}}-EI)$ 定义。

### 3.4 实时实现契约
实时实现必须遵循如下矩阵流程：

1. 子空间索引：
- 构造 $sector[(M_L,M_S)] \to indices$（Fock 基下的索引集合）。
- 对每个子空间构造嵌入矩阵 $R_{M_L,M_S}$，形状为
  `(dim_fock, dim_sector)`。

2. 升算符核求解：
- 构造
  $A_L = R_{L+1,S}^\dagger L_+ R_{L,S}$，
  $A_S = R_{L,S+1}^\dagger S_+ R_{L,S}$。
- 将二者堆叠为 $A=\begin{bmatrix}A_L\\A_S\end{bmatrix}$，求其核空间得到 $B_{L,S}$
  （列向量为子空间坐标下的正交 `hw` 候选）。
- 回嵌到全空间：
  $HW_{\mathrm{full}} = R_{L,S} B_{L,S}$。

3. 用投影 $H_{\mathrm{int}}$ 固定 $\alpha$：
- 构造 $H_{\mathrm{sub}} = R_{L,S}^\dagger H_{\mathrm{int}} R_{L,S}$。
- 投影到 `hw` 子空间：$H_{\mathrm{hw}} = B_{L,S}^\dagger H_{\mathrm{sub}} B_{L,S}$。
- 对角化 $H_{\mathrm{hw}}$，并用本征向量旋转 $B_{L,S} \leftarrow B_{L,S}U$。
- 最终最高权态：
  $\lvert hw_{\alpha,L,S}\rangle = R_{L,S} B_{L,S}(:,\alpha)$。

4. 降算符生成（带规范化递推）：
- 轨道降算符：
  $\lvert L,M_L-1;S,S\rangle = [L_- \lvert L,M_L;S,S\rangle] / \sqrt{L(L+1)-M_L(M_L-1)}$。
- 自旋降算符：
  $\lvert L,M_L;S,M_S-1\rangle = [S_- \lvert L,M_L;S,M_S\rangle] / \sqrt{S(S+1)-M_S(M_S-1)}$。
- 每一步都归一化，并施加固定相位规范。

5. 列拼装：
- 列顺序必须符合第 6 节排序约束。
- `V_fock` 的每一列对应一个 $\lvert \alpha L M_L S M_S\rangle$。

Code form:
```text
build sector embeddings R[ML,MS]
AL = R[L+1,S]^dag @ L_plus @ R[L,S]
AS = R[L,S+1]^dag @ S_plus @ R[L,S]
B_hw = nullspace(vstack([AL, AS]))
if dim(B_hw) > 1: diagonalize(B_hw^dag @ H_int_sub @ B_hw) to fix alpha
generate multiplets with L_minus and S_minus recursion
```

### 3.5 边界条件（MUST）
1. 若目标子空间 $\mathcal V_{L+1,S}$ 为空，则将 $A_L$ 视为“零行矩阵”；
   此时 $L_+$ 约束自动满足。
2. 若目标子空间 $\mathcal V_{L,S+1}$ 为空，则将 $A_S$ 视为“零行矩阵”；
   此时 $S_+$ 约束自动满足。
3. 核空间求解始终使用纵向堆叠
   $A=\begin{bmatrix}A_L\\A_S\end{bmatrix}$。若其中一块为空，只对非空块求核空间。
4. 即使在边界上，只要最高权候选维数大于 1，仍必须按 3.2 用投影 $H_{\mathrm{int}}$
   对角化来固定 $\alpha$。

## 4) Fock 基展开契约
每个 LSMS 态都在
`./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
定义的 Fock 基中展开：

Math:
$$
\lvert \psi_a^{\mathrm{LSMS}} \rangle
= \sum_{\mu} V^{\mathrm{LSMS}}_{\mu a}\,\lvert \mu \rangle_{\mathrm{fock}}.
$$

存储契约：
- `V_fock.shape = (dim_fock, n_states)`
- 第 `a` 列对应一个 LSMS 态。

## 5) 标签契约（MUST）
每个态列必须包含以下标签字段：
- `alpha`
- `L`
- `twoS`
- `ML`
- `MS`

可选文本标签（`label`/term symbol）可存在，但不能替代上述字段。

## 6) 规范排序（MUST）
LSMS 列顺序必须是确定性的：
1. 先按 `(alpha, L, twoS)` 分组，其中 `alpha` 按实时构建的确定性规则给出
   （若采用 CFP，实现也必须映射到同一顺序）；
2. 每个项内按 `ML` 升序；
3. 再按 `MS` 升序。

若实现采用其他顺序，必须在 metadata 显式给出可逆映射。

## 7) 能量契约
LSMS 能量输出必须采用“按项分解”并与 LSMS 态列一一对应。

每列必须给出以下系数：
- `coef_F0`
- `coef_F2`
- `coef_F4`
- `coef_F6`

### 7.1 算符期望值规则（MUST）
定义库仑分解：

Math:
$$
H_{\mathrm{int}} = \sum_{k\in\{0,2,4,6\}} F^k\,\hat O_k.
$$

对第 $a$ 列 LSMS 态向量 $\lvert \psi_a^{\mathrm{LSMS}}\rangle$：

Math:
$$
\mathrm{coef\_F_k}[a]
=
\langle \psi_a^{\mathrm{LSMS}} \rvert \hat O_k \lvert \psi_a^{\mathrm{LSMS}} \rangle,
\quad k\in\{0,2,4,6\}.
$$

对第 `a` 列 LSMS 态，总能量按下式重构：

Math:
$$
E^{\mathrm{LSMS}}_a
=
F0\cdot \mathrm{coef\_F0}[a]
+ F2\cdot \mathrm{coef\_F2}[a]
+ F4\cdot \mathrm{coef\_F4}[a]
+ F6\cdot \mathrm{coef\_F6}[a].
$$

说明：
- 本阶段仅包含库仑贡献（不含 SOC/CEF）。
- 若实现额外存储总能量数组，该数组仅为派生量，必须与上式数值一致。

### 7.2 $H_{\mathrm{int}}$ 的子空间对角性检查（MUST）
在 LSMS 输出态上构造投影矩阵：

Math:
$$
\left(H_{\mathrm{int}}^{\mathrm{LSMS}}\right)_{ab}
=
\langle \psi_a^{\mathrm{LSMS}} \rvert H_{\mathrm{int}} \lvert \psi_b^{\mathrm{LSMS}} \rangle.
$$

在 LSMS 输出子空间内，其非对角元必须足够小：

Math:
$$
\max_{a\neq b}
\left|
\left(H_{\mathrm{int}}^{\mathrm{LSMS}}\right)_{ab}
\right|
\le \varepsilon_{\mathrm{diag}}.
$$

可选诊断（推荐）：同时报告每个 $\hat O_k$ 投影矩阵的非对角范数，用于监控基底质量。

## 8) 校验
- 正交归一校验：$V_{fock}^\dagger V_{fock} = I$（在容差内）。
- 维度校验：态列数必须等于标签数。
- 基底校验：`basis_id` 必须与 Fock 基契约一致。
- 必须满足第 7.2 节的 `H_int` 投影非对角检查。

### 8.1 下游接口元数据（MUST）
MUST:
- LSMS 输出必须显式提供 `basis_id` 与 `orbital_order_id`，用于下游绑定。
- 元数据必须保留当前扇区 `n_orb` 与 `n_ele`。

Code form:
```text
lsms_meta_required = {basis_id, orbital_order_id, n_orb, n_ele, state_order_id}
```

Validation:
- 缺少必需元数据会导致下游 `03-01/04-01/04-02` 接口失败。

## 9) 到 LSJM 的接口（03-01 阶段）
LSMS 输出是后续 LSJM 规范中定义的直接输入。

接口语义要求：
1. `V^{\mathrm{LSMS}}` 必须包含完整多重态
   $\lvert \alpha L M_L S M_S\rangle$，且排序确定。
2. LSJM 通过分块 CG 变换构造：

Math:
$$
V^{\mathrm{LSJM}} = V^{\mathrm{LSMS}} U_{\mathrm{LSMS}\to\mathrm{LSJM}}.
$$

3. LSMS 的相互作用能部分 $E^{\mathrm{int}}(\alpha,L,S)$ 作为 LSJM 能量计算
   的基线应按系数数组（`coef_F0/F2/F4/F6`）传递，而不是预先合并成单一标量数组。
