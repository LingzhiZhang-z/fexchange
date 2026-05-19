# 04-01-FOPT_PREPROCESSING

本文件定义 FOPT 预处理 Levels `L0`、`L1` 与 `L2`。
FOPT 全局范围定义在
`./standards/en/04-fopt/04-00-FOPT_FORMALISM.md`。
底层 determinant 符号遵循
`./standards/en/01-core/01-00-FOCK_SLATER.md` 与
`./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`。

## 0) 变量类别（子模块作用域，MUST）
本文件只覆盖 FOPT `L0/L1/L2`。

逐层定义：
- `L0`：输入 `{charge_sectors, n_f_orb, n_p_orb}`；输出
  `{F_create_raw, P_annihilate_raw}`。
- `L1`：输入 `{F_create_raw, P_annihilate_raw, U_f, U_p, R_f, R_p}`；
  输出 `{F_create_rot, P_annihilate_rot}`。
- `L2`：输入 `{F_create_rot, P_annihilate_rot, t_r_lambda, charge_pairs}`；
  输出 `{V_plus}`。

约束：
- `L0/L1` 不得消费 hopping 矩阵。
- `L2` 不得消费 resolvent、路径列表或 W/Kramers projector。

Code form:
```text
FOPT_L0_L2_inputs_exclude = {resolvents, four_hop_paths, W, kramer_labels}
```

Validation:
- 任何接受 `t_r_lambda` 的 `L0/L1` 入口都是无效的。
- 任何接受 resolvent 或路径枚举的 `L2` 入口都是无效的。

## 1) Level 0: Raw Local Primitives（MUST）
MUST:
- 在正则 f-shell determinant 基上构造 f-site creation primitives。
- 在正则 ligand p-shell determinant 基上构造 ligand annihilation primitives。
- 使用 bra-ket 矩阵元约定，态矢量为列向量。
- 保留从 core 规范继承的 determinant 排序、parity-below-index 符号、
  dtype 策略与轨道指标顺序。
- `L0` 不得应用 site 标签、配体标签、local-frame rotation、Wannier rotation、
  hopping 矩阵或局域工作基投影。

Math:
$$
F_{\mathrm{raw}}^{\dagger,a}[N_f]_{\alpha\beta}
=
\langle \alpha^{N_f+1}|f_a^\dagger|\beta^{N_f}\rangle.
$$

Math:
$$
P_{\mathrm{raw}}^{b}[N_p]_{\rho\delta}
=
\langle \rho^{N_p-1}|p_b|\delta^{N_p}\rangle.
$$

Code form:
```text
F_create_raw[N_f][a].shape = (dim_f(N_f+1), dim_f(N_f))
P_annihilate_raw[N_p][b].shape = (dim_p(N_p-1), dim_p(N_p))
```

Index:
- `a` 是正则 f spin-orbital index。
- `b` 是正则 ligand p spin-orbital index。
- `alpha,beta` 是正则 f determinant-sector indices。
- `rho,delta` 是正则 ligand determinant-sector indices。

Validation:
- `F_create_raw[N_f][a]` 对无效 `N_f` 或 `a` 必须失败。
- `P_annihilate_raw[N_p][b]` 对无效 `N_p` 或 `b` 必须失败。
- `P_annihilate_raw[N_p][b]` 必须等于从 `N_p-1` 到 `N_p` 的对应 ligand
  creation matrix 的伴随。
- 对已占据轨道做 creation、对空轨道做 annihilation 必须给出零矩阵元。

## 2) Level 1: Working-Basis and Frame Rotation（MUST）
MUST:
- 将 raw primitives 旋转/绑定到 site-specific f working bases 与
  ligand-specific p working bases。
- 在 primitive orbital axis 上应用物理单粒子 frame rotations。
- 显式保留 f-site 标签 `r` 与配体标签 `lambda`。
- 在元数据中记录所有 state-basis 与 one-particle-frame order ids。
- 本层不得乘以 hopping 矩阵。

Math:
$$
F_{\mathrm{rot}}^{r,\alpha}[N_f]
=
\sum_a
R_f[r]_{a\alpha}\,
\left(U_f[r,N_f+1]\right)^\dagger
F_{\mathrm{raw}}^{\dagger,a}[N_f]
U_f[r,N_f].
$$

Math:
$$
P_{\mathrm{rot}}^{\lambda,\beta}[N_p]
=
\sum_b
R_p[\lambda]_{b\beta}\,
\left(U_p[\lambda,N_p-1]\right)^\dagger
P_{\mathrm{raw}}^{b}[N_p]
U_p[\lambda,N_p].
$$

Code form:
```text
F_create_rot[r][N_f][alpha] = sum_a R_f[r][a,alpha] * U_f_out^dag @ F_create_raw[N_f][a] @ U_f_in
P_annihilate_rot[lambda][N_p][beta] = sum_b R_p[lambda][b,beta] * U_p_out^dag @ P_annihilate_raw[N_p][b] @ U_p_in
```

Index:
- `U_f[r,N]` 将 sector `N` 的正则 f determinants 映射到该 f-site 的选定工作基。
- `U_p[lambda,N]` 将 sector `N` 的正则 ligand determinants 映射到该配体的选定工作基。
- `R_f[r]` 将 hopping 使用的物理 f orbital labels 映射到正则 raw primitive labels。
- `R_p[lambda]` 将 hopping 使用的物理 ligand orbital labels 映射到正则 raw primitive labels。

Validation:
- 所有 `U_f` 与 `U_p` 矩阵必须列正交归一。
- `R_f` 与 `R_p` 的行数必须匹配 raw primitive orbital axes。
- 必须对每个 sector 和每个 site 显式检查输出 shape。
- 元数据必须包含 `f_state_order_id`、`p_state_order_id`、
  `f_orbital_order_id`、`p_orbital_order_id` 与
  `active_pair_order_id`。

## 3) Level 2: Active-Pair Forward Blocks（MUST）
MUST:
- 只构造 p-to-f hopping 的 `V_plus` blocks。
- 使用 active-pair 张量积顺序 `f < p`。
- 不包含任何块间费米嵌入符号。
- 完整 cluster 的 embedding signs 保留给未来 `L3`。
- 不存储反向 hopping blocks。

Math:
$$
V_{+}^{r\lambda}[N_f,N_p]
=
\sum_{\alpha\beta}
t_{r\lambda}^{\alpha\beta}
\left(
F_{\mathrm{rot}}^{r,\alpha}[N_f]
\otimes
P_{\mathrm{rot}}^{\lambda,\beta}[N_p]
\right).
$$

Code form:
```text
V_plus[r,lambda,N_f,N_p] = sum_alpha_beta t[alpha,beta] * kron(F_create_rot[r][N_f][alpha], P_annihilate_rot[lambda][N_p][beta])
```

Index:
- `alpha` 是 `t_r_lambda` 中的物理 f orbital index。
- `beta` 是 `t_r_lambda` 中的物理 ligand orbital index。
- 行按 `(f_out,p_out)` 排序。
- 列按 `(f_in,p_in)` 排序。

Validation:
- `V_plus.shape == (D_f[N_f+1] * D_p[N_p-1], D_f[N_f] * D_p[N_p])`。
- 线性检查：
  `V_plus(t1 + c*t2) == V_plus(t1) + c*V_plus(t2)`。
- 零 hopping 检查：
  `V_plus(0) == 0`。
- 伴随一致性检查：
  `dagger(V_plus[N_f-1,N_p+1])` 必须具有当前扇区反向 hop
  从 `(N_f,N_p)` 到 `(N_f-1,N_p+1)` 所需的 shape。
- `L2` 不得输出名为 `B`、`V_minus` 或 `reverse` 的 key。

## 4) Charge-Pair Selection（MUST）
MUST:
- 接受显式 iterable of charge pairs，以便确定性构造。
- 提供低能扇区派生的最小 pair set helper。
- 在分配矩阵前过滤或拒绝无效 pairs。

Code form:
```text
required_fopt_pairs(n, n_p_full=6) = valid_sorted({(n,n_p_full), (n-1,n_p_full), (n,n_p_full-1)})
```

Validation:
- 返回的 charge pairs 必须按 `(N_f,N_p)` 字典序排序。
- 重复项必须确定性去重。
- pair 只有在 `0 <= N_f < n_f_orb` 且 `1 <= N_p <= n_p_orb` 时有效。

## 5) 测试要求（MUST）
MUST:
- 测试 raw f creation 与 ligand annihilation 的 shape。
- 测试 ligand annihilation 与 creation-adjoint 的一致性。
- 使用 identity transforms 和至少一个非平凡 unitary rotation 测试 `L1` shape binding。
- 测试 `L2` zero hopping。
- 测试 `L2` 对 `t_r_lambda` 的线性。
- 在足够小的 toy system 上测试 active-pair shape，以便做确定性精确比较。
- 测试 reverse-hop adjoint relation，但不得存储 reverse block。

Code form:
```text
pytest tests/test_fopt_l0.py tests/test_fopt_l1.py tests/test_fopt_l2.py -q
```

Validation:
- 测试不得要求完整四阶路径枚举。
- 测试不得要求 resolvent 构造。
