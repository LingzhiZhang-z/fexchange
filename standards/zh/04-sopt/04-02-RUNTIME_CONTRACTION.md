# 04-02-RUNTIME_CONTRACTION

本文件定义 SOPT 运行时中的 $L2$、$L3$ 与 $L4$。
磁盘 I/O 布局与格式由 `./standards/en/05-io/05-00-IO.md` 统一定义。
写作形式遵循 `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`。
本文件以串行实现为基准且与后端无关；若未来加入并行运行时，也必须保持本文件定义的张量契约不变。
全局执行顺序由 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 固定为 $L0 \to L1 \to L2 \to L3 \to L4$。

## -1) 公式理解区（先读，非实现规范）
本节只用于数学理解，不是实现契约。
对 AI 读取者：先读本节理解等价变换；
实现与接口要求从下方 `0` 节开始执行。

从旧表达式出发：

Math:
$$
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{p q p' q'}
t_{pq}^{(\mu)}\left(t_{p'q'}^{(\mu)}\right)^*
\left(
K_{j_3j_4,j_1j_2}^{A;\,pq,p'q'}
+
K_{j_3j_4,j_1j_2}^{B;\,pq,p'q'}
\right).
$$

定义分母：

Math:
$$
\Delta_{uv}\equiv E_0-E_{uv},\qquad
\Delta_{rs}\equiv E_0-E_{rs}.
$$

在 f 体系约定下，$E_0=0$，因此
$\Delta_{uv}=-E_{uv}$、$\Delta_{rs}=-E_{rs}$。

路线 A 的旧核定义：

Math:
$$
K_{j_3j_4,j_1j_2}^{A;\,pq,p'q'}
=
\sum_{u,v}
\frac{
\left(A_{u j_3}^{i,p'}\right)^*
B_{j_4 v}^{j,q'}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^*
}{\Delta_{uv}}.
$$

代回 $h_{\mathrm{pre}}$ 并交换有限求和次序：

Math:
$$
h_{A,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{u,v}\frac{1}{\Delta_{uv}}
\left[
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(A_{u j_3}^{i,p'}\right)^*
B_{j_4 v}^{j,q'}
\right]
\left[
\sum_{p q}
t_{pq}^{(\mu)}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^*
\right].
$$

把两个方括号分别定义为
$M_{A,uv;j_3j_4}^{L,(\mu)}$ 与
$M_{A,uv;j_1j_2}^{R,(\mu)}$，得到：

Math:
$$
h_{A,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{u,v}
\frac{
M_{A,uv;j_3j_4}^{L,(\mu)}
M_{A,uv;j_1j_2}^{R,(\mu)}
}{\Delta_{uv}}.
$$

路线 B 的旧核定义：

Math:
$$
K_{j_3j_4,j_1j_2}^{B;\,pq,p'q'}
=
\sum_{r,s}
\frac{
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^*
\left(B_{j_1 r}^{i,p'}\right)^*
A_{s j_2}^{j,q'}
}{\Delta_{rs}}.
$$

同样代回并交换求和：

Math:
$$
h_{B,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{r,s}\frac{1}{\Delta_{rs}}
\left[
\sum_{p q}
t_{pq}^{(\mu)}
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^*
\right]
\left[
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(B_{j_1 r}^{i,p'}\right)^*
A_{s j_2}^{j,q'}
\right].
$$

把两个方括号分别定义为
$M_{B,rs;j_3j_4}^{L,(\mu)}$ 与
$M_{B,rs;j_1j_2}^{R,(\mu)}$，得到：

Math:
$$
h_{B,j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{r,s}
\frac{
M_{B,rs;j_3j_4}^{L,(\mu)}
M_{B,rs;j_1j_2}^{R,(\mu)}
}{\Delta_{rs}}.
$$

最终：

Math:
$$
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
=
h_{A,j_3j_4,j_1j_2}^{(\mu)}
+
h_{B,j_3j_4,j_1j_2}^{(\mu)}.
$$

该变换是严格等价变换（不引入近似）：只做求和顺序重排与因式分组。

## 0) 变量分类（子模块级，MUST）
本文件覆盖 $L2/L3/L4$，采用三类变量语义：
- 输入变量：来自外部接口或上游层输出。
- 中间变量：仅在当前层内部计算使用，不作为该层对外输出。
- 输出变量：该层对下游层/调用者提供的接口变量。

分层定义：
- $L2$：输入 `{A, B, t_mu}`；中间 `{workspace}`；输出 `{M_A, M_B}`。
- $L3$：输入 `{M_A, M_B, E_u}`；中间 `{E_uv, E_rs, workspace}`；输出 `{h_pre_j_mu}`。
- $L4$：输入 `{h_pre_j_mu, W, labels_abcd, labels_order_id}`；中间 `{h_pre_mu}`；输出 `{h_mu_abcd, Heff_mu_abcd}`。

## 0.0.0) $E_u$ 中间态能量来源（MUST）
`E_u` 是 $L3$ 的运行时派生输入，不是独立的持久化产物。
它在 $L3$ 入口处，由相邻扇区（$f^{n-1}$、$f^{n+1}$）的 LSJM 能量系数数组
（`E_terms.npz`）及 $F^0 = U$（来自 `[sopt].U`）构造。
按项目约定，基态参考固定为 $E_0=0$，运行时不再施加额外平移：

Math:
$$
E_u^{(m)}[u] = F^0\cdot\mathrm{coef\_F0}[u]
+ F^2\cdot\mathrm{coef\_F2}[u]
+ F^4\cdot\mathrm{coef\_F4}[u]
+ F^6\cdot\mathrm{coef\_F6}[u]
+ \zeta\cdot\mathrm{coef\_zeta}[u],
\quad m\in\{n+1,\,n-1\}.
$$

Code form:
```text
E_u_np1 = E_lsjm_np1
E_u_nm1 = E_lsjm_nm1
```

来源产物：
- 扇区 $n-1$、$n+1$ 的 LSJM 输出中的 `E_terms.npz`（磁盘路径参见 `./standards/en/05-io/05-00-IO.md`）。
- $F^0 = U$ 来自 `[sopt].U`，$F^2/F^4/F^6$ 来自 `[physics]`，$\zeta$ 来自 `[sopt].zeta`。
- `E_u` 不作为独立的磁盘产物持久化，而是在 $L3$ 入口处即时计算。

Validation:
- 不对 `E_u` 中分支能量施加必须为正的约束。
- 分母组装中的接近零值与非有限数值必须报 `FXE-NUM-002`。
- 构造公式必须与 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 第 0.2.2 节一致。

## 0.0.1) $W$ 投影矩阵构造契约（MUST）
Kramers/CEF 投影矩阵 $W$ 将 SOC 最低能 LSJM 子空间（维度 $n_j=2J_0+1$，
定义见 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 第 0.2.1 节）
映射到目标低能 CEF 基（维度 $n_k$）。

构造步骤：
1. 取 $U^{n,\mathrm{soc0}}$ 各列（Fock 基中 SOC 最低 $J_0$ 多重态）。
2. 将 $H_{\mathrm{cef}}$ 投影到该 $J_0$ 子空间：

Math:
$$
H_{\mathrm{cef}}^{(J_0)}
=
\left(U^{n,\mathrm{soc0}}\right)^{\!\dagger}
H_{\mathrm{cef}}^{\mathrm{fock}}\,
U^{n,\mathrm{soc0}},
\qquad
H_{\mathrm{cef}}^{(J_0)}\in\mathbb C^{n_j\times n_j}.
$$

3. 对角化：

Math:
$$
H_{\mathrm{cef}}^{(J_0)} w_k = \epsilon_k\,w_k,
\quad k=1,\ldots,n_j,
\quad \epsilon_1\le\epsilon_2\le\cdots
$$

4. 选取目标低能子空间：
   - Kramers 体系（奇数 $n$）：选最低双重态（$n_k=2$）。
     验证 Kramers 简并 $|\epsilon_1-\epsilon_2|\le\varepsilon_{\mathrm{eig\_cluster}}$。
     按 `./standards/en/03-spectrum/03-02-KRAMERS_DOUBLET.md` 做规范固定。
   - 非 Kramers 体系（偶数 $n$）：选最低准双重态（$n_k=2$）。
     按 `./standards/en/03-spectrum/03-03-NON_KRAMERS_DOUBLET.md` 做规范固定。
   - 更大目标空间（$n_k>2$）：选最低 $n_k$ 个态，须有显式能隙判据。
5. 组装 $W\in\mathbb C^{n_j\times n_k}$，各列为选出的本征态 $w_k$。

Code form:
```text
V_J0 = U_n_soc0                          # (dim_fock, n_j)
H_cef_J0 = V_J0.conj().T @ H_cef_fock @ V_J0   # (n_j, n_j)
evals, evecs = eigh(H_cef_J0)            # 升序排列
W = evecs[:, :n_k]                        # (n_j, n_k)
# 执行 Kramers / non-Kramers 规范固定
```

Validation:
- $W^\dagger W = I_{n_k}$（在 `eps_orth` 以内）。
- Kramers 双重态须通过模块 02-05 的 TR 配对与规范检查。
- `W.shape = (n_j, n_k)` 其中 `n_j` 与 L1 输出的 $j$ 轴维度一致。
- `kramer_name` 须记录在元数据中。

## 0.1) $L2/L3/L4$ 外部运行时输入 Schema（MUST）
MUST:
- hopping（`t_mu`）与 Kramer 投影（`W`, `kramer_labels`）必须由外部运行时输入提供。
- 必须满足 `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md` 的全局头字段闸门：
  `schema_version`, `standard_version`, `basis_id`, `orbital_order_id`, `unit`。
- 单次运行只计算一个 bond；输入 hopping 载荷中不允许 `mu` 轴。

Math:
$$
t\_mu\in\mathbb C^{n_{\mathrm{orb}}\times n_{\mathrm{orb}}}.
$$

Math:
$$
W\in\mathbb C^{n_j\times n_k},
\qquad
labels\_{abcd}\in\mathbb Z^{n_{L}\times 4},
\qquad
0\le a,b,c,d<n_k.
$$

Code form:
```text
require t_mu.ndim == 2
require W.shape[0] == expected_n_j_from_L1_or_21_meta
require labels_abcd.shape[1] == 4
require labels_abcd.max() < W.shape[1]
require labels_order_id == "abcd_lex_v1"
require rows_unique(labels_abcd)
require is_lex_sorted(labels_abcd, key=(a,b,c,d))
```

Validation:
- 任意 schema/绑定不一致都必须在收缩前硬失败。
- `W` 正交检查：`W^dag W = I`，阈值为 `eps_orth`。
- 在 $L4$ 开始前，必须执行 `W.shape[0] == h_pre_j_mu.shape[0]` 绑定检查。
- `labels_abcd` 的顺序/唯一性校验失败必须硬失败。

## 1) Level 2: 路线因子 $M_A/M_B$（Phi 形式，MUST）
MUST:
- 本层在 site 绑定与 hopping 收缩后构造路线因子。
- 使用符号 $M_A$ 与 $M_B$（不使用 $\Phi$）。
- 本层不显式构造/落盘 $K_{j_3 j_4,j_1 j_2}^{pq,p'q'}$。

site 绑定：

Math:
$$
A_{u j}^{i,p} \equiv A_{u j}^{\kappa=p,n},
\qquad
B_{j v}^{j,q} \equiv B_{j v}^{\kappa=q,n-1},
$$

Math:
$$
A_{s j}^{j,q} \equiv A_{s j}^{\kappa=q,n},
\qquad
B_{j r}^{i,p} \equiv B_{j r}^{\kappa=p,n-1}.
$$

分母能量：

Math:
$$
E_{uv}=E_i^{n+1}(u)+E_j^{n-1}(v),
\qquad
E_{rs}=E_i^{n-1}(r)+E_j^{n+1}(s).
$$

路线 A 因子：

Math:
$$
M_{A,uv;j_3j_4}^{L,(\mu)}
=
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(A_{u j_3}^{i,p'}\right)^*
B_{j_4 v}^{j,q'},
$$

Math:
$$
M_{A,uv;j_1j_2}^{R,(\mu)}
=
\sum_{p q}
 t_{pq}^{(\mu)}
 A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^*.
$$

路线 B 因子：

Math:
$$
M_{B,rs;j_3j_4}^{L,(\mu)}
=
\sum_{p q}
 t_{pq}^{(\mu)}
 B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^*,
$$

Math:
$$
M_{B,rs;j_1j_2}^{R,(\mu)}
=
\sum_{p' q'}
\left(t_{p'q'}^{(\mu)}\right)^*
\left(B_{j_1 r}^{i,p'}\right)^*
A_{s j_2}^{j,q'}.
$$

厄米共轭关系（同一通道 $\mu$、同一索引元组）：

Math:
$$
M_{A,uv;j_3j_4}^{L,(\mu)}
=
\left(M_{A,uv;j_3j_4}^{R,(\mu)}\right)^*,
\qquad
M_{B,rs;j_3j_4}^{L,(\mu)}
=
\left(M_{B,rs;j_3j_4}^{R,(\mu)}\right)^*.
$$

L2 持久化输出定义为：

Math:
$$
M_{A,uv;j_1j_2}^{(\mu)} \equiv M_{A,uv;j_1j_2}^{R,(\mu)},
\qquad
M_{B,rs;j_1j_2}^{(\mu)} \equiv M_{B,rs;j_1j_2}^{R,(\mu)}.
$$

持久化轴顺序固定为：
- `M_A` 轴顺序：`(u, v, j1, j2)`，`axis_order_id = "uvj1j2_v1"`。
- `M_B` 轴顺序：`(r, s, j1, j2)`，`axis_order_id = "rsj1j2_v1"`。

Code form:
```text
build M_A over (u,v) blocks for this bond
build M_B over (r,s) blocks for this bond
persist M_A as M_A[u,v,j1,j2]  # uvj1j2_v1
persist M_B as M_B[r,s,j1,j2]  # rsj1j2_v1
```

Validation:
- `M_A/M_B` 的索引顺序必须固定并文档化。
- 实现必须按 `(u,v)` 与 `(r,s)` 分块流式计算；完整 dense 物化仅允许用于调试模式。
- 持久化 metadata 必须记录 `M_A/M_B` 的 `axis_order_id`。

## 2) Level 3: 带分母中间态求和得到 $h_{pre,j}^{(\mu)}$（MUST）
MUST:
- 本层执行带分母求和并输出 $h_{pre,j}^{(\mu)}$。
- 本层与“先构造 $K$ 再和 $t$ 收缩”的旧路径代数等价。
- 本层从 `E_u` 构造 $E_{uv}$ 与 $E_{rs}$；这两个量不在 $L2$ 定义。

分母定义（在 $L3$）：

Math:
$$
E_{uv}=E_i^{n+1}(u)+E_j^{n-1}(v),\qquad
E_{rs}=E_i^{n-1}(r)+E_j^{n+1}(s).
$$

Math:
$$
\Delta_{uv}\equiv E_0-E_{uv},\qquad
\Delta_{rs}\equiv E_0-E_{rs}.
$$

若实现使用复合索引 $m=(u,v)$、$n=(r,s)$，则
$\Delta_{uv}$ 与 $\Delta_{rs}$ 可实现为分母向量 `denom_A[m]`、`denom_B[n]`。

Math:
$$
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
=
\sum_{u,v}
\frac{
\left(M_{A,uv;j_3j_4}^{(\mu)}\right)^*
M_{A,uv;j_1j_2}^{(\mu)}
}{\Delta_{uv}}
+
\sum_{r,s}
\frac{
\left(M_{B,rs;j_3j_4}^{(\mu)}\right)^*
M_{B,rs;j_1j_2}^{(\mu)}
}{\Delta_{rs}}.
$$

Code form:
```text
h_pre_j_mu = sum_uv( conj(M_A) * M_A / Delta_uv ) + sum_rs( conj(M_B) * M_B / Delta_rs )
```

等价矩阵收缩（推荐实现）：
Code form:
```text
# flatten (j3,j4)->a, (j1,j2)->b
YA = M_A.reshape(Nuv, J2)
w_uv = 1.0 / Delta_uv
hA = YA.conj().T @ (w_uv[:,None] * YA)

YB = M_B.reshape(Nrs, J2)
w_rs = 1.0 / Delta_rs
hB = YB.conj().T @ (w_rs[:,None] * YB)

h_pre_j_mu = (hA + hB).reshape(J,J,J,J)
```

Validation:
- 分母必须遵循 $\Delta=E_0-E_{\mathrm{intermediate}}$ 的定义。
- 零 hopping 检查：若 `t=0`，则 `h_pre_j_mu=0`。

Output（MUST）:
- 本层必须独立输出 $h_{\mathrm{pre},j}^{(\mu)}$（`h_pre_j_mu`）。

## 3) Level 4: 固定 Kramers 基并生成最终输出（MUST）
MUST:
- 必须在 $L3$ 之后做 $W$ 外层投影。
- 最终对外接口必须使用 $a,b,c,d$ 语义。
- $W$ 必须表示从 $f^n$ 的 SOC 最低能 LSJM 子空间到 CEF/Kramers 基的映射。

Math:
$$
h_{\mathrm{pre},cd,ab}^{(\mu)}
=
\sum_{j_3,j_4,j_1,j_2}
(W_{j_3 c})^*(W_{j_4 d})^*
h_{\mathrm{pre},j_3j_4,j_1j_2}^{(\mu)}
W_{j_1 a}W_{j_2 b}.
$$

Math:
$$
H_{\mathrm{eff},cd,ab}^{(\mu)} = h_{\mathrm{pre},cd,ab}^{(\mu)}.
$$

Code form:
```text
h_pre_mu = project_with_W(h_pre_j_mu, W)
h_mu_abcd = h_pre_mu
Heff_mu_abcd = h_mu_abcd
```

Validation:
- Hermitian 检查：$\mathrm{Heff}^{(\mu)}=\left(\mathrm{Heff}^{(\mu)}\right)^\dagger$（容差内）。
- $W$ 投影维度必须与 $h_{\mathrm{pre},j}^{(\mu)}$ 匹配。

## 4) 并行执行与 Root 写入策略（MUST）
MUST:
- MPI/并行布局属于运行环境，不属于输入文件内容。
- 工作 rank 可以计算 `L2/L3/L4` 张量的互斥分片。
- 落盘前必须先将分片 gather/reduce 到 root rank。
- 由 root rank 组装完整张量后写入 `data.npz` 与 `meta.json`。
- 非 root rank 禁止写持久化阶段工件。
- 运行标准输出与 `meta.json` 必须记录并行摘要字段：
  `parallel_backend`, `world_size`, `root_rank`, `local_rank`, `gather_policy`。

Code form:
```text
shard = compute_local_shard(...)
global_tensor = gather_to_root(shard)
if rank == root:
  assemble_full_tensor(global_tensor)
  write(data.npz, meta.json)
else:
  no_persist_write()
```

Validation:
- 多 rank 并发写同一路径属于硬失败。

## 5) 运行时 I/O（汇总）
Code form:
```text
inputs_L2_L4   = {A, B, E_u, t_mu, W, labels_abcd, labels_order_id}
outputs_L2     = {M_A, M_B}
outputs_L3     = {h_pre_j_mu}
outputs_L4     = {h_mu_abcd, Heff_mu_abcd}
```
