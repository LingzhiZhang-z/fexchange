# 04-02-RUNTIME_CONTRACTION_LEGACY

本文件是旧版 04-02 规范的冻结副本。
当前规范以 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md` 为准。

本文件定义旧版 SOPT 运行时中的 $L2$、$L3$ 与 $L4$。
磁盘 I/O 布局与格式由 `./standards/en/05-io/05-00-IO.md` 统一定义。
写作形式遵循 `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`。
全局执行顺序由 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 固定为 $L0 \to L1 \to L2 \to L3 \to L4$。

## 0) 变量分类（子模块级，MUST）
本文件覆盖 $L2/L3/L4$，采用三类变量语义：
- 输入变量：来自外部接口或上游层输出。
- 中间变量：仅在当前层内部计算使用，不作为该层对外输出。
- 输出变量：该层对下游层/调用者提供的接口变量。

分层定义：
- $L2$：输入 `{A, B, E_u}`；中间 `{E_uv, E_rs, KA, KB}`；输出 `{K}`。
- $L3$：输入 `{K, t_mu}`；中间 `{workspace}`；输出 `{h_pre_j_mu}`。
- $L4$：输入 `{h_pre_j_mu, W, labels_abcd}`；中间 `{h_pre_mu}`；输出 `{h_mu_abcd, Heff_mu_abcd}`。

## 0.1) $L3/L4$ 外部运行时输入 Schema（MUST）
MUST:
- hopping（`t_mu`, `mu_labels`）与 Kramer 投影（`W`, `kramer_labels`）必须由外部运行时输入提供。
- 必须满足 `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md` 的全局头字段闸门：
  `schema_version`, `standard_version`, `basis_id`, `orbital_order_id`, `unit`。
- 输出中的通道顺序必须与输入 `mu_labels` 顺序严格一致。

Math:
$$
t\_mu\in\mathbb C^{n_\mu\times n_{\mathrm{orb}}\times n_{\mathrm{orb}}}
\ \text{或}\ 
\mathbb C^{n_{\mathrm{orb}}\times n_{\mathrm{orb}}}
\ (\text{提升为 }n_\mu=1).
$$

Math:
$$
W\in\mathbb C^{n_j\times n_k},
\qquad
labels\_{abcd}\in\mathbb Z^{n_{\mathrm{out}}\times 4},
\qquad
0\le a,b,c,d<n_k.
$$

Code form:
```text
if t_mu.ndim == 2: t_mu = t_mu[None, :, :]
require len(mu_labels) == t_mu.shape[0]
require W.shape[0] == h_pre_j_mu.shape[0]     # j-axis binding
require labels_abcd.shape[1] == 4
require labels_abcd.max() < W.shape[1]
```

Index:
- `mu_labels[k]` 是通道轴 `k` 的规范标签。
- `W` 的 `j` 轴必须与 03-01 模块导出的 LSJM 低 SOC 子空间顺序一致。

Validation:
- 任意 schema/绑定不一致都必须在收缩前硬失败。
- `W` 正交检查：`W^dag W = I`，阈值为 `eps_orth`。

NPZ 键契约（当采用文件载荷时）：
Code form:
```text
hopping npz required keys:
  t_mu, mu_labels, n_orb, hopping_name,
  schema_version, standard_version, basis_id, orbital_order_id, unit

kramer npz required keys:
  W, kramer_labels, kramer_name, n_j,
  schema_version, standard_version, basis_id, orbital_order_id, unit

labels npz/json required keys:
  labels_abcd, labels_order_id
```

## 1) Legacy Level 2: 中间态求和与裸核 K（冻结定义）
MUST:
- 本层只构造不含 Kramers 指标的裸核 $K$。
- 本层进行 site 绑定：把 $L1$ 的 $\kappa$ 绑定到具体 site 轨道指标。
- 本层显式求和采用路线 A 的 $(u,v)$ 与路线 B 的 $(r,s)$。
- 本层只做中间态求和与分母处理；$W$ 不进入 $K$ 的定义。
- 本层输入能量应为单 site 形式 `E_u`（按 $n+1/n-1$ 扇区给出）；$E_{uv}/E_{rs}$ 由本层内部组合计算。
- 对 LSJM 中间态流程，`E_u` 必须由分项系数重构：
  $E_u = F0\,c_{u,F0}+F2\,c_{u,F2}+F4\,c_{u,F4}+F6\,c_{u,F6}+\zeta\,c_{u,\zeta}$。
- 其中 $c_{u,F0/F2/F4/F6/\zeta}$ 必须来自 03-01 模块能量分项输出
  （算符期望值系数），不能在本层用硬编码 SOC 解析式替代。
- 为保证 $K = K_A + K_B$ 可直接成立且下游统一使用 $t_{pq}t_{p'q'}^\ast$ 收缩，本节的 $K_B$ 采用“对齐槽位”定义（等价于自然 B 路核在 $(pq)\leftrightarrow(p'q')$ 上做一次交换）。
- 若使用两 site 复合中间态索引，可写作：$m=(u,v)$（路线 A），$n=(r,s)$（路线 B）。

Math:
$$
A_{u j}^{i,p} \equiv A_{u j}^{\kappa=p,n},
\qquad
B_{j v}^{j,q} \equiv B_{j v}^{\kappa=q,n-1}.
$$

Math:
$$
E_i^{n+1}(u)
\equiv
F0\,c_{u,F0}^{n+1}
+ F2\,c_{u,F2}^{n+1}
+ F4\,c_{u,F4}^{n+1}
+ F6\,c_{u,F6}^{n+1}
+ \zeta\,c_{u,\zeta}^{n+1}.
$$

Math:
$$
E_j^{n-1}(v)
\equiv
F0\,c_{v,F0}^{n-1}
+ F2\,c_{v,F2}^{n-1}
+ F4\,c_{v,F4}^{n-1}
+ F6\,c_{v,F6}^{n-1}
+ \zeta\,c_{v,\zeta}^{n-1}.
$$

Math:
$$
E_{uv} \equiv E_i^{n+1}(u)+E_j^{n-1}(v).
$$

Math:
$$
K_{j_3 j_4,\,j_1 j_2}^{A;\,pq,p'q'}
= \sum_{u,v}
\frac{
\left(A_{u j_3}^{i,p'}\right)^{\ast}
B_{j_4 v}^{j,q'}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^{\ast}
}{-E_{uv}}.
$$

Math:
$$
E_i^{n-1}(r)
\equiv
F0\,c_{r,F0}^{n-1}
+ F2\,c_{r,F2}^{n-1}
+ F4\,c_{r,F4}^{n-1}
+ F6\,c_{r,F6}^{n-1}
+ \zeta\,c_{r,\zeta}^{n-1}.
$$

Math:
$$
E_j^{n+1}(s)
\equiv
F0\,c_{s,F0}^{n+1}
+ F2\,c_{s,F2}^{n+1}
+ F4\,c_{s,F4}^{n+1}
+ F6\,c_{s,F6}^{n+1}
+ \zeta\,c_{s,\zeta}^{n+1}.
$$

Math:
$$
E_{rs} \equiv E_i^{n-1}(r)+E_j^{n+1}(s).
$$

Math:
$$
A_{s j}^{j,q} \equiv A_{s j}^{\kappa=q,n},
\qquad
B_{j r}^{i,p} \equiv B_{j r}^{\kappa=p,n-1}.
$$

Math:
$$
K_{j_3 j_4,\,j_1 j_2}^{B;\,pq,p'q'}
= \sum_{r,s}
\frac{
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^{\ast}
\left(B_{j_1 r}^{i,p'}\right)^{\ast}
A_{s j_2}^{j,q'}
}{-E_{rs}}.
$$

Math:
$$
K^{pq,p'q'}_{j_3 j_4,\,j_1 j_2}
=
K^{A;\,pq,p'q'}_{j_3 j_4,\,j_1 j_2}
+
K^{B;\,pq,p'q'}_{j_3 j_4,\,j_1 j_2}.
$$

从有效哈密顿量到 $A/B$ 裸核形式的展开:
Math:
$$
\left(h_{\mathrm{pre}}^{(\mu)}\right)_{j_3 j_4,\,j_1 j_2}
=
\sum_n
\frac{
\langle j_3,j_4|H_{\mathrm{hop}}^{(\mu)}|n\rangle
\langle n|H_{\mathrm{hop}}^{(\mu)}|j_1,j_2\rangle
}{-E_n},
$$

Math:
$$
\left(h_{\mathrm{pre}}^{(\mu)}\right)_{j_3 j_4,\,j_1 j_2}
=
\sum_{p q p' q'}
t_{pq}^{(\mu)}\,t_{p'q'}^{(\mu)\ast}\,
K_{j_3 j_4,\,j_1 j_2}^{pq,p'q'}.
$$

路线 A 的分子分解:
Math:
$$
\langle j_3,j_4|f_{jq'}^{\dagger}f_{ip'}|u,v\rangle
\langle u,v|f_{ip}^{\dagger}f_{jq}|j_1,j_2\rangle
=
\left(A_{u j_3}^{i,p'}\right)^{\ast}
B_{j_4 v}^{j,q'}
A_{u j_1}^{i,p}
\left(B_{j_2 v}^{j,q}\right)^{\ast}.
$$

路线 B 的分子分解:
Math:
$$
\langle j_3,j_4|f_{ip}^{\dagger}f_{jq}|r,s\rangle
\langle r,s|f_{jq'}^{\dagger}f_{ip'}|j_1,j_2\rangle
=
B_{j_3 r}^{i,p}
\left(A_{s j_4}^{j,q}\right)^{\ast}
\left(B_{j_1 r}^{i,p'}\right)^{\ast}
A_{s j_2}^{j,q'}.
$$

Code form:
```text
KA[j3,j4,j1,j2,p,q,p2,q2] = sum_{u,v} conj(A_i_p2[u,j3]) * B_j_q2[j4,v] * A_i_p[u,j1] * conj(B_j_q[j2,v]) / (-Euv[u,v])
KB[j3,j4,j1,j2,p,q,p2,q2] = sum_{r,s} B_i_p[j3,r] * conj(A_j_q[s,j4]) * conj(B_i_p2[j1,r]) * A_j_q2[s,j2] / (-Ers[r,s])
K = KA + KB
```

Index:
- $K[...,p,q,p',q']$：不含 Kramers 指标的裸核（$L2$ 产物）。
- $A/B$ 上标中的 `$i$/$j$` 仅表示 site 标签，不是 LSJM 子空间索引
  `$j_1,j_2,j_3,j_4$`。

Validation:
- $L2$ 输出的核索引顺序必须固定为 $[j_3,j_4,j_1,j_2,p,q,p',q']$（或明确记录等价顺序）。
- $W$ 不得进入 $K$ 的定义。
- 分母符号必须与 `04-00` 中 $E_0=0$ 约定一致。

Output（MUST）:
- 本层必须独立输出 $K$（及必要 shape/label 元数据）。
- 本层不得把 $E_{uv}$、$E_{rs}$ 作为对外持久化输出。
- 本层输出是 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md` 的直接输入。

## 2) Level 3: 固定 hopping 并收缩
MUST:
- 必须先用 hopping 与裸核 $K$ 收缩。
- 指标约定必须与 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 完全一致。

Math:
$$
h_{\mathrm{pre},\,j_3 j_4,\,j_1 j_2}^{(\mu)}
= \sum_{p q p' q'}
t_{pq}^{(\mu)}\,t_{p'q'}^{(\mu)\ast}
K_{j_3 j_4,\,j_1 j_2}^{pq,p'q'}.
$$

Code form:
```text
h_pre_j_mu[j3,j4,j1,j2] = sum_{p,q,p2,q2} t_mu[p,q] * conj(t_mu[p2,q2]) * K[j3,j4,j1,j2,p,q,p2,q2]
```

Index:
- $h_{\mathrm{pre},j}^{(\mu)}$：收缩后、Kramers 投影前的通道张量（定义在 $f^n$ 的 SOC 最低能 LSJM 子空间）。

Validation:
- $t$ 与 $K$ 的索引顺序必须一致。
- 零 hopping 检查：$t=0$ 时 $h_{\mathrm{pre},j}^{(\mu)}=0$。

Output（MUST）:
- 本层必须独立输出 $h_{\mathrm{pre},j}^{(\mu)}$（等价代码名：`h_pre_j_mu`）。
- 本层输出是 $L4$ 的直接输入。

## 3) Level 4: 固定 Kramers 基并生成最终输出
MUST:
- 必须在 $L3$ 之后做 $W$ 外层投影。
- 最终对外接口必须使用 $a,b,c,d$ 语义。
- $W$ 必须表示从 $f^n$ 的 SOC 最低能 LSJM 子空间到 CEF/Kramers 基的映射。

Math:
$$
h_{\mathrm{pre},\,cd,ab}^{(\mu)}
= \sum_{j_3,j_4,j_1,j_2}
(W_{j_3 c})^{\ast}(W_{j_4 d})^{\ast}
h_{\mathrm{pre},\,j_3 j_4,\,j_1 j_2}^{(\mu)}
W_{j_1 a}W_{j_2 b}.
$$

Math:
$$
H_{\mathrm{eff},\,cd,ab}^{(\mu)}
= h_{\mathrm{pre},\,cd,ab}^{(\mu)}.
$$

Code form:
```text
h_pre_mu = project_with_W(h_pre_j_mu, W)
h_mu_abcd = h_pre_mu
Heff_mu_abcd = h_mu_abcd
```

Index:
- $h_{\mathrm{pre},cd,ab}^{(\mu)}$：经 $W$ 投影后的通道张量。
- $h_{\mu,abcd}$：目标 Kramers 基中的通道张量。
- $\mathrm{Heff}_{abcd}^{(\mu)}$：固定单个 $\mu$-bond 通道的有效哈密顿量。

Validation:
- Hermitian 检查：$\mathrm{Heff}^{(\mu)}=\left(\mathrm{Heff}^{(\mu)}\right)^{\dagger}$（容差内）。
- $W$ 投影维度必须与 $h_{\mathrm{pre},j}^{(\mu)}$ 匹配。

Output（MUST）:
- 本层必须独立输出 $h_{\mu,abcd}$ 与 $\mathrm{Heff}_{abcd}^{(\mu)}$。
- 当本规范不引入额外后旋转时，二者数值相同。
- 当启用自旋模型导出时，$\mathrm{Heff}_{abcd}^{(\mu)}$ 是
  `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md` 的直接输入。

## 4) 运行时 I/O（分层汇总）
MUST:
- 输入必须包含 $\{A,B,E_u,K,t^{(\mu)},W,\mathrm{labels}_{abcd}\}$。

Code form:
```text
inputs_L2_L4  = {A, B, E_u, K, t_mu, W, labels_abcd}
outputs_L2    = {K}
outputs_L3    = {h_pre_j_mu}
outputs_L4    = {h_mu_abcd, Heff_mu_abcd}
```
