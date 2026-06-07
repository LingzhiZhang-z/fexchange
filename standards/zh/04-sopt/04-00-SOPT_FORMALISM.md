# 04-00-SOPT_FORMALISM

本文件定义 SOPT 的物理主干与最小可编程接口。
本文件不定义态构造细节，不定义分层实现细节（见 `./standards/en/04-sopt/04-01-PRECOMPUTE.md`、`./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md` 与 `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`）。
磁盘 I/O 布局与格式由 `./standards/en/05-io/05-00-IO.md` 统一定义。
写作形式遵循 `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`。
运行时阈值、确定性线性代数与全局输入闸门遵循
`./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md`。

## 0) 分层定义（MUST）
分层语义：
- $L0$：Fock 基原始跃迁层（仅在标准 Fock 基上构造跃迁元；不含 site 标签 $i/j$，不依赖外部态文件）。
- $L1$：局域跃迁顶点构造层（对 $f^{n+1}/f^{n-1}$ 中间态扇区做基变换，并将 $f^n$ 腿投影到 SOC 下最低能 LSJM 子空间）。
- $L2$：投影后的路线因子构造层（由 $A/B$、site 绑定、hopping 收缩和低能投影矩阵 $W$ 得到 $M_A/M_B$）。
- $L3$：带分母的固定 Kramers 基输出层（由投影后的 $M_A/M_B$ 生成单通道
  $\mathrm{Heff}^{(\mu)}$）。

执行顺序：
- 默认且强制顺序为 $L0 \to L1 \to L2 \to L3$，并以此作为最终输出。

职责边界：
- $L0/L1$ 的公式细节放在 `./standards/en/04-sopt/04-01-PRECOMPUTE.md`。
- $L2/L3$ 的公式细节放在 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`。
- SOPT 后处理的赝自旋-$\tfrac{1}{2}$ 映射放在 `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`。
- 本文件只保留跨层契约与共用符号。
- 站点标签 $i/j$ 从 $L2$ 才进入；$L0/L1$ 都是 site-agnostic 定义。
- 跃迁方向约定：计算与缓存统一采用“低电子数扇区 $\to$ 高电子数扇区”的 $f^\dagger$ 形式；反向跃迁由厄米共轭得到。
- 运行粒度：一次运行只计算一个 bond；$\mu$ 是该次运行的 bond 标签（不是输入张量轴）。

## 0.1) 命名规范（MUST）
- Fock 基态索引：$\alpha,\beta,\gamma,\chi$。
- $f^n$ 的 SOC 最低能 LSJM 子空间索引：$j_1,j_2,j_3,j_4$（注意这不是 site-$j$ 标签）。
- Kramers 低能态索引：$a,b,c,d$。
- 单 site 中间态索引：$u,v,r,s$。
- 两 site 中间态复合索引：$m,n$，其中路线 A 用 $m=(u,v)$，路线 B 用 $n=(r,s)$。
- 通用单站点轨道指标：$\kappa$（仅在未绑定 site 时使用）。
- site 绑定轨道指标：$p,p'$ 属于 site-$i$；$q,q'$ 属于 site-$j$。
- 作用域说明：本处 `p,p',q,q'` 是 SOPT 内部（约化后）记号，
  与 `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md` 中配体记号 `(o,p)`
  不同。

## 0.2) 核心符号（精简，MUST）
- 投影与哈密顿量：$P,Q,H_0,H_{\mathrm{hop}}^{(\mu)},H_{\mathrm{eff}}^{(2)}$。
- 能量/分母记号：$E_0,E_n,E_{uv},E_{rs},\Delta_{uv},\Delta_{rs}$，其中
  $\Delta_{uv}=E_0-E_{uv}$、$\Delta_{rs}=E_0-E_{rs}$。
- 有效张量：$h_{cd,ab}^{(\mu)}$ 与 $\mathrm{Heff}_{cd,ab}^{(\mu)}$（固定单个 $\mu$ 通道）。
- 费米算符：$f,\hat f,\hat N_i$，以及嵌入规则
  $\hat f_{ip}=f_{ip}\otimes I$、$\hat f_{jq}=(-1)^{\hat N_i}\otimes f_{jq}$。
- hopping 与缓存：$t_{pq}^{(\mu)},t_{p'q'}^{(\mu)\ast},M_A,M_B,W,U^{n,\mathrm{soc0}}$。
- 代码别名约定：代码块中的 `t_mu[p,q]` 仅是实现变量名，语义等价于数学记号 $t_{pq}^{(\mu)}$。

## 0.2.1) SOC 最低子空间精确定义（MUST）
SOPT 中引用的"$f^n$ SOC 最低能 LSJM 子空间"（也称"$f^n$ LSJM 基态多重态"）精确定义如下：

1. 确定基态 LS 项：LSMS（模块 03-00）中库仑本征值 $\varepsilon_{\alpha}$ 最小的 $(\alpha_0, L_0, S_0)$ 项。
   若同一 $(L_0,S_0)$ 有多个 $\alpha$，取 $\varepsilon_\alpha$ 最小者；
   若数值上相等，按 `./standards/en/01-core/01-01-STATE_VECTOR.md` 的确定性规则处理。
2. 确定基态 $J$：在基态 LS 项内应用 Hund 第三规则：
   - $n \le 2\ell$（少于半满，$\ell=3$）：$J_0 = |L_0-S_0|$。
   - $n > 2\ell$（超过半满）：$J_0 = L_0+S_0$。
   - $n = 2\ell+1 = 7$（恰好半满）：$L_0=0$，故 $J_0 = S_0$。
3. SOC 最低子空间由 $2J_0+1$ 个态组成：

Math:
$$
\mathcal S_{\mathrm{soc0}}^{(n)}
= \bigl\{\lvert \alpha_0,L_0,S_0,J_0,M\rangle : M=-J_0,\ldots,J_0\bigr\}.
$$

4. $U^{n,\mathrm{soc0}}\in\mathbb C^{d_{\mathrm{fock}}\times(2J_0+1)}$ 的列即为上述态，
   其列序继承自 `./standards/en/03-spectrum/03-01-LSJM.md` 的正则 LSJM 序。

Code form:
```text
ground_term = argmin_alpha_L_S(E_coulomb[alpha,L,S])
(alpha0, L0, S0) = ground_term
if n <= 2*ell: J0 = abs(L0 - S0)
elif n > 2*ell: J0 = L0 + S0
n_j = 2*J0 + 1
U_n_soc0 = V_lsjm_fock[:, columns_for(alpha0,L0,S0,J0)]
U_n_soc0.shape = (dim_fock, n_j)
```

Validation:
- $n_j = 2J_0+1$ 须记录在元数据中并传递给下游模块。
- $U^{n,\mathrm{soc0}\dagger}U^{n,\mathrm{soc0}}=I$（在 `eps_orth` 以内）。
- 基态项识别须跨运行确定性一致。

## 0.2.2) 中间态能量构造（MUST）
`E_u` 中间态能量取自相邻电子数扇区（$f^{n+1}$ 和 $f^{n-1}$）的 LSJM 本征值
（$H_{\mathrm{int}}+H_{\mathrm{soc}}$）。
按项目约定，基态参考固定为 $E_0=0$，运行时能量必须已经在该参考下给出（不得再做额外平移）。

Math:
$$
E_u^{(n+1)}[u] = E^{\mathrm{LSJM}}_{n+1}(u),
$$

$$
E_v^{(n-1)}[v] = E^{\mathrm{LSJM}}_{n-1}(v),
$$

其中：
- $E^{\mathrm{LSJM}}_{m}(u) = \sum_{k} F^k\cdot\mathrm{coef\_F_k}[u] + \zeta\cdot\mathrm{coef\_zeta}[u]$
  为扇区 $f^m$ 态 $u$ 的 LSJM 总能量。

Code form:
```text
E_u_np1   = E_lsjm_np1              # f^{n+1} 所有态
E_u_nm1   = E_lsjm_nm1              # f^{n-1} 所有态
```

Index:
- $u$：遍历 $f^{n+1}$ 所有 LSJM 态。
- $v$：遍历 $f^{n-1}$ 所有 LSJM 态。

Validation:
- 不对分支能量 $E_u^{(n+1)}$ 或 $E_v^{(n-1)}$ 施加必须为正的约束。
- 接近零的分母（$|\Delta_{uv}|<\varepsilon_{\mathrm{zero}}$）或非有限数值必须报 `FXE-NUM-002`。
- 构造需要扇区 $n-1$ 与 $n+1$ 的 $F^0,F^2,F^4,F^6,\zeta$。

## 0.3) 变量类别（MUST）
定义：
- 输入变量（Input）：来自其它模块/接口，或由外部调用者传入。
- 中间变量（Intermediate）：仅在当前模块计算过程中临时使用，不作为跨模块对外接口。
- 输出变量（Output）：当前模块对外提供给下游模块/调用者的变量。

大模块（SOPT 全链路，$L0 \to L3$）：
- 输入变量：`E_u`、`U_np1`、`U_n_soc0`、`U_nm1`、`t_mu`、`W`、`labels_abcd`、`labels_order_id`。
- 中间变量：`X`、`Y`、`A`、`B`、`E_uv`、`E_rs`、`M_A`、`M_B`。
- 输出变量：`h_mu_abcd`、`Heff_mu_abcd`。

小模块（按层级）：
- $L0$：输入 `{}`；中间 `{sign/workspace}`；输出 `{X, Y}`。
- $L1$：输入 `{X, Y, U_np1, U_n_soc0, U_nm1}`；中间 `{workspace}`；输出 `{A, B}`。
- $L2$：输入 `{A, B, t_mu, W}`；中间 `{workspace}`；输出 projected local basis 中的 `{M_A, M_B}`。
- $L3$：输入 `{M_A, M_B, E_u, labels_abcd, labels_order_id}`；中间
  `{h_mu_abcd}`；输出 `{h_mu_abcd, Heff_mu_abcd}`。

## 1) 核心 SOPT 规则
MUST:
- 有效相互作用必须使用二阶投影微扰形式。
- 每个键/方向通道 $\mu$ 必须生成一个 $h^{(\mu)}_{cd,ab}$。

Math:
$$
H_{\mathrm{eff}}^{(2)}
= -P H_{\mathrm{hop}} Q (QH_0Q-E_0)^{-1} Q H_{\mathrm{hop}} P.
$$

Math:
$$
h^{(\mu)}_{cd,ab}
= \sum_n
\frac{
\langle c,d \mid H_{\mathrm{hop}}^{(\mu)} \mid n\rangle
\langle n \mid H_{\mathrm{hop}}^{(\mu)} \mid a,b\rangle
}{E_0-E_n}.
$$

Code form:
```text
h_mu[c,d,a,b] = sum_n hop_out[c,d,n] * hop_in[n,a,b] / (E0 - En[n])
```

Index:
- $a,b,c,d$：两站点低能投影态。
- $n$：两站点 LSJM 中间态。
- $\mu$：键/方向通道标签。

Validation:
- `hop_out` 与 `hop_in` 在 $n$ 维度必须对齐。
- `h_mu` 与投影基顺序必须一一对应。

## 2) 分母与参考能规则
MUST:
- 只能使用分母约定 $E_0 - E_n$。
- f 壳层计算中固定 $E_0 = 0$。
- 禁止在同一实现中混用其它符号约定。

Math:
$$
E_0 = 0,\qquad
\frac{1}{E_0-E_n} = -\frac{1}{E_n}.
$$

Code form:
```text
denom[n] = -En[n]
```

Index:
- $E_n$：本规范 SOPT 约定下的中间态能量。

Validation:
- 必须做分母符号抽样核验（与直接微扰表达对比）。

## 3) 中间态与两条路线规则
MUST:
- 必须包含两条虚跃迁路线：addition/removal 与 removal/addition。
- 本版本低能态固定在 $f^n$ 扇区，并且 $f^n$ 只保留 SOC 下最低能 LSJM 子空间（即 $f^n$ 的 LSJM 基态多重态）。

Math:
$$
\lvert a,b\rangle \equiv \lvert a\rangle_i^{(n)} \otimes \lvert b\rangle_j^{(n)},
\qquad
\lvert c,d\rangle \equiv \lvert c\rangle_i^{(n)} \otimes \lvert d\rangle_j^{(n)}.
$$

Math:
$$
\text{路线 A: }\lvert u,v\rangle=
\lvert u\rangle_i^{(n+1)}\otimes \lvert v\rangle_j^{(n-1)},
\quad
E_{uv}=E_i^{(n+1)}(u)+E_j^{(n-1)}(v).
$$

Math:
$$
\text{路线 B: }\lvert r,s\rangle=
\lvert r\rangle_i^{(n-1)}\otimes \lvert s\rangle_j^{(n+1)},
\quad
E_{rs}=E_i^{(n-1)}(r)+E_j^{(n+1)}(s).
$$

Code form:
```text
routeA_state = (u,v)
routeB_state = (r,s)
```

Index:
- $u,v,r,s$：仅用于中间态标签。

Validation:
- 路线 A/B 的中间态集合必须满足粒子数扇区约束。

## 4) 费米子分级规则
MUST:
- 站点顺序固定为 $i < j$。
- site-$j$ 算符必须采用分级嵌入。

Math:
$$
\hat f_{ip}=f_{ip}\otimes I,\qquad
\hat f_{jq}=(-1)^{\hat N_i}\otimes f_{jq}.
$$

Math:
$$
\langle u,v\rvert \hat f_{ip}^{\dagger}\hat f_{jq}\lvert a,b\rangle
=(-1)^{n_a}
\langle u\rvert f_{ip}^{\dagger}\lvert a\rangle
\langle v\rvert f_{jq}\lvert b\rangle.
$$

Math:
$$
\langle c,d\rvert \hat f_{jr}^{\dagger}\hat f_{is}\lvert u,v\rangle
=(-1)^{n_c}
\langle c\rvert f_{is}\lvert u\rangle
\langle d\rvert f_{jr}^{\dagger}\lvert v\rangle.
$$

Code form:
```text
f_i(p) = kron(f_i_p, I_j)
f_j(q) = kron(parity_i, f_j_q)
```

Index:
- $n_a,n_c$：低能态在 site-$i$ 的粒子数。

Validation:
- 必须验证跨站点反对易关系。
- 在固定 $f^n$ 子空间，应验证总奇偶因子化为 $+1$。

## 5) hopping 形式规则
MUST:
- $p,p'$ 仅是 site-$i$ 轨道指标。
- $q,q'$ 仅是 site-$j$ 轨道指标。
- $p,q$ 表示 $j \to i$；$p',q'$ 表示 $i \to j$。
- 本节只定义 hopping 形式，不定义路线因子、收缩或基变换细节（见 `04-01` 与 `04-02`）。

Math:
$$
H_{\mathrm{hop}}^{(\mu)}
= \sum_{p q} t_{pq}^{(\mu)}\, f_{ip}^{\dagger} f_{jq}
+ \sum_{p' q'} t_{p'q'}^{(\mu)\ast}\, f_{jq'}^{\dagger} f_{ip'}.
$$

Code form:
```text
H_hop_mu = sum_{p,q} t_mu[p,q] * f_i_dag[p] * f_j[q]
         + sum_{p2,q2} conj(t_mu[p2,q2]) * f_j_dag[q2] * f_i[p2]
```

Index:
- $t_{pq}^{(\mu)}$：site-$j$ 到 site-$i$ 的 hopping 幅度。
- $t_{p'q'}^{(\mu)\ast}$：反向（site-$i$ 到 site-$j$）的共轭幅度。

Validation:
- $H_{\mathrm{hop}}^{(\mu)}$ 必须满足厄米性。
- 实现中不得出现 $p/q$ 跨站点误用。

## 6) 最小 I/O 与运行时校验
MUST:
- 最终执行顺序在本文件中固定为 $L0 \to L1 \to L2 \to L3$。
- 本文件只定义 SOPT 局部分层（`L0..L3`）。
- 全局运行窗口（`LMSM..L3`）由
  `./standards/en/05-io/05-04-RUN_INPUT.md` 与
  `./standards/en/05-io/05-00-IO.md` 统一定义。
- $L0$ 必须可由代码在运行时直接生成，不要求外部文件输入。
- $L0$ 输出必须是与 site 无关的统一局域跃迁元；site 差异不得在 $L0$ 引入。
- 缓存接口中不得同时存储一对互为共轭的正反向跃迁元；仅允许存“低 $\to$ 高”方向，反向按需共轭恢复。
- 大模块输入必须包含：`E_u`、`U_np1`、`U_n_soc0`、`U_nm1`、`t_mu`、`W`、`labels_abcd`、`labels_order_id`。
- 大模块输出必须包含：`h_mu_abcd`、`Heff_mu_abcd`。
- `Heff_mu_abcd` 可作为 04-03 模块输入，做自旋-$\tfrac{1}{2}$ 后处理映射。
- 大模块中间变量（`X/Y/A/B/E_uv/E_rs/M_A/M_B`）默认不得作为最终对外接口。
- 若采用分层执行，允许将上游层输出作为下游层输入；该情形下它们仍属于大模块语义下的中间变量。
- $E_{uv}/E_{rs}$ 仅是 $L3$ 内部组合量，由 `E_u` 计算得到，不得作为对外持久化输出。
- 最终对外输出必须以 $a,b,c,d$ 为索引语义；$u,v,r,s$ 仅作为中间态内部索引，不作为最终输出接口。
- `labels_abcd` 的正则顺序固定为字典序 `(a,b,c,d)`，等价嵌套循环：
  `for a in 0..n_k-1, for b in 0..n_k-1, for c in 0..n_k-1, for d in 0..n_k-1`。
  该顺序标识固定为 `labels_order_id = "abcd_lex_v1"`。
- 若做标签子集/截断，必须保持该相对顺序，并在 metadata 中同时记录
  `labels_abcd` 与 `labels_order_id`。

Math:
$$
\text{Final order: } L0 \rightarrow L1 \rightarrow L2 \rightarrow L3.
$$

Math:
$$
H_{\mathrm{eff},\,cd,ab}^{(\mu)}=h_{cd,ab}^{(\mu)}.
$$

Code form:
```text
module_inputs       = {E_u, U_np1, U_n_soc0, U_nm1, t_mu, W, labels_abcd, labels_order_id}
module_internal     = {X, Y, A, B, E_uv, E_rs, M_A, M_B}
module_outputs      = {h_mu_abcd, Heff_mu_abcd}
submodule_handoff   = {L0: X/Y, L1: A/B, L2: M_A&M_B, L3: h_mu_abcd&Heff_mu_abcd}
labels_order_id     = "abcd_lex_v1"
```

Index:
- $u,v,r,s$：中间态内部索引，仅用于 $L1/L2/L3$ 求和与缓存。
- $a,b,c,d$：最终低能输出索引，必须作为对外接口。

Validation:
- `04-01` 与 `04-02` 的定义必须与本文件第 1-5 节保持一致。
- Hermitian 检查：$\mathrm{Heff}^{(\mu)}=\left(\mathrm{Heff}^{(\mu)}\right)^\dagger$（容差内）。
- 零 hopping 检查：$t=0 \Rightarrow \mathrm{Heff}^{(\mu)}=0$。
- 所有收缩步骤的 shape/dtype 必须一致。
