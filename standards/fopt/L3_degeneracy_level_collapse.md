# L3 简并能级合并(degeneracy-level collapse)：逐 path 的精确加速

本文档定义 FOPT L3 的**第二个化简(公式②)**，与 spectator-$\delta$ 结构化简
(`standards/fopt/L3_full_cluster_expansion.md`，公式①)**正交**：

- **公式①** 把完整四站点 cluster 求和化成 local-index einsum —— 决定 **算哪一个张量收缩**。
- **公式②(本文)** 把该 einsum 中对 $f^{n\pm1}$ 中间态的求和，按**简并能级**先合并再过 resolvent
  —— 决定 **怎么把那个收缩算得又快又省**，且是 **严格精确(非近似)** 的重排。

综合收益(`fexchange/fopt/contraction.py:19-53`)：$f^5$ 下 $n_u\!\sim\!3003\to n_{\rm lev}\!\sim\!293$，
约 **100× 加速、30× 省内存**，结果 **bit-exact**。

> 本文与①一样**逐 path 组织**(§4)。32 条 path 归为 5 个 process,$P_2/P_4/P_5$ 各含 A/B 两 pattern；
> **同一 process+pattern 的所有 path 共享同一收缩拓扑与同一组被合并的轴**，只在输入顶点(site/ligand)
> 与 cluster sign 上不同。所以"逐 path"在②里就是"逐 process(+pattern)"。

---

## 0. 统一记号

沿用①：cluster $(f_1,f_2,p_1,p_2)$，顶点 $A_{r\lambda}=V_+$、$B_{r\lambda}=V_-=A^\dagger$。中间 $f$-sector：

$$
A\in f^{n+1}\ \text{multiplets},\qquad B\in f^{n-1}\ \text{multiplets}.
$$

- $M_1,M_2,M_3,M_4$：①化简后留下的投影顶点张量(`V_n[:,:,:,0]` 等切片)，**携带该中间态的全部
  $M$ 量子数依赖**(外部 doublet 指标 $a,b,c,d$ 与 ligand 指标 $G,H,K\in p^{5}/p^{4}$)。
- resolvent $w(E)=-\dfrac{1}{E-kE_0}$，$k\in\{1,2\}$。代码 `_resolvent`(`contraction.py:252-266`)：
  返回 `-1/denom`，对 $|E-kE_0|<\texttt{EPS\_ZERO}=10^{-12}$ 或非有限值**硬报错**。
- $L,M$：被合并后的**能级**索引(代码里 einsum 的 `L/M` 或 `i/j`)。

**核心观察：$w$ 只通过能量 $E$ 依赖中间态，不依赖其 $M$ 量子数。**

---

## 1. 简并结构

$H_{\rm int}+H_{\rm soc}$ 的本征态按 $(\alpha,L,S,J)$ 多重态组织；多重态内 $2J+1$ 个 $|\dots,M\rangle$ 态
**严格简并**。四阶求和对 $A$(遍历整个 $f^{n+1}$ sector)中，**同能级的所有态共享同一个 $w$**。

---

## 2. 核心恒等式(分配律)

$$
\boxed{\;
\sum_{A} M[A]\,w(E_A)\,M'[A]
=
\sum_{L} w(e_L)\Big(\sum_{A\in L} M[A]\,M'[A]\Big)
\;}
$$

$A\in L\Rightarrow E_A=e_L$，$w(E_A)=w(e_L)$ 是级内常数提出。右边内层 $\sum_{A\in L}$ 就是 **per-level Gram
预求和**(代码 `_sum_by_level`)，在乘 resolvent **之前**完成。精确重排,非近似。

---

## 3. 两-sector / 多维 resolvent 下仍精确(通用论证)

四阶 resolvent 常同时耦合两个被合并的 $f$ 轴(及未合并的 ligand 轴)。一般地,对固定的
$(L,M,\text{ligand 指标})$，每个 resolvent 因子都只看 $e_L,e_M$ 和**显式**的 ligand 能量,因此是级内常数；
于是对 $A\in L$ 与 $B\in M$ 的两个级内求和**各自独立 factor 出来**，分别成为下文的 $\mathrm{TA},\mathrm{TB}$。
**resolvent 的维数(2-D/3-D/4-D)与该因子化无关** —— 只要它耦合的两个轴在级内能量恒定。

---

## 4. 逐 process(path)的能级合并

每个 process 给出：**被合并的 $f$ 轴**(`_sum_by_level` 作用在 einsum 的轴 0)、$\mathrm{TA}/\mathrm{TB}$ 的
级内 Gram 预求和、**化简后最终 einsum**(代码原样)、resolvent 维数、path 数与 pattern 判据、代码位置。
einsum 大写 `A,B`=未合并的逐态 $f$ 索引(只出现在 $\mathrm{TA}/\mathrm{TB}$ 内部),`L,M`(或 `i,j`)=合并后的
能级索引,`G,H,K`=ligand,`a,b,c,d`=doublet。

### 4.1 $P_1$ —— alternating 单 ligand(`contraction.py:269-351`)
- 被合并：$A\in f^{n+1}\to L$(`levels_np1`)，$B\in f^{n-1}\to M$(`levels_nm1`)。
- 级内 Gram:
  ```
  TA = sum_by_level( einsum("AHa,AGc->AHaGc", M4*, M1), levels_np1 )
  TB = sum_by_level( einsum("bHB,dGB->BbHdG", M3 , M2*), levels_nm1 )
  ```
- 化简后 einsum:`"LHaGc,LG,LM,LH,MbHdG->abcd"`(TA, G_s1, G_s2, G_s3, TB)。
- resolvent:全 2-D —— `G_s1[L,G]`, `G_s2[L,M]`, `G_s3[L,H]`。
- paths:4 条(`_PROCESS1_PATHS`，单 pattern)。

### 4.2 $P_2$ —— onion 单 ligand,经 $p^4$ 中央态(`contraction.py:354-442`)
两个中间 $f$-sector **都是 $f^{n+1}$**,故 $\mathrm{TA},\mathrm{TB}$ **都用 `levels_np1`**。
- **Pattern A**(FIFO,`r_3=r_1`):
  ```
  TA = sum_by_level( einsum("AHbK,AGc->AHbKGc", M3*, M1), levels_np1 )
  TB = sum_by_level( einsum("BKa,BHdG->BKaHdG", M4*, M2), levels_np1 )
  einsum("LHbKGc,LG,LMH,MK,MKaHdG->abcd", TA,G_s1,G_s2,G_s3,TB)
  ```
- **Pattern B**(LIFO):
  ```
  TA = sum_by_level( einsum("AKa,AGc->AKaGc", M4*, M1), levels_np1 )
  TB = sum_by_level( einsum("BHbK,BHdG->BHbKdG", M3*, M2), levels_np1 )
  einsum("LKaGc,LG,LMH,LK,MHbKdG->abcd", TA,G_s1,G_s2,G_s3,TB)
  ```
- resolvent:`G_s2[L,M,H]` 是 **3-D**(中央态多出 $p^4$ 光子 $\gamma_2$)。
- paths:8 条(`_PROCESS2_PATHS`，`pattern=A iff r3==r1`)。

### 4.3 $P_3$ —— alternating 双(交叉)ligand(`contraction.py:445-519`)
收缩拓扑与 $P_1$ **完全相同**(同 einsum)，仅两个 ligand 能量不同($G_{s1}$ 用 lig$_a$、$G_{s3}$ 用 lig$_b$)。
- 被合并:$A\to L$(`levels_np1`)，$B\to M$(`levels_nm1`)。
  ```
  TA = sum_by_level( einsum("AHa,AGc->AHaGc", M4*, M1), levels_np1 )
  TB = sum_by_level( einsum("bHB,dGB->BbHdG", M3 , M2*), levels_nm1 )
  einsum("LHaGc,LG,LM,LH,MbHdG->abcd", TA,G_s1,G_s2,G_s3,TB)
  ```
- resolvent:全 2-D（`G_s1[L,G]` lig$_a$,`G_s2[L,M]`,`G_s3[L,H]` lig$_b$）。
- paths:4 条(`_PROCESS3_PATHS`，单 pattern)。

### 4.4 $P_4$ —— onion 双 ligand(crossed,交叉借还)(`contraction.py:522-616`)
两中间 $f$-sector **都是 $f^{n+1}$** → $\mathrm{TA},\mathrm{TB}$ 都用 `levels_np1`。**含全框架唯一的 4-D resolvent。**
- **Pattern A**(FIFO):
  ```
  TA = sum_by_level( einsum("AHb,AGc->AHbGc", M3*, M1), levels_np1 )
  TB = sum_by_level( einsum("BGa,BHd->BGaHd", M4*, M2), levels_np1 )
  einsum("iHbGc,iG,ijGH,jG,jGaHd->abcd", TA,G_s1,G_s2,G_s3,TB)   # i=L, j=M
  ```
- **Pattern B**(LIFO):
  ```
  TA = sum_by_level( einsum("AHa,AGc->AHaGc", M4*, M1), levels_np1 )
  TB = sum_by_level( einsum("BGb,BHd->BGbHd", M3*, M2), levels_np1 )
  einsum("iHaGc,iG,ijGH,iH,jGbHd->abcd", TA,G_s1,G_s2,G_s3,TB)
  ```
- resolvent:`G_s2[A,B,G,H]` 是 **4-D**($S_2$ 同时激发两个 $f$ 站点与两个 ligand $p^5$)。
- paths:8 条(`_PROCESS4_PATHS`，`pattern=A iff r_c==r_a`)。

### 4.5 $P_5$ —— onion 双 ligand(uncrossed,各借各还)(`contraction.py:619-689`)
两中间 $f$-sector **都是 $f^{n+1}$** → 都用 `levels_np1`；与 $P_4$ 共享 4-D 的 $S_2$,仅 $S_3$ 不同。
- **Pattern A**(FIFO,native 输出轴 `->bacd`,与①的 §6.8 Pattern A 对应):
  ```
  TA = sum_by_level( einsum("AGa,AGc->AGac", M3*, M1), levels_np1 )
  TB = sum_by_level( einsum("BHb,BHd->BHbd", M4*, M2), levels_np1 )
  einsum("iGac,iG,ijGH,jH,jHbd->bacd", TA,G_s1,G_s2,G_s3,TB)
  ```
- **Pattern B**(LIFO,①的 §6.8 Pattern B,$G_3\equiv G_1$):
  ```
  TA = sum_by_level( einsum("AGa,AGc->AGac", M4*, M1), levels_np1 )
  TB = sum_by_level( einsum("BHb,BHd->BHbd", M3*, M2), levels_np1 )
  einsum("iGac,iG,ijGH,iG,jHbd->abcd", TA,G_s1,G_s2,G_s3,TB)
  ```
- resolvent:`G_s2[A,B,G,H]` 4-D。
- paths:8 条(`_PROCESS5_PATHS`，`pattern=A iff r_c==r_a`)。

> **每条 path 的 sign**(与能级合并正交,沿用①的 sign 表):
> `_cluster_sign(n_ele, sign_const) = (-1)^(10*n_ele + sign_const)`(`contraction.py:757-758`)，
> `sign_const` 在各 `_PROCESS{1..5}_PATHS` 表里(`:700-754`)。

**逐 path 小结**:被合并的轴只取决于 process(P1/P3 的 $f^{n-1}$ 腿用 `levels_nm1`，其余全 `levels_np1`);
A/B pattern 只改 $\mathrm{TA}/\mathrm{TB}$ 里 $M_3/M_4$ 的配对与 $G_{s3}$ 的索引,**不改"按能级合并"这件事**;
具体 8/4 条 path 只改输入顶点(site/ligand)与 sign,**复用同一收缩与同一合并**。

---

## 5. 收缩什么、不收缩什么

- **被合并**:仅 $f$ 多重态指标 $A\in f^{n+1}$ / $B\in f^{n-1}$，永远是 `_sum_by_level` 的 **axis 0**。
- **spectator(不合并)**:ligand $G,H,K\in p^{5}/p^{4}$ 与 doublet $a,b,c,d$，随 $\mathrm{TA}/\mathrm{TB}$ 保留到最终 einsum。
- 结构保护:`_sum_by_level` 形状守卫(`values.shape[0]!=levels.n_states -> ValueError`，`:115-118`)使"误合并 ligand 轴"不可能。

---

## 6. 代码映射(机制层)

| 概念 | 代码 | 位置 |
|---|---|---|
| 能级分组对象 | `_EnergyLevels`(`energy`/`order`/`starts`/`n_states`) | `contraction.py:56-73` |
| 切级(gap-to-start) | `_energy_levels(E, tol=EPS_EIG_CLUSTER)` | `:76-104` |
| per-level Gram 预求和 | `_sum_by_level` = `np.take(order)` + `np.add.reduceat(starts)` | `:107-120` |
| resolvent | `_resolvent` 返回 `-1/denom`，守卫 `|denom|<EPS_ZERO` | `:252-266` |
| 单次构建并下传 | `build_L3` 在 `:798-799` 构建 `levels_np1/levels_nm1`，各 process 另有 fallback(`:302/304,397,483/485,565,658`) | `:761,798-799` |

**切级规则**:某能量与**本级首个(最小)成员**之差 $>\texttt{tol}$ 时开新级(gap-to-start，**非** single-linkage)，
级直径 $\le\texttt{tol}$,级能量取**簇均值**。与 `spectrum/ion_ed._clusters`(`ion_ed.py:287-291`)**逻辑一致、同 tol、同均值**，
保证 FOPT 侧分级 = 产生 canonical 能量时的分级。

---

## 7. 精确前提(exactness preconditions)

级内若有真实 spread,级均 resolvent 才会偏离逐态。生产路径下 **级内 spread $\equiv 0$**,故 bit-exact:

1. **ED 路径**:`ion_ed` 把简并簇内**每个**成员赋同一 `e_group=mean(evals[lo:hi])`(`ion_ed.py:144-146,173`)→ spread 严格 0。
2. **RS 路径**:中间态能量线性于 `coef_F0/F2/F4/F6/zeta`(`spectrum/energy.py:69-78`)，只依赖 $(\alpha,L,S,J)$、与 $M$ 无关 → 多重态 $2J+1$ 态逐比特相同。
3. **数值窗口**:$\texttt{EPS\_EIG\_CLUSTER}=10^{-10}\gg\texttt{EPS\_ZERO}=10^{-12}$。仅"非规范化喂入 + 真实 spread 落在 $(10^{-12},10^{-10})$"才会偏离/翻转近零守卫;当前**无任何代码路径**产生(审计注入 $9\times10^{-11}$ → $\sim1.1\times10^{-11}$ 偏差,证明前提 load-bearing 但不可达)。
4. **Graceful degradation**:若给中间 sector 加 CEF 解除简并 → 更多更小的级(极限:每态一级,合并=恒等)→ **仍精确**,只是收缩比下降。

---

## 8. 数值 / 测试验证

- **独立 oracle**:`tests/test_fopt_l2.py` 的 `_ps_process1..5`(`:404-455`)是**逐态**四阶 einsum(无能级合并)，结构上独立于合并实现。
- **回归**:`test_level_collapse_matches_reference_per_path_for_exact_degeneracies`(多态简并级)与 `..._without_degeneracies`(级=单态、合并=恒等)均 `rtol=1e-12`、**逐 process / 逐 path** 匹配 oracle。全套 `python -m pytest tests/test_fopt_l2.py` → **13 passed**。
- **审计独立复算**:真·多态简并级下,合并 vs 逐态最大差 $\sim10^{-16}$(P4-A 1.1e-16, P5-A 1.8e-16, P2-A 2.5e-16, P1 7.8e-16)。
- **已知测试缺口(minor,不可达)**:无用例构造"级内真实微小 spread(非逐比特相同)"来打 §7 的 precondition;可选补一个 spread $\in(\texttt{EPS\_ZERO},\texttt{EPS\_EIG\_CLUSTER})$ 的回归。

---

## 9. 与公式① 的关系

$$
\underbrace{\text{完整 cluster sum}}_{\text{四阶 ligand 介导超交换}}
\xrightarrow[\text{①: spectator-}\delta]{}
\underbrace{\text{local-index einsum(逐 path)}}_{\text{算哪个张量收缩}}
\xrightarrow[\text{②: degeneracy-level collapse(逐 path)}]{}
\underbrace{\text{per-level Gram + resolvent}}_{\text{把该收缩算快,bit-exact}}.
$$

①定**结构**(每条 path 收缩什么),②在每条 path 的结构上把 $f$ 中间态按**能级**合并。二者正交、各有独立背书:
① doc↔code 审计 + per-state oracle;② 分配律 + 同一 oracle 的 degenerate / non-degenerate 用例。
