# L3 五类 path 的推导与简并加速

本文档把 L3 四阶 ligand 介导超交换的两层化简**合并、逐 path 组织**:每条 path 先做
**spectator-δ 结构推导**(完整四站点 cluster sum → local-index einsum,即"最简单的精确形式"),
再做 **degeneracy-level collapse**(按中间态简并能级合并,精确加速)。本文取代并合并
`L3_full_cluster_expansion.md`(①)与 `L3_degeneracy_level_collapse.md`(②)。

- **①(结构化简)**:用每个 hopping 的 spectator Kronecker-δ **精确**消去不变的 block 指标,把完整四站点
  cluster sum 化成逐态 einsum。**不是**把四站点近似成两站点;Type 2 的 $p_1^4$、Type 4/5 的 $p_1^5p_2^5$
  中央态都显式保留。
- **②(简并加速)**:resolvent 只通过**能量**依赖中间态,原子 $H_{\rm int}+H_{\rm soc}$ 让每个
  $(\alpha,L,S,J)$ 多重态严格简并,故按能级合并 $f$ 中间态、再过 resolvent,与逐态求和 **bit 等价**
  (分配律)。$f^5$ 下约 **100× 加速、30× 省内存**。

---

## 0. 统一记号与完整链式公式

### 0.1 记号

四个 block 固定顺序 $(f_1,f_2,p_1,p_2)$。cluster 基
$|i_1,i_2,\rho_1,\rho_2\rangle=|f_1;i_1\rangle\otimes|f_2;i_2\rangle\otimes|p_1;\rho_1\rangle\otimes|p_2;\rho_2\rangle$。
外部低能态 $|I\rangle=|c,d,\Omega_1,\Omega_2\rangle$、$|F\rangle=|a,b,\Omega_1,\Omega_2\rangle$;
$a,c\in f_1^n$ doublet,$b,d\in f_2^n$ doublet,$\Omega_1=p_1^6,\ \Omega_2=p_2^6$。

顶点:$A_{r\lambda}=V_+(r,\lambda):p_\lambda\to f_r$;$B_{r\lambda}=V_-=A_{r\lambda}^\dagger:f_r\to p_\lambda$。
L2 forward vertex
$M_{r\lambda}^{(N_f,N_p)}[u,\eta,v,\xi]=\langle f_r^{N_f+1},u;\,p_\lambda^{N_p-1},\eta|A_{r\lambda}|f_r^{N_f},v;\,p_\lambda^{N_p},\xi\rangle$;
$B$ 的矩阵元用 $M^\ast$。代码切片 $M_1,M_2,M_3,M_4=\texttt{V\_n[:,:,:,0]}$。

每步乘 cluster embedding sign $s_k$;一条 path 的 sign $\sigma=s_1s_2s_3s_4$,代码
$\text{sign}=(-1)^{10\,n_{\rm ele}+\text{sign\_const}}$(`_cluster_sign`,`contraction.py:757-758`)。

**einsum 指标约定**(§N.3):大写 `A,B` = 未合并的逐态 $f$ 索引(只在 TA/TB 内部);`L,M`(或 `i,j`)=
合并后的**能级**索引;`G,H,K` = ligand($p^5/p^4$);`a,b,c,d` = doublet。

### 0.2 完整四站点 path-chain 公式

对任意四跳 path $h=(h_1,h_2,h_3,h_4)$:

$$
H[h]_{ab,cd}=\sum_{S_1,S_2,S_3}
\langle F|h_4|S_3\rangle\,G(S_3)\,\langle S_3|h_3|S_2\rangle\,G(S_2)\,\langle S_2|h_2|S_1\rangle\,G(S_1)\,\langle S_1|h_1|I\rangle,
$$

完整中间态 $S_1=|u_1,u_2,\rho_1,\rho_2\rangle$、$S_2=|v_1,v_2,\sigma_1,\sigma_2\rangle$、
$S_3=|w_1,w_2,\tau_1,\tau_2\rangle$,$G$ 为 resolvent $w(E)=-1/(E-kE_0)$。下面每条 path 都从此式出发,
逐顶点的 spectator-δ 把它精确约化(§N.2),再按简并能级合并(§N.3)。

---

## 1. Path-1 / Process P1 — alternating 单 ligand

### 1.1 Path 与 charge sequence

四个顶点(取自① §2.1)为

$$
h_1=A_{11},\qquad h_2=B_{21},\qquad h_3=A_{21},\qquad h_4=B_{11},
$$

即

$$
p_1\to f_1,\qquad f_2\to p_1,\qquad p_1\to f_2,\qquad f_1\to p_1 .
$$

charge sequence 为

$$
(n,n,6,6)\rightarrow(n{+}1,n,5,6)\rightarrow(n{+}1,n{-}1,6,6)\rightarrow(n{+}1,n,5,6)\rightarrow(n,n,6,6).
$$

拓扑:同一个 ligand $p_1$ 先后被借走/还回两次(alternating 单 ligand)——$f_1$ 站点在整条链上一直处于激发($f^{n+1}$,"long"),$f_2$ 站点只在中段短暂变成 $f^{n-1}$("short"),两次跃迁都经由同一个 $p_1$ 空穴。

### 1.2 Spectator-δ 推导 → 逐态 einsum(最简单的精确形式)

每个顶点只作用在两个 block,其余 spectator block 给出 Kronecker-$\delta$(取自① §2.2–§2.5)。

第一步 $A_{11}$($p_1\to f_1$):

$$
\langle u_1,u_2,\rho_1,\rho_2|A_{11}|c,d,\Omega_1,\Omega_2\rangle
= s_1\,M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,\delta_{u_2,d}\,\delta_{\rho_2,\Omega_2}.
$$

第二步 $B_{21}$($f_2\to p_1$):

$$
\langle v_1,v_2,\sigma_1,\sigma_2|B_{21}|u_1,u_2,\rho_1,\rho_2\rangle
= s_2\,\big[M_{21}^{(n-1,6)}[u_2,\rho_1,v_2,\sigma_1]\big]^*\,\delta_{v_1,u_1}\,\delta_{\sigma_2,\rho_2}.
$$

第三步 $A_{21}$($p_1\to f_2$):

$$
\langle w_1,w_2,\tau_1,\tau_2|A_{21}|v_1,v_2,\sigma_1,\sigma_2\rangle
= s_3\,M_{21}^{(n-1,6)}[w_2,\tau_1,v_2,\sigma_1]\,\delta_{w_1,v_1}\,\delta_{\tau_2,\sigma_2}.
$$

第四步 $B_{11}$($f_1\to p_1$):

$$
\langle a,b,\Omega_1,\Omega_2|B_{11}|w_1,w_2,\tau_1,\tau_2\rangle
= s_4\,\big[M_{11}^{(n,6)}[w_1,\tau_1,a,\Omega_1]\big]^*\,\delta_{b,w_2}\,\delta_{\Omega_2,\tau_2}.
$$

合并全部 8 个 $\delta$(① §2.6)给出

$$
u_2=d,\qquad \rho_2=\sigma_2=\tau_2=\Omega_2,\qquad
v_1=w_1=u_1,\qquad \sigma_1=\Omega_1,\qquad w_2=b,
$$

只剩 4 个真正的自由求和指标,重命名为

$$
\alpha=u_1\in f_1^{n+1},\quad \beta=v_2\in f_2^{n-1},\quad \gamma=\rho_1\in p_1^5,\quad \gamma'=\tau_1\in p_1^5 .
$$

三个完整中间态被精确约化为

$$
S_1=(\alpha,d,\gamma,\Omega_2),\qquad
S_2=(\alpha,\beta,\Omega_1,\Omega_2),\qquad
S_3=(\alpha,b,\gamma',\Omega_2),
$$

故 resolvent 只依赖 $(\alpha,\gamma),(\alpha,\beta),(\alpha,\gamma')$:

$$
G_1[\alpha,\gamma]=G(\alpha,d,\gamma,\Omega_2),\quad
G_2[\alpha,\beta]=G(\alpha,\beta,\Omega_1,\Omega_2),\quad
G_3[\alpha,\gamma']=G(\alpha,b,\gamma',\Omega_2).
$$

化简后的逐态闭式(① §2.7)为

$$
\begin{aligned}
H_{T1}[a,b,c,d]
=\sigma_1\sum_{\alpha,\beta,\gamma,\gamma'}
&\big[M_{11}^{(n,6)}[\alpha,\gamma',a,\Omega_1]\big]^*\,G_3[\alpha,\gamma']\\
&\times M_{21}^{(n-1,6)}[b,\gamma',\beta,\Omega_1]\,G_2[\alpha,\beta]\\
&\times \big[M_{21}^{(n-1,6)}[d,\gamma,\beta,\Omega_1]\big]^*\,G_1[\alpha,\gamma]\\
&\times M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1].
\end{aligned}
$$

对应逐态 einsum(对每个 $f$ 中间态逐态求和,尚未做简并加速):

```python
H_type1_path1 = sigma_1 * np.einsum(
    'xha,xh,bhy,xy,dgy,xg,xgc->abcd',
    B11_n6, G3,
    A21_nm1_6, G2,
    B21_nm1_6, G1,
    A11_n6,
    optimize=True,
)
```

其中

```text
B11_n6[x,h,a]       = (M11^(n,6)[x,h,a,Ω1])*
A21_nm1_6[b,h,y]    =  M21^(n-1,6)[b,h,y,Ω1]
B21_nm1_6[d,g,y]    = (M21^(n-1,6)[d,g,y,Ω1])*
A11_n6[x,g,c]       =  M11^(n,6)[x,g,c,Ω1]
```

### 1.3 按简并性加速(degeneracy-level collapse)

代码中以局域 index 命名:$A=\alpha$、$B=\beta$、$G=\gamma$、$H=\gamma'$,逐态 einsum 等价为(② 索引约定下)

$$
H[a,b,c,d]=\sum_{A,B,G,H} M_4^*[A,H,a]\,G_{s3}[A,H]\,M_3[b,H,B]\,G_{s2}[A,B]\,M_2^*[d,G,B]\,G_{s1}[A,G]\,M_1[A,G,c].
$$

被按能级合并的两条 $f$ 轴(② §4.1):$A\in f^{n+1}\to L$(用 `levels_np1`)、$B\in f^{n-1}\to M$(用 `levels_nm1`)。`_sum_by_level` 作用在 einsum 的轴 0。级内 Gram 预求和:

```python
TA = _sum_by_level(np.einsum("AHa,AGc->AHaGc", M4.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("bHB,dGB->BbHdG", M3, M2.conj(), optimize=True), levels_nm1)
```

化简后最终 einsum(代码原样,逐字):

```python
return np.einsum("LHaGc,LG,LM,LH,MbHdG->abcd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

resolvent 三个因子全是 2-D:$G_{s1}[L,G]$、$G_{s2}[L,M]$、$G_{s3}[L,H]$(其中 $s_1,s_3$ 各激发 1 个 $f$ 站点,扣 $1\cdot E_0$;$s_2$ 激发 2 个 $f$ 站点,扣 $2\cdot E_0$)。本 process 单 pattern,故无 Pattern A/B 之分。

严格精确的依据:resolvent 只通过能量依赖中间态,对固定 $(L,M,\text{ligand})$ 它在能级内是常数;于是 $A$ 轴的级内 Gram 求和 factor 成 TA、$B$ 轴的级内 Gram 求和 factor 成 TB,合并能级与逐态求和给出 bit 等价的结果。

### 1.4 代码 / path 表 / sign

代码函数:`fexchange/fopt/contraction.py:269`(`_path_amplitude_process1`,主体 269–351)。

path 数与 pattern 判据:4 条,见 `_PROCESS1_PATHS`(`fexchange/fopt/contraction.py:700-705`),条目格式 `(f_first, ligand, sign_const)`,约定 `f_first = r1 = r4`、`f_other = r2 = r3`;本 process 为单 pattern(无 A/B)。

```python
_PROCESS1_PATHS: list[tuple[int, int, int]] = [
    (1, 1, 2),
    (1, 2, 26),
    (2, 1, 0),
    (2, 2, 24),
]
```

sign:$\text{sign}=(-1)^{10\,n_{\text{ele}}+\text{sign\_const}}$,`sign_const` 取自 path 表第三列(`_cluster_sign`,`contraction.py:757`)。

## 2. Path-2 / Process P2 — onion 单 ligand(经 p^4 中央态)

### 2.1 Path 与 charge sequence

取 ① §3.1 的代表 path1(Type 2)$A_{11}A_{21}B_{11}B_{21}$,四个顶点为

$$
h_1=A_{11},\qquad h_2=A_{21},\qquad h_3=B_{11},\qquad h_4=B_{21},
$$

即

$$
p_1\to f_1,\qquad p_1\to f_2,\qquad f_1\to p_1,\qquad f_2\to p_1.
$$

charge sequence 是

$$
(n,n,6,6)\rightarrow(n+1,n,5,6)\rightarrow(n+1,n+1,4,6)\rightarrow(n,n+1,5,6)\rightarrow(n,n,6,6).
$$

拓扑:同一个 ligand($p_1$)被连续借两次电子去激发两个 $f$ 站点,中央态含 $p_1^4$;然后两个 $f$ 各自把电子还回 $p_1$。这是"洋葱(onion)单 ligand"——光子轨迹 $p^6\to p^5(\gamma_1)\to p^4(\gamma_2)\to p^5(\gamma_3)\to p^6$,中间显式经过一个 $p^4$ 中央虚态(不像 $P_1$ 那样在每一步还回 ligand)。

### 2.2 Spectator-δ 推导 → 逐态 einsum(最简单的精确形式)

逐顶点写出 hopping matrix element 及其 spectator Kronecker-$\delta$(① §3.2–3.5)。

**第一步 $A_{11}$**(active sector $(f_1^n,p_1^6)\to(f_1^{n+1},p_1^5)$):

$$
\langle u_1,u_2,\rho_1,\rho_2|A_{11}|c,d,\Omega_1,\Omega_2\rangle
= s_1\,M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,\delta_{u_2,d}\,\delta_{\rho_2,\Omega_2}.
$$

spectator 钉住 $f_2,p_2$:$u_2=d,\ \rho_2=\Omega_2$;且 $u_1\in f_1^{n+1},\ \rho_1\in p_1^5$。

**第二步 $A_{21}$**(active sector $(f_2^n,p_1^5)\to(f_2^{n+1},p_1^4)$):

$$
\langle v_1,v_2,\sigma_1,\sigma_2|A_{21}|u_1,u_2,\rho_1,\rho_2\rangle
= s_2\,M_{21}^{(n,5)}[v_2,\sigma_1,u_2,\rho_1]\,\delta_{v_1,u_1}\,\delta_{\sigma_2,\rho_2}.
$$

spectator 钉住 $f_1,p_2$:$v_1=u_1,\ \sigma_2=\rho_2=\Omega_2$;且 $v_2\in f_2^{n+1},\ \sigma_1\in p_1^4$。结合得 $S_2=(u_1,v_2,\sigma_1,\Omega_2)$。

**第三步 $B_{11}$**(active sector $(f_1^{n+1},p_1^4)\to(f_1^n,p_1^5)$,用 $M_{11}^{(n,5)}$ 的 dagger):

$$
\langle w_1,w_2,\tau_1,\tau_2|B_{11}|v_1,v_2,\sigma_1,\sigma_2\rangle
= s_3\,\big[M_{11}^{(n,5)}[v_1,\sigma_1,w_1,\tau_1]\big]^*\,\delta_{w_2,v_2}\,\delta_{\tau_2,\sigma_2}.
$$

spectator 钉住 $f_2,p_2$:$w_2=v_2,\ \tau_2=\sigma_2=\Omega_2$;且 $w_1\in f_1^n,\ \tau_1\in p_1^5$。结合得 $S_3=(w_1,v_2,\tau_1,\Omega_2)$。

**第四步 $B_{21}$**(active sector $(f_2^{n+1},p_1^5)\to(f_2^n,p_1^6)$):

$$
\langle a,b,\Omega_1,\Omega_2|B_{21}|w_1,w_2,\tau_1,\tau_2\rangle
= s_4\,\big[M_{21}^{(n,6)}[w_2,\tau_1,b,\Omega_1]\big]^*\,\delta_{a,w_1}\,\delta_{\Omega_2,\tau_2}.
$$

spectator 钉住 $f_1,p_2$:$w_1=a,\ \tau_2=\Omega_2$;且 $w_2\in f_2^{n+1},\ \tau_1\in p_1^5$。

**合并所有 δ**(① §3.6)。全部 constraints:

$$
u_2=d,\quad \rho_2=\Omega_2,\qquad v_1=u_1,\quad \sigma_2=\rho_2=\Omega_2,
$$
$$
w_2=v_2,\quad \tau_2=\sigma_2=\Omega_2,\qquad w_1=a.
$$

剩余自由求和指标 $u_1,v_2,\rho_1,\sigma_1,\tau_1$,重命名为

$$
\alpha=u_1\in f_1^{n+1},\quad \beta=v_2\in f_2^{n+1},\quad
\gamma_1=\rho_1\in p_1^5,\quad \gamma_2=\sigma_1\in p_1^4,\quad \gamma_3=\tau_1\in p_1^5.
$$

约化后的三个中间态(注意 $S_2$ 显式保留 $p_1^4$——onion 并不删 ligand,而是经 $p^4$ 中央虚态,比 $P_1$ 多出一个自由 ligand 指标 $\gamma_2$):

$$
S_1=(\alpha,d,\gamma_1,\Omega_2),\qquad
S_2=(\alpha,\beta,\gamma_2,\Omega_2),\qquad
S_3=(a,\beta,\gamma_3,\Omega_2).
$$

化简后的逐态闭式(① §3.7):

$$
\begin{aligned}
H_{T2}[a,b,c,d]
=\sigma_2\!\!\sum_{\alpha,\beta,\gamma_1,\gamma_2,\gamma_3}\!\!
&\big[M_{21}^{(n,6)}[\beta,\gamma_3,b,\Omega_1]\big]^*\,G_3[\beta,\gamma_3]\\
&\times\big[M_{11}^{(n,5)}[\alpha,\gamma_2,a,\gamma_3]\big]^*\,G_2[\alpha,\beta,\gamma_2]\\
&\times M_{21}^{(n,5)}[\beta,\gamma_2,d,\gamma_1]\,G_1[\alpha,\gamma_1]\\
&\times M_{11}^{(n,6)}[\alpha,\gamma_1,c,\Omega_1],
\end{aligned}
$$

其中 resolvent

$$
G_1[\alpha,\gamma_1]=G(\alpha,d,\gamma_1,\Omega_2),\quad
G_2[\alpha,\beta,\gamma_2]=G(\alpha,\beta,\gamma_2,\Omega_2),\quad
G_3[\beta,\gamma_3]=G(a,\beta,\gamma_3,\Omega_2).
$$

对应逐态 einsum(① §3.7,尚未做能级合并,对每个 $f$ 中间态逐态求和):

```python
H_type2_path1 = sigma_2 * np.einsum(
    'yhb,yh,xzah,xyz,yzdg,xg,xgc->abcd',
    B21_n6, G3,
    B11_n5, G2,
    A21_n5, G1,
    A11_n6,
    optimize=True,
)
```

```text
B21_n6[y,h,b]   = (M21^(n,6)[y,h,b,Ω1])*
B11_n5[x,z,a,h] = (M11^(n,5)[x,z,a,h])*
A21_n5[y,z,d,g] =  M21^(n,5)[y,z,d,g]
A11_n6[x,g,c]   =  M11^(n,6)[x,g,c,Ω1]
```

index 对应:$x=\alpha,\ y=\beta,\ g=\gamma_1\in p_1^5,\ z=\gamma_2\in p_1^4,\ h=\gamma_3\in p_1^5$。

### 2.3 按简并性加速(degeneracy-level collapse)

依据(② §3、§4.2):resolvent $w(E)=-1/(E-kE_0)$ 只通过能量 $E$ 依赖中间态,不依赖其 $M$ 量子数。$P_2$ 的两个中间 $f$-sector **都是 $f^{n+1}$**,故被合并的两条 $f$ 轴 $A,B$ **都用 `levels_np1`**(以 `_sum_by_level` 作用在 einsum 的 axis 0):$A\in f^{n+1}\to L$,$B\in f^{n+1}\to M$。ligand 指标 $G,H,K\in p^5/p^4$ 与 doublet $a,b,c,d$ 是 spectator,不合并,保留到最终 einsum。

resolvent 因子(代码 `contraction.py:400–404, 424, 436`):

- $G_{s1}[L,G]$ 是 **2-D**(只激发一个 $f$ 站点 + 一个 $p^5$);
- $G_{s2}[L,M,H]$ 是 **3-D**(中央态同时激发两个 $f$ 站点,且多出 $p^4$ 光子 $\gamma_2=H$,这是 onion 经 $p^4$ 中央态的特征);
- $G_{s3}$ 是 **2-D**(只剩一个 $f$ 站点仍激发 + 一个 $p^5$):Pattern A 为 $G_{s3}[M,K]$,Pattern B 为 $G_{s3}[L,K]$。

**Pattern A**(FIFO,`r3==r1`,$V_3$ lower $V_1$ 的站点;$V_1\!\leftrightarrow\!V_3$ 共享 $A$、$V_2\!\leftrightarrow\!V_4$ 共享 $B$),级内 Gram 预求和与最终 einsum(代码原样,`contraction.py:429–431`):

```python
TA = _sum_by_level(np.einsum("AHbK,AGc->AHbKGc", M3.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("BKa,BHdG->BKaHdG", M4.conj(), M2, optimize=True), levels_np1)
return np.einsum("LHbKGc,LG,LMH,MK,MKaHdG->abcd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

**Pattern B**(LIFO,$V_3$ lower $V_2$ 的站点;$V_1\!\leftrightarrow\!V_4$ 共享 $A$、$V_2\!\leftrightarrow\!V_3$ 共享 $B$,$M_3^*/M_4^*$ 的 high-$f$ 索引相对 FIFO 互换 $A\!\leftrightarrow\!B$),代码原样(`contraction.py:439–441`):

```python
TA = _sum_by_level(np.einsum("AKa,AGc->AKaGc", M4.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("BHbK,BHdG->BHbKdG", M3.conj(), M2, optimize=True), levels_np1)
return np.einsum("LKaGc,LG,LMH,LK,MHbKdG->abcd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

其中投影顶点张量切片 $M_1=V_1[:,:,:,0]$、$M_2=V_2$、$M_3=V_3$、$M_4=V_4[:,:,:,0]$。

为何严格精确:对固定的 $(L,M,\text{ligand }G,H,K)$,每个 resolvent 因子只看级能量 $e_L,e_M$ 和显式的 ligand 能量,因此是级内常数,可由分配律 $\sum_A M[A]\,w(E_A)\,M'[A]=\sum_L w(e_L)\big(\sum_{A\in L}M[A]M'[A]\big)$ 提到求和外。于是对 $A\in L$ 与 $B\in M$ 的两个级内求和各自独立 factor,分别成为 $\mathrm{TA},\mathrm{TB}$ 的 per-level Gram 预求和(在乘 resolvent 之前完成)。resolvent 的维数(2-D/3-D)与此因子化无关——只要它耦合的轴在级内能量恒定。这是精确重排,非近似,结果 bit-exact。

### 2.4 代码 / path 表 / sign

- 代码函数:`fexchange/fopt/contraction.py:354`(`_path_amplitude_process2`,Pattern A 在 `:429–431`,Pattern B 在 `:439–441`)。
- path 数与 pattern 判据:`_PROCESS2_PATHS`(`contraction.py:709–718`)共 **8 条**,表项格式 `(r1, r2, r3, r4, ligand, sign_const)`;**`pattern = "A" iff r3 == r1`**($V_3$ lower $V_1$ 的站点),否则 `"B"`。同一 process+pattern 的所有 path 共享同一收缩拓扑与同一组被合并轴(均 `levels_np1`),只在输入顶点(site/ligand)与 cluster sign 上不同。
- sign:`_cluster_sign(n_ele, sign_const) = (-1)^(10*n_ele + sign_const)`(`contraction.py:757–758`),其中 `sign_const` 取自 `_PROCESS2_PATHS` 表的第 6 列。8 条 path 的 `sign_const` 依次为 `3, 4, 2, 3`(ligand=1)与 `27, 28, 26, 27`(ligand=2)。

## 3. Path-3 / Process P3 — alternating 双(交叉)ligand

### 3.1 Path 与 charge sequence

取自① §4.1(Type 3 path1 $A_{11}B_{21}A_{22}B_{12}$):四个顶点为

$$
h_1=A_{11},\qquad h_2=B_{21},\qquad h_3=A_{22},\qquad h_4=B_{12},
$$

即依次 $p_1\to f_1,\ f_2\to p_1,\ p_2\to f_2,\ f_1\to p_2$。charge sequence 是

$$
(n,n,6,6)\rightarrow(n+1,n,5,6)\rightarrow(n+1,n-1,6,6)\rightarrow(n+1,n,6,5)\rightarrow(n,n,6,6).
$$

拓扑:$r_X$ 粒子($f^{n+1}$)贯穿全链,$r_Y$ 空穴($f^{n-1}$)嵌在中段,两次 $p^5$ 空穴远足落在**不同的两个 ligand**(lig$_a$ 在 $s_1$、lig$_b$ 在 $s_3$)—— 这是 cross-ligand $K$-type path(alternating 双交叉 ligand,单 pattern)。

### 3.2 Spectator-δ 推导 → 逐态 einsum(最简单的精确形式)

cluster 基 $|i_1,i_2,\rho_1,\rho_2\rangle$,逐顶点写出 spectator Kronecker-δ(取自① §4.2–§4.5):

**$h_1=A_{11}$**(§4.2,active $(f_1,p_1)$,$p_2$ block 旁观):
$$
\langle u_1,u_2,\rho_1,\rho_2|A_{11}|c,d,\Omega_1,\Omega_2\rangle
= s_1\,M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,\delta_{u_2,d}\,\delta_{\rho_2,\Omega_2},
$$
钉住 $u_2=d,\ \rho_2=\Omega_2$;非零要求 $u_1\in f_1^{n+1},\ \rho_1\in p_1^5$。

**$h_2=B_{21}$**(§4.3,active $(f_2,p_1)$):
$$
\langle v_1,v_2,\sigma_1,\sigma_2|B_{21}|u_1,u_2,\rho_1,\rho_2\rangle
= s_2\big[M_{21}^{(n-1,6)}[u_2,\rho_1,v_2,\sigma_1]\big]^*\,\delta_{v_1,u_1}\,\delta_{\sigma_2,\rho_2},
$$
钉住 $v_1=u_1,\ \sigma_2=\Omega_2$;另 $\sigma_1=\Omega_1\in p_1^6$、$v_2\in f_2^{n-1}$。得 $S_2=(u_1,v_2,\Omega_1,\Omega_2)$。

**$h_3=A_{22}$**(§4.4,active $(f_2,p_2)$):
$$
\langle w_1,w_2,\tau_1,\tau_2|A_{22}|v_1,v_2,\sigma_1,\sigma_2\rangle
= s_3\,M_{22}^{(n-1,6)}[w_2,\tau_2,v_2,\sigma_2]\,\delta_{w_1,v_1}\,\delta_{\tau_1,\sigma_1},
$$
钉住 $w_1=v_1,\ \tau_1=\sigma_1=\Omega_1$;另 $w_2\in f_2^n,\ \tau_2\in p_2^5$。得 $S_3=(u_1,w_2,\Omega_1,\tau_2)$。

**$h_4=B_{12}$**(§4.5,active $(f_1,p_2)$):
$$
\langle a,b,\Omega_1,\Omega_2|B_{12}|w_1,w_2,\tau_1,\tau_2\rangle
= s_4\big[M_{12}^{(n,6)}[w_1,\tau_2,a,\Omega_2]\big]^*\,\delta_{b,w_2}\,\delta_{\Omega_1,\tau_1},
$$
钉住 $w_2=b,\ \tau_1=\Omega_1$;另 $w_1\in f_1^{n+1},\ \tau_2\in p_2^5$。

**合并所有 δ(§4.6):**
$$
u_2=d,\ \rho_2=\Omega_2;\quad v_1=u_1,\ \sigma_1=\Omega_1,\ \sigma_2=\Omega_2;\quad w_1=v_1=u_1,\ \tau_1=\sigma_1=\Omega_1;\quad w_2=b.
$$
剩余四个自由求和指标 $u_1,v_2,\rho_1,\tau_2$,重命名
$$
u_1=\alpha\in f_1^{n+1},\quad v_2=\beta\in f_2^{n-1},\quad \rho_1=\gamma\in p_1^5,\quad \tau_2=\delta\in p_2^5,
$$
约化后三个中间态为
$$
S_1=(\alpha,d,\gamma,\Omega_2),\qquad S_2=(\alpha,\beta,\Omega_1,\Omega_2),\qquad S_3=(\alpha,b,\Omega_1,\delta).
$$
与 process 1 结构同形(对 4 个 dummy 求和),唯一区别是两个 $p^5$ 光子指标 $\gamma,\delta$ **落在不同 ligand**。

**化简后的逐态闭式(① §4.7):**
$$
\begin{aligned}
H_{T3}[a,b,c,d]=\sigma_3\sum_{\alpha,\beta,\gamma,\delta}
&\big[M_{12}^{(n,6)}[\alpha,\delta,a,\Omega_2]\big]^*\,G_3[\alpha,\delta]\\
&\times M_{22}^{(n-1,6)}[b,\delta,\beta,\Omega_2]\,G_2[\alpha,\beta]\\
&\times\big[M_{21}^{(n-1,6)}[d,\gamma,\beta,\Omega_1]\big]^*\,G_1[\alpha,\gamma]\\
&\times M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1],
\end{aligned}
$$
其中 resolvent 切片
$$
G_1[\alpha,\gamma]=G(\alpha,d,\gamma,\Omega_2),\quad
G_2[\alpha,\beta]=G(\alpha,\beta,\Omega_1,\Omega_2),\quad
G_3[\alpha,\delta]=G(\alpha,b,\Omega_1,\delta).
$$
对应逐态 einsum(① §4.7,$x=\alpha,\ y=\beta,\ g=\gamma\in p_1^5,\ h=\delta\in p_2^5$):
```python
H_type3_path1 = sigma_3 * np.einsum(
    'xha,xh,bhy,xy,dgy,xg,xgc->abcd',
    B12_n6, G3,
    A22_nm1_6, G2,
    B21_nm1_6, G1,
    A11_n6,
    optimize=True,
)
```
```text
B12_n6[x,h,a]    = (M12^(n,6)[x,h,a,Ω2])*
A22_nm1_6[b,h,y] =  M22^(n-1,6)[b,h,y,Ω2]
B21_nm1_6[d,g,y] = (M21^(n-1,6)[d,g,y,Ω1])*
A11_n6[x,g,c]    =  M11^(n,6)[x,g,c,Ω1]
```
此式对 $f$ 中间态 $\alpha\in f^{n+1}$、$\beta\in f^{n-1}$ 逐态求和,尚未做简并加速。

### 3.3 按简并性加速(degeneracy-level collapse)

取自② §4.3。收缩拓扑与 $P_1$ **完全相同**(同一 einsum),仅两个 ligand 能量不同:$G_{s1}$ 用 lig$_a$、$G_{s3}$ 用 lig$_b$。

**被合并的 $f$ 轴:** $A\in f^{n+1}\to L$(用 `levels_np1`),$B\in f^{n-1}\to M$(用 `levels_nm1`);`_sum_by_level` 永远作用在 einsum 的 axis 0。

**TA、TB 级内 Gram 预求和**(单 pattern):
```
TA = sum_by_level( einsum("AHa,AGc->AHaGc", M4*, M1), levels_np1 )
TB = sum_by_level( einsum("bHB,dGB->BbHdG", M3 , M2*), levels_nm1 )
```

**化简后最终 einsum(代码原样):**
```
einsum("LHaGc,LG,LM,LH,MbHdG->abcd", TA,G_s1,G_s2,G_s3,TB)
```

**resolvent 维数:** 全 2-D —— `G_s1[L,G]`(lig$_a$ 的 $p^5$,即 $S_1$ 单 $f$ 站点 + lig$_a$ 光子)、`G_s2[L,M]`(两 $f$ 站点的中央态)、`G_s3[L,H]`(lig$_b$ 的 $p^5$,$S_3$ 单 $f$ 站点 + lig$_b$ 光子)。

**为何严格精确:** $w(E)=-1/(E-kE_0)$ 只通过能量依赖中间态,不依赖其 $M$ 量子数。对固定 $(L,M,\text{ligand 指标 }G,H)$,每个 resolvent 因子只看 $e_L,e_M$ 与显式 ligand 能量,故是级内常数;于是对 $A\in L$、$B\in M$ 的两次级内求和**各自独立 factor 出来**,分别成为 TA、TB(级内 Gram 预求和,在乘 resolvent 之前完成),即 $\sum_A M\,w(E_A)\,M'=\sum_L w(e_L)\big(\sum_{A\in L}M\,M'\big)$。这是精确重排,非近似。

### 3.4 代码 / path 表 / sign

- 代码函数:`_path_amplitude_process3`(`fexchange/fopt/contraction.py:445-519`);最终 einsum 在 `:517-519`,resolvent 切片在 `:487-492`。
- path 数与 pattern 判据:`_PROCESS3_PATHS`(`contraction.py:721-726`)共 **4 条 path,单 pattern**;表项为 `(f_first, lig_a, lig_b, sign_const)`:
  ```
  (1, 1, 2, 14),
  (1, 2, 1, 14),
  (2, 1, 2, 12),
  (2, 2, 1, 12),
  ```
  4 条 path 复用同一收缩与同一被合并轴(`levels_np1`/`levels_nm1`),只改输入顶点(`f_first` 选 $r_X$ 站点、`lig_a`/`lig_b` 选两次远足的 ligand)与 sign。
- sign:`_cluster_sign(n_ele, sign_const) = (-1)^(10*n_ele + sign_const)`(`contraction.py:757-758`),`sign_const` 取自上表第 4 列($f_{\text{first}}=1$ 时 14;$f_{\text{first}}=2$ 时 12),与能级合并正交。

## 4. Path-4 / Process P4 — onion 双 ligand crossed(交叉借还)

### 4.1 Path 与 charge sequence

取自① §5.1（Type 4 path1 $A_{11}A_{22}B_{12}B_{21}$）:

$$
h_1=A_{11},\qquad h_2=A_{22},\qquad h_3=B_{12},\qquad h_4=B_{21},
$$

即

$$
p_1\to f_1,\qquad p_2\to f_2,\qquad f_1\to p_2,\qquad f_2\to p_1.
$$

charge sequence:

$$
(n,n,6,6)\to(n{+}1,n,5,6)\to(n{+}1,n{+}1,5,5)\to(n,n{+}1,5,6)\to(n,n,6,6).
$$

中央态 $f_1^{n+1}f_2^{n+1}p_1^5p_2^5$ 四个 block 全部活跃。拓扑：两个 f-site 各从一个 ligand 借电子（$V_1,V_2$），再**交叉**地把电子还回**对方**借过的 ligand（$V_3$ 把 $f_1$ 还给 $p_2$、$V_4$ 把 $f_2$ 还给 $p_1$）——这就是 two-ligand crossed onion / ring path，全框架唯一含 4-D resolvent 的 process。

### 4.2 Spectator-δ 推导 → 逐态 einsum(最简单的精确形式)

取自① §5.2–5.6。从 §1 完整四站点链
$$
H[h]_{ab,cd}=\sum_{S_1,S_2,S_3}\langle F|h_4|S_3\rangle G(S_3)\langle S_3|h_3|S_2\rangle G(S_2)\langle S_2|h_2|S_1\rangle G(S_1)\langle S_1|h_1|I\rangle
$$
出发,逐个顶点写出 spectator Kronecker-$\delta$:

**$V_1=A_{11}$**（作用在 $f_1,p_1$,spectator $f_2,p_2$）:
$$
\langle u_1,u_2,\rho_1,\rho_2|A_{11}|c,d,\Omega_1,\Omega_2\rangle=s_1\,M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,\delta_{u_2,d}\,\delta_{\rho_2,\Omega_2}.
$$

**$V_2=A_{22}$**（作用在 $f_2,p_2$,spectator $f_1,p_1$）:
$$
\langle v_1,v_2,\sigma_1,\sigma_2|A_{22}|u_1,u_2,\rho_1,\rho_2\rangle=s_2\,M_{22}^{(n,6)}[v_2,\sigma_2,u_2,\rho_2]\,\delta_{v_1,u_1}\,\delta_{\sigma_1,\rho_1}.
$$

**$V_3=B_{12}$**（作用在 $f_1,p_2$,spectator $f_2,p_1$）:
$$
\langle w_1,w_2,\tau_1,\tau_2|B_{12}|v_1,v_2,\sigma_1,\sigma_2\rangle=s_3\,\big[M_{12}^{(n,6)}[v_1,\sigma_2,w_1,\tau_2]\big]^*\,\delta_{w_2,v_2}\,\delta_{\tau_1,\sigma_1}.
$$

**$V_4=B_{21}$**（作用在 $f_2,p_1$,spectator $f_1,p_2$）:
$$
\langle a,b,\Omega_1,\Omega_2|B_{21}|w_1,w_2,\tau_1,\tau_2\rangle=s_4\,\big[M_{21}^{(n,6)}[w_2,\tau_1,b,\Omega_1]\big]^*\,\delta_{a,w_1}\,\delta_{\Omega_2,\tau_2}.
$$

**合并 δ**（① §5.6）:
$$
u_2=d,\ \rho_2=\Omega_2;\quad v_1=u_1,\ \sigma_1=\rho_1;\quad w_2=v_2,\ \tau_1=\sigma_1=\rho_1,\ \tau_2=\Omega_2;\quad w_1=a.
$$

剩余四个自由求和指标:
$$
u_1=\alpha\in f_1^{n+1},\quad v_2=\beta\in f_2^{n+1},\quad \rho_1=\gamma\in p_1^5,\quad \sigma_2=\delta\in p_2^5,
$$

约化中间态
$$
S_1=(\alpha,d,\gamma,\Omega_2),\qquad S_2=(\alpha,\beta,\gamma,\delta),\qquad S_3=(a,\beta,\gamma,\Omega_2).
$$
$S_2$ 同时含 $f_1^{n+1},f_2^{n+1},p_1^5,p_2^5$ 四个 active index —— 这是真四站点中央态，不是两站点近似。

**化简后的逐态闭式**（① §5.7）:
$$
\begin{aligned}
H_{P4}[a,b,c,d]=\sigma\sum_{\alpha,\beta,\gamma,\delta}
&\big[M_{21}^{(n,6)}[\beta,\gamma,b,\Omega_1]\big]^*G_3[\beta,\gamma]\\
&\times\big[M_{12}^{(n,6)}[\alpha,\delta,a,\Omega_2]\big]^*G_2[\alpha,\beta,\gamma,\delta]\\
&\times M_{22}^{(n,6)}[\beta,\delta,d,\Omega_2]\,G_1[\alpha,\gamma]\\
&\times M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1],
\end{aligned}
$$
$$
G_1[\alpha,\gamma]=G(\alpha,d,\gamma,\Omega_2),\quad G_2[\alpha,\beta,\gamma,\delta]=G(\alpha,\beta,\gamma,\delta),\quad G_3[\beta,\gamma]=G(a,\beta,\gamma,\Omega_2).
$$
对应**逐态 einsum**（① §5.7，尚未做简并加速，对每个 $f$ 中间态 $\alpha,\beta$ 逐态求和）:
```python
H_type4_path1 = sigma * np.einsum(
    'ygb,yg,xha,xygh,yhd,xg,xgc->abcd',
    B21_n6, G3,
    B12_n6, G2,
    A22_n6, G1,
    A11_n6,
    optimize=True,
)
```
其中 `B21_n6[y,g,b]=(M21^(n,6)[y,g,b,Ω1])*`,`B12_n6[x,h,a]=(M12^(n,6)[x,h,a,Ω2])*`,`A22_n6[y,h,d]=M22^(n,6)[y,h,d,Ω2]`,`A11_n6[x,g,c]=M11^(n,6)[x,g,c,Ω1]`;index 对应 `x=α∈f1^(n+1), y=β∈f2^(n+1), g=γ∈p1^5, h=δ∈p2^5`。$G_1$ 是 1 f-site,$G_2$ 是 4-D 全激发分母,$G_3$ 是 1 f-site;$E_0$ 偏移 $s_1,s_3$ 减 $1\cdot E_0$,$s_2$ 减 $2\cdot E_0$。

### 4.3 按简并性加速(degeneracy-level collapse)

取自② §4.4。两个中间 $f$-sector **都是 $f^{n+1}$**,故 $\mathrm{TA},\mathrm{TB}$ 被合并的 $f$ 轴**都用 `levels_np1`**(即把逐态 einsum 的 $\alpha$/$\beta$ 轴按能级合并;`_sum_by_level` 作用在 axis 0)。ligand $G,H\in p^5$ 与 doublet $a,b,c,d$ 不合并,随 $\mathrm{TA}/\mathrm{TB}$ 保留到最终 einsum。这是全框架**唯一含 4-D resolvent** 的 process。

**Pattern A**（FIFO,$V_3$ 落 $V_1$ 的 site,$r_c=r_a$）:
```python
TA = _sum_by_level(np.einsum("AHb,AGc->AHbGc", M3.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("BGa,BHd->BGaHd", M4.conj(), M2, optimize=True), levels_np1)
return np.einsum("iHbGc,iG,ijGH,jG,jGaHd->abcd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

**Pattern B**（LIFO；与 A 同 δ-collapse、同四自由指标,$M_3^*/M_4^*$ 的 high-$f$ 标号 $A\leftrightarrow B$ 互换,$G_{s3}$ 索引随之换）:
```python
TA = _sum_by_level(np.einsum("AHa,AGc->AHaGc", M4.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("BGb,BHd->BGbHd", M3.conj(), M2, optimize=True), levels_np1)
return np.einsum("iHaGc,iG,ijGH,iH,jGbHd->abcd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

resolvent 各因子维数(`i=L`、`j=M` 为合并后能级索引):$G_{s1}[L,G]$ **2-D**(1 f-site)、$G_{s3}$ **2-D**(1 f-site;A 用 $[j,G]$ 即 $[M,\text{lig}_a]$,B 用 $[i,H]$ 即 $[L,\text{lig}_b]$)、$G_{s2}[A,B,G,H]\!\to\![i,j,G,H]$ 是 **4-D**($S_2$ 同时激发两个 $f$ 站点与两个 ligand $p^5$)。

严格精确的理由:resolvent 只通过能量依赖中间态,对固定 $(L,M,\text{ligand})$ 每个因子(含 4-D 的 $G_{s2}$)只看 $e_L,e_M$ 与显式 ligand 能量,是级内常数;于是对 $A\in L$、$B\in M$ 的两个级内 Gram 求和**各自 factor 出来**成 $\mathrm{TA}/\mathrm{TB}$（② §2 分配律、§3 多维论证）,resolvent 维数与该因子化无关。

### 4.4 代码 / path 表 / sign

- 代码函数:`fexchange/fopt/contraction.py:522-616`(`_path_amplitude_process4`,Pattern A `:597-605`、Pattern B `:606-615`)。
- path 数与 pattern 判据:8 条(`_PROCESS4_PATHS`,`contraction.py:730-739`),格式 `(r_a, lig_a, r_b, lig_b, r_c, lig_c, r_d, lig_d, sign_const)`;`pattern = "A" iff r_c == r_a`（$V_3$ 落 $V_1$ 的 site）,否则 `"B"`。
- sign:`sign = (-1)^(10*n_ele + sign_const)`（`_cluster_sign`,`contraction.py:757-758`）。8 条 path 的 `sign_const` 来自 path 表:Pattern A 行（`r_c==r_a`）取 `13`,Pattern B 行取 `15`:
  ```
  (1,1,2,2, 1,2,2,1, 13)  A      (1,1,2,2, 2,1,1,2, 15)  B
  (2,2,1,1, 1,2,2,1, 13)  A      (2,2,1,1, 2,1,1,2, 15)  B
  (1,2,2,1, 1,1,2,2, 15)  B      (1,2,2,1, 2,2,1,1, 15)  B
  (2,1,1,2, 1,1,2,2, 13)  A      (2,1,1,2, 2,2,1,1, 13)  A
  ```

## 5. Path-5 / Process P5 — onion 双 ligand uncrossed(各借各还)

### 5.1 Path 与 charge sequence

取自 ① §6.1：path1 的四个顶点为

$$
h_1=A_{11},\qquad h_2=A_{22},\qquad h_3=B_{11},\qquad h_4=B_{22},
$$

即 $p_1\to f_1,\ p_2\to f_2,\ f_1\to p_1,\ f_2\to p_2$。charge sequence 为

$$
(n,n,6,6)\to(n+1,n,5,6)\to(n+1,n+1,5,5)\to(n,n+1,6,5)\to(n,n,6,6),
$$

中央态 $f_1^{n+1}f_2^{n+1}p_1^5p_2^5$。这是 two-ligand uncrossed onion path：与 Type 4(crossed)不同,第三、四步 $B_{11},B_{22}$ 让每个 f-site 各自把电子还回**自己**借来的 ligand(各借各还),而非交叉还 $B_{12},B_{21}$。

### 5.2 Spectator-δ 推导 → 逐态 einsum(最简单的精确形式)

取自 ① §6.2–6.7。逐个顶点的 spectator Kronecker-δ(每个顶点钉住未参与的 block index):

**第一步 $A_{11}$**($p_1\to f_1$,active $f_1,p_1$,spectator $f_2,p_2$):

$$
\langle u_1,u_2,\rho_1,\rho_2|A_{11}|c,d,\Omega_1,\Omega_2\rangle
= s_1\,M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,\delta_{u_2,d}\,\delta_{\rho_2,\Omega_2},
$$

钉住 $u_2=d,\ \rho_2=\Omega_2$(且 $u_1\in f_1^{n+1},\ \rho_1\in p_1^5$)。

**第二步 $A_{22}$**($p_2\to f_2$,active $f_2,p_2$,spectator $f_1,p_1$):

$$
\langle v_1,v_2,\sigma_1,\sigma_2|A_{22}|u_1,u_2,\rho_1,\rho_2\rangle
= s_2\,M_{22}^{(n,6)}[v_2,\sigma_2,u_2,\rho_2]\,\delta_{v_1,u_1}\,\delta_{\sigma_1,\rho_1},
$$

钉住 $v_1=u_1,\ \sigma_1=\rho_1$(且 $v_2\in f_2^{n+1},\ \sigma_2\in p_2^5$)。至此四个 block 全部活跃。

**第三步 $B_{11}=A_{11}^\dagger$**($f_1\to p_1$,用 $\left[M_{11}^{(n,6)}\right]^*$,spectator $f_2,p_2$):

$$
\langle w_1,w_2,\tau_1,\tau_2|B_{11}|v_1,v_2,\sigma_1,\sigma_2\rangle
= s_3\left[M_{11}^{(n,6)}[v_1,\sigma_1,w_1,\tau_1]\right]^*\delta_{w_2,v_2}\,\delta_{\tau_2,\sigma_2},
$$

钉住 $w_2=v_2,\ \tau_2=\sigma_2,\ \tau_1=\Omega_1\in p_1^6$(且 $w_1\in f_1^n$)。

**第四步 $B_{22}=A_{22}^\dagger$**($f_2\to p_2$,用 $\left[M_{22}^{(n,6)}\right]^*$,spectator $f_1,p_1$):

$$
\langle a,b,\Omega_1,\Omega_2|B_{22}|w_1,w_2,\tau_1,\tau_2\rangle
= s_4\left[M_{22}^{(n,6)}[w_2,\tau_2,b,\Omega_2]\right]^*\delta_{a,w_1}\,\delta_{\Omega_1,\tau_1},
$$

钉住 $w_1=a,\ \tau_1=\Omega_1$(且 $w_2\in f_2^{n+1},\ \tau_2\in p_2^5$)。

**合并所有 δ**(§6.6)。约束为

$$
u_2=d,\ \rho_2=\Omega_2;\quad v_1=u_1,\ \sigma_1=\rho_1;\quad w_2=v_2,\ \tau_2=\sigma_2,\ \tau_1=\Omega_1;\quad w_1=a.
$$

剩余自由求和指标 $u_1,v_2,\rho_1,\sigma_2$,重命名

$$
u_1=\alpha\in f_1^{n+1},\quad v_2=\beta\in f_2^{n+1},\quad \rho_1=\gamma\in p_1^5,\quad \sigma_2=\delta\in p_2^5.
$$

**约化后的三个完整中间态**:

$$
S_1=(\alpha,d,\gamma,\Omega_2),\qquad
S_2=(\alpha,\beta,\gamma,\delta),\qquad
S_3=(a,\beta,\Omega_1,\delta).
$$

$S_1,S_2$ 与 Type 4 逐字相同;唯一区别在 $S_3$:Type 4 为 $(a,\beta,\gamma,\Omega_2)$($p_1$ 仍激发、$p_2$ 复原),Type 5 为 $(a,\beta,\Omega_1,\delta)$($p_1$ 复原、$p_2$ 仍激发)。$S_2$ 含四个 active index,故 Type 5 同样不是两站点中间态。

**化简后的逐态闭式**(§6.7,对每个 f 中间态逐态求和,尚未做简并加速):

$$
\begin{aligned}
H_{T5}[a,b,c,d]
= \sigma_5 \sum_{\alpha,\beta,\gamma,\delta}
&\left[M_{22}^{(n,6)}[\beta,\delta,b,\Omega_2]\right]^* G_3[\beta,\delta]\\
&\times \left[M_{11}^{(n,6)}[\alpha,\gamma,a,\Omega_1]\right]^* G_2[\alpha,\beta,\gamma,\delta]\\
&\times M_{22}^{(n,6)}[\beta,\delta,d,\Omega_2]\, G_1[\alpha,\gamma]\\
&\times M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1],
\end{aligned}
$$

其中 resolvent

$$
G_1[\alpha,\gamma]=G(\alpha,d,\gamma,\Omega_2),\quad
G_2[\alpha,\beta,\gamma,\delta]=G(\alpha,\beta,\gamma,\delta),\quad
G_3[\beta,\delta]=G(a,\beta,\Omega_1,\delta).
$$

$G_1$ 是 1 f-site($f_1$+$p_1$ 激发),$G_2$ 是 2 f-site 的 4-D 分母(两 f 两 p 全激发,与 Type 4 同形),$G_3$ 是 1 f-site($f_2$+$p_2$ 激发);$E_0$ 偏移 $s_1,s_3$ 减 $1\cdot E_0$,$s_2$ 减 $2\cdot E_0$。对应逐态 einsum:

```python
H_type5_path1 = sigma_5 * np.einsum(
    'yhb,yh,xga,xygh,yhd,xg,xgc->abcd',
    B22_n6, G3,
    B11_n6, G2,
    A22_n6, G1,
    A11_n6,
    optimize=True,
)
```

```text
B22_n6[y,h,b] = (M22^(n,6)[β,δ,b,Ω2])*    x = α ∈ f1^(n+1)
B11_n6[x,g,a] = (M11^(n,6)[α,γ,a,Ω1])*    y = β ∈ f2^(n+1)
A22_n6[y,h,d] =  M22^(n,6)[β,δ,d,Ω2]      g = γ ∈ p1^5
A11_n6[x,g,c] =  M11^(n,6)[α,γ,c,Ω1]      h = δ ∈ p2^5
```

对照 Type 4 einsum `'ygb,yg,xha,xygh,yhd,xg,xgc->abcd'`:唯一变化是 $B$-vertex 的 p-索引(Type 4 的 $B_{12}$ 用 $p_2{=}h$、$B_{21}$ 用 $p_1{=}g$;Type 5 的 $B_{11}$ 用 $p_1{=}g$、$B_{22}$ 用 $p_2{=}h$),随之 $G_3$ 从 $[y,g]$ 变为 $[y,h]$。返回顺序对调($A_{11}A_{22}B_{22}B_{11}$)给出 pattern B,结构同形($G_3\equiv G_1$)。

**数值验证(§6.8)**:按 §1 完整 12-重 cluster sum(逐步矩阵元 + spectator-δ 直接构造)对照化简后 einsum,随机复数 $M_{11},M_{22},G_1,G_2,G_3$ 下到机器精度相等:Pattern A `'yhb,yh,xga,xygh,yhd,xg,xgc->abcd'` max|full−einsum| ≈ 8e-15 → PASS;Pattern B `'xga,xg,yhb,xygh,yhd,xg,xgc->abcd'`($G_3\equiv G_1$) ≈ 8e-15 → PASS。

### 5.3 按简并性加速(degeneracy-level collapse)

取自 ② §4.5。两个中间 f-sector **都是 $f^{n+1}$** → TA、TB 都按 `levels_np1` 合并(`_sum_by_level` 作用在 einsum 轴 0);与 $P_4$ 共享 4-D 的 $S_2$,仅 $S_3$ 不同。被合并的轴是 f 多重态指标 $A,B\in f^{n+1}$(合并后记为级索引 $i=L,\ j=M$),ligand 轴 $G,H$ 与 doublet $a,b,c,d$ 不合并。

**Pattern A**(FIFO,native 输出轴 `->bacd`,与 ① §6.8 Pattern A 对应):

```python
TA = _sum_by_level(np.einsum("AGa,AGc->AGac", M3.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("BHb,BHd->BHbd", M4.conj(), M2, optimize=True), levels_np1)
return np.einsum("iGac,iG,ijGH,jH,jHbd->bacd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

**Pattern B**(LIFO,① §6.8 Pattern B,$G_3\equiv G_1$):

```python
TA = _sum_by_level(np.einsum("AGa,AGc->AGac", M4.conj(), M1, optimize=True), levels_np1)
TB = _sum_by_level(np.einsum("BHb,BHd->BHbd", M3.conj(), M2, optimize=True), levels_np1)
return np.einsum("iGac,iG,ijGH,iG,jHbd->abcd", TA, G_s1, G_s2, G_s3, TB, optimize=True)
```

resolvent 各因子维数:$G_{s1}[i,G]$ 与 $G_{s3}$(Pattern A 为 $G_{s3}[j,H]$、Pattern B 为 $G_{s3}[i,G]\equiv G_{s1}$)均为 **2-D**;$G_{s2}[i,j,G,H]$ 是 **4-D**($S_2$ 同时激发两 f 站点与两 ligand $p^5$,全框架与 $P_4$ 共有的 4-D resolvent)。

严格精确的依据:resolvent 只通过能量依赖中间态。对固定 $(L,M,\text{ligand}=G,H)$,每个 resolvent 因子只看级能量 $e_L,e_M$ 和显式 ligand 能量,在级内是常数;于是分配律 $\sum_A M\,w(E_A)\,M' = \sum_L w(e_L)\big(\sum_{A\in L}M\,M'\big)$ 把对 $A\in L,\ B\in M$ 的级内 Gram 求和各自 factor 成 TA/TB,在乘 resolvent 之前完成(`_sum_by_level`),与 resolvent 维数(2-D/4-D)无关。

### 5.4 代码 / path 表 / sign

- 代码函数:`/Users/lingzhi/Documents/Code/fexchange/fexchange/fopt/contraction.py:619-689`(`_path_amplitude_process5`)。
- path 表:`_PROCESS5_PATHS`(`contraction.py:745-754`),共 **8 条 path**,每条编码 $(r_a,\text{lig}_a,r_b,\text{lig}_b,r_c,\text{lig}_c,r_d,\text{lig}_d,\text{sign\_const})$;pattern 判据 **A iff $r_c=r_a$**(即 $V_3$ 降回 $V_1$ 的 f-site),否则 B。
- sign:`sign = (-1)^(10*n_ele + sign_const)`(`_cluster_sign`,`contraction.py:757-758`),`sign_const` 取自 `_PROCESS5_PATHS` 各行(8 条为 14,14,14,14,14,16,12,14),源自 `standards/fopt/fopt_path_type5.md` 的 paths 1..8。

---

## 6. 五类最终对照

| Type | path1 | 完整中间态 $S_1$ | $S_2$ | $S_3$ | 剩余求和指标 |
|---|---|---|---|---|---|
| 1 | $A_{11}B_{21}A_{21}B_{11}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\Omega_1,\Omega_2)$ | $(\alpha,b,\gamma',\Omega_2)$ | $\alpha,\beta,\gamma,\gamma'$ |
| 2 | $A_{11}A_{21}B_{11}B_{21}$ | $(\alpha,d,\gamma_1,\Omega_2)$ | $(\alpha,\beta,\gamma_2,\Omega_2)$ | $(a,\beta,\gamma_3,\Omega_2)$ | $\alpha,\beta,\gamma_1,\gamma_2,\gamma_3$ |
| 3 | $A_{11}B_{21}A_{22}B_{12}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\Omega_1,\Omega_2)$ | $(\alpha,b,\Omega_1,\delta)$ | $\alpha,\beta,\gamma,\delta$ |
| 4 | $A_{11}A_{22}B_{12}B_{21}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\gamma,\delta)$ | $(a,\beta,\gamma,\Omega_2)$ | $\alpha,\beta,\gamma,\delta$ |
| 5 | $A_{11}A_{22}B_{11}B_{22}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\gamma,\delta)$ | $(a,\beta,\Omega_1,\delta)$ | $\alpha,\beta,\gamma,\delta$ |

Type 4 与 Type 5 的 $S_1,S_2$ 相同,仅 $S_3$ 不同:Type 4(crossed)$S_3$ 留 $p_1{=}\gamma$;
Type 5(uncrossed,各借各还)$S_3$ 留 $p_2{=}\delta$。

> **结论**:L3 没有把四站点虚过程近似成两站点(五类皆然);它用每个 hopping 的 spectator Kronecker-δ
> **精确**消去不变的 block 指标。最终只剩 $H[a,b,c,d]$ 是因为外部 $P$-space 里 $p_1=p_1^6,\ p_2=p_2^6$
> 是固定闭壳层,而非虚过程中丢掉了 ligand。

## 7. 精确前提(degeneracy-level collapse)

级内若有真实 spread,级均 resolvent 才会偏离逐态;生产路径下 **级内 spread $\equiv 0$**,故 bit-exact:

1. **ED 路径**:`ion_ed` 给简并簇内每个成员同一 `e_group=mean(evals[lo:hi])`(`ion_ed.py:144-146,173`)。
2. **RS 路径**:中间态能量线性于 `coef_F0/F2/F4/F6/zeta`(`energy.py:69-78`),只依赖 $(\alpha,L,S,J)$、与 $M$ 无关 → 多重态 $2J+1$ 态逐比特相同。
3. **数值窗口**:$\texttt{EPS\_EIG\_CLUSTER}=10^{-10}\gg\texttt{EPS\_ZERO}=10^{-12}$;仅"非规范化喂入 + 真实 spread 落在 $(10^{-12},10^{-10})$"才会偏离/翻转近零守卫,当前无代码路径产生(审计注入 $9\times10^{-11}$ → $\sim1.1\times10^{-11}$ 偏差,证明前提 load-bearing 但不可达)。
4. **Graceful degradation**:若中间 sector 加 CEF 解除简并 → 更多更小的级(极限每态一级,合并=恒等)→ 仍精确,只是收缩比下降。

切级规则:`_energy_levels`(`contraction.py:76-104`)按"与本级首个成员之差 $>$ tol"开新级(gap-to-start,非 single-linkage),级直径 $\le$ tol,级能量取簇均值;与 `ion_ed._clusters`(`ion_ed.py:287-291`)逻辑一致。`_sum_by_level`(`:107-120`)= `np.take(order)` + `np.add.reduceat(starts)`;同一个 `levels` 对象同时供 resolvent 与分组,保证一致。

## 8. 数值与测试验证

- **结构化简(①)**:§N.2 各式经独立数值校验,完整 12 重 cluster sum vs 化简 einsum 到机器精度($\sim8\times10^{-15}$)。Type 5 的两个 pattern 在 §5.3 给出,记为 `'yhb,yh,xga,xygh,yhd,xg,xgc->abcd'`(A)与 `'xga,xg,yhb,xygh,yhd,xg,xgc->abcd'`(B,$G_3\equiv G_1$)。
- **简并加速(②)**:独立 oracle `tests/test_fopt_l2.py::_ps_process1..5`(`:404-455`,逐态四阶 einsum,无能级合并)。`test_level_collapse_matches_reference_per_path_for_exact_degeneracies`(多态简并级)与 `..._without_degeneracies`(级=单态、合并=恒等)均 `rtol=1e-12` 逐 process/逐 path 匹配。
- 真·多态简并级下,合并 vs 逐态最大差 $\sim10^{-16}$(P4-A 1.1e-16,P5-A 1.8e-16,P2-A 2.5e-16,P1 7.8e-16)。
- 全套:`python -m pytest tests/test_fopt_l2.py` → **13 passed**。

## 9. 与代码/标准的对应

| 文档 path | 代码 process | 函数 (contraction.py) | path 表 |
|---|---|---|---|
| Path-1 | P1 | `_path_amplitude_process1:269` | `_PROCESS1_PATHS:700` (4) |
| Path-2 | P2 (A/B) | `_path_amplitude_process2:354` | `_PROCESS2_PATHS:709` (8) |
| Path-3 | P3 | `_path_amplitude_process3:445` | `_PROCESS3_PATHS:721` (4) |
| Path-4 | P4 (A/B) | `_path_amplitude_process4:522` | `_PROCESS4_PATHS:730` (8) |
| Path-5 | P5 (A/B) | `_path_amplitude_process5:619` | `_PROCESS5_PATHS:745` (8) |

合并机制层:`build_L3`(`contraction.py:761`)在 `:798-799` 构建 `levels_np1/levels_nm1` 并下传各 process;`spin12_map` 把最终 `H[a,b,c,d]` 投到 pseudospin-1/2 exchange。
