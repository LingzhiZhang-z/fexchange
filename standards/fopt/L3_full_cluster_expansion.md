# L3 五类 path1 的完整四站点展开与 spectator-$\delta$ 化简

本文只做一件事：对五类 representative path 各写一个例子，从完整四站点中间态

$$
S_1=(u_1,u_2,\rho_1,\rho_2),\quad
S_2=(v_1,v_2,\sigma_1,\sigma_2),\quad
S_3=(w_1,w_2,\tau_1,\tau_2)
$$

开始，逐步写出每个 hopping matrix element 中的 spectator Kronecker $\delta$，然后把完整 cluster sum 化简成 L3 的 local-index einsum 形式。

---

## 0. 统一记号

四个 block 的顺序固定为

$$
(f_1,\ f_2,\ p_1,\ p_2).
$$

完整 cluster basis 写成

$$
|i_1,i_2,\rho_1,\rho_2\rangle
=
|f_1;i_1\rangle
\otimes
|f_2;i_2\rangle
\otimes
|p_1;\rho_1\rangle
\otimes
|p_2;\rho_2\rangle.
$$

外部低能态是

$$
|I\rangle=|c,d,\Omega_1,\Omega_2\rangle,
$$

$$
|F\rangle=|a,b,\Omega_1,\Omega_2\rangle,
$$

其中

$$
a,c\in f_1^n\ {\rm doublet},\qquad
b,d\in f_2^n\ {\rm doublet},
$$

$$
\Omega_1=p_1^6,\qquad
\Omega_2=p_2^6.
$$

定义

$$
A_{r\lambda}\equiv V_+(r,\lambda):p_\lambda\rightarrow f_r,
$$

$$
B_{r\lambda}\equiv V_-(r,\lambda)=A_{r\lambda}^{\dagger}:f_r\rightarrow p_\lambda.
$$

L2 forward vertex 定义为

$$
M_{r\lambda}^{(N_f,N_p)}
[u,\eta,v,\xi]
=
\langle f_r^{N_f+1},u;\ p_\lambda^{N_p-1},\eta|
A_{r\lambda}
|f_r^{N_f},v;\ p_\lambda^{N_p},\xi\rangle.
$$

所以 $B_{r\lambda}$ 的 matrix element 用 $M_{r\lambda}^{\dagger}$ 给出。例如

$$
\langle f_r^{N_f},v;\ p_\lambda^{N_p},\xi|
B_{r\lambda}
|f_r^{N_f+1},u;\ p_\lambda^{N_p-1},\eta\rangle
=
\left[
M_{r\lambda}^{(N_f,N_p)}
[u,\eta,v,\xi]
\right]^*.
$$

每一步还应乘 L3 的 cluster embedding sign

$$
s_k=s_{r\lambda}(\mathbf q_{k-1}),
$$

最终一条 path 的 sign 是

$$
\sigma=s_1s_2s_3s_4.
$$

下面推导主要关注 spectator-$\delta$ 如何把 full cluster sum 化成 local-index sum。sign 可在最后整体乘上。

---

## 1. 完整四站点 path-chain 公式

对任意一条四跳 path

$$
h=(h_1,h_2,h_3,h_4),
$$

其 contribution 是

$$
H[h]_{ab,cd}
=
\sum_{S_1,S_2,S_3}
\langle F|h_4|S_3\rangle
G(S_3)
\langle S_3|h_3|S_2\rangle
G(S_2)
\langle S_2|h_2|S_1\rangle
G(S_1)
\langle S_1|h_1|I\rangle .
$$

三个完整中间态是

$$
S_1=|u_1,u_2,\rho_1,\rho_2\rangle,
$$

$$
S_2=|v_1,v_2,\sigma_1,\sigma_2\rangle,
$$

$$
S_3=|w_1,w_2,\tau_1,\tau_2\rangle.
$$

也就是完整写为

$$
\begin{aligned}
H[h]_{ab,cd}
=&
\sum_{u_1,u_2,\rho_1,\rho_2}
\sum_{v_1,v_2,\sigma_1,\sigma_2}
\sum_{w_1,w_2,\tau_1,\tau_2}
\\
&\quad
\langle a,b,\Omega_1,\Omega_2|h_4|w_1,w_2,\tau_1,\tau_2\rangle
G(w_1,w_2,\tau_1,\tau_2)
\\
&\quad\times
\langle w_1,w_2,\tau_1,\tau_2|h_3|v_1,v_2,\sigma_1,\sigma_2\rangle
G(v_1,v_2,\sigma_1,\sigma_2)
\\
&\quad\times
\langle v_1,v_2,\sigma_1,\sigma_2|h_2|u_1,u_2,\rho_1,\rho_2\rangle
G(u_1,u_2,\rho_1,\rho_2)
\\
&\quad\times
\langle u_1,u_2,\rho_1,\rho_2|h_1|c,d,\Omega_1,\Omega_2\rangle .
\end{aligned}
$$

下面四个例子都是从这个式子出发。

---

# 2. Type 1 path1：$A_{11}B_{21}A_{21}B_{11}$

## 2.1 Path 与 charge sequence

$$
h_1=A_{11},\qquad
h_2=B_{21},\qquad
h_3=A_{21},\qquad
h_4=B_{11}.
$$

即

$$
p_1\to f_1,\qquad
f_2\to p_1,\qquad
p_1\to f_2,\qquad
f_1\to p_1.
$$

charge sequence 是

$$
(n,n,6,6)
\rightarrow
(n+1,n,5,6)
\rightarrow
(n+1,n-1,6,6)
\rightarrow
(n+1,n,5,6)
\rightarrow
(n,n,6,6).
$$

---

## 2.2 第一步 $A_{11}$

$$
A_{11}:p_1\rightarrow f_1.
$$

它只作用在 $f_1,p_1$，所以

$$
\begin{aligned}
&
\langle u_1,u_2,\rho_1,\rho_2|
A_{11}
|c,d,\Omega_1,\Omega_2\rangle
\\
&=
s_1\,
M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,
\delta_{u_2,d}\,
\delta_{\rho_2,\Omega_2}.
\end{aligned}
$$

非零条件：

$$
u_1\in f_1^{n+1},\qquad
\rho_1\in p_1^5,\qquad
u_2=d,\qquad
\rho_2=\Omega_2.
$$

---

## 2.3 第二步 $B_{21}$

$$
B_{21}:f_2\rightarrow p_1.
$$

它只作用在 $f_2,p_1$，所以

$$
\begin{aligned}
&
\langle v_1,v_2,\sigma_1,\sigma_2|
B_{21}
|u_1,u_2,\rho_1,\rho_2\rangle
\\
&=
s_2\,
\left[
M_{21}^{(n-1,6)}[u_2,\rho_1,v_2,\sigma_1]
\right]^*
\delta_{v_1,u_1}
\delta_{\sigma_2,\rho_2}.
\end{aligned}
$$

非零条件：

$$
v_2\in f_2^{n-1},\qquad
\sigma_1=\Omega_1\in p_1^6,
$$

$$
v_1=u_1,\qquad
\sigma_2=\rho_2.
$$

结合第一步：

$$
S_2=(u_1,v_2,\Omega_1,\Omega_2).
$$

---

## 2.4 第三步 $A_{21}$

$$
A_{21}:p_1\rightarrow f_2.
$$

它只作用在 $f_2,p_1$，所以

$$
\begin{aligned}
&
\langle w_1,w_2,\tau_1,\tau_2|
A_{21}
|v_1,v_2,\sigma_1,\sigma_2\rangle
\\
&=
s_3\,
M_{21}^{(n-1,6)}[w_2,\tau_1,v_2,\sigma_1]\,
\delta_{w_1,v_1}
\delta_{\tau_2,\sigma_2}.
\end{aligned}
$$

非零条件：

$$
w_2\in f_2^n,\qquad
\tau_1\in p_1^5,
$$

$$
w_1=v_1,\qquad
\tau_2=\sigma_2.
$$

结合前面：

$$
S_3=(u_1,w_2,\tau_1,\Omega_2).
$$

---

## 2.5 第四步 $B_{11}$

$$
B_{11}:f_1\rightarrow p_1.
$$

它只作用在 $f_1,p_1$，所以

$$
\begin{aligned}
&
\langle a,b,\Omega_1,\Omega_2|
B_{11}
|w_1,w_2,\tau_1,\tau_2\rangle
\\
&=
s_4\,
\left[
M_{11}^{(n,6)}[w_1,\tau_1,a,\Omega_1]
\right]^*
\delta_{b,w_2}
\delta_{\Omega_2,\tau_2}.
\end{aligned}
$$

非零条件：

$$
w_1\in f_1^{n+1},\qquad
\tau_1\in p_1^5,
$$

$$
w_2=b,\qquad
\tau_2=\Omega_2.
$$

---

## 2.6 所有 $\delta$ 合并

四步给出的全部 spectator constraints 是

$$
u_2=d,
\qquad
\rho_2=\Omega_2,
$$

$$
v_1=u_1,
\qquad
\sigma_1=\Omega_1,
\qquad
\sigma_2=\rho_2=\Omega_2,
$$

$$
w_1=v_1=u_1,
\qquad
\tau_2=\sigma_2=\Omega_2,
$$

$$
w_2=b.
$$

剩余自由求和指标：

$$
u_1,\quad v_2,\quad \rho_1,\quad \tau_1.
$$

重命名为

$$
u_1=\alpha\in f_1^{n+1},
$$

$$
v_2=\beta\in f_2^{n-1},
$$

$$
\rho_1=\gamma\in p_1^5,
$$

$$
\tau_1=\gamma'\in p_1^5.
$$

所以三个完整中间态被 $\delta$ 精确约束为

$$
S_1=(\alpha,d,\gamma,\Omega_2),
$$

$$
S_2=(\alpha,\beta,\Omega_1,\Omega_2),
$$

$$
S_3=(\alpha,b,\gamma',\Omega_2).
$$

---

## 2.7 化简后的 Type 1 公式

$$
\begin{aligned}
H_{T1}[a,b,c,d]
=
\sigma_1
\sum_{\alpha,\beta,\gamma,\gamma'}
&
\left[
M_{11}^{(n,6)}[\alpha,\gamma',a,\Omega_1]
\right]^*
G_3[\alpha,\gamma']
\\
&\times
M_{21}^{(n-1,6)}[b,\gamma',\beta,\Omega_1]
G_2[\alpha,\beta]
\\
&\times
\left[
M_{21}^{(n-1,6)}[d,\gamma,\beta,\Omega_1]
\right]^*
G_1[\alpha,\gamma]
\\
&\times
M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1].
\end{aligned}
$$

若低能 $f^n$ doublet 和 $p^6$ closed shell 能量取作 0，则

$$
G_1[\alpha,\gamma]
=
G(\alpha,d,\gamma,\Omega_2),
$$

$$
G_2[\alpha,\beta]
=
G(\alpha,\beta,\Omega_1,\Omega_2),
$$

$$
G_3[\alpha,\gamma']
=
G(\alpha,b,\gamma',\Omega_2).
$$

对应 einsum：

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

---

# 3. Type 2 path1：$A_{11}A_{21}B_{11}B_{21}$

## 3.1 Path 与 charge sequence

$$
h_1=A_{11},\qquad
h_2=A_{21},\qquad
h_3=B_{11},\qquad
h_4=B_{21}.
$$

即

$$
p_1\to f_1,\qquad
p_1\to f_2,\qquad
f_1\to p_1,\qquad
f_2\to p_1.
$$

charge sequence 是

$$
(n,n,6,6)
\rightarrow
(n+1,n,5,6)
\rightarrow
(n+1,n+1,4,6)
\rightarrow
(n,n+1,5,6)
\rightarrow
(n,n,6,6).
$$

中央态含

$$
p_1^4.
$$

---

## 3.2 第一步 $A_{11}$

同 Type 1 第一步：

$$
\begin{aligned}
&
\langle u_1,u_2,\rho_1,\rho_2|
A_{11}
|c,d,\Omega_1,\Omega_2\rangle
\\
&=
s_1\,
M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]\,
\delta_{u_2,d}\,
\delta_{\rho_2,\Omega_2}.
\end{aligned}
$$

非零条件：

$$
u_1\in f_1^{n+1},\qquad
\rho_1\in p_1^5,\qquad
u_2=d,\qquad
\rho_2=\Omega_2.
$$

---

## 3.3 第二步 $A_{21}$

$$
A_{21}:p_1\rightarrow f_2.
$$

此时 active sector 是

$$
(f_2^n,p_1^5)\rightarrow(f_2^{n+1},p_1^4).
$$

所以

$$
\begin{aligned}
&
\langle v_1,v_2,\sigma_1,\sigma_2|
A_{21}
|u_1,u_2,\rho_1,\rho_2\rangle
\\
&=
s_2\,
M_{21}^{(n,5)}[v_2,\sigma_1,u_2,\rho_1]\,
\delta_{v_1,u_1}
\delta_{\sigma_2,\rho_2}.
\end{aligned}
$$

非零条件：

$$
v_2\in f_2^{n+1},
\qquad
\sigma_1\in p_1^4,
$$

$$
v_1=u_1,
\qquad
\sigma_2=\rho_2=\Omega_2.
$$

结合第一步：

$$
S_2=(u_1,v_2,\sigma_1,\Omega_2).
$$

---

## 3.4 第三步 $B_{11}$

$$
B_{11}:f_1\rightarrow p_1.
$$

此时 active sector 是

$$
(f_1^{n+1},p_1^4)\rightarrow(f_1^n,p_1^5).
$$

用 $M_{11}^{(n,5)}$ 的 dagger：

$$
\begin{aligned}
&
\langle w_1,w_2,\tau_1,\tau_2|
B_{11}
|v_1,v_2,\sigma_1,\sigma_2\rangle
\\
&=
s_3\,
\left[
M_{11}^{(n,5)}[v_1,\sigma_1,w_1,\tau_1]
\right]^*
\delta_{w_2,v_2}
\delta_{\tau_2,\sigma_2}.
\end{aligned}
$$

非零条件：

$$
w_1\in f_1^n,
\qquad
\tau_1\in p_1^5,
$$

$$
w_2=v_2,
\qquad
\tau_2=\sigma_2=\Omega_2.
$$

结合前面：

$$
S_3=(w_1,v_2,\tau_1,\Omega_2).
$$

---

## 3.5 第四步 $B_{21}$

$$
B_{21}:f_2\rightarrow p_1.
$$

此时 active sector 是

$$
(f_2^{n+1},p_1^5)\rightarrow(f_2^n,p_1^6).
$$

所以

$$
\begin{aligned}
&
\langle a,b,\Omega_1,\Omega_2|
B_{21}
|w_1,w_2,\tau_1,\tau_2\rangle
\\
&=
s_4\,
\left[
M_{21}^{(n,6)}[w_2,\tau_1,b,\Omega_1]
\right]^*
\delta_{a,w_1}
\delta_{\Omega_2,\tau_2}.
\end{aligned}
$$

非零条件：

$$
w_2\in f_2^{n+1},
\qquad
\tau_1\in p_1^5,
$$

$$
w_1=a,
\qquad
\tau_2=\Omega_2.
$$

---

## 3.6 所有 $\delta$ 合并

全部 constraints 是

$$
u_2=d,\qquad \rho_2=\Omega_2,
$$

$$
v_1=u_1,\qquad \sigma_2=\rho_2=\Omega_2,
$$

$$
w_2=v_2,\qquad \tau_2=\sigma_2=\Omega_2,
$$

$$
w_1=a.
$$

剩余自由求和指标：

$$
u_1,\quad v_2,\quad \rho_1,\quad \sigma_1,\quad \tau_1.
$$

重命名为

$$
u_1=\alpha\in f_1^{n+1},
$$

$$
v_2=\beta\in f_2^{n+1},
$$

$$
\rho_1=\gamma_1\in p_1^5,
$$

$$
\sigma_1=\gamma_2\in p_1^4,
$$

$$
\tau_1=\gamma_3\in p_1^5.
$$

于是三个完整中间态是

$$
S_1=(\alpha,d,\gamma_1,\Omega_2),
$$

$$
S_2=(\alpha,\beta,\gamma_2,\Omega_2),
$$

$$
S_3=(a,\beta,\gamma_3,\Omega_2).
$$

注意这里 $S_2$ 中确实包含 $p_1^4$。因此 Type 2 并没有把 ligand 删掉，而是显式保留了 $p^4$ virtual state。

---

## 3.7 化简后的 Type 2 公式

$$
\begin{aligned}
H_{T2}[a,b,c,d]
=
\sigma_2
\sum_{\alpha,\beta,\gamma_1,\gamma_2,\gamma_3}
&
\left[
M_{21}^{(n,6)}[\beta,\gamma_3,b,\Omega_1]
\right]^*
G_3[\beta,\gamma_3]
\\
&\times
\left[
M_{11}^{(n,5)}[\alpha,\gamma_2,a,\gamma_3]
\right]^*
G_2[\alpha,\beta,\gamma_2]
\\
&\times
M_{21}^{(n,5)}[\beta,\gamma_2,d,\gamma_1]
G_1[\alpha,\gamma_1]
\\
&\times
M_{11}^{(n,6)}[\alpha,\gamma_1,c,\Omega_1].
\end{aligned}
$$

其中

$$
G_1[\alpha,\gamma_1]
=
G(\alpha,d,\gamma_1,\Omega_2),
$$

$$
G_2[\alpha,\beta,\gamma_2]
=
G(\alpha,\beta,\gamma_2,\Omega_2),
$$

$$
G_3[\beta,\gamma_3]
=
G(a,\beta,\gamma_3,\Omega_2).
$$

对应 einsum：

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

其中

```text
B21_n6[y,h,b]       = (M21^(n,6)[y,h,b,Ω1])*
B11_n5[x,z,a,h]     = (M11^(n,5)[x,z,a,h])*
A21_n5[y,z,d,g]     =  M21^(n,5)[y,z,d,g]
A11_n6[x,g,c]       =  M11^(n,6)[x,g,c,Ω1]
```

index 对应：

```text
x = α
y = β
g = γ1 ∈ p1^5
z = γ2 ∈ p1^4
h = γ3 ∈ p1^5
```

---

# 4. Type 3 path1：$A_{11}B_{21}A_{22}B_{12}$

## 4.1 Path 与 charge sequence

$$
h_1=A_{11},\qquad
h_2=B_{21},\qquad
h_3=A_{22},\qquad
h_4=B_{12}.
$$

即

$$
p_1\to f_1,\qquad
f_2\to p_1,\qquad
p_2\to f_2,\qquad
f_1\to p_2.
$$

charge sequence 是

$$
(n,n,6,6)
\rightarrow
(n+1,n,5,6)
\rightarrow
(n+1,n-1,6,6)
\rightarrow
(n+1,n,6,5)
\rightarrow
(n,n,6,6).
$$

这是 cross-ligand $K$-type path。

---

## 4.2 第一步 $A_{11}$

同 Type 1 第一步：

$$
\begin{aligned}
&
\langle u_1,u_2,\rho_1,\rho_2|
A_{11}
|c,d,\Omega_1,\Omega_2\rangle
\\
&=
s_1
M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]
\delta_{u_2,d}
\delta_{\rho_2,\Omega_2}.
\end{aligned}
$$

非零条件：

$$
u_1\in f_1^{n+1},
\qquad
\rho_1\in p_1^5,
$$

$$
u_2=d,
\qquad
\rho_2=\Omega_2.
$$

---

## 4.3 第二步 $B_{21}$

同 Type 1 第二步：

$$
\begin{aligned}
&
\langle v_1,v_2,\sigma_1,\sigma_2|
B_{21}
|u_1,u_2,\rho_1,\rho_2\rangle
\\
&=
s_2
\left[
M_{21}^{(n-1,6)}[u_2,\rho_1,v_2,\sigma_1]
\right]^*
\delta_{v_1,u_1}
\delta_{\sigma_2,\rho_2}.
\end{aligned}
$$

非零条件：

$$
v_2\in f_2^{n-1},
\qquad
\sigma_1=\Omega_1\in p_1^6,
$$

$$
v_1=u_1,
\qquad
\sigma_2=\Omega_2.
$$

所以

$$
S_2=(u_1,v_2,\Omega_1,\Omega_2).
$$

---

## 4.4 第三步 $A_{22}$

$$
A_{22}:p_2\rightarrow f_2.
$$

active sector 是

$$
(f_2^{n-1},p_2^6)\rightarrow(f_2^n,p_2^5).
$$

所以

$$
\begin{aligned}
&
\langle w_1,w_2,\tau_1,\tau_2|
A_{22}
|v_1,v_2,\sigma_1,\sigma_2\rangle
\\
&=
s_3
M_{22}^{(n-1,6)}[w_2,\tau_2,v_2,\sigma_2]
\delta_{w_1,v_1}
\delta_{\tau_1,\sigma_1}.
\end{aligned}
$$

非零条件：

$$
w_2\in f_2^n,
\qquad
\tau_2\in p_2^5,
$$

$$
w_1=v_1,
\qquad
\tau_1=\sigma_1=\Omega_1.
$$

所以

$$
S_3=(u_1,w_2,\Omega_1,\tau_2).
$$

---

## 4.5 第四步 $B_{12}$

$$
B_{12}:f_1\rightarrow p_2.
$$

active sector 是

$$
(f_1^{n+1},p_2^5)\rightarrow(f_1^n,p_2^6).
$$

所以

$$
\begin{aligned}
&
\langle a,b,\Omega_1,\Omega_2|
B_{12}
|w_1,w_2,\tau_1,\tau_2\rangle
\\
&=
s_4
\left[
M_{12}^{(n,6)}[w_1,\tau_2,a,\Omega_2]
\right]^*
\delta_{b,w_2}
\delta_{\Omega_1,\tau_1}.
\end{aligned}
$$

非零条件：

$$
w_1\in f_1^{n+1},
\qquad
\tau_2\in p_2^5,
$$

$$
w_2=b,
\qquad
\tau_1=\Omega_1.
$$

---

## 4.6 所有 $\delta$ 合并

constraints 是

$$
u_2=d,\qquad \rho_2=\Omega_2,
$$

$$
v_1=u_1,\qquad \sigma_1=\Omega_1,\qquad \sigma_2=\Omega_2,
$$

$$
w_1=v_1=u_1,\qquad \tau_1=\sigma_1=\Omega_1,
$$

$$
w_2=b.
$$

剩余自由求和指标：

$$
u_1,\quad v_2,\quad \rho_1,\quad \tau_2.
$$

重命名为

$$
u_1=\alpha\in f_1^{n+1},
$$

$$
v_2=\beta\in f_2^{n-1},
$$

$$
\rho_1=\gamma\in p_1^5,
$$

$$
\tau_2=\delta\in p_2^5.
$$

三个完整中间态是

$$
S_1=(\alpha,d,\gamma,\Omega_2),
$$

$$
S_2=(\alpha,\beta,\Omega_1,\Omega_2),
$$

$$
S_3=(\alpha,b,\Omega_1,\delta).
$$

---

## 4.7 化简后的 Type 3 公式

$$
\begin{aligned}
H_{T3}[a,b,c,d]
=
\sigma_3
\sum_{\alpha,\beta,\gamma,\delta}
&
\left[
M_{12}^{(n,6)}[\alpha,\delta,a,\Omega_2]
\right]^*
G_3[\alpha,\delta]
\\
&\times
M_{22}^{(n-1,6)}[b,\delta,\beta,\Omega_2]
G_2[\alpha,\beta]
\\
&\times
\left[
M_{21}^{(n-1,6)}[d,\gamma,\beta,\Omega_1]
\right]^*
G_1[\alpha,\gamma]
\\
&\times
M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1].
\end{aligned}
$$

其中

$$
G_1[\alpha,\gamma]
=
G(\alpha,d,\gamma,\Omega_2),
$$

$$
G_2[\alpha,\beta]
=
G(\alpha,\beta,\Omega_1,\Omega_2),
$$

$$
G_3[\alpha,\delta]
=
G(\alpha,b,\Omega_1,\delta).
$$

对应 einsum：

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

其中

```text
B12_n6[x,h,a]       = (M12^(n,6)[x,h,a,Ω2])*
A22_nm1_6[b,h,y]    =  M22^(n-1,6)[b,h,y,Ω2]
B21_nm1_6[d,g,y]    = (M21^(n-1,6)[d,g,y,Ω1])*
A11_n6[x,g,c]       =  M11^(n,6)[x,g,c,Ω1]
```

index 对应：

```text
x = α
y = β
g = γ ∈ p1^5
h = δ ∈ p2^5
```

---

# 5. Type 4 path1：$A_{11}A_{22}B_{12}B_{21}$

## 5.1 Path 与 charge sequence

$$
h_1=A_{11},\qquad
h_2=A_{22},\qquad
h_3=B_{12},\qquad
h_4=B_{21}.
$$

即

$$
p_1\to f_1,\qquad
p_2\to f_2,\qquad
f_1\to p_2,\qquad
f_2\to p_1.
$$

charge sequence 是

$$
(n,n,6,6)
\rightarrow
(n+1,n,5,6)
\rightarrow
(n+1,n+1,5,5)
\rightarrow
(n,n+1,5,6)
\rightarrow
(n,n,6,6).
$$

中央态是

$$
f_1^{n+1}f_2^{n+1}p_1^5p_2^5.
$$

这是 two-ligand crossed onion / ring path。

---

## 5.2 第一步 $A_{11}$

同前：

$$
\begin{aligned}
&
\langle u_1,u_2,\rho_1,\rho_2|
A_{11}
|c,d,\Omega_1,\Omega_2\rangle
\\
&=
s_1
M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]
\delta_{u_2,d}
\delta_{\rho_2,\Omega_2}.
\end{aligned}
$$

非零条件：

$$
u_1\in f_1^{n+1},
\qquad
\rho_1\in p_1^5,
$$

$$
u_2=d,
\qquad
\rho_2=\Omega_2.
$$

---

## 5.3 第二步 $A_{22}$

$$
A_{22}:p_2\rightarrow f_2.
$$

active sector 是

$$
(f_2^n,p_2^6)\rightarrow(f_2^{n+1},p_2^5).
$$

所以

$$
\begin{aligned}
&
\langle v_1,v_2,\sigma_1,\sigma_2|
A_{22}
|u_1,u_2,\rho_1,\rho_2\rangle
\\
&=
s_2
M_{22}^{(n,6)}[v_2,\sigma_2,u_2,\rho_2]
\delta_{v_1,u_1}
\delta_{\sigma_1,\rho_1}.
\end{aligned}
$$

非零条件：

$$
v_2\in f_2^{n+1},
\qquad
\sigma_2\in p_2^5,
$$

$$
v_1=u_1,
\qquad
\sigma_1=\rho_1.
$$

结合第一步：

$$
S_2=(u_1,v_2,\rho_1,\sigma_2).
$$

此时四个 block 都在中央态中活跃：

$$
f_1^{n+1},\quad f_2^{n+1},\quad p_1^5,\quad p_2^5.
$$

---

## 5.4 第三步 $B_{12}$

$$
B_{12}:f_1\rightarrow p_2.
$$

active sector 是

$$
(f_1^{n+1},p_2^5)\rightarrow(f_1^n,p_2^6).
$$

所以

$$
\begin{aligned}
&
\langle w_1,w_2,\tau_1,\tau_2|
B_{12}
|v_1,v_2,\sigma_1,\sigma_2\rangle
\\
&=
s_3
\left[
M_{12}^{(n,6)}[v_1,\sigma_2,w_1,\tau_2]
\right]^*
\delta_{w_2,v_2}
\delta_{\tau_1,\sigma_1}.
\end{aligned}
$$

非零条件：

$$
w_1\in f_1^n,
\qquad
\tau_2=\Omega_2\in p_2^6,
$$

$$
w_2=v_2,
\qquad
\tau_1=\sigma_1.
$$

所以

$$
S_3=(w_1,v_2,\rho_1,\Omega_2).
$$

---

## 5.5 第四步 $B_{21}$

$$
B_{21}:f_2\rightarrow p_1.
$$

active sector 是

$$
(f_2^{n+1},p_1^5)\rightarrow(f_2^n,p_1^6).
$$

所以

$$
\begin{aligned}
&
\langle a,b,\Omega_1,\Omega_2|
B_{21}
|w_1,w_2,\tau_1,\tau_2\rangle
\\
&=
s_4
\left[
M_{21}^{(n,6)}[w_2,\tau_1,b,\Omega_1]
\right]^*
\delta_{a,w_1}
\delta_{\Omega_2,\tau_2}.
\end{aligned}
$$

非零条件：

$$
w_2\in f_2^{n+1},
\qquad
\tau_1\in p_1^5,
$$

$$
w_1=a,
\qquad
\tau_2=\Omega_2.
$$

---

## 5.6 所有 $\delta$ 合并

constraints 是

$$
u_2=d,\qquad \rho_2=\Omega_2,
$$

$$
v_1=u_1,\qquad \sigma_1=\rho_1,
$$

$$
w_2=v_2,\qquad \tau_1=\sigma_1=\rho_1,\qquad \tau_2=\Omega_2,
$$

$$
w_1=a.
$$

剩余自由求和指标：

$$
u_1,\quad v_2,\quad \rho_1,\quad \sigma_2.
$$

重命名为

$$
u_1=\alpha\in f_1^{n+1},
$$

$$
v_2=\beta\in f_2^{n+1},
$$

$$
\rho_1=\gamma\in p_1^5,
$$

$$
\sigma_2=\delta\in p_2^5.
$$

三个完整中间态是

$$
S_1=(\alpha,d,\gamma,\Omega_2),
$$

$$
S_2=(\alpha,\beta,\gamma,\delta),
$$

$$
S_3=(a,\beta,\gamma,\Omega_2).
$$

这里 $S_2$ 明确含有四个 active indices。因此 Type 4 并不是两站点中间态；它只是最后把 $p_1,p_2$ 外部闭壳层收缩回去。

---

## 5.7 化简后的 Type 4 公式

$$
\begin{aligned}
H_{T4}[a,b,c,d]
=
\sigma_4
\sum_{\alpha,\beta,\gamma,\delta}
&
\left[
M_{21}^{(n,6)}[\beta,\gamma,b,\Omega_1]
\right]^*
G_3[\beta,\gamma]
\\
&\times
\left[
M_{12}^{(n,6)}[\alpha,\delta,a,\Omega_2]
\right]^*
G_2[\alpha,\beta,\gamma,\delta]
\\
&\times
M_{22}^{(n,6)}[\beta,\delta,d,\Omega_2]
G_1[\alpha,\gamma]
\\
&\times
M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1].
\end{aligned}
$$

其中

$$
G_1[\alpha,\gamma]
=
G(\alpha,d,\gamma,\Omega_2),
$$

$$
G_2[\alpha,\beta,\gamma,\delta]
=
G(\alpha,\beta,\gamma,\delta),
$$

$$
G_3[\beta,\gamma]
=
G(a,\beta,\gamma,\Omega_2).
$$

$G_1$ 是 1 f-site（$f_1$ 激发 + $p_1$ 激发），$G_2$ 是 2 f-site 的
4-D 分母（两 f 两 p 全激发），$G_3$ 是 1 f-site（$f_2$ 激发 +
$p_1$ 激发）。对应 $E_0$ 偏移：$s_1,s_3$ 减 $1\cdot E_0$，$s_2$ 减
$2\cdot E_0$。

对应 einsum：

```python
H_type4_path1 = sigma_4 * np.einsum(
    'ygb,yg,xha,xygh,yhd,xg,xgc->abcd',
    B21_n6, G3,
    B12_n6, G2,
    A22_n6, G1,
    A11_n6,
    optimize=True,
)
```

其中

```text
B21_n6[y,g,b]       = (M21^(n,6)[y,g,b,Ω1])*
B12_n6[x,h,a]       = (M12^(n,6)[x,h,a,Ω2])*
A22_n6[y,h,d]       =  M22^(n,6)[y,h,d,Ω2]
A11_n6[x,g,c]       =  M11^(n,6)[x,g,c,Ω1]
```

index 对应：

```text
x = α ∈ f1^(n+1)
y = β ∈ f2^(n+1)
g = γ ∈ p1^5
h = δ ∈ p2^5
```

---

# 6. Type 5 path1：$A_{11}A_{22}B_{11}B_{22}$

## 6.1 Path 与 charge sequence

$$
h_1=A_{11},\qquad
h_2=A_{22},\qquad
h_3=B_{11},\qquad
h_4=B_{22}.
$$

即

$$
p_1\to f_1,\qquad
p_2\to f_2,\qquad
f_1\to p_1,\qquad
f_2\to p_2.
$$

charge sequence 是

$$
(n,n,6,6)
\rightarrow
(n+1,n,5,6)
\rightarrow
(n+1,n+1,5,5)
\rightarrow
(n,n+1,6,5)
\rightarrow
(n,n,6,6).
$$

中央态是

$$
f_1^{n+1}f_2^{n+1}p_1^5p_2^5.
$$

这是 two-ligand uncrossed onion path：与 Type 4 不同，第三、四步
$B_{11},B_{22}$ 让每个 f-site 各自把电子还回**自己**借来的 ligand
（Type 4 是交叉还 $B_{12},B_{21}$）。

---

## 6.2 第一步 $A_{11}$

同 Type 4 第一步：

$$
\begin{aligned}
&
\langle u_1,u_2,\rho_1,\rho_2|
A_{11}
|c,d,\Omega_1,\Omega_2\rangle
\\
&=
s_1
M_{11}^{(n,6)}[u_1,\rho_1,c,\Omega_1]
\delta_{u_2,d}
\delta_{\rho_2,\Omega_2}.
\end{aligned}
$$

非零条件：

$$
u_1\in f_1^{n+1},
\qquad
\rho_1\in p_1^5,
$$

$$
u_2=d,
\qquad
\rho_2=\Omega_2.
$$

---

## 6.3 第二步 $A_{22}$

同 Type 4 第二步：

$$
A_{22}:p_2\rightarrow f_2.
$$

active sector 是

$$
(f_2^n,p_2^6)\rightarrow(f_2^{n+1},p_2^5).
$$

所以

$$
\begin{aligned}
&
\langle v_1,v_2,\sigma_1,\sigma_2|
A_{22}
|u_1,u_2,\rho_1,\rho_2\rangle
\\
&=
s_2
M_{22}^{(n,6)}[v_2,\sigma_2,u_2,\rho_2]
\delta_{v_1,u_1}
\delta_{\sigma_1,\rho_1}.
\end{aligned}
$$

非零条件：

$$
v_2\in f_2^{n+1},
\qquad
\sigma_2\in p_2^5,
$$

$$
v_1=u_1,
\qquad
\sigma_1=\rho_1.
$$

结合第一步：

$$
S_2=(u_1,v_2,\rho_1,\sigma_2).
$$

四个 block 全部活跃：

$$
f_1^{n+1},\quad f_2^{n+1},\quad p_1^5,\quad p_2^5.
$$

（到此与 Type 4 §5.2–5.3 一字不差。）

---

## 6.4 第三步 $B_{11}$

$$
B_{11}:f_1\rightarrow p_1.
$$

active sector 是

$$
(f_1^{n+1},p_1^5)\rightarrow(f_1^n,p_1^6).
$$

$B_{11}=A_{11}^{\dagger}$，matrix element 用 $\left[M_{11}^{(n,6)}\right]^*$；
vertex 作用在 $f_1,p_1$，spectator 是 $f_2,p_2$。所以

$$
\begin{aligned}
&
\langle w_1,w_2,\tau_1,\tau_2|
B_{11}
|v_1,v_2,\sigma_1,\sigma_2\rangle
\\
&=
s_3
\left[
M_{11}^{(n,6)}[v_1,\sigma_1,w_1,\tau_1]
\right]^*
\delta_{w_2,v_2}
\delta_{\tau_2,\sigma_2}.
\end{aligned}
$$

非零条件：

$$
w_1\in f_1^n,
\qquad
\tau_1=\Omega_1\in p_1^6,
$$

$$
w_2=v_2,
\qquad
\tau_2=\sigma_2.
$$

所以

$$
S_3=(w_1,v_2,\Omega_1,\sigma_2).
$$

与 Type 4 §5.4 的唯一区别：Type 4 第三步 $B_{12}$ 关掉 $p_2$、保留
$p_1=\rho_1$；这里 $B_{11}$ 关掉 $p_1$（$\tau_1=\Omega_1$）、保留
$p_2=\sigma_2$。

---

## 6.5 第四步 $B_{22}$

$$
B_{22}:f_2\rightarrow p_2.
$$

active sector 是

$$
(f_2^{n+1},p_2^5)\rightarrow(f_2^n,p_2^6).
$$

所以

$$
\begin{aligned}
&
\langle a,b,\Omega_1,\Omega_2|
B_{22}
|w_1,w_2,\tau_1,\tau_2\rangle
\\
&=
s_4
\left[
M_{22}^{(n,6)}[w_2,\tau_2,b,\Omega_2]
\right]^*
\delta_{a,w_1}
\delta_{\Omega_1,\tau_1}.
\end{aligned}
$$

非零条件：

$$
w_2\in f_2^{n+1},
\qquad
\tau_2\in p_2^5,
$$

$$
w_1=a,
\qquad
\tau_1=\Omega_1.
$$

---

## 6.6 所有 $\delta$ 合并

constraints 是

$$
u_2=d,\qquad \rho_2=\Omega_2,
$$

$$
v_1=u_1,\qquad \sigma_1=\rho_1,
$$

$$
w_2=v_2,\qquad \tau_2=\sigma_2,\qquad \tau_1=\Omega_1,
$$

$$
w_1=a.
$$

剩余自由求和指标：

$$
u_1,\quad v_2,\quad \rho_1,\quad \sigma_2.
$$

重命名为

$$
u_1=\alpha\in f_1^{n+1},
$$

$$
v_2=\beta\in f_2^{n+1},
$$

$$
\rho_1=\gamma\in p_1^5,
$$

$$
\sigma_2=\delta\in p_2^5.
$$

三个完整中间态是

$$
S_1=(\alpha,d,\gamma,\Omega_2),
$$

$$
S_2=(\alpha,\beta,\gamma,\delta),
$$

$$
S_3=(a,\beta,\Omega_1,\delta).
$$

$S_1,S_2$ 与 Type 4 逐字相同；只有 $S_3$ 不同：Type 4 是
$(a,\beta,\gamma,\Omega_2)$（$p_1$ 仍激发、$p_2$ 复原），这里是
$(a,\beta,\Omega_1,\delta)$（$p_1$ 复原、$p_2$ 仍激发）。
$S_2$ 同样含四个 active index，故 Type 5 也不是两站点中间态。

---

## 6.7 化简后的 Type 5 公式

$$
\begin{aligned}
H_{T5}[a,b,c,d]
=
\sigma_5
\sum_{\alpha,\beta,\gamma,\delta}
&
\left[
M_{22}^{(n,6)}[\beta,\delta,b,\Omega_2]
\right]^*
G_3[\beta,\delta]
\\
&\times
\left[
M_{11}^{(n,6)}[\alpha,\gamma,a,\Omega_1]
\right]^*
G_2[\alpha,\beta,\gamma,\delta]
\\
&\times
M_{22}^{(n,6)}[\beta,\delta,d,\Omega_2]
G_1[\alpha,\gamma]
\\
&\times
M_{11}^{(n,6)}[\alpha,\gamma,c,\Omega_1].
\end{aligned}
$$

其中

$$
G_1[\alpha,\gamma]
=
G(\alpha,d,\gamma,\Omega_2),
$$

$$
G_2[\alpha,\beta,\gamma,\delta]
=
G(\alpha,\beta,\gamma,\delta),
$$

$$
G_3[\beta,\delta]
=
G(a,\beta,\Omega_1,\delta).
$$

$G_1$ 是 1 f-site（$f_1$ 激发 + $p_1$ 激发），$G_2$ 是 2 f-site 的
4-D 分母（两 f 两 p 全激发，与 Type 4 同形），$G_3$ 是 1 f-site
（$f_2$ 激发 + $p_2$ 激发）。对应 $E_0$ 偏移：$s_1,s_3$ 减 $1\cdot E_0$，
$s_2$ 减 $2\cdot E_0$（同 Type 4）。

对应 einsum：

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

其中

```text
B22_n6[y,h,b]       = (M22^(n,6)[β,δ,b,Ω2])*
B11_n6[x,g,a]       = (M11^(n,6)[α,γ,a,Ω1])*
A22_n6[y,h,d]       =  M22^(n,6)[β,δ,d,Ω2]
A11_n6[x,g,c]       =  M11^(n,6)[α,γ,c,Ω1]
```

index 对应：

```text
x = α ∈ f1^(n+1)
y = β ∈ f2^(n+1)
g = γ ∈ p1^5
h = δ ∈ p2^5
```

（对照 Type 4 einsum `'ygb,yg,xha,xygh,yhd,xg,xgc->abcd'`：唯一变化是
$B$-vertex 的 p-索引——Type 4 的 $B_{12}$ 用 $p_2{=}h$、$B_{21}$ 用
$p_1{=}g$；Type 5 的 $B_{11}$ 用 $p_1{=}g$、$B_{22}$ 用 $p_2{=}h$。
随之 $G_3$ 从 Type 4 的 $[y,g]$ 变为 $[y,h]$。）

返回顺序对调（先还 $f_2$ 再还 $f_1$，即 $A_{11}A_{22}B_{22}B_{11}$）
得到 pattern B，结构同形，由对称性给出（与 Type 4 的 pattern A/B 关系
相同），在 `contraction.py` 的 Process-5 中实现。

### 6.8 数值验证记录

§6.2–6.7 的 δ-collapse、$S_1,S_2,S_3$、$G_1,G_2,G_3$ 及 einsum 串经
独立数值校验：按 §1 完整 12-重 cluster sum（用 §6.2–6.5 的逐步矩阵元
+ spectator-$\delta$ 直接构造）对照化简后的 einsum，随机复数
$M_{11},M_{22},G_1,G_2,G_3$ 下两者到机器精度相等。

- Pattern A `'yhb,yh,xga,xygh,yhd,xg,xgc->abcd'`：max|full−einsum| ≈ 8e-15 → PASS
- Pattern B `'xga,xg,yhb,xygh,yhd,xg,xgc->abcd'`（$G_3\equiv G_1$）：≈ 8e-15 → PASS

cluster sign（`fopt_path_type5.md` 的 8 条 path）由 `docs/fopt_l3_review_prompt.md §B`
规则的模拟器生成；该模拟器先对 Type 1–4 全部 24 条已知 path 复现其
`Total=10n+const`（$n=1$ 与 $n=3$ 均一致）后才用于 Type 5。

---

# 7. 五个 type 的最终对照

| Type | path1 | 完整中间态 $S_1$ | 完整中间态 $S_2$ | 完整中间态 $S_3$ | 剩余求和指标 |
|---|---|---|---|---|---|
| Type 1 | $A_{11}B_{21}A_{21}B_{11}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\Omega_1,\Omega_2)$ | $(\alpha,b,\gamma',\Omega_2)$ | $\alpha,\beta,\gamma,\gamma'$ |
| Type 2 | $A_{11}A_{21}B_{11}B_{21}$ | $(\alpha,d,\gamma_1,\Omega_2)$ | $(\alpha,\beta,\gamma_2,\Omega_2)$ | $(a,\beta,\gamma_3,\Omega_2)$ | $\alpha,\beta,\gamma_1,\gamma_2,\gamma_3$ |
| Type 3 | $A_{11}B_{21}A_{22}B_{12}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\Omega_1,\Omega_2)$ | $(\alpha,b,\Omega_1,\delta)$ | $\alpha,\beta,\gamma,\delta$ |
| Type 4 | $A_{11}A_{22}B_{12}B_{21}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\gamma,\delta)$ | $(a,\beta,\gamma,\Omega_2)$ | $\alpha,\beta,\gamma,\delta$ |
| Type 5 | $A_{11}A_{22}B_{11}B_{22}$ | $(\alpha,d,\gamma,\Omega_2)$ | $(\alpha,\beta,\gamma,\delta)$ | $(a,\beta,\Omega_1,\delta)$ | $\alpha,\beta,\gamma,\delta$ |

Type 4 与 Type 5 的 $S_1,S_2$ 相同，仅 $S_3$ 不同：Type 4（crossed）
$S_3$ 留 $p_1{=}\gamma$，Type 5（uncrossed，各借各还）$S_3$ 留
$p_2{=}\delta$。

关键结论：

$$
\boxed{
\text{L3 并没有把四站点虚过程近似成两站点虚过程（五类皆然）。}
}
$$

它做的是：

$$
\boxed{
\text{从完整四站点 sum 出发，用每个 hopping 的 spectator Kronecker }\delta
\text{ 精确消去不变的 block indices。}
}
$$

因此，Type 2 的 $p_1^4$，Type 4、Type 5 的 $p_1^5p_2^5$ 中央态都显式保留在求和与 denominator 里。

最终输出只有 $H[a,b,c,d]$，是因为外部 $P$-space 中

$$
p_1=p_1^6,\qquad p_2=p_2^6
$$

是固定闭壳层态；它不是因为虚过程中 ligand 被丢掉。
